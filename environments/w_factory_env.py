"""
W工厂生产调度系统 - 仿真环境核心
包含SimPy仿真逻辑和PettingZoo多智能体环境接口
"""

import simpy
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from .w_factory_config import *


SILENT_MODE = True  # 设置为True时，完全禁用调试输出

# =============================================================================
# 1. 数据结构定义 (Data Structures)
# =============================================================================

class Part:
    """零件类 - 表示生产中的一个零件"""
    def __init__(self, part_id: int, product_type: str, order_id: int, 
                 due_date: float, priority: int):
        self.part_id = part_id
        self.product_type = product_type
        self.order_id = order_id
        self.due_date = due_date
        self.priority = priority
        self.current_step = 0
        self.start_time = 0
        self.completion_time = None
        self.processing_history = []
        
    def get_current_station(self) -> Optional[str]:
        """获取当前需要加工的工作站"""
        route = get_route_for_product(self.product_type)
        if self.current_step < len(route):
            return route[self.current_step]["station"]
        return None
    
    def get_processing_time(self) -> float:
        """获取当前工序的加工时间"""
        route = get_route_for_product(self.product_type)
        if self.current_step < len(route):
            return route[self.current_step]["time"]
        return 0
    
    def is_completed(self) -> bool:
        """检查零件是否完成所有工序"""
        route = get_route_for_product(self.product_type)
        return self.current_step >= len(route)

class Order:
    """订单类"""
    def __init__(self, order_id: int, product: str, quantity: int, 
                 priority: int, due_date: float, arrival_time: float = 0):
        self.order_id = order_id
        self.product = product
        self.quantity = quantity
        self.priority = priority
        self.due_date = due_date
        self.arrival_time = arrival_time
        self.parts = []
        self.completed_parts = 0
        
    def create_parts(self) -> List[Part]:
        """为订单创建零件"""
        self.parts = []
        for i in range(self.quantity):
            part_id = self.order_id * 1000 + i
            part = Part(part_id, self.product, self.order_id, 
                       self.due_date, self.priority)
            part.start_time = self.arrival_time
            self.parts.append(part)
        return self.parts

# =============================================================================
# 2. SimPy仿真核心 (SimPy Simulation Core)
# =============================================================================

class WFactorySim:
    """W工厂仿真核心类 - 基于SimPy的离散事件仿真"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 定义智能体列表
        self.agents = [f"agent_{station}" for station in WORKSTATIONS.keys()]
        
        # 调试级别控制
        self.debug_level = self.config.get('debug_level', 'INFO')  # DEBUG, INFO, WARNING, ERROR
        
        # 训练模式标志，控制输出冗余度
        self._training_mode = self.config.get('training_mode', False)
        
        # 减少输出冗余
        if self._training_mode:
            self.debug_level = 'WARNING'
        
        # 仿真环境
        self.env = simpy.Environment()
        self.current_time = 0
        self.simulation_ended = False
        
        # 状态跟踪
        self.active_parts: List[Part] = []
        self.completed_parts: List[Part] = []
        self.orders: List[Order] = []
        
        # 资源和队列
        self.queues: Dict[str, simpy.Store] = {}
        self.resources: Dict[str, simpy.Resource] = {}
        self.equipment_status: Dict[str, Dict[str, Any]] = {}
        
        # 性能指标
        self._start_times: Dict[int, float] = {}
        self._end_times: Dict[int, float] = {}
        self._equipment_busy_time: Dict[str, float] = defaultdict(float)
        
        # 订单级别跟踪系统
        self.order_progress = {}  # 订单进度跟踪
        self.order_completion_times = {}  # 订单完成时间
        

        self.stats: Dict[str, Any] = {
            'last_completed_count': 0,
            'completed_orders': 0,
            'last_completed_orders': 0,
            'makespan': 0,
            'total_tardiness': 0,
            'max_tardiness': 0,
            'equipment_utilization': {},
            'queue_lengths': defaultdict(list),
            'total_parts': 0
        }
        
        # 用于快速查找下游工作站的缓存
        self._downstream_map = self._create_downstream_map()
        
        self._initialize_resources()
        self._initialize_orders()
    
    def reset(self):
        """重置仿真状态"""
        self.env = simpy.Environment()
        self.current_time = 0
        self.simulation_ended = False
        
        # 清空所有状态
        self.active_parts.clear()
        self.completed_parts.clear()
        self.orders.clear()
        self.queues.clear()
        self.resources.clear()
        self.equipment_status.clear()

        
        # 重置订单跟踪
        self.order_progress.clear()
        self.order_completion_times.clear()
        
        # 重新初始化
        self._initialize_resources()
        self._initialize_orders()
        
        # 完整重置stats字典
        self.stats = {
            'last_completed_count': 0,
            'completed_orders': 0,
            'last_completed_orders': 0,
            'makespan': 0,
            'total_tardiness': 0,
            'max_tardiness': 0,
            'equipment_utilization': {},
            'queue_lengths': defaultdict(list),
            'total_parts': 0
        }
    
    def _initialize_resources(self):
        """初始化设备资源和队列"""
        for station_name, station_config in WORKSTATIONS.items():
            # 创建SimPy资源（设备）
            capacity = station_config["count"]
            self.resources[station_name] = simpy.Resource(self.env, capacity=capacity)
            
            # 创建输入队列
            self.queues[station_name] = simpy.Store(self.env, capacity=QUEUE_CAPACITY)
            
            # 初始化设备状态
            self.equipment_status[station_name] = {
                'busy_count': 0,
                'total_busy_time': 0,
                'last_status_change': 0,
                'is_failed': False,
                'failure_end_time': 0,
                # 新增：用于精确统计并行设备的忙碌"面积"（机器-分钟）
                'busy_machine_time': 0.0,
                'last_event_time': 0.0,
            }
            
            # 启动设备处理进程
            self.env.process(self._equipment_process(station_name))
    
    def _initialize_orders(self):
        """初始化订单（支持课程学习）"""
        # 支持课程学习的订单缩放
        orders_scale = self.config.get('orders_scale', 1.0)
        time_scale = self.config.get('time_scale', 1.0)
        
        # 如果启用课程学习，按比例调整订单
        actual_orders = []
        if orders_scale < 1.0:
            # 计算需要多少个零件
            total_parts_needed = int(sum(o["quantity"] for o in BASE_ORDERS) * orders_scale)
            parts_added = 0
            
            # 优先选择不同产品类型的订单，保持多样性
            for order_data in BASE_ORDERS:
                if parts_added >= total_parts_needed:
                    break
                
                # 调整订单数量
                adjusted_quantity = min(order_data["quantity"], total_parts_needed - parts_added)
                if adjusted_quantity > 0:
                    adjusted_order = order_data.copy()
                    adjusted_order["quantity"] = adjusted_quantity
                    adjusted_order["due_date"] = order_data["due_date"] * time_scale  # 放宽时间限制
                    actual_orders.append(adjusted_order)
                    parts_added += adjusted_quantity
        else:
            actual_orders = BASE_ORDERS
        
        # 创建订单对象
        for i, order_data in enumerate(actual_orders):
            order = Order(
                order_id=i,
                product=order_data["product"],
                quantity=order_data["quantity"],
                priority=order_data["priority"],
                due_date=order_data["due_date"],
                arrival_time=0
            )
            self.orders.append(order)
            
            # 创建零件并添加到仿真中
            parts = order.create_parts()
            for part in parts:
                self.env.process(self._part_process(part))
                self.active_parts.append(part)
    
    def _part_process(self, part: Part):
        """零件的生产流程进程 - 简化版本"""
        # 将零件放入第一个工作站的队列
        first_station = part.get_current_station()
        if first_station:
            yield self.queues[first_station].put(part)
    

    
    def _equipment_process(self, station_name: str):
        """设备处理进程 - 处理设备故障等事件"""
        while True:
            if EQUIPMENT_FAILURE["enabled"]:
                # 随机设备故障
                failure_interval = np.random.exponential(
                    EQUIPMENT_FAILURE["mtbf_hours"] * 60
                )
                yield self.env.timeout(failure_interval)
                
                if random.random() < EQUIPMENT_FAILURE["failure_probability"]:
                    # 设备故障
                    self.equipment_status[station_name]['is_failed'] = True
                    repair_time = np.random.exponential(
                        EQUIPMENT_FAILURE["mttr_minutes"]
                    )
                    self.equipment_status[station_name]['failure_end_time'] = (
                        self.env.now + repair_time
                    )
                    
                    yield self.env.timeout(repair_time)
                    self.equipment_status[station_name]['is_failed'] = False
            else:
                # 静态训练模式：设备不会故障，只需要等待仿真结束
                yield self.env.timeout(SIMULATION_TIME)  # 等待仿真结束
    
    def _update_equipment_status(self, station_name: str, busy: bool):
        """更新设备状态"""
        status = self.equipment_status[station_name]
        current_time = self.env.now
        
        # 在变更 busy_count 之前，先结算从上次事件到现在的忙碌"面积"
        previous_busy_count = status['busy_count']
        last_event_time = status.get('last_event_time', 0.0)
        if current_time > last_event_time:
            elapsed = current_time - last_event_time
            # 积分：elapsed * previous_busy_count（机器-分钟）
            status['busy_machine_time'] = status.get('busy_machine_time', 0.0) + elapsed * previous_busy_count
        status['last_event_time'] = current_time
        
        if busy:
            status['busy_count'] += 1
        else:
            status['busy_count'] = max(0, status['busy_count'] - 1)
            # 累计忙碌时间
            if status['busy_count'] == 0:
                status['total_busy_time'] += (
                    current_time - status['last_status_change']
                )
        
        status['last_status_change'] = current_time
    
    def _create_downstream_map(self) -> Dict[str, str]:
        """创建下游工作站映射，用于快速查询"""
        downstream_map = {}
        routes = list(PRODUCT_ROUTES.values())
        for route in routes:
            for i in range(len(route) - 1):
                current_station = route[i]["station"]
                next_station = route[i+1]["station"]
                if current_station not in downstream_map:
                    downstream_map[current_station] = next_station
        return downstream_map
    
    def _update_order_progress(self):
        """更新订单进度跟踪"""
        for order in self.orders:
            completed_parts = sum(1 for part in self.completed_parts 
                                if part.order_id == order.order_id)
            progress_rate = completed_parts / order.quantity if order.quantity > 0 else 0
            self.order_progress[order.order_id] = progress_rate
            
            # 检查订单是否完成
            if progress_rate >= 1.0 and order.order_id not in self.order_completion_times:
                self.order_completion_times[order.order_id] = self.current_time
                self.stats['completed_orders'] += 1

    def _update_completion_stats(self, part: Part):
        """更新完成统计"""
        # 计算延期
        tardiness = max(0, part.completion_time - part.due_date)
        self.stats['total_tardiness'] += tardiness
        self.stats['max_tardiness'] = max(self.stats['max_tardiness'], tardiness)
        
        # 更新makespan
        self.stats['makespan'] = max(self.stats['makespan'], part.completion_time)
        
        self.stats['total_parts'] += 1
    
    def get_state_for_agent(self, agent_id: str) -> np.ndarray:
        """
        获取智能体的观测状态 - 全面增强版
        - 包含自身队列中前N个零件的详细信息
        - 包含下游工作站的队列信息
        """
        station_name = agent_id.replace("agent_", "")

        # --- 增强状态特征 ---
        state_features = []
        
        # 1. 自身设备状态 (1-2个特征)
        # 归一化设备忙碌数
        busy_ratio = self.equipment_status[station_name]['busy_count'] / WORKSTATIONS[station_name]['count']
        state_features.append(busy_ratio)
        # 设备是否故障
        state_features.append(1.0 if self.equipment_status[station_name]['is_failed'] else 0.0)

        # 2. 自身队列的详细信息 (N * 4个特征)
        queue = self.queues[station_name].items
        num_parts_to_observe = ENHANCED_OBS_CONFIG["top_n_parts"]
        
        for i in range(num_parts_to_observe):
            if i < len(queue):
                part = queue[i]
                # 特征a: 归一化剩余处理时间
                total_route_time = sum(step['time'] for step in get_route_for_product(part.product_type))
                remaining_time = sum(get_route_for_product(part.product_type)[step_idx]['time'] for step_idx in range(part.current_step, len(get_route_for_product(part.product_type))))
                state_features.append(remaining_time / (total_route_time + 1e-6))
                
                # 特征b: 归一化延期紧迫性
                time_to_due = part.due_date - self.env.now
                urgency = max(0, -time_to_due) / (ENHANCED_OBS_CONFIG["time_feature_normalization"] + 1e-6)
                state_features.append(min(urgency, 1.0))

                # 特征c: 优先级
                state_features.append(part.priority / 5.0) # 假设优先级最大为5

                # 特征d: 下一站是否是终点
                state_features.append(1.0 if part.current_step + 1 >= len(get_route_for_product(part.product_type)) else 0.0)

            else:
                # 如果队列中没有足够的零件，用0填充
                state_features.extend([0.0] * 4)

        # 3. 下游工作站信息 (1个特征)
        if ENHANCED_OBS_CONFIG["include_downstream_info"]:
            downstream_station = self._downstream_map.get(station_name)
            if downstream_station:
                downstream_queue_len = len(self.queues[downstream_station].items)
                normalized_downstream_queue = min(downstream_queue_len / QUEUE_CAPACITY, 1.0)
                state_features.append(normalized_downstream_queue)
            else:
                # 如果没有下游（如包装台），则用0填充
                state_features.append(0.0)
        
        return np.array(state_features, dtype=np.float32)

    def get_global_state(self) -> np.ndarray:
        """获取全局状态，拼接所有智能体的局部观察"""
        all_obs = []
        # 确保智能体顺序固定
        for agent_id in sorted(self.agents):
            all_obs.append(self.get_state_for_agent(agent_id))
        return np.concatenate(all_obs, axis=0)

    def step_with_actions(self, actions: Dict[str, int]) -> Dict[str, float]:
        """执行一步仿真，传入智能体动作"""
        # 记录执行前状态
        prev_completed = len(self.completed_parts)
        prev_total_steps = sum(part.current_step for part in self.active_parts)
        
        # 执行智能体动作
        actions_executed = 0
        for agent_id, action in actions.items():
            station_name = agent_id.replace("agent_", "")

            # V7 扩展动作空间 (0=IDLE, 1=处理第1个, 2=处理第2个, ...)
            if action > 0:
                part_index = action - 1
                if part_index < len(self.queues[station_name].items):
                    self._process_part_at_station(station_name, part_index=part_index)
                    actions_executed += 1
        
        # 推进仿真 - 减少步长以获得更精细的控制
        try:
            self.env.run(until=self.env.now + 1)  # 每步推进1分钟而不是5分钟
        except simpy.core.EmptySchedule:
            self.simulation_ended = True
        
        self.current_time = self.env.now
        
        # 计算奖励
        rewards = self.get_rewards(actions)
        
        # 训练模式下完全静默调试信息
        if not self._training_mode and self.debug_level == 'DEBUG':
            new_completed = len(self.completed_parts)
            new_total_steps = sum(part.current_step for part in self.active_parts)
            
            if new_completed > prev_completed or new_total_steps > prev_total_steps:
                print(f"🎯 进度更新: 完成零件 {prev_completed}->{new_completed}, 总工序 {prev_total_steps}->{new_total_steps}")
                print(f"   执行动作数: {actions_executed}, 奖励: {list(rewards.values())}")
        
        return rewards
    
    def _process_part_at_station(self, station_name: str, part_index: int = 0):
        """
        在指定工作站处理零件 - 增强版
        - 可以选择处理队列中的特定零件
        """
        if part_index >= len(self.queues[station_name].items):
            return # 索引越界，不处理
            
        # 获取队列中的特定零件
        part = self.queues[station_name].items[part_index]
        
        # 检查设备是否可用
        if self.equipment_status[station_name]['busy_count'] < WORKSTATIONS[station_name]['count']:
            # 从队列中移除零件
            self.queues[station_name].items.pop(part_index)
            
            # 启动处理进程
            self.env.process(self._execute_processing(station_name, part))
    
    def _execute_processing(self, station_name: str, part: Part):
        """执行零件加工"""
        # 请求设备资源
        with self.resources[station_name].request() as request:
            yield request
            
            # 更新设备状态
            self._update_equipment_status(station_name, busy=True)
            
            # 执行加工
            processing_time = part.get_processing_time()
            yield self.env.timeout(processing_time)
            
            # 更新设备状态
            self._update_equipment_status(station_name, busy=False)
            
            # 零件完成当前工序
            part.current_step += 1
            
            # 检查是否完成所有工序
            if part.is_completed():
                part.completion_time = self.env.now
                self.completed_parts.append(part)
                # 🔧 关键修复：从活跃零件列表中移除完成的零件
                if part in self.active_parts:
                    self.active_parts.remove(part)
                self._update_completion_stats(part)
            else:
                # 移动到下一个工作站
                next_station = part.get_current_station()
                if next_station:
                    yield self.queues[next_station].put(part)
    
    def get_rewards(self, actions: Dict[str, int]) -> Dict[str, float]:
        """简洁目标导向的奖励函数 - 5个核心组件"""
        rewards = {f"agent_{station}": 0.0 for station in WORKSTATIONS.keys()}
        
        # 获取基础统计数据
        total_required = sum(order.quantity for order in self.orders)
        current_completed = len(self.completed_parts)
        
        # 在奖励计算前更新一次统计数据
        current_completed = len(self.completed_parts)
        new_completed_parts = current_completed - self.stats.get('last_completed_count', 0)
        self.stats['last_completed_count'] = current_completed

        # === 1. 零件完成奖励 - 主要驱动力 ===
        if new_completed_parts > 0:
            part_reward = new_completed_parts * REWARD_CONFIG["part_completion_reward"]
            # 零件完成奖励主要给包装台（最后工序）
            rewards["agent_包装台"] += part_reward
        
        # === 2. 订单完成奖励 - 协调激励 ===
        new_completed_orders = self.stats['completed_orders'] - self.stats.get('last_completed_orders', 0)
        if new_completed_orders > 0:
            order_reward = new_completed_orders * REWARD_CONFIG["order_completion_reward"]
            # 订单完成奖励平分给所有智能体（鼓励协作）
            for agent_id in rewards:
                rewards[agent_id] += order_reward / len(WORKSTATIONS)
            self.stats['last_completed_orders'] = self.stats['completed_orders']

        # === 3. 新增：持续时间压力惩罚 (Continuous Time Pressure Penalty) ===
        continuous_lateness_penalty = 0
        for part in self.active_parts:
            if self.current_time > part.due_date:
                # 零件已延期，施加持续惩罚
                continuous_lateness_penalty += REWARD_CONFIG["continuous_lateness_penalty"]

        if continuous_lateness_penalty < 0:
            # 将惩罚平分给所有智能体，提供持续的负向反馈
            for agent_id in rewards:
                rewards[agent_id] += continuous_lateness_penalty / len(WORKSTATIONS)
        
        # === 4. 闲置惩罚与工作激励 (Bug修复版) ===
        # 奖励逻辑基于智能体“动作”，而非“状态”，杜绝躺平漏洞
        for agent_id, action in actions.items():
            station_name = agent_id.replace("agent_", "")
            work_is_available = len(self.queues[station_name].items) > 0

            if action > 0:  # 智能体选择“工作”
                if work_is_available:
                    # 当有工作时选择工作，给予奖励
                    rewards[agent_id] += REWARD_CONFIG["work_bonus"]
            else:  # 智能体选择“闲置” (action == 0)
                if work_is_available:
                    # 当有工作时选择闲置，给予惩罚
                    rewards[agent_id] += REWARD_CONFIG["idle_penalty"]
        
        # === 5. 终局完成率奖励/惩罚 - 全局目标 ===
        if self.is_done():
            completion_rate = (current_completed / total_required) * 100 if total_required > 0 else 0
            
            # --- 终局奖励/惩罚组件 ---
            final_reward_component = 0
            
            # 组件a: 完成率 & 完工大奖
            if completion_rate >= 100:
                final_reward_component += 100 * REWARD_CONFIG["final_completion_bonus_per_percent"]
                # 发放巨额的“完工大奖”
                final_reward_component += REWARD_CONFIG.get("final_all_parts_completion_bonus", 500.0)
            else:
                incomplete_percent = 100 - completion_rate
                final_reward_component += incomplete_percent * REWARD_CONFIG["final_incompletion_penalty_per_percent"]
            
            # 组件b: 延期 (Tardiness) - 综合计算所有订单
            total_tardiness = 0
            for order in self.orders:
                if order.order_id in self.order_completion_times:
                    completion_time = self.order_completion_times[order.order_id]
                    total_tardiness += max(0, completion_time - order.due_date)
                else:
                    # 对于未完成的订单，延期时间从截止日期算到仿真结束
                    total_tardiness += max(0, self.current_time - order.due_date)
            
            final_reward_component += total_tardiness * REWARD_CONFIG["final_tardiness_penalty"]
            
            # --- 将总的终局奖励/惩罚平分 ---
            for agent_id in rewards:
                rewards[agent_id] += final_reward_component / len(WORKSTATIONS)
        
        # 🔧 更新统计（为下次计算准备）
        self._update_order_progress()
        
        return rewards
    
    def is_done(self) -> bool:
        """检查仿真是否结束 - 优先任务完成，时间作为备用条件"""
        # 优先检查任务完成，而不是时间耗尽
        
        # 条件1: 所有订单完成 (主要完成条件)
        total_required = sum(order.quantity for order in self.orders)
        if len(self.completed_parts) >= total_required:
            if not hasattr(self, '_completion_logged'):
                # 训练模式下完全静默
                if not SILENT_MODE and not self._training_mode:
                    print(f"🎉 所有订单完成! 完成{len(self.completed_parts)}/{total_required}个零件，用时{self.current_time:.1f}分钟")
                self._completion_logged = True
            return True
        
        # 条件2: 手动结束仿真
        if self.simulation_ended:
            return True
        
        # 条件3: 时间耗尽 (备用条件，增加时间限制)
        # 给智能体更多时间完成任务，避免总是超时截断
        max_time = SIMULATION_TIME * 2.0  # 从1.5增加到2.0，给更充足的时间
        if self.current_time >= max_time:
            if not hasattr(self, '_timeout_logged'):
                # 训练模式下完全静默
                if not SILENT_MODE and not self._training_mode:
                    print(f"⏰ 时间耗尽! 完成{len(self.completed_parts)}/{total_required}个零件，用时{self.current_time:.1f}分钟")
                self._timeout_logged = True
            return True
        
        return False
    
    def get_final_stats(self) -> Dict[str, Any]:
        """获取最终统计结果，修复设备利用率计算异常"""
        # 关键修复：强制结算所有设备的最终忙碌时间
        for station_name, status in self.equipment_status.items():
            # 结算从 last_event_time 到当前时间的忙碌面积
            if self.current_time > status.get('last_event_time', 0.0):
                elapsed = self.current_time - status.get('last_event_time', 0.0)
                busy_count = status.get('busy_count', 0)
                status['busy_machine_time'] = status.get('busy_machine_time', 0.0) + elapsed * busy_count
                status['last_event_time'] = self.current_time
            
            # 计算该工作站的设备利用率
            capacity = WORKSTATIONS[station_name]['count']
            if self.current_time > 0 and capacity > 0:
                utilization = status.get('busy_machine_time', 0.0) / (self.current_time * capacity)
            else:
                utilization = 0.0
            self.stats['equipment_utilization'][station_name] = utilization
        
        # 更可靠的平均利用率计算
        util_values = list(self.stats['equipment_utilization'].values())
        if util_values:
            mean_utilization = float(np.mean(util_values))
                # 移除调试信息，保持训练日志简洁
            if mean_utilization < 0.001 and len(self.completed_parts) > 0:
                # 静默处理异常情况，避免日志冗余
                pass
        else:
            mean_utilization = 0.0
        
        # 新增：计算延期统计
        total_tardiness = 0
        late_orders_count = 0
        for order in self.orders:
            if order.order_id in self.order_completion_times:
                completion_time = self.order_completion_times[order.order_id]
                if completion_time > order.due_date:
                    tardiness = completion_time - order.due_date
                    total_tardiness += tardiness
                    late_orders_count += 1
        
        # 关键修复：正确计算makespan，解决1200分钟显示问题
        total_required = sum(order.quantity for order in self.orders)
        
        if len(self.completed_parts) == total_required:
            # 所有零件都完成了，makespan是最后一个零件的完成时间
            if self.completed_parts:
                makespan = max(part.completion_time for part in self.completed_parts if part.completion_time is not None)
            else:
                makespan = self.current_time
        else:
            # 关键修复：未完成所有零件时，显示最后完成零件的时间
            if self.completed_parts:
                # 如果有零件完成，显示最后完成零件的时间
                makespan = max(part.completion_time for part in self.completed_parts if part.completion_time is not None)
            else:
                # 关键：如果没有零件完成，显示0而不是1200
                makespan = 0.0
            self.stats['timeout_occurred'] = True
            self.stats['incomplete_parts'] = total_required - len(self.completed_parts)
        
        # 更新统计字段
        self.stats['mean_utilization'] = mean_utilization
        self.stats['total_tardiness'] = total_tardiness
        self.stats['total_parts'] = len(self.completed_parts)
        self.stats['makespan'] = makespan
        
        return self.stats

# =============================================================================
# 3. PettingZoo多智能体环境接口 (PettingZoo Multi-Agent Environment)
# =============================================================================

class WFactoryEnv(ParallelEnv):
    """W工厂多智能体强化学习环境 - 基于PettingZoo"""
    
    metadata = {
        "render_modes": ["human"],
        "name": "w_factory_v1",
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config if config else {}
        self.sim = WFactorySim(self.config)
        self.agents = self.sim.agents
        self.possible_agents = self.sim.agents
        
        # 新增全局状态空间
        self._setup_spaces()
        obs_shape = self._get_obs_shape()
        num_agents = len(self.agents)
        self.global_state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape[0] * num_agents,), dtype=np.float32)
        
        self.max_steps = self.sim.config.get("MAX_SIM_STEPS", 1500)
        self.step_count = 0
        self.render_mode = None
    
    # 重写observation_space和action_space方法
    def observation_space(self, agent: str = None):
        return self._observation_spaces[agent]
    
    def action_space(self, agent: str = None):
        return self._action_spaces[agent]
        
    def _get_obs_shape(self) -> Tuple[int,]:
        # 创建一个临时的、功能齐全的仿真实例来获取状态维度
        temp_sim = WFactorySim(self.config)
        temp_sim.reset()
        # 假设所有智能体的观测空间相同
        agent_id = temp_sim.agents[0]
        obs = temp_sim.get_state_for_agent(agent_id)
        return obs.shape

    def _setup_spaces(self):
        obs_shape = self._get_obs_shape()
        self._observation_spaces = {
            agent: gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
            )
            for agent in self.agents
        }
        action_size = ACTION_CONFIG_ENHANCED["action_space_size"]
        self._action_spaces = {agent: gym.spaces.Discrete(action_size) for agent in self.agents}
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.sim.reset()
        self.step_count = 0
        self.agents = self.possible_agents[:]
        
        self.observations = {agent: self.sim.get_state_for_agent(agent) for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # 在info中添加全局状态
        global_state = self.sim.get_global_state()
        for agent_id in self.agents:
            self.infos[agent_id]['global_state'] = global_state
            
        return self.observations, self.infos
    
    def step(self, actions: Dict[str, int]):
        """执行一步"""
        if not self.sim:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # 执行仿真步骤
        rewards = self.sim.step_with_actions(actions)
        self.step_count += 1
        
        # 获取新的观测
        observations = {
            agent: self.sim.get_state_for_agent(agent)
            for agent in self.agents
        }
        
        # 检查是否结束
        terminations = {agent: self.sim.is_done() for agent in self.agents}
        truncations = {agent: self.step_count >= self.max_steps for agent in self.agents}
        
        # 信息
        infos = {agent: {} for agent in self.agents}
        if self.sim.is_done():
            final_stats = self.sim.get_final_stats()
            for agent in self.agents:
                infos[agent]["final_stats"] = final_stats
        
        # 在info中添加全局状态
        global_state = self.sim.get_global_state()
        for agent_id in self.agents:
            infos[agent_id]['global_state'] = global_state

        if self.render_mode == "human":
            self.render()
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self, mode="human"):
        self.render_mode = mode
        if mode == "human":
            print(f"仿真时间: {self.sim.current_time:.1f}")
            print(f"完成零件数: {len(self.sim.completed_parts)}")
            for station_name in WORKSTATIONS.keys():
                queue_len = len(self.sim.queues[station_name].items)
                busy_count = self.sim.equipment_status[station_name]['busy_count']
                print(f"{station_name}: 队列={queue_len}, 忙碌设备={busy_count}")
    
    def close(self):
        """关闭环境"""
        pass

# =============================================================================
# 4. 环境工厂函数 (Environment Factory Functions)
# =============================================================================

def make_parallel_env(config: Dict[str, Any] = None):
    """直接创建PettingZoo环境"""
    # 仅在主进程中显示环境创建日志，避免worker重复输出
    import os
    if config and any(key in config for key in ['orders_scale', 'time_scale', 'stage_name']) and os.getpid() == os.getppid():
        print(f"🏭 创建环境 - 课程学习配置: {config.get('stage_name', 'Unknown')}")
        print(f"   订单比例: {config.get('orders_scale', 1.0)}, 时间比例: {config.get('time_scale', 1.0)}")
    
    return WFactoryEnv(config) 