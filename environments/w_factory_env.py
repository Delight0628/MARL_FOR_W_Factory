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
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

# Ray RLlib imports
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    RAY_AVAILABLE = True
except ImportError:
    # 如果Ray不可用，创建一个虚拟基类
    class MultiAgentEnv:
        pass
    RAY_AVAILABLE = False

from .w_factory_config import *

# 🔧 V9.1强化：全局静默模式控制 - 训练时完全静默
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
        
        # 调试级别控制
        self.debug_level = self.config.get('debug_level', 'INFO')  # DEBUG, INFO, WARNING, ERROR
        
        # 🔧 新增：训练模式标志，控制输出冗余度
        self._training_mode = self.config.get('training_mode', False)
        
        # 🔧 V9.1修复：训练模式下强制使用WARNING级别，减少输出冗余
        if self._training_mode:
            self.debug_level = 'WARNING'
        
        # 仿真环境
        self.env = simpy.Environment()
        self.current_time = 0
        self.simulation_ended = False
        
        # 设备和队列
        self.resources = {}
        self.queues = {}
        self.equipment_status = {}
        
        # 订单和零件管理
        self.orders = []
        self.active_parts = []
        self.completed_parts = []
        self.part_counter = 0
        
        # 统计数据
        self.stats = {
            'makespan': 0,
            'total_tardiness': 0,
            'max_tardiness': 0,
            'equipment_utilization': {},
            'queue_lengths': defaultdict(list),
            'completed_orders': 0,
            'total_parts': 0
        }
        
        # 🔧 V9新增：订单级别跟踪系统
        self.order_progress = {}  # 订单进度跟踪
        self.order_completion_times = {}  # 订单完成时间
        self.last_order_progress_milestones = {}  # 上次奖励的进度里程碑
        
        # 🔧 V9新增：瓶颈和关键路径分析
        self._bottleneck_stations = self._identify_bottleneck_stations()
        self._critical_parts = set()  # 关键路径上的零件
        
        # 🔧 V7 新增：用于快速查找下游工作站的缓存
        self._downstream_map = self._create_downstream_map()
        
        # 智能体决策接口
        self.agent_decisions = {}
        self.pending_decisions = set()
        
        self._initialize_resources()
        self._initialize_orders()
    
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
                # 新增：用于精确统计并行设备的忙碌“面积”（机器-分钟）
                'busy_machine_time': 0.0,
                'last_event_time': 0.0,
            }
            
            # 启动设备处理进程
            self.env.process(self._equipment_process(station_name))
    
    def _initialize_orders(self):
        """初始化订单（支持课程学习）"""
        # 🔧 V16：支持课程学习的订单缩放
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
        
        # 在变更 busy_count 之前，先结算从上次事件到现在的忙碌“面积”
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
        """🔧 V7 新增：创建下游工作站映射，用于快速查询"""
        downstream_map = {}
        routes = list(PRODUCT_ROUTES.values())
        for route in routes:
            for i in range(len(route) - 1):
                current_station = route[i]["station"]
                next_station = route[i+1]["station"]
                if current_station not in downstream_map:
                    downstream_map[current_station] = next_station
        return downstream_map
    
    def _identify_bottleneck_stations(self) -> set:
        """🔧 V9新增：识别瓶颈工作站"""
        station_loads = {}
        for station_name, station_config in WORKSTATIONS.items():
            total_load = 0
            for order in BASE_ORDERS:
                route = get_route_for_product(order["product"])
                for step in route:
                    if step["station"] == station_name:
                        total_load += step["time"] * order["quantity"]
            # 考虑并行处理能力
            station_loads[station_name] = total_load / station_config["count"]
        
        # 识别负荷最高的工作站作为瓶颈
        max_load = max(station_loads.values())
        bottlenecks = {station for station, load in station_loads.items() 
                      if load >= max_load * 0.8}  # 负荷达到最高负荷80%的都算瓶颈
        return bottlenecks
    
    def _update_order_progress(self):
        """🔧 V9新增：更新订单进度跟踪"""
        for order in self.orders:
            completed_parts = sum(1 for part in self.completed_parts 
                                if part.order_id == order.order_id)
            progress_rate = completed_parts / order.quantity if order.quantity > 0 else 0
            self.order_progress[order.order_id] = progress_rate
            
            # 检查订单是否完成
            if progress_rate >= 1.0 and order.order_id not in self.order_completion_times:
                self.order_completion_times[order.order_id] = self.current_time
                self.stats['completed_orders'] += 1
    
    def _identify_critical_parts(self) -> set:
        """🔧 V9新增：识别关键路径上的零件"""
        critical_parts = set()
        
        # 识别即将到期的订单的零件
        for part in self.active_parts:
            time_to_due = part.due_date - self.current_time
            if time_to_due <= 100:  # 100分钟内到期
                critical_parts.add(part.part_id)
        
        # 识别瓶颈工作站的零件
        for part in self.active_parts:
            current_station = part.get_current_station()
            if current_station in self._bottleneck_stations:
                critical_parts.add(part.part_id)
        
        return critical_parts

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
        获取智能体的观测状态 - 🔧 V7 全面增强版
        - 包含自身队列中前N个零件的详细信息
        - 包含下游工作站的队列信息
        """
        station_name = agent_id.replace("agent_", "")

        # 如果不启用增强观测，则返回旧版状态
        if not ENHANCED_OBS_CONFIG.get("enabled", False):
            # 队列长度（归一化）
            queue_length = len(self.queues[station_name].items)
            normalized_queue_length = min(queue_length / QUEUE_CAPACITY, 1.0)
            # 设备状态（0=空闲，1=忙碌）
            equipment_busy = float(self.equipment_status[station_name]['busy_count'] > 0)
            return np.array([normalized_queue_length, equipment_busy], dtype=np.float32)

        # --- V7 增强状态特征 ---
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

    def step_with_actions(self, actions: Dict[str, int]) -> Dict[str, float]:
        """执行一步仿真，传入智能体动作"""
        # 记录执行前状态
        prev_completed = len(self.completed_parts)
        prev_total_steps = sum(part.current_step for part in self.active_parts)
        
        # 执行智能体动作
        actions_executed = 0
        for agent_id, action in actions.items():
            station_name = agent_id.replace("agent_", "")

            # 兼容旧版动作空间 (0=IDLE, 1=PROCESS)
            if not ACTION_CONFIG_ENHANCED.get("enabled", False):
                if action == 1 and len(self.queues[station_name].items) > 0:
                    # 处理队列中的第一个零件
                    self._process_part_at_station(station_name, part_index=0)
                    actions_executed += 1
            else:
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
        rewards = self.get_rewards()
        
        # 🔧 V9.1修复：训练模式下完全静默调试信息
        if not self._training_mode and self.debug_level == 'DEBUG':
            new_completed = len(self.completed_parts)
            new_total_steps = sum(part.current_step for part in self.active_parts)
            
            if new_completed > prev_completed or new_total_steps > prev_total_steps:
                print(f"🎯 进度更新: 完成零件 {prev_completed}->{new_completed}, 总工序 {prev_total_steps}->{new_total_steps}")
                print(f"   执行动作数: {actions_executed}, 奖励: {list(rewards.values())}")
        
        return rewards
    
    def _process_part_at_station(self, station_name: str, part_index: int = 0):
        """
        在指定工作站处理零件 - 🔧 V7 增强版
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
    
    def get_rewards(self) -> Dict[str, float]:
        """🔧 V22 智能奖励重构：恢复核心引导奖励 + 动态放大"""
        rewards = {}
        
        # 获取课程阶段信息，用于动态调整奖励
        orders_scale = self.config.get('orders_scale', 1.0)
        
        # 核心：更新订单进度和关键路径分析
        self._update_order_progress()
        self._critical_parts = self._identify_critical_parts()
        
        # 仿真结束时的未完成订单严厉惩罚
        final_incomplete_penalty = 0
        if self.is_done():
            incomplete_orders = 0
            for order in self.orders:
                if order.order_id not in self.order_completion_times:
                    incomplete_orders += 1
            if incomplete_orders > 0:
                final_incomplete_penalty = incomplete_orders * REWARD_CONFIG["incomplete_order_final_penalty"]

        # --- 奖励计算 ---
        
        # 1. 订单完成奖励 (最高优先级)
        new_completed_orders = self.stats['completed_orders'] - self.stats.get('last_completed_orders', 0)
        order_completion_reward = 0
        if new_completed_orders > 0:
            order_completion_reward = new_completed_orders * REWARD_CONFIG["order_completion_reward"]
            self.stats['last_completed_orders'] = self.stats['completed_orders']

        # 2. 零件完成奖励 (V22 核心恢复)
        new_part_completions = len(self.completed_parts) - self.stats.get('last_completed_count', 0)
        part_completion_reward = 0
        if new_part_completions > 0:
            part_completion_reward = new_part_completions * REWARD_CONFIG["part_completion_reward"]
            self.stats['last_completed_count'] = len(self.completed_parts)
            # 🔧 V22 动态奖励放大: 在早期阶段，为关键的“路标”行为提供更强的正反馈
            if orders_scale <= 0.5:
                scale_factor = (2.0 - orders_scale * 2) # e.g., scale=0.2 -> x1.6, scale=0.5 -> x1.0
                part_completion_reward *= scale_factor

        # 3. 工序进展奖励
        current_total_steps = sum(part.current_step for part in self.active_parts)
        last_total_steps = self.stats.get('last_total_steps', 0)
        step_progress = current_total_steps - last_total_steps
        step_reward = 0
        if step_progress > 0:
            step_reward = step_progress * REWARD_CONFIG["step_reward"]
            self.stats['last_total_steps'] = current_total_steps
        
        # 4. 订单进度里程碑奖励 (V22 核心恢复)
        order_progress_reward = 0
        for order_id, progress in self.order_progress.items():
            last_milestone = self.last_order_progress_milestones.get(order_id, 0)
            current_milestone = int(progress * 4)
            
            if current_milestone > last_milestone:
                milestone_reward = (current_milestone - last_milestone) * REWARD_CONFIG["order_progress_bonus"]
                order_progress_reward += milestone_reward
                self.last_order_progress_milestones[order_id] = current_milestone
        
        # 🔧 V22 动态奖励放大
        if order_progress_reward > 0 and orders_scale <= 0.5:
            scale_factor = (2.0 - orders_scale * 2)
            order_progress_reward *= scale_factor
        
        # 5. 订单效率奖励
        order_efficiency_reward = 0
        for order_id, completion_time in self.order_completion_times.items():
            if order_id not in self.stats.get('rewarded_orders', set()):
                order = next((o for o in self.orders if o.order_id == order_id), None)
                if order and completion_time <= order.due_date:
                    efficiency = max(0, (order.due_date - completion_time) / order.due_date)
                    order_efficiency_reward += efficiency * REWARD_CONFIG["order_efficiency_bonus"]
                if 'rewarded_orders' not in self.stats: self.stats['rewarded_orders'] = set()
                self.stats['rewarded_orders'].add(order_id)
        
        # 6. 订单延期惩罚
        order_tardiness_penalty = 0
        for order in self.orders:
            if order.order_id in self.order_completion_times:
                completion_time = self.order_completion_times[order.order_id]
                if completion_time > order.due_date:
                    tardiness = completion_time - order.due_date
                    order_tardiness_penalty += REWARD_CONFIG["order_tardiness_penalty"] * (tardiness / 60)
        
        if orders_scale >= 0.7:
            efficiency_multiplier = 1.0 + (orders_scale - 0.7) * 3.0
            order_tardiness_penalty *= efficiency_multiplier
        
        # 7. 订单遗弃惩罚
        order_abandonment_penalty = 0
        for order_id, progress in self.order_progress.items():
            if progress < 1.0:
                last_progress_time = self.stats.get(f'last_progress_time_{order_id}', 0)
                if progress > self.stats.get(f'last_progress_{order_id}', 0):
                    self.stats[f'last_progress_time_{order_id}'] = self.current_time
                    self.stats[f'last_progress_{order_id}'] = progress
                elif self.current_time - last_progress_time > REWARD_CONFIG["order_abandonment_threshold"]:
                    order_abandonment_penalty += REWARD_CONFIG["order_abandonment_penalty"]
        
        # 8. 塑形奖励
        shaping_reward = 0
        if REWARD_CONFIG.get("shaping_enabled", False):
            # 1. 连续完成同一订单的奖励
            if not hasattr(self, 'last_completed_order_id'):
                self.last_completed_order_id = None
            
            # 检查最新完成的零件是否属于同一订单
            if new_part_completions > 0 and len(self.completed_parts) > 0:
                latest_part = self.completed_parts[-1]
                if self.last_completed_order_id == latest_part.order_id:
                    shaping_reward += REWARD_CONFIG["same_order_bonus"] * new_part_completions
                self.last_completed_order_id = latest_part.order_id
            
            # 2. 紧急订单处理奖励
            for part in self.active_parts:
                if part.due_date - self.current_time < 100:  # 100分钟内到期
                    if part.current_step > 0:  # 有进展
                        shaping_reward += REWARD_CONFIG["urgent_order_bonus"] / len(self.active_parts)
            
            # 3. 生产线流畅性奖励
            active_stations = sum(1 for s in WORKSTATIONS.keys() 
                                 if self.equipment_status[s]['busy_count'] > 0)
            if active_stations > len(WORKSTATIONS) * 0.6:  # 60%以上设备在工作
                shaping_reward += REWARD_CONFIG["flow_smoothness_bonus"]
            
            # 4. 队列均衡奖励
            queue_lengths = [len(self.queues[s].items) for s in WORKSTATIONS.keys()]
            if len(queue_lengths) > 0:
                queue_variance = np.var(queue_lengths)
                if queue_variance < 5:  # 队列长度差异小
                    shaping_reward += REWARD_CONFIG["queue_balance_bonus"]
            
            # 5. 提前完成奖励
            for order_id, completion_time in self.order_completion_times.items():
                if order_id not in self.stats.get('shaping_rewarded_orders', set()):
                    order = next((o for o in self.orders if o.order_id == order_id), None)
                    if order and completion_time < order.due_date * 0.8:  # 提前20%完成
                        shaping_reward += REWARD_CONFIG["early_completion_bonus"]
                        if 'shaping_rewarded_orders' not in self.stats:
                            self.stats['shaping_rewarded_orders'] = set()
                        self.stats['shaping_rewarded_orders'].add(order_id)
        
        # --- 奖励分配 ---
        if not hasattr(self, 'idle_counters'):
            self.idle_counters = {station: 0 for station in WORKSTATIONS.keys()}
        
        for station_name in WORKSTATIONS.keys():
            agent_id = f"agent_{station_name}"
            agent_reward = 0.0
            
            is_active = (len(self.queues[station_name].items) > 0 or 
                        self.equipment_status[station_name]['busy_count'] > 0)
            
            if is_active:
                self.idle_counters[station_name] = 0
                
                if step_reward > 0:
                    agent_reward += step_reward / len(WORKSTATIONS)
                
                station_critical_parts = [part for part in self.queues[station_name].items 
                                        if part.part_id in self._critical_parts]
                if station_critical_parts:
                    agent_reward += REWARD_CONFIG["critical_path_bonus"] * len(station_critical_parts) / 10
                
                if station_name in self._bottleneck_stations and len(self.queues[station_name].items) > 0:
                    agent_reward += REWARD_CONFIG["bottleneck_priority_bonus"] / 10
            else:
                self.idle_counters[station_name] += 1
                if self.idle_counters[station_name] > REWARD_CONFIG["idle_penalty_threshold"]:
                    agent_reward += REWARD_CONFIG["idle_penalty"]
            
            if order_completion_reward > 0:
                if station_name == "包装台":
                    agent_reward += order_completion_reward * 0.4
                else:
                    agent_reward += order_completion_reward * 0.6 / (len(WORKSTATIONS) - 1)
            
            if part_completion_reward > 0 and station_name == "包装台":
                agent_reward += part_completion_reward
            
            if order_progress_reward > 0:
                agent_reward += order_progress_reward / len(WORKSTATIONS)
            
            if order_efficiency_reward > 0:
                agent_reward += order_efficiency_reward / len(WORKSTATIONS)
            
            if shaping_reward > 0:
                agent_reward += shaping_reward / len(WORKSTATIONS)
            
            agent_reward += order_tardiness_penalty * REWARD_CONFIG["penalty_scale_factor"] / len(WORKSTATIONS)
            agent_reward += order_abandonment_penalty * REWARD_CONFIG["penalty_scale_factor"] / len(WORKSTATIONS)
            agent_reward += final_incomplete_penalty / len(WORKSTATIONS)
            
            agent_reward *= REWARD_CONFIG["reward_scale_factor"]
            
            rewards[agent_id] = agent_reward
        
        # 移除V21的日志，避免干扰
        return rewards
    
    def is_done(self) -> bool:
        """检查仿真是否结束 - 优先任务完成，时间作为备用条件"""
        # 🔧 修复：优先检查任务完成，而不是时间耗尽
        
        # 条件1: 所有订单完成 (主要完成条件)
        total_required = sum(order.quantity for order in self.orders)
        if len(self.completed_parts) >= total_required:
            if not hasattr(self, '_completion_logged'):
                # 🔧 V9.1强化：训练模式下完全静默
                if not SILENT_MODE and not self._training_mode:
                    print(f"🎉 所有订单完成! 完成{len(self.completed_parts)}/{total_required}个零件，用时{self.current_time:.1f}分钟")
                self._completion_logged = True
            return True
        
        # 条件2: 手动结束仿真
        if self.simulation_ended:
            return True
        
        # 条件3: 时间耗尽 (备用条件，增加时间限制)
        # 🔧 V8修复：给智能体更多时间完成任务，避免总是超时截断
        max_time = SIMULATION_TIME * 2.0  # 🔧 V8修复：从1.5增加到2.0，给更充足的时间
        if self.current_time >= max_time:
            if not hasattr(self, '_timeout_logged'):
                # 🔧 V9.1强化：训练模式下完全静默
                if not SILENT_MODE and not self._training_mode:
                    print(f"⏰ 时间耗尽! 完成{len(self.completed_parts)}/{total_required}个零件，用时{self.current_time:.1f}分钟")
                self._timeout_logged = True
            return True
        
        return False
    
    def get_final_stats(self) -> Dict[str, Any]:
        """获取最终统计结果"""
        # 计算设备利用率
        for station_name, status in self.equipment_status.items():
            # 在统计前结算从 last_event_time 到当前时间的忙碌面积
            if self.current_time > status.get('last_event_time', 0.0):
                elapsed = self.current_time - status.get('last_event_time', 0.0)
                status['busy_machine_time'] = status.get('busy_machine_time', 0.0) + elapsed * status['busy_count']
                status['last_event_time'] = self.current_time
            
            capacity = WORKSTATIONS[station_name]['count']
            if self.current_time > 0 and capacity > 0:
                # 平均设备利用率 = 忙碌机器时间总量 / (总时间 * 设备数量)
                utilization = status.get('busy_machine_time', 0.0) / (self.current_time * capacity)
            else:
                utilization = 0.0
            self.stats['equipment_utilization'][station_name] = utilization
        
        # 便捷字段与聚合
        try:
            # 平均设备利用率（各工作站平均）
            util_values = list(self.stats['equipment_utilization'].values())
            mean_utilization = float(np.mean(util_values)) if len(util_values) > 0 else 0.0
        except Exception:
            mean_utilization = 0.0
        
        # 为评估脚本提供更直观的键名（不移除原字段）
        self.stats['tardiness'] = self.stats.get('total_tardiness', 0)
        self.stats['completed_parts'] = self.stats.get('total_parts', 0)
        self.stats['mean_utilization'] = mean_utilization
        
        return self.stats
    
    def get_completion_stats(self) -> Dict[str, Any]:
        """获取完成统计信息 - V5新增"""
        total_required = sum(order.quantity for order in self.orders)
        completed_count = len(self.completed_parts)
        completion_rate = (completed_count / total_required) * 100 if total_required > 0 else 0
        
        # 设备利用率统计（使用忙碌面积口径）
        utilization_stats = {}
        for station_name, status in self.equipment_status.items():
            # 结算未计入的忙碌面积
            if self.current_time > status.get('last_event_time', 0.0):
                elapsed = self.current_time - status.get('last_event_time', 0.0)
                status['busy_machine_time'] = status.get('busy_machine_time', 0.0) + elapsed * status['busy_count']
                status['last_event_time'] = self.current_time
            capacity = WORKSTATIONS[station_name]['count']
            if self.current_time > 0 and capacity > 0:
                utilization = status.get('busy_machine_time', 0.0) / (self.current_time * capacity)
            else:
                utilization = 0.0
            utilization_stats[station_name] = utilization
        
        # 按产品类型统计完成情况
        product_completion = {}
        for order in self.orders:
            product_type = order.product
            if product_type not in product_completion:
                product_completion[product_type] = {'required': 0, 'completed': 0}
            product_completion[product_type]['required'] += order.quantity
        
        for part in self.completed_parts:
            product_type = part.product_type
            if product_type in product_completion:
                product_completion[product_type]['completed'] += 1
        
        # 🔧 新增：延期分析 (项目核心目标)
        tardiness_info = {
            'late_orders': 0,
            'max_tardiness': 0,
            'total_tardiness': 0,
            'on_time_orders': 0
        }
        
        # 🔧 V12修复：分析订单延期情况（使用真实的订单完成时间）
        for order in self.orders:
            # 使用订单的实际完成时间，如果未完成则使用当前时间
            if order.order_id in self.order_completion_times:
                order_completion_time = self.order_completion_times[order.order_id]
            else:
                # 未完成的订单，使用当前时间作为“假想完成时间”
                order_completion_time = self.current_time
            
            if order_completion_time > order.due_date:
                tardiness = order_completion_time - order.due_date
                tardiness_info['late_orders'] += 1
                tardiness_info['total_tardiness'] += tardiness
                tardiness_info['max_tardiness'] = max(tardiness_info['max_tardiness'], tardiness)
            else:
                tardiness_info['on_time_orders'] += 1
        
        # 计算平均延期时间
        if tardiness_info['late_orders'] > 0:
            tardiness_info['avg_tardiness'] = tardiness_info['total_tardiness'] / tardiness_info['late_orders']
        else:
            tardiness_info['avg_tardiness'] = 0
        
        return {
            'total_required': total_required,
            'completed_count': completed_count,
            'completion_rate': completion_rate,
            'current_time': self.current_time,
            'utilization_stats': utilization_stats,
            'product_completion': product_completion,
            'is_naturally_done': self.is_done(),
            'tardiness_info': tardiness_info,  # 🔧 新增延期分析
            'total_orders': len(self.orders),  # 🔧 新增订单总数
            'makespan': self.current_time  # 🔧 新增Makespan指标
        }

# =============================================================================
# 3. PettingZoo多智能体环境接口 (PettingZoo Multi-Agent Environment)
# =============================================================================

class WFactoryEnv(ParallelEnv):
    """W工厂多智能体强化学习环境 - 基于PettingZoo"""
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "w_factory_v1",
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # 智能体定义
        self.possible_agents = [f"agent_{station}" for station in WORKSTATIONS.keys()]
        self.agents = self.possible_agents[:]
        
        # 🔧 V7 新增：根据配置动态决定空间大小
        self._setup_spaces()

        # 仿真环境
        self.sim = None
        self.episode_count = 0
    
    # 🔧 修复PettingZoo警告：重写observation_space和action_space方法
    def observation_space(self, agent: str = None):
        """获取观测空间"""
        if agent is None:
            return self.observation_spaces
        return self.observation_spaces.get(agent)
    
    def action_space(self, agent: str = None):
        """获取动作空间"""
        if agent is None:
            return self.action_spaces
        return self.action_spaces.get(agent)
        
    def _get_obs_shape(self) -> Tuple[int,]:
        """🔧 V7 新增：动态计算观测空间维度"""
        if not ENHANCED_OBS_CONFIG.get("enabled", False):
            return (2,)
        
        shape = 0
        # 1. 自身设备状态
        shape += 2
        # 2. 自身队列详细信息
        shape += ENHANCED_OBS_CONFIG["top_n_parts"] * 4
        # 3. 下游工作站信息
        if ENHANCED_OBS_CONFIG["include_downstream_info"]:
            shape += 1
        
        return (shape,)

    def _setup_spaces(self):
        """🔧 V7 新增：根据配置设置动作和观测空间"""
        
        # --- 动作空间 ---
        if ACTION_CONFIG_ENHANCED.get("enabled", False):
            action_size = ACTION_CONFIG_ENHANCED["action_space_size"]
        else:
            action_size = ACTION_CONFIG["action_space_size"]
            
        self.action_spaces = {
            agent: gym.spaces.Discrete(action_size)
            for agent in self.possible_agents
        }

        # --- 观测空间 ---
        obs_shape = self._get_obs_shape()
        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 创建新的仿真实例
        self.sim = WFactorySim(self.config)
        self.agents = self.possible_agents[:]
        self.episode_count += 1
        
        # 获取初始观测
        observations = {
            agent: self.sim.get_state_for_agent(agent)
            for agent in self.agents
        }
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]):
        """执行一步"""
        if not self.sim:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # 执行仿真步骤
        rewards = self.sim.step_with_actions(actions)
        
        # 获取新的观测
        observations = {
            agent: self.sim.get_state_for_agent(agent)
            for agent in self.agents
        }
        
        # 检查是否结束
        terminations = {agent: self.sim.is_done() for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        # 信息
        infos = {agent: {} for agent in self.agents}
        if self.sim.is_done():
            final_stats = self.sim.get_final_stats()
            for agent in self.agents:
                infos[agent]["final_stats"] = final_stats
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self, mode="human"):
        """渲染环境（可选实现）"""
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

def make_env(config: Dict[str, Any] = None):
    """创建W工厂环境实例"""
    env = WFactoryEnv(config)
    return env

class WFactoryGymEnv(MultiAgentEnv):
    """W工厂环境的Ray RLlib MultiAgentEnv适配器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # 创建PettingZoo环境
        self.pz_env = WFactoryEnv(config)
        
        # Ray RLlib MultiAgentEnv必需属性
        self._agent_ids = set(self.pz_env.possible_agents)
        self._spaces_in_preferred_format = True
        
        # 设置动作和观测空间
        self.action_spaces = self.pz_env.action_spaces
        self.observation_spaces = self.pz_env.observation_spaces
        
        # 兼容性属性
        self.agents = self.pz_env.possible_agents
        self.possible_agents = self.pz_env.possible_agents
        self._num_agents = len(self.agents)
        
        # 单智能体兼容性（使用第一个智能体的空间）
        first_agent = self.pz_env.possible_agents[0]
        self.action_space = self.pz_env.action_spaces[first_agent]
        self.observation_space = self.pz_env.observation_spaces[first_agent]
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        observations, infos = self.pz_env.reset(seed=seed, options=options)
        
        # 确保返回的观测包含所有活跃智能体
        # Ray RLlib期望观测字典包含所有智能体
        for agent in self.possible_agents:
            if agent not in observations:
                # 如果某个智能体不在观测中，添加默认观测
                observations[agent] = self.observation_spaces[agent].sample() * 0  # 零观测
            if agent not in infos:
                infos[agent] = {}
        
        return observations, infos
    
    def step(self, action_dict):
        """执行一步"""
        # Ray RLlib直接传递智能体名称作为键的动作字典
        # 如果传入的是数字索引，需要转换
        if action_dict and isinstance(list(action_dict.keys())[0], int):
            # 数字索引格式，转换为智能体名称
            actions = {}
            for i, agent in enumerate(self.agents):
                if i in action_dict:
                    actions[agent] = action_dict[i]
                else:
                    actions[agent] = 0  # 默认动作
        else:
            # 已经是智能体名称格式
            actions = action_dict
        
        # 执行步骤
        observations, rewards, terminations, truncations, infos = self.pz_env.step(actions)
        
        # 确保所有智能体都有对应的返回值
        for agent in self.possible_agents:
            if agent not in observations:
                observations[agent] = self.observation_spaces[agent].sample() * 0
            if agent not in rewards:
                rewards[agent] = 0.0
            if agent not in terminations:
                terminations[agent] = False
            if agent not in truncations:
                truncations[agent] = False
            if agent not in infos:
                infos[agent] = {}
        
        # Ray RLlib需要特殊的终止状态处理
        # 添加"__all__"键来指示是否所有智能体都完成
        terminations["__all__"] = all(terminations.values()) if terminations else False
        truncations["__all__"] = all(truncations.values()) if truncations else False
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self, mode="human"):
        """渲染环境"""
        return self.pz_env.render(mode)
    
    def close(self):
        """关闭环境"""
        self.pz_env.close()
    
    # Ray RLlib 2.48.0 MultiAgentEnv必需方法
    def get_agent_ids(self):
        """获取智能体ID集合"""
        return self._agent_ids
    
    def get_observation_space(self, agent_id: str = None):
        """获取观测空间"""
        if agent_id is None:
            return self.observation_spaces
        return self.observation_spaces.get(agent_id)
    
    def get_action_space(self, agent_id: str = None):
        """获取动作空间"""
        if agent_id is None:
            return self.action_spaces
        return self.action_spaces.get(agent_id)
    
    def observation_space_contains(self, x: dict):
        """检查观测是否在观测空间内"""
        for agent_id, obs in x.items():
            if agent_id not in self.observation_spaces:
                return False
            if not self.observation_spaces[agent_id].contains(obs):
                return False
        return True
    
    def action_space_contains(self, x: dict):
        """检查动作是否在动作空间内"""
        for agent_id, action in x.items():
            if agent_id not in self.action_spaces:
                return False
            if not self.action_spaces[agent_id].contains(action):
                return False
        return True
    
    def action_space_sample(self, agent_ids: list = None):
        """从动作空间采样"""
        if agent_ids is None:
            agent_ids = list(self._agent_ids)
        return {
            agent_id: self.action_spaces[agent_id].sample()
            for agent_id in agent_ids
            if agent_id in self.action_spaces
        }
    
    def observation_space_sample(self, agent_ids: list = None):
        """从观测空间采样"""
        if agent_ids is None:
            agent_ids = list(self._agent_ids)
        return {
            agent_id: self.observation_spaces[agent_id].sample()
            for agent_id in agent_ids
            if agent_id in self.observation_spaces
        }
    
    @property
    def num_agents(self):
        """智能体数量属性（只读）"""
        return self._num_agents
    
    @num_agents.setter
    def num_agents(self, value):
        """允许Ray RLlib设置num_agents属性"""
        self._num_agents = value

def make_parallel_env(config: Dict[str, Any] = None):
    """创建并行环境（用于训练）"""
    # 🔧 V17优化：仅在主进程中显示环境创建日志，避免worker重复输出
    import os
    if config and any(key in config for key in ['orders_scale', 'time_scale', 'stage_name']) and os.getpid() == os.getppid():
        print(f"🏭 创建环境 - 课程学习配置: {config.get('stage_name', 'Unknown')}")
        print(f"   订单比例: {config.get('orders_scale', 1.0)}, 时间比例: {config.get('time_scale', 1.0)}")
    
    # 检查是否需要Ray RLlib兼容的环境
    import inspect
    frame = inspect.currentframe()
    try:
        # 检查调用栈中是否有Ray相关的模块
        caller_frame = frame.f_back
        while caller_frame:
            caller_filename = caller_frame.f_code.co_filename
            if 'ray' in caller_filename.lower() or 'rllib' in caller_filename.lower():
                # Ray RLlib调用，返回Gymnasium兼容环境
                return WFactoryGymEnv(config)
            caller_frame = caller_frame.f_back
        
        # 非Ray调用，返回原始PettingZoo环境
        return WFactoryEnv(config)
    finally:
        del frame

def make_parallel_env_for_ray(config: Dict[str, Any] = None):
    """专门为Ray RLlib创建环境"""
    return WFactoryGymEnv(config)

def make_parallel_env_pettingzoo(config: Dict[str, Any] = None):
    """创建原始PettingZoo环境"""
    return WFactoryEnv(config)

def make_aec_env(config: Dict[str, Any] = None):
    """创建AEC环境（Agent-Environment-Cycle）"""
    env = make_env(config)
    env = parallel_to_aec(env)
    return env 