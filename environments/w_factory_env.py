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
                'failure_end_time': 0
            }
            
            # 启动设备处理进程
            self.env.process(self._equipment_process(station_name))
    
    def _initialize_orders(self):
        """初始化订单"""
        for i, order_data in enumerate(BASE_ORDERS):
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
        """获取智能体的观测状态"""
        station_name = agent_id.replace("agent_", "")
        
        # 队列长度（归一化）
        queue_length = len(self.queues[station_name].items)
        normalized_queue_length = min(queue_length / QUEUE_CAPACITY, 1.0)
        
        # 设备状态（0=空闲，1=忙碌）
        equipment_busy = float(self.equipment_status[station_name]['busy_count'] > 0)
        
        return np.array([normalized_queue_length, equipment_busy], dtype=np.float32)
    
    def step_with_actions(self, actions: Dict[str, int]) -> Dict[str, float]:
        """执行一步仿真，传入智能体动作"""
        # 记录执行前状态
        prev_completed = len(self.completed_parts)
        prev_total_steps = sum(part.current_step for part in self.active_parts)
        
        # 执行智能体动作
        actions_executed = 0
        for agent_id, action in actions.items():
            station_name = agent_id.replace("agent_", "")
            
            if action == 1 and len(self.queues[station_name].items) > 0:
                # 处理队列中的第一个零件
                self._process_part_at_station(station_name)
                actions_executed += 1
        
        # 推进仿真 - 减少步长以获得更精细的控制
        try:
            self.env.run(until=self.env.now + 1)  # 每步推进1分钟而不是5分钟
        except simpy.core.EmptySchedule:
            self.simulation_ended = True
        
        self.current_time = self.env.now
        
        # 计算奖励
        rewards = self.get_rewards()
        
        # 调试信息
        new_completed = len(self.completed_parts)
        new_total_steps = sum(part.current_step for part in self.active_parts)
        
        if self.debug_level == 'DEBUG' and (new_completed > prev_completed or new_total_steps > prev_total_steps):
            print(f"🎯 进度更新: 完成零件 {prev_completed}->{new_completed}, 总工序 {prev_total_steps}->{new_total_steps}")
            print(f"   执行动作数: {actions_executed}, 奖励: {list(rewards.values())}")
        
        return rewards
    
    def _process_part_at_station(self, station_name: str):
        """在指定工作站处理零件"""
        if len(self.queues[station_name].items) == 0:
            return
            
        # 获取队列中的第一个零件
        part = self.queues[station_name].items[0]
        
        # 检查设备是否可用
        if self.equipment_status[station_name]['busy_count'] < WORKSTATIONS[station_name]['count']:
            # 从队列中移除零件
            self.queues[station_name].items.remove(part)
            
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
        """计算奖励 - 🔧 修复版：移除过度复杂的时间压力机制"""
        rewards = {}
        
        # 🔧 V4修复：大幅提升基础奖励，确保正奖励基础
        base_reward = REWARD_CONFIG["base_reward"]  # 从0.01提升到0.5
        
        # 完成奖励
        new_completions = len(self.completed_parts) - self.stats.get('last_completed_count', 0)
        completion_reward = 0
        if new_completions > 0:
            completion_reward = new_completions * REWARD_CONFIG["completion_reward"]
            self.stats['last_completed_count'] = len(self.completed_parts)
            
            # 🔧 新增：提前完成奖励
            if self.current_time < SIMULATION_TIME * 0.8:  # 在80%时间内完成
                completion_reward += REWARD_CONFIG["early_completion_bonus"]
        
        # 🔧 增强工序完成奖励 - 使用新的配置值
        current_total_steps = sum(part.current_step for part in self.active_parts)
        last_total_steps = self.stats.get('last_total_steps', 0)
        step_progress = current_total_steps - last_total_steps
        step_reward = 0
        if step_progress > 0:
            step_reward = step_progress * REWARD_CONFIG["step_reward"]  # 🔧 使用配置中的3.0
            self.stats['last_total_steps'] = current_total_steps
        
        # 🔧 新增：效率奖励 - 基于设备利用率
        efficiency_reward = 0
        total_utilization = 0
        for station_name, status in self.equipment_status.items():
            if status['busy_count'] > 0:
                utilization = min(status['busy_count'] / WORKSTATIONS[station_name]['count'], 1.0)
                total_utilization += utilization
        
        if len(WORKSTATIONS) > 0:
            avg_utilization = total_utilization / len(WORKSTATIONS)
            if avg_utilization > 0.6:  # 高利用率奖励
                efficiency_reward = avg_utilization * REWARD_CONFIG["efficiency_bonus"]
        
        # 🔧 V4关键修复：延期惩罚逻辑重构
        tardiness_penalty = 0
        if self.stats['max_tardiness'] > 0:
            # 只有当延期超过阈值时才惩罚，且不影响所有智能体
            if REWARD_CONFIG.get("tardiness_penalty_per_agent", True):
                # 旧逻辑：影响所有智能体
                tardiness_penalty = REWARD_CONFIG["tardiness_penalty"] * min(self.stats['max_tardiness'] / 60, 2.0)
            else:
                # 🔧 新逻辑：延期惩罚只影响相关工作站，且大幅减少
                tardiness_penalty = REWARD_CONFIG["tardiness_penalty"] * REWARD_CONFIG["penalty_scale_factor"]
        
        # 🔧 V4关键修复：空闲惩罚频率控制
        # 初始化空闲计数器
        if not hasattr(self, 'idle_counters'):
            self.idle_counters = {station: 0 for station in WORKSTATIONS.keys()}
        
        # 🔧 智能奖励分配机制 - V4平衡版
        for station_name in WORKSTATIONS.keys():
            agent_id = f"agent_{station_name}"
            agent_reward = base_reward  # 🔧 所有智能体都有大幅提升的基础奖励
            
            # 检查工作站是否活跃
            is_active = (len(self.queues[station_name].items) > 0 or 
                        self.equipment_status[station_name]['busy_count'] > 0)
            
            if is_active:
                # 重置空闲计数器
                self.idle_counters[station_name] = 0
                
                # 工序奖励：只给有活动的工作站
                if step_reward > 0:
                    agent_reward += step_reward / len(WORKSTATIONS)  # 平均分配工序奖励
                
                # 🔧 效率奖励：给活跃的工作站
                if efficiency_reward > 0:
                    station_utilization = min(self.equipment_status[station_name]['busy_count'] / WORKSTATIONS[station_name]['count'], 1.0)
                    agent_reward += efficiency_reward * station_utilization
            else:
                # 🔧 V4修复：空闲惩罚频率控制
                self.idle_counters[station_name] += 1
                
                # 只有连续空闲超过阈值才开始惩罚
                if self.idle_counters[station_name] > REWARD_CONFIG["idle_penalty_threshold"]:
                    # 应用惩罚缩放因子，大幅减少惩罚
                    scaled_idle_penalty = REWARD_CONFIG["idle_penalty"] * REWARD_CONFIG["penalty_scale_factor"]
                    agent_reward += scaled_idle_penalty
            
            # 🔧 完成奖励：只给最后完成工序的工作站 (包装台)
            if completion_reward > 0 and station_name == "包装台":
                agent_reward += completion_reward  # 只有包装台获得完成奖励
            
            # 🔧 V4修复：延期惩罚不再影响所有智能体
            if not REWARD_CONFIG.get("tardiness_penalty_per_agent", True):
                # 新逻辑：延期惩罚只影响包装台（最终负责交付的工作站）
                if station_name == "包装台" and tardiness_penalty != 0:
                    agent_reward += tardiness_penalty
            else:
                # 旧逻辑：所有智能体共同承担（已大幅减少）
                agent_reward += tardiness_penalty
            
            # 🔧 应用整体奖励缩放
            agent_reward *= REWARD_CONFIG["reward_scale_factor"]
            
            rewards[agent_id] = agent_reward
        
        # 🔧 V4调试信息 - 显示平衡效果
        if self.debug_level == 'DEBUG' and (new_completions > 0 or step_progress > 0 or efficiency_reward > 0):
            total_positive = base_reward * len(WORKSTATIONS) + completion_reward + step_reward + efficiency_reward
            total_negative = abs(tardiness_penalty * len(WORKSTATIONS)) + abs(REWARD_CONFIG["idle_penalty"] * REWARD_CONFIG["penalty_scale_factor"])
            print(f"🏆 V4平衡奖励详情:")
            print(f"   正奖励: 基础={base_reward:.2f}×{len(WORKSTATIONS)}, 完成={completion_reward:.1f}, 工序={step_reward:.1f}, 效率={efficiency_reward:.1f}")
            print(f"   负奖励: 延期={tardiness_penalty:.1f}, 空闲惩罚={REWARD_CONFIG['idle_penalty'] * REWARD_CONFIG['penalty_scale_factor']:.3f}")
            print(f"   平衡比例: 正奖励={total_positive:.1f} vs 负奖励={total_negative:.1f}")
            if completion_reward > 0:
                print(f"   🎉 完成奖励只给包装台智能体: {completion_reward:.1f}")
        
        # 🔧 V5新增：时间压力奖励机制
        
        return rewards
    
    def is_done(self) -> bool:
        """检查仿真是否结束 - 优先任务完成，时间作为备用条件"""
        # 🔧 修复：优先检查任务完成，而不是时间耗尽
        
        # 条件1: 所有订单完成 (主要完成条件)
        total_required = sum(order.quantity for order in self.orders)
        if len(self.completed_parts) >= total_required:
            if not hasattr(self, '_completion_logged'):
                print(f"🎉 所有订单完成! 完成{len(self.completed_parts)}/{total_required}个零件，用时{self.current_time:.1f}分钟")
                self._completion_logged = True
            return True
        
        # 条件2: 手动结束仿真
        if self.simulation_ended:
            return True
        
        # 条件3: 时间耗尽 (备用条件，增加时间限制)
        # 🔧 增加时间限制，给任务完成更多机会
        max_time = SIMULATION_TIME * 1.5  # 增加50%时间缓冲
        if self.current_time >= max_time:
            if not hasattr(self, '_timeout_logged'):
                print(f"⏰ 时间耗尽! 完成{len(self.completed_parts)}/{total_required}个零件，用时{self.current_time:.1f}分钟")
                self._timeout_logged = True
            return True
        
        return False
    
    def get_final_stats(self) -> Dict[str, Any]:
        """获取最终统计结果"""
        # 计算设备利用率
        for station_name, status in self.equipment_status.items():
            if self.current_time > 0:
                utilization = status['total_busy_time'] / self.current_time
                self.stats['equipment_utilization'][station_name] = utilization
        
        return self.stats

    def get_completion_stats(self) -> Dict[str, Any]:
        """获取完成统计信息 - V5新增"""
        total_required = sum(order.quantity for order in self.orders)
        completed_count = len(self.completed_parts)
        completion_rate = (completed_count / total_required) * 100 if total_required > 0 else 0
        
        # 设备利用率统计
        utilization_stats = {}
        for station_name, status in self.equipment_status.items():
            if self.current_time > 0:
                utilization = status['total_busy_time'] / self.current_time
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
        
        # 分析订单延期情况
        for order in self.orders:
            order_completion_time = self.current_time  # 当前时间作为完成时间
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
        
        # 动作和观测空间
        self.action_spaces = {
            agent: gym.spaces.Discrete(ACTION_CONFIG["action_space_size"])
            for agent in self.possible_agents
        }
        
        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        # 仿真环境
        self.sim = None
        self.episode_count = 0
        
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