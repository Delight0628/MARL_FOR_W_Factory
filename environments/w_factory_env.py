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
        self.env = simpy.Environment()
        
        # 仿真状态
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
        # 执行智能体动作
        for agent_id, action in actions.items():
            station_name = agent_id.replace("agent_", "")
            
            if action == 1 and len(self.queues[station_name].items) > 0:
                # 处理队列中的第一个零件
                self._process_part_at_station(station_name)
        
        # 推进仿真
        try:
            self.env.run(until=self.env.now + 1)
        except simpy.core.EmptySchedule:
            self.simulation_ended = True
        
        self.current_time = self.env.now
        
        # 计算奖励
        rewards = self._calculate_rewards()
        
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
                self._update_completion_stats(part)
            else:
                # 移动到下一个工作站
                next_station = part.get_current_station()
                if next_station:
                    yield self.queues[next_station].put(part)
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """计算智能体奖励"""
        rewards = {}
        
        # 全局共享奖励
        global_reward = 0
        
        # 完成奖励
        new_completions = len(self.completed_parts) - self.stats.get('last_completed', 0)
        global_reward += new_completions * REWARD_CONFIG["completion_reward"]
        self.stats['last_completed'] = len(self.completed_parts)
        
        # 延期惩罚
        if self.stats['max_tardiness'] > 0:
            global_reward += REWARD_CONFIG["tardiness_penalty"]
        
        # 为所有智能体分配相同的全局奖励
        for station_name in WORKSTATIONS.keys():
            agent_id = f"agent_{station_name}"
            rewards[agent_id] = global_reward
        
        return rewards
    
    def is_done(self) -> bool:
        """检查仿真是否结束"""
        return (self.simulation_ended or 
                self.current_time >= SIMULATION_TIME or
                len(self.completed_parts) >= sum(order.quantity for order in self.orders))
    
    def get_final_stats(self) -> Dict[str, Any]:
        """获取最终统计结果"""
        # 计算设备利用率
        for station_name, status in self.equipment_status.items():
            if self.current_time > 0:
                utilization = status['total_busy_time'] / self.current_time
                self.stats['equipment_utilization'][station_name] = utilization
        
        return self.stats

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

def make_parallel_env(config: Dict[str, Any] = None):
    """创建并行环境（用于训练）"""
    env = make_env(config)
    return env

def make_aec_env(config: Dict[str, Any] = None):
    """创建AEC环境（Agent-Environment-Cycle）"""
    env = make_env(config)
    env = parallel_to_aec(env)
    return env 