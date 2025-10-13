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

# --- V3 融合版：新增的辅助函数 ---
def _calculate_part_total_remaining_processing_time(part: 'Part') -> float:
    """计算一个零件所有剩余工序的总加工时间"""
    route = get_route_for_product(part.product_type)
    if part.current_step >= len(route):
        return 0.0
    return sum(step['time'] for i, step in enumerate(route) if i >= part.current_step)

def calculate_slack_time(part: 'Part', current_time: float, queues: Dict[str, Any] = None, workstations: Dict[str, Dict] = None) -> float:
    """
    计算零件的松弛时间 (Slack Time) - 改进版本
    
    Args:
        part: 零件对象
        current_time: 当前时间
        queues: 工作站队列字典（可选）
        workstations: 工作站配置字典（可选）
    
    Returns:
        松弛时间（分钟）。正值表示有余量，负值表示可能延期
    """
    remaining_processing_time = _calculate_part_total_remaining_processing_time(part)
    
    # 基础松弛时间（原始计算）
    basic_slack = (part.due_date - current_time) - remaining_processing_time
    
    return basic_slack

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
        self.contribution_map: Dict[str, float] = {}
        
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
        
        # 新增：用于生成甘特图的加工历史记录
        self.gantt_chart_history: List[Dict[str, Any]] = []
        
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

        # 终局奖励发放标记（防重复）
        self.final_bonus_awarded = False
        self.final_bonus_value = 0.0

        # 🔧 新增：迟期总量缓存与候选缓存（保证同一步一致性）
        self._last_overdue_sum: float = 0.0
        self._cached_candidates: Dict[str, List[Dict[str, Any]]] = {}
        
        # 用于快速查找下游工作站的缓存
        self._downstream_map = self._create_downstream_map()
        
        self._initialize_resources()
        
        # --- 方案三：引入环境随机性 ---
        # 备份基础订单，以便在重置时重新引入随机性
        self._base_orders_template = [o.copy() for o in BASE_ORDERS]
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
        
        # 新增：清空甘特图历史
        self.gantt_chart_history.clear()
        
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

        # 重置终局奖励标记
        self.final_bonus_awarded = False
        self.final_bonus_value = 0.0

        # 重置迟期与候选缓存
        self._last_overdue_sum = 0.0
        self._cached_candidates.clear()
    
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
        """初始化订单（支持课程学习、自定义订单和环境随机性）"""
        # 🔧 修复：优先使用自定义订单配置
        if 'custom_orders' in self.config:
            # 使用自定义订单，忽略课程学习缩放
            actual_orders_config = self.config['custom_orders']
            is_randomized = False
        else:
            # --- 方案三：引入环境随机性 ---
            orders_scale = self.config.get('orders_scale', 1.0)
            time_scale = self.config.get('time_scale', 1.0)
            is_randomized = self.config.get('randomize_env', False)

            base_orders_template = self._base_orders_template
            
            # 如果启用课程学习，按比例调整订单
            actual_orders_config = []
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
                        actual_orders_config.append(adjusted_order)
                        parts_added += adjusted_quantity
            else:
                actual_orders_config = base_orders_template

        # 创建订单对象
        for i, order_data in enumerate(actual_orders_config):
            order_data_copy = order_data.copy()

            # --- 方案三：如果启用了随机化，则添加扰动 ---
            if is_randomized:
                due_date_jitter = np.random.uniform(-15, 15)
                arrival_time_jitter = np.random.uniform(0, 10)
                order_data_copy['due_date'] += due_date_jitter
                # 修复：使用订单索引作为基础到达时间，而不是不存在的'start_time'
                base_arrival_time = order_data_copy.get('arrival_time', 0)
                order_data_copy['arrival_time'] = base_arrival_time + arrival_time_jitter
            else:
                # 修复：确保有默认的到达时间
                order_data_copy['arrival_time'] = order_data_copy.get('arrival_time', 0)

            order = Order(
                order_id=i,
                product=order_data_copy["product"],
                quantity=order_data_copy["quantity"],
                priority=order_data_copy["priority"],
                due_date=order_data_copy["due_date"],
                arrival_time=order_data_copy['arrival_time']
            )
            self.orders.append(order)
            
            # 创建零件并添加到仿真中
            parts = order.create_parts()
            for part in parts:
                self.env.process(self._part_process(part))
                self.active_parts.append(part)
    
    def _part_process(self, part: Part):
        """零件的生产流程进程 - 简化版本"""
        # 在达到计划到达时间前等待
        if hasattr(part, 'start_time') and part.start_time > self.env.now:
            yield self.env.timeout(part.start_time - self.env.now)
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
        方案B：全局优化观测状态
        - 包含四大部分：
          1. 智能体自身特征 (8维)
          2. 全局宏观特征 (7维)
          3. 当前队列摘要统计 (40维)
          4. 候选工件详细特征 (150维)
        总维度 = 205维
        """
        station_name = agent_id.replace("agent_", "")
        
        # --- 1. 智能体自身特征 (Agent Features) - 8维 ---
        agent_features_list = []
        station_types = list(WORKSTATIONS.keys())
        station_index = station_types.index(station_name)
        agent_features_list.extend([1.0 if i == station_index else 0.0 for i in range(len(station_types))])
        
        capacity = WORKSTATIONS[station_name]['count']
        agent_features_list.append(capacity / 5.0)  # 归一化能力
        
        busy_ratio = self.equipment_status[station_name]['busy_count'] / capacity
        agent_features_list.append(busy_ratio)
        agent_features_list.append(1.0 if self.equipment_status[station_name]['is_failed'] else 0.0)
        
        agent_features = np.array(agent_features_list, dtype=np.float32)

        # --- 2. 全局宏观特征 (Global Features) - 7维 ---
        time_normalized = self.env.now / SIMULATION_TIME
        total_parts_in_system = sum(order.quantity for order in self.orders)
        wip_normalized = len(self.active_parts) / total_parts_in_system if total_parts_in_system > 0 else 0.0
        
        # 计算全局最小/平均松弛度
        if self.active_parts:
            slack_times = [calculate_slack_time(p, self.env.now, self.queues, WORKSTATIONS) 
                          for p in self.active_parts]
            min_slack_normalized = min(slack_times) / ENHANCED_OBS_CONFIG["time_slack_norm"]
            avg_slack_normalized = np.mean(slack_times) / ENHANCED_OBS_CONFIG["time_slack_norm"]
            
            # 延期零件数占比
            late_parts_ratio = sum(1 for s in slack_times if s < 0) / len(slack_times)
        else:
            min_slack_normalized = 0.0
            avg_slack_normalized = 0.0
            late_parts_ratio = 0.0
        
        # 瓶颈工作站拥堵度
        max_queue_len = max(len(self.queues[s].items) for s in WORKSTATIONS.keys())
        bottleneck_congestion = max_queue_len / ENHANCED_OBS_CONFIG["w_station_capacity_norm"]
        
        # 当前队列长度
        current_queue_len = len(self.queues[station_name].items)
        queue_len_normalized = current_queue_len / ENHANCED_OBS_CONFIG["w_station_capacity_norm"]
        
        global_features = np.array([
            time_normalized,
            wip_normalized,
            np.clip(min_slack_normalized, -1.0, 1.0),
            np.clip(avg_slack_normalized, -1.0, 1.0),
            np.clip(bottleneck_congestion, 0, 1.0),
            late_parts_ratio,
            np.clip(queue_len_normalized, 0, 1.0),
        ], dtype=np.float32)

        # --- 3. 当前队列摘要统计 (Queue Summary) - 40维 ---
        queue_summary = self._get_queue_summary_features(station_name)
        
        # --- 4. 候选工件详细特征 (Candidate Workpieces) - 150维 ---
        candidate_features = self._get_candidate_features(station_name)
        
        # 组合所有特征
        full_obs = np.concatenate([agent_features, global_features, queue_summary, candidate_features])
        return full_obs.flatten()
    def _get_queue_summary_features(self, station_name: str) -> np.ndarray:
        """
        方案B：获取队列摘要统计特征 (40维 = 8特征 × 5统计量)
        对队列中所有工件计算统计特征
        """
        queue = self.queues[station_name].items
        
        if not queue:
            # 空队列返回零向量
            return np.zeros(40, dtype=np.float32)
        
        # 收集各种特征
        slack_times = []
        processing_times = []
        remaining_ops = []
        remaining_total_times = []
        downstream_congestions = []
        priorities = []
        is_late_flags = []
        is_final_ops = []
        
        for part in queue:
            slack = calculate_slack_time(part, self.env.now, self.queues, WORKSTATIONS)
            slack_times.append(slack)
            processing_times.append(part.get_processing_time())
            
            route = get_route_for_product(part.product_type)
            remaining_ops_count = len(route) - part.current_step
            remaining_ops.append(remaining_ops_count)
            
            remaining_total_times.append(_calculate_part_total_remaining_processing_time(part))
            
            # 下游拥堵
            if part.current_step < len(route) - 1:
                downstream_station = route[part.current_step + 1]["station"]
                congestion = len(self.queues[downstream_station].items)
                downstream_congestions.append(congestion)
            else:
                downstream_congestions.append(0)
            
            priorities.append(part.priority)
            is_late_flags.append(1.0 if slack < 0 else 0.0)
            is_final_ops.append(1.0 if remaining_ops_count <= 1 else 0.0)
        
        # 计算5种统计量：min, max, mean, std, median
        def compute_stats(values):
            if not values:
                return [0.0, 0.0, 0.0, 0.0, 0.0]
            arr = np.array(values)
            return [
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.median(arr)),
            ]
        
        # 归一化并收集统计
        features = []
        
        # 1. 松弛度统计
        slack_norm = [s / ENHANCED_OBS_CONFIG["time_slack_norm"] for s in slack_times]
        features.extend(compute_stats(slack_norm))
        
        # 2. 加工时间统计
        proc_norm = [p / ENHANCED_OBS_CONFIG["max_op_duration_norm"] for p in processing_times]
        features.extend(compute_stats(proc_norm))
        
        # 3. 剩余工序统计
        ops_norm = [o / ENHANCED_OBS_CONFIG["max_bom_ops_norm"] for o in remaining_ops]
        features.extend(compute_stats(ops_norm))
        
        # 4. 剩余总时间统计
        time_norm = [t / ENHANCED_OBS_CONFIG["total_remaining_time_norm"] for t in remaining_total_times]
        features.extend(compute_stats(time_norm))
        
        # 5. 下游拥堵统计
        cong_norm = [c / ENHANCED_OBS_CONFIG["w_station_capacity_norm"] for c in downstream_congestions]
        features.extend(compute_stats(cong_norm))
        
        # 6. 优先级统计
        prio_norm = [p / 5.0 for p in priorities]
        features.extend(compute_stats(prio_norm))
        
        # 7. 延期标记统计
        features.extend(compute_stats(is_late_flags))
        
        # 8. 最终工序标记统计
        features.extend(compute_stats(is_final_ops))
        
        return np.array(features, dtype=np.float32)
    
    def _get_candidate_features(self, station_name: str) -> np.ndarray:
        """
        方案B：获取候选工件详细特征 (150维 = 15维 × 10工件)
        采用多样性采样策略
        """
        candidates = self._get_candidate_workpieces(station_name)
        
        feature_list = []
        candidate_dim = ENHANCED_OBS_CONFIG["candidate_feature_dim"]
        
        for i in range(ENHANCED_OBS_CONFIG["num_candidate_workpieces"]):
            if i < len(candidates):
                part = candidates[i]['part']
                features = self._get_workpiece_obs(part, current_station=station_name)
            else:
                # 空槽位用零填充
                features = np.zeros(candidate_dim, dtype=np.float32)
            feature_list.append(features)
        
        return np.concatenate(feature_list)
    
    def _get_candidate_workpieces(self, station_name: str) -> List[Dict[str, Any]]:
        """
        方案B：获取候选工件列表（多样性采样）
        
        核心思想：打破FIFO锁定，提供全局视野
        - 通过多样性采样确保agent能看到队列中不同类型的工件
        - 不再受限于队列前几个位置，实现真正的全局优化
        
        采样策略：
        1. 最紧急的N个：按EDD策略选择（松弛度最小）
        2. 最短的N个：按SPT策略选择（加工时间最短）
        3. 随机N个：随机采样（保证探索多样性）
        
        返回格式：[{"part": Part, "index": int, "category": str}, ...]
        """
        queue = self.queues[station_name].items
        
        if not queue:
            # 空队列清空缓存
            self._cached_candidates[station_name] = []
            return []
        
        # 若本步已有缓存，直接返回，确保观测与执行一致
        if station_name in self._cached_candidates and self._cached_candidates[station_name]:
            return self._cached_candidates[station_name]

        candidates = []
        used_indices = set()
        
        # 1. 最紧急的N个工件
        num_urgent = ENHANCED_OBS_CONFIG["num_urgent_candidates"]
        sorted_by_slack = sorted(enumerate(queue), 
                                key=lambda x: calculate_slack_time(x[1], self.env.now, self.queues, WORKSTATIONS))
        for idx, part in sorted_by_slack[:num_urgent]:
            if idx not in used_indices:
                candidates.append({"part": part, "index": idx, "category": "urgent"})
                used_indices.add(idx)
        
        # 2. 最短加工时间的N个工件
        num_short = ENHANCED_OBS_CONFIG["num_short_candidates"]
        sorted_by_time = sorted(enumerate(queue), 
                               key=lambda x: x[1].get_processing_time())
        for idx, part in sorted_by_time[:num_short]:
            if idx not in used_indices:
                candidates.append({"part": part, "index": idx, "category": "short"})
                used_indices.add(idx)
        
        # 3. 随机采样N个工件（确保多样性）
        num_random = ENHANCED_OBS_CONFIG["num_random_candidates"]
        available_indices = [i for i in range(len(queue)) if i not in used_indices]
        if available_indices:
            # 🔧 确定性随机：基于(站点, 当前时间, 队列part_id序列)生成种子
            seed_tuple = (station_name, int(self.env.now), tuple(p.part_id for p in queue))
            seed = hash(seed_tuple) & 0xffffffff
            rng = random.Random(seed)
            sample_size = min(num_random, len(available_indices))
            sampled_indices = rng.sample(available_indices, sample_size)
            for idx in sampled_indices:
                candidates.append({"part": queue[idx], "index": idx, "category": "random"})
                used_indices.add(idx)
        
        # 缓存本步候选以保证一致性
        self._cached_candidates[station_name] = candidates
        return candidates
    
    def _select_workpiece_by_action(self, station_name: str, action: int) -> Optional[Tuple[Part, int]]:
        """
        方案B：根据动作选择工件（全局优化核心）
        
        关键创新：动作直接映射到全局工件选择策略
        - 不再通过FIFO索引选择，而是通过策略或候选列表选择
        - 彻底打破了SimPy队列的FIFO约束
        
        动作映射：
        - 0: IDLE（不处理）
        - 1: URGENT（选择全队列最紧急的工件）
        - 2: SHORT（选择全队列最短的工件）
        - 3: BALANCE（选择下游最空闲的工件）
        - 4: FIFO（选择队首工件，作为baseline）
        - 5: RANDOM（随机选择）
        - 6-15: 候选工件1-10（从多样性采样列表中选择）
        
        返回：(选中的工件, 在队列中的索引) 或 None
        """
        queue = self.queues[station_name].items
        
        if not queue or action == 0:
            return None
        
        # 策略型动作
        if action == 1:  # URGENT (EDD)
            min_slack = float('inf')
            selected_idx = 0
            for idx, part in enumerate(queue):
                slack = calculate_slack_time(part, self.env.now, self.queues, WORKSTATIONS)
                if slack < min_slack:
                    min_slack = slack
                    selected_idx = idx
            return (queue[selected_idx], selected_idx)
        
        elif action == 2:  # SHORT (SPT)
            min_time = float('inf')
            selected_idx = 0
            for idx, part in enumerate(queue):
                proc_time = part.get_processing_time()
                if proc_time < min_time:
                    min_time = proc_time
                    selected_idx = idx
            return (queue[selected_idx], selected_idx)
        
        elif action == 3:  # BALANCE (负载均衡)
            # 选择下游负载最轻的工件
            min_downstream_load = float('inf')
            selected_idx = 0
            for idx, part in enumerate(queue):
                route = get_route_for_product(part.product_type)
                if part.current_step < len(route) - 1:
                    downstream_station = route[part.current_step + 1]["station"]
                    downstream_load = len(self.queues[downstream_station].items)
                else:
                    downstream_load = 0
                if downstream_load < min_downstream_load:
                    min_downstream_load = downstream_load
                    selected_idx = idx
            return (queue[selected_idx], selected_idx)
        
        elif action == 4:  # FIFO
            return (queue[0], 0)
        
        elif action == 5:  # RANDOM
            selected_idx = random.randint(0, len(queue) - 1)
            return (queue[selected_idx], selected_idx)
        
        # 候选工件动作 (6-15)
        elif 6 <= action <= 15:
            candidates = self._get_candidate_workpieces(station_name)
            candidate_idx = action - 6
            if candidate_idx < len(candidates):
                candidate_info = candidates[candidate_idx]
                part = candidate_info['part']
                # 需要找到这个工件在当前队列中的实际索引
                for idx, queue_part in enumerate(queue):
                    if queue_part.part_id == part.part_id:
                        return (part, idx)
        
        return None
    
    def _get_workpiece_obs(self, part: Part, current_station: str = None) -> np.ndarray:
        """
        方案B：获取单个工件的特征 (15维)
        简化版本，不包含one-hot编码
        """
        # 特征1: 是否存在
        exists = 1.0
        
        # 特征2: 时间松弛度
        time_slack = calculate_slack_time(part, self.env.now, self.queues, WORKSTATIONS)
        normalized_time_slack = time_slack / ENHANCED_OBS_CONFIG["time_slack_norm"]
        
        # 特征3: 剩余工序数
        route = get_route_for_product(part.product_type)
        remaining_ops = len(route) - part.current_step
        normalized_remaining_ops = remaining_ops / ENHANCED_OBS_CONFIG["max_bom_ops_norm"]
        
        # 特征4: 剩余总加工时间
        total_remaining_time = _calculate_part_total_remaining_processing_time(part)
        normalized_total_remaining_time = total_remaining_time / ENHANCED_OBS_CONFIG["total_remaining_time_norm"]

        # 特征5: 当前工序加工时间
        current_op_duration = part.get_processing_time()
        normalized_op_duration = current_op_duration / ENHANCED_OBS_CONFIG["max_op_duration_norm"]
        
        # 特征6: 下游拥堵情况
        downstream_congestion = 0.0
        if part.current_step < len(route) - 1:
            downstream_station = route[part.current_step + 1]["station"]
            if downstream_station in self.queues:
                congestion = len(self.queues[downstream_station].items) / ENHANCED_OBS_CONFIG["w_station_capacity_norm"]
                downstream_congestion = np.clip(congestion, 0, 1.0)
        
        # 特征7: 订单优先级
        priority = part.priority / 5.0

        # 特征8: 是否为最终工序
        is_final_op = 1.0 if remaining_ops <= 1 else 0.0
        
        # 特征9-12: 产品类型编码（简化为产品ID）
        product_types = list(PRODUCT_ROUTES.keys())
        product_id = 0.0
        if part.product_type in product_types:
            product_id = float(product_types.index(part.product_type)) / len(product_types)
        
        # 特征13: 是否延期
        is_late = 1.0 if time_slack < 0 else 0.0
        
        # 特征14: 瓶颈感知
        is_next_bottleneck = 0.0
        if part.current_step < len(route) - 1:
            next_station = route[part.current_step + 1]["station"]
            max_queue_len = max(len(self.queues[s].items) for s in WORKSTATIONS.keys()) if WORKSTATIONS else 1
            next_queue_len = len(self.queues[next_station].items)
            if next_queue_len >= max_queue_len * 0.8:
                is_next_bottleneck = 1.0
        
        # 特征15: 全局紧急度对比
        relative_urgency = 0.0
        if self.active_parts:
            min_slack = min(calculate_slack_time(p, self.env.now, self.queues, WORKSTATIONS) 
                           for p in self.active_parts)
            if abs(min_slack) > 1e-6:
                relative_urgency = np.clip(time_slack / min_slack, -2.0, 2.0)

        feature_list = [
            exists,
            np.clip(normalized_time_slack, -1.0, 1.0),
            np.clip(normalized_remaining_ops, 0, 1.0),
            np.clip(normalized_total_remaining_time, 0, 1.0),
            np.clip(normalized_op_duration, 0, 1.0),
            downstream_congestion,
            priority,
            is_final_op,
            product_id,
            is_late,
            is_next_bottleneck,
            relative_urgency,
        ]
        
        return np.array(feature_list, dtype=np.float32)


    def get_global_state(self) -> np.ndarray:
        """🔧 MAPPO关键修复：获取真正的全局状态，包含环境全局信息而非局部观察拼接"""
        global_features = []
        
        # 1. 环境时间信息
        time_normalized = self.env.now / SIMULATION_TIME
        global_features.append(time_normalized)
        
        # 2. 全局任务进度
        total_parts_in_system = sum(order.quantity for order in self.orders)
        completed_parts_ratio = len(self.completed_parts) / total_parts_in_system if total_parts_in_system > 0 else 0.0
        active_parts_ratio = len(self.active_parts) / total_parts_in_system if total_parts_in_system > 0 else 0.0
        global_features.extend([completed_parts_ratio, active_parts_ratio])
        
        # 3. 所有工作站的汇总状态（顺序与agents一致）
        for station_name in WORKSTATIONS.keys():
            # 队列长度归一化
            queue_len = len(self.queues[station_name].items)
            queue_len_norm = queue_len / ENHANCED_OBS_CONFIG["w_station_capacity_norm"]
            global_features.append(np.clip(queue_len_norm, 0, 1.0))
            
            # 设备忙碌率
            capacity = WORKSTATIONS[station_name]['count']
            busy_ratio = self.equipment_status[station_name]['busy_count'] / capacity
            global_features.append(busy_ratio)
            
            # 设备故障状态
            is_failed = 1.0 if self.equipment_status[station_name]['is_failed'] else 0.0
            global_features.append(is_failed)
        
        # 4. 全局订单紧急度统计
        critical_parts_count = 0
        urgent_parts_count = 0
        
        for part in self.active_parts:
            slack_time = calculate_slack_time(part, self.env.now, self.queues, WORKSTATIONS)
            if slack_time < -60:  # 严重延期
                critical_parts_count += 1
            elif slack_time < 0:  # 一般延期
                urgent_parts_count += 1
        
        critical_parts_ratio = critical_parts_count / len(self.active_parts) if self.active_parts else 0.0
        urgent_parts_ratio = urgent_parts_count / len(self.active_parts) if self.active_parts else 0.0
        global_features.extend([critical_parts_ratio, urgent_parts_ratio])
        
        # 5. 全局KPI趋势（修复：使用累积利用率而非瞬时值）
        cumulative_utilizations = []
        for station_name in WORKSTATIONS.keys():
            status = self.equipment_status[station_name]
            capacity = WORKSTATIONS[station_name]['count']
            
            # 专家修复：计算到当前时间的累积利用率，提供稳定信号
            # 结算从 last_event_time 到当前时间的忙碌面积
            if self.env.now > status.get('last_event_time', 0.0):
                elapsed = self.env.now - status.get('last_event_time', 0.0)
                busy_count = status.get('busy_count', 0)
                # 这个更新是临时的，不会写回status字典，仅用于计算当前全局状态
                current_busy_machine_time = status.get('busy_machine_time', 0.0) + elapsed * busy_count
            else:
                current_busy_machine_time = status.get('busy_machine_time', 0.0)

            if self.env.now > 0 and capacity > 0:
                utilization = current_busy_machine_time / (self.env.now * capacity)
                cumulative_utilizations.append(np.clip(utilization, 0.0, 1.0))
            else:
                cumulative_utilizations.append(0.0)
                
        avg_cumulative_utilization = np.mean(cumulative_utilizations) if cumulative_utilizations else 0.0
        
        global_features.append(avg_cumulative_utilization)
        
        return np.array(global_features, dtype=np.float32)

    def step_with_actions(self, actions: Dict[str, int]) -> Dict[str, float]:
        """
        方案B：执行一步仿真，使用全局工件选择
        """
        # 记录执行前状态
        prev_completed = len(self.completed_parts)
        prev_total_steps = sum(part.current_step for part in self.active_parts)
        
        # 执行智能体动作
        actions_executed = 0
        decision_time = self.env.now
        action_context: Dict[str, Dict[str, Any]] = {}

        for agent_id, action in actions.items():
            station_name = agent_id.replace("agent_", "")
            pre_queue_snapshot = list(self.queues[station_name].items)

            context = {
                "queue_len_before": len(pre_queue_snapshot),
                "queue_snapshot": pre_queue_snapshot,
                "decision_time": decision_time,
                "action": action,
                "selected_part": None,
                "processed": False,
                "started_parts": []  # 记录本步该agent启动的所有零件及其决策时slack
            }
            action_context[agent_id] = context

            # 方案B：使用新的全局工件选择机制
            if action > 0:
                result = self._select_workpiece_by_action(station_name, action)
                if result is not None:
                    selected_part, part_index = result
                    context["selected_part"] = selected_part
                    context["selected_part_slack"] = calculate_slack_time(selected_part, decision_time, self.queues, WORKSTATIONS)
                    context["orig_index_before"] = part_index
                    self._process_part_at_station(station_name, part_index=part_index)
                    context["processed"] = True
                    # 记录启动的零件
                    context["started_parts"].append({
                        "part_id": selected_part.part_id,
                        "slack": context["selected_part_slack"]
                    })
                    # 🔧 修复：仅在当前决策步“乐观预估可用名额”范围内补充启动，避免预加载清空队列
                    available_capacity = max(0, WORKSTATIONS[station_name]['count'] - self.equipment_status[station_name]['busy_count'])
                    # 已经启动了一个
                    remaining_slots = max(0, available_capacity - 1)
                    for _ in range(remaining_slots):
                        if len(self.queues[station_name].items) == 0:
                            break
                        extra = self._select_workpiece_by_action(station_name, action)
                        if extra is None:
                            break
                        extra_part, extra_index = extra
                        extra_slack = calculate_slack_time(extra_part, decision_time, self.queues, WORKSTATIONS)
                        self._process_part_at_station(station_name, part_index=extra_index)
                        # 记录并行启动的额外零件
                        context["started_parts"].append({
                            "part_id": extra_part.part_id,
                            "slack": extra_slack
                        })
                        # 不重复累计 actions_executed，这些属于同一次决策下的并行启动
            if context.get("processed"):
                actions_executed += 1
        
        # 推进仿真
        try:
            self.env.run(until=self.env.now + 1)
        except simpy.core.EmptySchedule:
            self.simulation_ended = True
        
        self.current_time = self.env.now
        
        # 计算奖励
        rewards = self.get_rewards(actions, action_context)

        # 本步结束后清空候选缓存（下一步将重建）
        self._cached_candidates.clear()
        
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
            start_time = self.env.now
            processing_time = part.get_processing_time()
            yield self.env.timeout(processing_time)
            finish_time = self.env.now
            
            # 新增：记录加工历史用于生成甘特图
            self.gantt_chart_history.append({
                "Task": f"Part-{part.part_id}",
                "Start": start_time,
                "Finish": finish_time,
                "Duration": finish_time - start_time,
                "Resource": station_name,
                "Product": part.product_type,
                "Part ID": part.part_id,
                "Order ID": part.order_id
            })
            
            # 更新设备状态
            self._update_equipment_status(station_name, busy=False)
            
            # 专家修复 V3.1：修正错误的属性访问，应为 part.contribution_map
            part.contribution_map[station_name] = part.contribution_map.get(station_name, 0.0) + processing_time
            
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
    
    def get_rewards(self, actions: Dict[str, int], action_context: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """融合版奖励函数：子目标主干 + 行为底线 + WIP + 启发式护栏(退火) + 超越EDD塑造"""
        rewards = {f"agent_{station}": 0.0 for station in WORKSTATIONS.keys()}
        
        # 读取退火与护栏配置
        anneal_cfg = globals().get('REWARD_ANNEALING_CONFIG', {"ANNEALING_END_EPISODE": 500})
        guard_cfg = globals().get('HEURISTIC_GUARDRAILS_CONFIG', {"enabled": False})
        
        current_episode = int(self.config.get('current_episode', 0))
        anneal_end = max(1, int(anneal_cfg.get('ANNEALING_END_EPISODE', 500)))
        shaping_strength = max(0.0, 1.0 - (current_episode / anneal_end))
        
        # === -1. 迟期差分密集奖励（主信号，归一+均分）===
        delta_coeff = REWARD_CONFIG.get("dense_tardiness_delta_coeff", 0.0)
        if delta_coeff != 0.0:
            current_overdue_sum = 0.0
            if self.active_parts:
                for p in self.active_parts:
                    slack = calculate_slack_time(p, self.env.now, self.queues, WORKSTATIONS)
                    if slack < 0:
                        current_overdue_sum += (-slack)
            # 增量：S(t-1) - S(t)，下降即正奖励
            delta = (self._last_overdue_sum - current_overdue_sum)
            # 归一：按time_slack_norm缩放，再按agent数均分
            norm = ENHANCED_OBS_CONFIG.get("time_slack_norm", 480.0)
            per_agent_reward = delta_coeff * (delta / (norm + 1e-6)) / max(1, len(WORKSTATIONS))
            if per_agent_reward != 0.0:
                for agent_id in rewards.keys():
                    rewards[agent_id] += per_agent_reward
            # 更新缓存
            self._last_overdue_sum = current_overdue_sum

        # === -0. 事件驱动：等待迟期惩罚（按工作站归因）===
        waiting_coeff = REWARD_CONFIG.get("waiting_overdue_penalty_per_part", 0.0)
        if waiting_coeff != 0.0:
            for station_name in WORKSTATIONS.keys():
                tardy_waiting_count = 0
                for part in self.queues[station_name].items:
                    slack = calculate_slack_time(part, self.env.now, self.queues, WORKSTATIONS)
                    if slack < 0:
                        tardy_waiting_count += 1
                if tardy_waiting_count > 0:
                    rewards[f"agent_{station_name}"] += waiting_coeff * tardy_waiting_count

        # === -0.1 事件驱动：启动迟期奖励（按动作归因）===
        start_overdue_coeff = REWARD_CONFIG.get("start_overdue_reward_coeff", 0.0)
        if start_overdue_coeff != 0.0:
            norm = ENHANCED_OBS_CONFIG.get("time_slack_norm", 480.0)
            for agent_id, context in action_context.items():
                started_parts = context.get("started_parts", [])
                for sp in started_parts:
                    slack_at_decision = float(sp.get("slack", 0.0))
                    if slack_at_decision < 0:
                        rewards[agent_id] += start_overdue_coeff * ((-slack_at_decision) / (norm + 1e-6))

        # === 0. 无效动作与不必要闲置：行为底线 ===
        for agent_id, action in actions.items():
            context = action_context.get(agent_id, {})
            queue_len_before = context.get("queue_len_before", 0)
            if action > 0:
                if context.get("selected_part") is None:
                    rewards[agent_id] += REWARD_CONFIG.get("invalid_action_penalty", 0.0)
            else:
                if queue_len_before > 0:
                    rewards[agent_id] += REWARD_CONFIG.get("unnecessary_idle_penalty", 0.0)
        
        # === 🔧 新增0.5：瓶颈感知奖励（退火）===
        bottleneck_bonus = REWARD_CONFIG.get("bottleneck_awareness_bonus", 0.0)
        if bottleneck_bonus > 0:
            # 识别瓶颈工作站（队列最长的）
            max_queue_len = 0
            bottleneck_station = None
            for station_name in WORKSTATIONS.keys():
                queue_len = len(self.queues[station_name].items)
                if queue_len > max_queue_len:
                    max_queue_len = queue_len
                    bottleneck_station = station_name
            
            # 奖励那些处理了即将进入瓶颈的零件的agent
            if bottleneck_station:
                for agent_id, action in actions.items():
                    if action > 0:
                        context = action_context.get(agent_id, {})
                        selected_part = context.get("selected_part")
                        if selected_part:
                            route = get_route_for_product(selected_part.product_type)
                            # 检查该零件的下一个工作站是否是瓶颈
                            if selected_part.current_step < len(route) - 1:
                                next_station = route[selected_part.current_step + 1]["station"]
                                if next_station == bottleneck_station:
                                    rewards[agent_id] += bottleneck_bonus * shaping_strength
        
        # === 🔧 新增0.6：短工序优先奖励（SPT策略融合，退火）===
        sjf_bonus = REWARD_CONFIG.get("short_job_first_bonus", 0.0)
        if sjf_bonus > 0:
            for agent_id, action in actions.items():
                if action > 0:  # 非闲置动作
                    context = action_context.get(agent_id, {})
                    selected_part = context.get("selected_part")
                    if selected_part:
                        # 计算选中零件的当前工序时长
                        current_op_duration = selected_part.get_processing_time()
                        # 计算时间松弛度（正值表示宽裕）
                        time_slack = calculate_slack_time(selected_part, self.env.now, self.queues, WORKSTATIONS)
                        
                        # 只有在交期较宽裕（松弛度>60分钟）且选择了短工序时才奖励
                        # 这避免了为了短工序而牺牲紧急订单
                        if time_slack > 60 and current_op_duration <= 20:  # 短工序定义：≤20分钟
                            # 工序越短，奖励越高
                            bonus_scale = (20 - current_op_duration) / 20.0  # 0-1之间
                            rewards[agent_id] += (sjf_bonus * shaping_strength) * bonus_scale
        
        # === 🔧 新增0.7：负载均衡奖励（退火）===
        lb_bonus = REWARD_CONFIG.get("load_balancing_bonus", 0.0)
        if lb_bonus > 0:
            for agent_id, action in actions.items():
                if action > 0:
                    context = action_context.get(agent_id, {})
                    selected_part = context.get("selected_part")
                    if selected_part:
                        route = get_route_for_product(selected_part.product_type)
                        # 检查下一个工位是否会因此次决策过载
                        if selected_part.current_step + 1 < len(route):
                            next_station = route[selected_part.current_step + 1]["station"]
                            next_queue_len = len(self.queues[next_station].items)
                            # 计算所有工作站的平均队列长度
                            avg_queue_len = sum(len(q.items) for q in self.queues.values()) / len(self.queues)
                            
                            # 如果下游队列显著低于平均水平，给予奖励
                            if next_queue_len < avg_queue_len * 0.8:
                                rewards[agent_id] += lb_bonus * shaping_strength
        
        # === 1. 事件驱动奖励：新完成零件按时/延期 ===
        # 专家修复 V3：实现基于贡献时间的加权信用分配
        current_completed = len(self.completed_parts)
        new_completed_parts_count = current_completed - self.stats.get('last_completed_count', 0)
        self.stats['last_completed_count'] = current_completed
        
        if new_completed_parts_count > 0:
            recent_completed = self.completed_parts[-new_completed_parts_count:]
            for part in recent_completed:
                tardiness = max(0.0, part.completion_time - part.due_date)
                
                # 🔧 新增：提前完成奖励
                early_bonus_mult = REWARD_CONFIG.get("early_completion_bonus_multiplier", 1.0)
                
                # 确定奖励值
                if tardiness > 0:
                    part_reward = REWARD_CONFIG.get("tardiness_penalty_scaler", -1.0) * (tardiness / 480.0)
                else:
                    base_reward = REWARD_CONFIG.get("on_time_completion_reward", 0.0)
                    # 提前完成的时间越多，奖励越高
                    early_time = part.due_date - part.completion_time
                    if early_time > 0:
                        part_reward = base_reward * (1.0 + (early_time / part.due_date) * (early_bonus_mult - 1.0))
                    else:
                        part_reward = base_reward
                
                # 🔧 修复：基于调度决策重要性的均匀信用分配
                # 避免按加工时间分配造成的学习信号偏差
                if part.contribution_map:
                    # 均匀分配给所有参与加工的工作站
                    participating_stations = list(part.contribution_map.keys())
                    equal_weight = 1.0 / len(participating_stations)
                    for station_name in participating_stations:
                        agent_id = f"agent_{station_name}"
                        if agent_id in rewards:
                            rewards[agent_id] += part_reward * equal_weight
        
        # === 2. 启发式护栏（仅在极端错误时介入，且随训练退火） ===
        if guard_cfg.get('enabled', False) and shaping_strength > 0.0:
            # 核心逻辑：仅在非课程学习模式，或课程学习的“完整挑战”阶段启用护栏
            activate_guardrails = True
            
            # 检查是否在课程学习模式下
            is_in_curriculum = 'stage_name' in self.config
            if is_in_curriculum:
                # 如果在课程学习中，只有“完整挑战”阶段才激活
                if self.config.get('stage_name') != "完整挑战":
                    activate_guardrails = False
            
            if activate_guardrails:
                critical_thr = float(guard_cfg.get('critical_slack_threshold', -60.0))
                safe_thr = float(guard_cfg.get('safe_slack_threshold', 120.0))
                penalty_base = float(guard_cfg.get('critical_choice_penalty', 0.5))
                
                for agent_id, action in actions.items():
                    if action <= 0:
                        continue
                    context = action_context.get(agent_id, {})
                    selected_part = context.get("selected_part")
                    if selected_part is None:
                        continue
                    decision_time = context.get("decision_time", self.current_time)
                    queue_snapshot = context.get("queue_snapshot", [])
                    chosen_slack = context.get("selected_part_slack")
                    if chosen_slack is None:
                        chosen_slack = calculate_slack_time(selected_part, decision_time, self.queues, WORKSTATIONS)
                    
                    # 是否存在"火烧眉毛"的零件
                    exists_critical = any(calculate_slack_time(p, decision_time, self.queues, WORKSTATIONS) < critical_thr for p in queue_snapshot)
                    # 选择是否“很安全”的零件
                    chosen_is_safe = chosen_slack > safe_thr
                    
                    if exists_critical and chosen_is_safe:
                        rewards[agent_id] -= penalty_base * shaping_strength
        
        # === 3. 本地化拥堵惩罚 (已移除，保持奖励配置一致性) ===
        
        # === 4. 终局大奖（全部完成） ===
        if self.is_done():
            total_required = sum(order.quantity for order in self.orders)
            if len(self.completed_parts) >= total_required:
                # 防重复发放：仅首次触发时发放终局奖励
                if not self.final_bonus_awarded:
                    final_bonus = REWARD_CONFIG.get("final_all_parts_completion_bonus", 0.0)
                    for agent_id in rewards:
                        rewards[agent_id] += final_bonus
                    self.final_bonus_awarded = True
                    self.final_bonus_value = final_bonus * len(rewards)
        
        # 更新订单进度与统计
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
                # 订单已完成
                completion_time = self.order_completion_times[order.order_id]
                if completion_time > order.due_date:
                    tardiness = completion_time - order.due_date
                    total_tardiness += tardiness
                    late_orders_count += 1
            else:
                # 订单未完成，延期时间从交期算到仿真结束
                tardiness = max(0, self.current_time - order.due_date)
                total_tardiness += tardiness
                if tardiness > 0:
                    late_orders_count += 1
        
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
                # 关键：如果没有零件完成，则将makespan设为当前耗尽的时间
                makespan = self.current_time
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
        
        # 🔧 MAPPO修复：重新设计全局状态空间
        self._setup_spaces()
        obs_shape = self._get_obs_shape()
        
        # 计算真正的全局状态维度
        # 1. 环境时间：1维
        # 2. 全局任务进度：2维 (completed_ratio, active_ratio)
        # 3. 工作站状态：5个工作站 × 3个特征 = 15维
        # 4. 紧急度统计：2维 (critical_ratio, urgent_ratio)  
        # 5. 全局KPI：1维 (avg_cumulative_utilization) - 专家修复V2
        global_state_dim = 1 + 2 + len(WORKSTATIONS) * 3 + 2 + 1
        self.global_state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(global_state_dim,), dtype=np.float32)
        
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
    try:
        import multiprocessing as _mp
        is_main_process = (_mp.current_process().name == 'MainProcess')
    except Exception:
        is_main_process = True
    return WFactoryEnv(config) 