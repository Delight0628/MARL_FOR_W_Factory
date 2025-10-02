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
    计算零件的松弛时间 (Slack Time) - 改进版本，考虑队列等待时间
    
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
    
    # 如果提供了队列和工作站信息，则考虑等待时间
    if queues is not None and workstations is not None:
        try:
            from .w_factory_config import calculate_estimated_waiting_time, WORKSTATIONS
            estimated_waiting = calculate_estimated_waiting_time(part, current_time, queues, WORKSTATIONS)
            # 修正后的松弛时间 = 基础松弛时间 - 估算等待时间
            return basic_slack - estimated_waiting
        except (ImportError, Exception):
            # 如果导入失败或计算出错，回退到基础计算
            pass
    
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
        self.processing_history = []
        # 专家修复 V3：追踪贡献时间，用于加权信用分配
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
        V3 融合版：获取智能体的最优观测状态
        - 包含三大部分：
          1. 智能体自身特征 (我是谁，我的状态如何)
          2. 全局宏观特征 (工厂整体情况如何)
          3. 队列中工件的详细特征 (我面前的任务是什么)
        """
        station_name = agent_id.replace("agent_", "")
        
        # --- 1. 智能体自身特征 (Agent Features) ---
        # 恢复智能体身份和能力信息
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

        # --- 2. 全局宏观特征 (Global Features) ---
        # 采用新版更优的全局特征
        time_normalized = self.env.now / SIMULATION_TIME
        total_parts_in_system = sum(order.quantity for order in self.orders)
        wip_normalized = len(self.active_parts) / total_parts_in_system if total_parts_in_system > 0 else 0.0
        
        global_features = np.array([
            time_normalized,
            wip_normalized
        ], dtype=np.float32)

        # --- 3. 队列中工件的详细特征 (Workpiece Features) ---
        # 专家修复 V3：由于One-Hot编码导致特征维度变化，需重新计算空槽位的维度
        workpiece_feature_dim = 8 + len(PRODUCT_ROUTES)
        
        workpiece_features_list = []
        
        # 🔧 关键修复：确保观测空间与动作空间一致
        # 如果启用了排序视图，则状态观测也必须基于排序后的队列
        queue_view_enabled = bool(globals().get('QUEUE_VIEW_CONFIG', {}).get("enabled", False))

        if queue_view_enabled:
            # 使用排序后的视图来构建观测
            sorted_view = self._get_sorted_queue_view(station_name)
            for i in range(ENHANCED_OBS_CONFIG["obs_slot_size"]):
                if i < len(sorted_view):
                    # 从排序后的视图中获取零件
                    part = sorted_view[i]["part"]
                    workpiece_features = self._get_workpiece_obs(part)
                else:
                    # 空槽位用0填充
                    workpiece_features = np.zeros(workpiece_feature_dim, dtype=np.float32)
                workpiece_features_list.append(workpiece_features)
        else:
            # 保持原始逻辑：使用原始队列顺序
            queue = self.queues[station_name].items
            for i in range(ENHANCED_OBS_CONFIG["obs_slot_size"]):
                if i < len(queue):
                    part = queue[i]
                    workpiece_features = self._get_workpiece_obs(part)
                else:
                    # 使用0填充空槽位, 第一个特征"exists"为0
                    workpiece_features = np.zeros(workpiece_feature_dim, dtype=np.float32)
                workpiece_features_list.append(workpiece_features)
        obs_queue = np.concatenate(workpiece_features_list)
        
        # 组合所有特征
        full_obs = np.concatenate([agent_features, global_features, obs_queue])
        return full_obs.flatten()

    def _get_sorted_queue_view(self, station_name: str, queue_items: Optional[List['Part']] = None):
        """
        返回按"紧急度优先"排序后的视图（仅用于状态与动作映射）：
        排序键: (是否已/将延期优先, 松弛时间小优先, 残余工序少优先, 下游拥堵小优先)
        返回: 列表[ {"part": Part, "orig_index": int, "features": np.ndarray(9,), "key": tuple } ]
        """
        queue_items = queue_items if queue_items is not None else self.queues[station_name].items
        view = []
        for idx, part in enumerate(queue_items):
            feats = self._get_workpiece_obs(part)  # 9维
            # 从特征中提取排序关键信息
            slack_norm = feats[1]  # 时间松弛度
            rem_ops_norm = feats[2]  # 剩余工序数
            downstream = feats[5]  # 下游拥堵情况
            
            # 判断是否已延期或即将延期
            time_slack = calculate_slack_time(part, self.env.now, self.queues, WORKSTATIONS)
            is_late_soon = 1.0 if time_slack < 0 else 0.0
            
            # 已/将延期优先 -> late_flag 越小越优
            late_flag = 0.0 if is_late_soon >= 0.5 else 1.0
            key = (late_flag, slack_norm, rem_ops_norm, downstream)
            view.append({"part": part, "orig_index": idx, "features": feats, "key": key})
        
        view.sort(key=lambda x: x["key"]) 
        top_k = ENHANCED_OBS_CONFIG["obs_slot_size"]
        return view[:top_k]

    def _get_workpiece_obs(self, part: Part) -> np.ndarray:
        """V3 融合版：获取单个工件的最优观测特征"""
        
        # 特征1: 是否存在
        exists = 1.0
        
        # 特征2: 时间松弛度（改进版本，考虑队列等待时间）
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
        
        # 特征6: 是否即将延期 (二进制信号) - V4版优化: 移除此冗余特征
        # is_late_soon = 1.0 if time_slack < 0 else 0.0

        # 特征7: 下游拥堵情况 (现为特征6) —— 按零件工艺动态计算下一工位
        downstream_congestion = 0.0
        route = get_route_for_product(part.product_type)
        if part.current_step < len(route) - 1:
            downstream_station = route[part.current_step + 1]["station"]
            if downstream_station in self.queues:
                congestion = len(self.queues[downstream_station].items) / ENHANCED_OBS_CONFIG["w_station_capacity_norm"]
                downstream_congestion = np.clip(congestion, 0, 1.0)
        
        # --- 恢复的关键特征 ---
        # 特征8: 订单优先级 (现为特征7)
        priority = part.priority / 5.0 # 假设最高优先级为5

        # 特征9: 是否为最终工序 (现为特征8)
        is_final_op = 1.0 if remaining_ops <= 1 else 0.0

        # 特征10: 零件类型编码 (现为特征9)
        product_types = list(PRODUCT_ROUTES.keys())
        product_index = product_types.index(part.product_type) if part.product_type in product_types else -1
        product_type_encoded = (product_index + 1) / len(product_types)

        # 专家修复 V3：实现产品类型的One-Hot编码
        product_types = list(PRODUCT_ROUTES.keys())
        num_product_types = len(product_types)
        product_type_one_hot = np.zeros(num_product_types, dtype=np.float32)
        if part.product_type in product_types:
            product_index = product_types.index(part.product_type)
            product_type_one_hot[product_index] = 1.0

        feature_list = [
            exists,
            np.clip(normalized_time_slack, -1.0, 1.0),
            np.clip(normalized_remaining_ops, 0, 1.0),
            np.clip(normalized_total_remaining_time, 0, 1.0),
            np.clip(normalized_op_duration, 0, 1.0),
            downstream_congestion,
            priority,
            is_final_op,
        ]
        
        return np.concatenate([np.array(feature_list, dtype=np.float32), product_type_one_hot])


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
        
        # 3. 所有工作站的汇总状态（固定顺序）
        for station_name in sorted(WORKSTATIONS.keys()):
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
        """执行一步仿真，传入智能体动作"""
        # 记录执行前状态
        prev_completed = len(self.completed_parts)
        prev_total_steps = sum(part.current_step for part in self.active_parts)
        
        # 执行智能体动作
        actions_executed = 0
        queue_view_enabled = bool(globals().get('QUEUE_VIEW_CONFIG', {}).get("enabled", False))
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
                "processed": False
            }
            action_context[agent_id] = context

            # V8 支持紧急度排序视图的动作空间 (0=IDLE, 1=处理最紧急, 2=处理次紧急, ...)
            if action > 0:
                chosen_view_idx = action - 1
                if queue_view_enabled:
                    sorted_view = self._get_sorted_queue_view(station_name, queue_items=pre_queue_snapshot)
                    context["sorted_view"] = sorted_view
                    if chosen_view_idx < len(sorted_view):
                        orig_index = sorted_view[chosen_view_idx]["orig_index"]
                        if orig_index < len(self.queues[station_name].items):
                            selected_part = sorted_view[chosen_view_idx]["part"]
                            context["selected_part"] = selected_part
                            context["selected_part_slack"] = calculate_slack_time(selected_part, decision_time, self.queues, WORKSTATIONS)
                            context["orig_index_before"] = orig_index
                            self._process_part_at_station(station_name, part_index=orig_index)
                            context["processed"] = True
                else:
                    if chosen_view_idx < len(pre_queue_snapshot):
                        selected_part = pre_queue_snapshot[chosen_view_idx]
                        context["selected_part"] = selected_part
                        context["selected_part_slack"] = calculate_slack_time(selected_part, decision_time, self.queues, WORKSTATIONS)
                        context["orig_index_before"] = chosen_view_idx
                        self._process_part_at_station(station_name, part_index=chosen_view_idx)
                        context["processed"] = True

            if context.get("processed"):
                actions_executed += 1
        
        # 推进仿真 - 减少步长以获得更精细的控制
        try:
            self.env.run(until=self.env.now + 1)  # 每步推进1分钟而不是5分钟
        except simpy.core.EmptySchedule:
            self.simulation_ended = True
        
        self.current_time = self.env.now
        
        # 计算奖励
        rewards = self.get_rewards(actions, action_context)
        
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
        """融合版奖励函数：子目标主干 + 行为底线 + WIP + 启发式护栏(退火)"""
        rewards = {f"agent_{station}": 0.0 for station in WORKSTATIONS.keys()}
        
        # 读取退火与护栏配置
        anneal_cfg = globals().get('REWARD_ANNEALING_CONFIG', {"ANNEALING_END_EPISODE": 500})
        guard_cfg = globals().get('HEURISTIC_GUARDRAILS_CONFIG', {"enabled": False})
        
        current_episode = int(self.config.get('current_episode', 0))
        anneal_end = max(1, int(anneal_cfg.get('ANNEALING_END_EPISODE', 500)))
        shaping_strength = max(0.0, 1.0 - (current_episode / anneal_end))
        
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
        
        # === 1. 事件驱动奖励：新完成零件按时/延期 ===
        # 专家修复 V3：实现基于贡献时间的加权信用分配
        current_completed = len(self.completed_parts)
        new_completed_parts_count = current_completed - self.stats.get('last_completed_count', 0)
        self.stats['last_completed_count'] = current_completed
        
        if new_completed_parts_count > 0:
            recent_completed = self.completed_parts[-new_completed_parts_count:]
            for part in recent_completed:
                tardiness = max(0.0, part.completion_time - part.due_date)
                
                # 确定奖励值
                if tardiness > 0:
                    part_reward = REWARD_CONFIG.get("tardiness_penalty_scaler", -1.0) * (tardiness / 480.0)
                else:
                    part_reward = REWARD_CONFIG.get("on_time_completion_reward", 0.0)
                
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
                # 关键：如果没有零件完成，显示0而不是1200
                makespan = 0
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
    if config and any(key in config for key in ['orders_scale', 'time_scale', 'stage_name']) and os.getpid() == os.getppid():
        print(f"🏭 创建环境 - 课程学习配置: {config.get('stage_name', 'Unknown')}")
        print(f"   订单比例: {config.get('orders_scale', 1.0)}, 时间比例: {config.get('time_scale', 1.0)}")
    
    return WFactoryEnv(config) 