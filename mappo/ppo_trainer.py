"""
MAPPO训练器模块
==================
实现SimplePPOTrainer类，管理完整的训练流程

核心功能：
1. 两阶段渐进式训练（基础泛化 → 动态事件强化）
2. 课程学习支持（逐步增加任务难度）
3. 自适应熵调整（探索-利用平衡）
4. 分阶段模型保存（双达标最佳模型跟踪）
5. TensorBoard实时监控
6. 并行经验采集（ProcessPoolExecutor）
"""

import os
import sys
import time
import random
import socket
import numpy as np
import tensorflow as tf
import gymnasium as gym
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *
from mappo.ppo_buffer import ExperienceBuffer
from mappo.ppo_network import PPONetwork
from mappo.ppo_worker import _collect_experience_wrapper

TENSORBOARD_AVAILABLE = hasattr(tf.summary, "create_file_writer")


class RunningMeanStd:
    """
    🔧 新增：运行均值和标准差归一化器
    用于对观测、回报等进行在线归一化，提升训练稳定性
    """
    def __init__(self, shape: tuple, epsilon: float = 1e-8):
        self.shape = shape
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 0
    
    def update(self, x: np.ndarray):
        """更新统计量"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        # 增量更新
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean += delta * batch_count / max(total_count, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / max(total_count, 1)
        self.var = M2 / max(total_count, 1)
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """归一化输入"""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)


class SimplePPOTrainer:
    """
    MAPPO自适应训练器
    
    实现两阶段渐进式训练策略：
    
    【阶段一】基础能力训练（随机订单泛化）
    - 目标：学习基本调度能力
    - 方法：随机订单 + 多任务混合（含BASE_ORDERS锚点）
    - 毕业标准：连续N次达到分数/完成率/延期阈值
    
    【阶段二】动态事件鲁棒性训练
    - 目标：提升应对突发事件的能力
    - 方法：随机订单 + 设备故障 + 紧急插单
    - 完成标准：连续M次达到更高的性能指标
    
    自适应机制：
    - 学习率衰减：PolynomialDecay
    - 熵系数调整：根据停滞等级阶梯式提升/衰减
    - 课程学习：逐步增加订单数量和时间压力
    """
    
    def __init__(self, initial_lr: float, total_train_episodes: int, steps_per_episode: int, training_targets: dict = None, models_root_dir: Optional[str] = None, logs_root_dir: Optional[str] = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        
        # 使用配置文件的系统资源配置
        self.num_workers = SYSTEM_CONFIG["num_parallel_workers"]
        print(f"使用 {self.num_workers} 个并行环境进行数据采集")
        
        # 使用配置文件的TensorFlow线程配置
        tf.config.threading.set_inter_op_parallelism_threads(SYSTEM_CONFIG["tf_inter_op_threads"])
        tf.config.threading.set_intra_op_parallelism_threads(SYSTEM_CONFIG["tf_intra_op_threads"])
        print(f"TensorFlow将使用 {SYSTEM_CONFIG['tf_inter_op_threads']}个inter线程, {SYSTEM_CONFIG['tf_intra_op_threads']}个intra线程")
        
        # 环境探测
        # 之前的代码依赖动态配置，现在我们直接创建
        temp_env = make_parallel_env()
        self.state_dim = temp_env.observation_space(temp_env.possible_agents[0]).shape[0]
        # 直接使用环境的动作空间对象以支持 MultiDiscrete
        self.action_space = temp_env.action_space(temp_env.possible_agents[0])
        self.agent_ids = temp_env.possible_agents
        self.num_agents = len(self.agent_ids)
        # Critic智能体条件化：将智能体one-hot并入全局状态输入维度
        self.global_state_dim = temp_env.global_state_space.shape[0] + self.num_agents
        temp_env.close()
        
        print("环境空间检测:")
        print(f"   观测维度: {self.state_dim}")
        print(f"   动作空间: {self.action_space}")
        print(f"   智能体数量: {len(self.agent_ids)}")
        print(f"   全局状态维度(含agent one-hot): {self.global_state_dim}")
        
        # 移除动态参数调整
        optimized_episodes = total_train_episodes
        optimized_steps = steps_per_episode
        # 评估步长与采集步长对齐，避免训练/评估不一致
        self.max_steps_for_eval = optimized_steps
        
        # 使用配置文件的学习率调度配置
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=LEARNING_RATE_CONFIG["initial_lr"],
            decay_steps=optimized_episodes * optimized_steps,
            end_learning_rate=LEARNING_RATE_CONFIG["end_lr"],
            power=LEARNING_RATE_CONFIG["decay_power"]
        )

        # 共享网络（传递动作空间对象，支持MultiDiscrete）
        self.shared_network = PPONetwork(
            state_dim=self.state_dim,
            action_space=self.action_space,
            lr=self.lr_schedule,
            global_state_dim=self.global_state_dim
        )
        
        # 训练统计
        self.episode_rewards = []
        self.training_losses = []
        self.iteration_times = []  # 记录每轮训练时间
        self.kpi_history = []      # 记录每轮KPI历史
        self.initial_lr = initial_lr  # 保存初始学习率
        self.start_time = time.time()
        self.start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        

        self.final_stage_best_kpi = {
            'mean_completed_parts': -1.0,
            'mean_makespan': float('inf'),
            'mean_utilization': 0.0,
            'mean_tardiness': float('inf')
        }
        self.final_stage_best_score = float('-inf')
        self.final_stage_best_episode = -1 # 记录最佳KPI的回合数
        
        # "双达标"最佳KPI跟踪器
        self.best_kpi_dual_objective = {
            'mean_completed_parts': -1.0,
            'mean_makespan': float('inf'),
            'mean_utilization': 0.0,
            'mean_tardiness': float('inf')
        }
        self.best_score_dual_objective = float('-inf')
        self.best_episode_dual_objective = -1

        # 训练流程由配置文件驱动
        self.training_flow_config = TRAINING_FLOW_CONFIG
        self.training_targets = self.training_flow_config["general_params"] # 通用参数
        
        # 自适应训练状态跟踪
        self.adaptive_state = {
            "target_achieved_count": 0,          # 连续达到目标的次数
            "best_performance": 0.0,             # 历史最佳性能
            "last_improvement_episode": 0,       # 上次改进的轮数
            "performance_history": [],           # 性能历史记录
            "training_phase": "exploration",     # 当前训练阶段：exploration, exploitation, fine_tuning
            "stagnation_counter": 0,             # 停滞计数器
            "last_stagnation_performance": -1.0, # 上一次停滞时的性能
        }
        # --- 方案二：升级自适应熵所需变量 ---
        self.epochs_without_improvement = 0
        self.stagnation_level = 0  # 新增：停滞等级，用于阶梯式提升熵
        
        # --- 新增：基础训练 + 随机领域强化 阶段管理 ---
        self.foundation_training_completed = False  # 基础训练是否完成
        self.generalization_phase_active = False   # 是否进入泛化强化阶段
        self.foundation_achievement_count = 0      # 基础训练连续达标次数
        self.generalization_achievement_count = 0  # 泛化阶段连续达标次数
        
        # --- 新增：为新两阶段方案的独立模型保存追踪 ---
        self.best_score_foundation_phase = float('-inf')    # 基础训练阶段最佳分数
        self.best_kpi_foundation_phase = {}         # 基础训练阶段最佳KPI
        self.best_episode_foundation_phase = -1    # 基础训练阶段最佳回合
        
        self.best_score_generalization_phase = float('-inf')  # 泛化强化阶段最佳分数
        self.best_kpi_generalization_phase = {}       # 泛化强化阶段最佳KPI
        self.best_episode_generalization_phase = -1  # 泛化强化阶段最佳回合
        
        # --- 新增：课程学习阶段的自适应毕业跟踪器 ---
        self.curriculum_stage_achievement_count = 0
        
        # 初始化动态训练参数
        self.current_entropy_coeff = PPO_NETWORK_CONFIG["entropy_coeff"] # 初始化动态熵系数
        self.current_learning_rate = LEARNING_RATE_CONFIG["initial_lr"] # 使用正确的学习率配置
        
        # 熵系数退火计划（改进版）
        self.entropy_decay_rate = 0.9995  # 🔧 更慢的衰减率，保持更长时间的探索
        self.min_entropy_coeff = 0.05     # 🔧 更高的最小熵系数，避免过早收敛
        
        
        # 回合事件日志记录器
        self.episode_events = []
        
        # 创建保存目录 (以训练开始时间创建专用文件夹)
        self.base_models_dir = models_root_dir if models_root_dir else "mappo/ppo_models"
        self.models_dir = f"{self.base_models_dir}/{self.start_time_str}"
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"模型保存目录: {self.models_dir}")
        
        # TensorBoard支持
        self.tensorboard_dir = (
            os.path.join(logs_root_dir, "tensorboard_logs", self.timestamp)
            if logs_root_dir else f"mappo/tensorboard_logs/{self.timestamp}"
        )
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        if TENSORBOARD_AVAILABLE:
            self.train_writer = None
            self.current_tensorboard_run_name = None
            # 为本次运行分配唯一端口
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("127.0.0.1", 0))
                self.tensorboard_port = sock.getsockname()[1]
                sock.close()
            except Exception:
                # 回退到常见端口范围内的伪随机端口
                self.tensorboard_port = 6006 + (hash(self.timestamp) % 1000)
            print(f"📊 TensorBoard命令: tensorboard --logdir=\"{self.tensorboard_dir}\" --port={self.tensorboard_port}")
        else:
            self.train_writer = None
            print("⚠️  TensorBoard不可用")
        
        # 10-22-10-52 修复：切换到进程池，彻底解决TensorFlow权重冲突问题
        # 说明：使用进程池确保每个worker在完全独立的Python进程中运行
        # 优点：完全隔离，避免TensorFlow变量名冲突和权重加载问题
        try:
            import multiprocessing as _mp
            _mp.set_start_method('spawn', force=True)
        except Exception:
            pass
        self.pool = ProcessPoolExecutor(max_workers=self.num_workers)

        # 🔧 初始化训练所需的关键成员
        self.seed = RANDOM_SEED
        self.total_steps = 0
        self.network_config = PPO_NETWORK_CONFIG
        
        # 🔧 新增：归一化器（观测、回报归一化，优势白化）
        self.obs_normalizer = RunningMeanStd(shape=(self.state_dim,))
        self.global_obs_normalizer = RunningMeanStd(shape=(self.global_state_dim,))
        self.reward_normalizer = RunningMeanStd(shape=(1,))
        self.normalize_obs = True
        self.normalize_rewards = True
        self.normalize_advantages = True  # 优势白化（已在buffer中实现，这里只是标志）
        
        # 10-23-18-00 核心改进：多任务混合机制贯穿两个阶段
        # 从foundation_phase和generalization_phase分别读取配置
        # 两阶段都使用25% BASE_ORDERS worker作为稳定锚点
        self.foundation_multi_task_config = TRAINING_FLOW_CONFIG["foundation_phase"].get("multi_task_mixing", {"enabled": False, "base_worker_fraction": 0.0, "randomize_base_env": False})
        self.generalization_multi_task_config = TRAINING_FLOW_CONFIG["generalization_phase"].get("multi_task_mixing", {"enabled": False, "base_worker_fraction": 0.0, "randomize_base_env": False})
        
        # 计算BASE_ORDERS worker数量（两阶段使用相同配置）
        base_fraction = float(self.foundation_multi_task_config.get("base_worker_fraction", 0.0))
        base_fraction = min(max(base_fraction, 0.0), 1.0)
 
    
    def should_continue_training(self, episode: int, current_score: float, completion_rate: float) -> tuple:
        """基于训练流程配置的阶段标准评估是否继续训练"""
        general = self.training_flow_config["general_params"]
        state = self.adaptive_state

        # 基本限制检查
        if episode >= general["max_episodes"]:
            return False, f"已达到最大训练轮数({general['max_episodes']})", 0

        # 按阶段选择标准
        if self.generalization_phase_active:
            criteria = self.training_flow_config["generalization_phase"]["completion_criteria"]
        else:
            criteria = self.training_flow_config["foundation_phase"]["graduation_criteria"]

        target_score = criteria["target_score"]
        min_completion_rate = criteria.get("min_completion_rate", 100.0)
        target_consistency = criteria["target_consistency"]

        # 达标计数逻辑
        if completion_rate >= min_completion_rate and current_score >= target_score:
            state["target_achieved_count"] += 1
            print(f"🎯 达标: 完成率 {completion_rate:.1f}% & 分数 {current_score:.3f} (连续第{state['target_achieved_count']}/{target_consistency}次)")
            if state["target_achieved_count"] >= target_consistency:
                return False, f"连续{target_consistency}次达到阶段标准", 0
        else:
            state["target_achieved_count"] = 0

        # 早停逻辑（基于分数停滞）
        state["performance_history"].append(current_score)
        if len(state["performance_history"]) > general["performance_window"]:
            state["performance_history"].pop(0)

        if current_score > state["best_performance"]:
            state["best_performance"] = current_score
            state["last_improvement_episode"] = episode

        improvement_gap = episode - state["last_improvement_episode"]
        if improvement_gap >= general["early_stop_patience"]:
            if len(state["performance_history"]) >= general["performance_window"]:
                recent_avg_score = sum(state["performance_history"]) / len(state["performance_history"])
                if recent_avg_score < target_score * 0.8:
                    return False, f"连续{improvement_gap}轮无改进，且平均分数低于{target_score*0.8:.3f}", 0

        return True, f"当前分数 {current_score:.3f}, 完成率 {completion_rate:.1f}%", 0
    
    def check_foundation_training_completion(self, kpi_results: Dict[str, float], current_score: float, curriculum_config: Optional[Dict] = None) -> bool:
        """
        检查基础训练是否达到毕业标准，由配置文件驱动
        
        10-27-17-30 修复：使用实际评估订单的目标零件数，而非固定的BASE_ORDERS，确保完成率计算与实际任务一致
        """
        criteria = self.training_flow_config["foundation_phase"]["graduation_criteria"]
        
        # 10-27-17-30 修复：使用_get_target_parts获取正确的目标零件数（与实际评估订单一致）
        total_parts_target = self._get_target_parts(curriculum_config)
        completion_rate_kpi = (kpi_results.get('mean_completed_parts', 0) / total_parts_target) * 100 if total_parts_target > 0 else 0
        # 🔧 上限裁剪，避免因动态插单导致>100%的显示与判定
        completion_rate_kpi = float(min(100.0, completion_rate_kpi))
        
        target_score = criteria["target_score"]
        stability_goal = criteria["target_consistency"]
        tardiness_threshold = criteria["tardiness_threshold"]
        min_completion_rate = criteria["min_completion_rate"]
        current_tardiness = kpi_results.get('mean_tardiness', float('inf'))

        conditions_met = {
            f"完成率达标(>={min_completion_rate}%)": completion_rate_kpi >= min_completion_rate,
            f"分数达标(>={target_score})": current_score >= target_score,
            f"延期达标(<={tardiness_threshold}min)": current_tardiness <= tardiness_threshold
        }

        if all(conditions_met.values()):
            self.foundation_achievement_count += 1
            print(f"🎯 基础训练达标: 完成率 {completion_rate_kpi:.1f}%, 分数 {current_score:.3f}, 延期 {current_tardiness:.1f}min (连续第{self.foundation_achievement_count}/{stability_goal}次)")
        else:
            if self.foundation_achievement_count > 0:
                reasons = [k for k, v in conditions_met.items() if not v]
                print(f"❌ 基础训练连续达标中断. 未达标项: {', '.join(reasons)}")
            self.foundation_achievement_count = 0

        if self.foundation_achievement_count >= stability_goal:
            print(f"🏆 基础训练完成！连续{stability_goal}次达到所有标准，准备进入泛化强化阶段。")
            return True
        return False
    
    def check_generalization_training_completion(self, current_score: float, completion_rate: float) -> bool:
        """检查泛化训练是否已达到最终训练完成的条件，由配置文件驱动"""
        criteria = self.training_flow_config["generalization_phase"]["completion_criteria"]
        
        target_score = criteria["target_score"]
        stability_goal = criteria["target_consistency"]
        min_completion_rate = criteria["min_completion_rate"]
        
        if completion_rate >= min_completion_rate and current_score >= target_score:
            self.generalization_achievement_count += 1
            print(f"🌟 泛化阶段达标: 完成率 {completion_rate:.1f}% & 分数 {current_score:.3f} (连续第{self.generalization_achievement_count}/{stability_goal}次)")
            
            if self.generalization_achievement_count >= stability_goal:
                print(f"🎉 泛化训练完成！模型已具备优秀的泛化能力。")
                return True
        else:
            self.generalization_achievement_count = 0
        
        return False
    
    def create_environment(self, curriculum_stage=None):
        """创建环境（支持课程学习）"""
        config = {}
        
        # 🔧 V16：实现课程学习的环境配置
        # 核心重构：课程学习逻辑现在由 TRAINING_FLOW_CONFIG 控制
        cl_config = self.training_flow_config["foundation_phase"]["curriculum_learning"]
        if curriculum_stage is not None and cl_config["enabled"]:
            stages = cl_config["stages"]
            stage = stages[curriculum_stage] if curriculum_stage < len(stages) else stages[-1]
            config['curriculum_stage'] = stage
            config['orders_scale'] = stage.get('orders_scale', 1.0)
            config['time_scale'] = stage.get('time_scale', 1.0)
            print(f"📚 课程学习阶段 {curriculum_stage+1}: {stage['name']} (订单比例: {stage['orders_scale']}, 时间倍数: {stage['time_scale']})")
        
        # 统一注入 MAX_SIM_STEPS
        config['MAX_SIM_STEPS'] = self.max_steps_for_eval
        env = make_parallel_env(config)
        buffers = {
            agent: ExperienceBuffer() 
            for agent in env.possible_agents
        }
        return env, buffers
    
    def collect_and_process_experience(self, num_steps: int, curriculum_config: Dict[str, Any] = None) -> Tuple[float, Optional[Dict[str, np.ndarray]]]:
        """
        并行采集经验并计算GAE优势函数
        
        核心设计：统一MDP保证训练稳定性
        - 同一回合所有worker使用相同订单配置（避免梯度冲突）
        - 回合间轮换任务类型（BASE_ORDERS vs 随机订单）
        - 根据训练阶段启用动态事件（设备故障、紧急插单）
        
        执行流程：
        1. 根据当前阶段生成本回合统一订单配置
        2. 并行提交N个worker任务到进程池
        3. 收集所有worker返回的经验缓冲区
        4. 对每个智能体的缓冲区计算GAE优势函数
        5. 聚合所有数据为统一训练批次
        
        Args:
            num_steps: 每个worker采集的最大步数
            curriculum_config: 课程学习配置（可选）
            
        Returns:
            tuple: (avg_reward, batch)
                - avg_reward: 所有worker的平均奖励
                - batch: 训练批次字典或None（失败时）
        """
        # 核心设计：统一MDP
        # 1. 同一回合所有worker使用相同订单配置（避免梯度冲突）
        # 2. 回合间轮换任务类型（BASE_ORDERS vs 随机订单）
        # 3. 每个新回合重新生成随机订单（保证泛化）
        
        # 根据当前阶段选择本回合使用的订单配置
        if not self.foundation_training_completed:
            # 阶段一：基础训练（随机订单泛化）
            current_mixing_config = self.foundation_multi_task_config
        else:
            # 阶段二：泛化强化（动态事件）
            current_mixing_config = self.generalization_multi_task_config
        
        # 10-23-20-00 核心改进：回合级别的任务轮换（而非worker级别）
        # 使用回合数来决定本回合使用哪种订单配置
        use_base_orders_this_episode = False
        if current_mixing_config.get("enabled", False):
            # 根据base_worker_fraction决定使用BASE_ORDERS的频率
            base_fraction = current_mixing_config.get("base_worker_fraction", 0.25)
            # 每4回合中有1回合使用BASE_ORDERS（如果base_fraction=0.25）
            cycle_length = int(1.0 / base_fraction) if base_fraction > 0 else 999999
            episode_in_cycle = (self.total_steps // num_steps) % cycle_length
            use_base_orders_this_episode = (episode_in_cycle == 0)
        
        # 生成本回合统一的订单配置
        if use_base_orders_this_episode:
            # 本回合所有worker使用BASE_ORDERS
            episode_orders = BASE_ORDERS
            episode_tag = "BASE_ORDERS"
        else:
            # 10-27-16-45 修复：为随机订单生成引入确定性种子（按回合索引），提高可复现性
            # 说明：仅在调用 generate_random_orders() 前暂时设置随机种子，调用后恢复原状态
            # 这样可以确保相同的训练seed与回合序号得到一致的订单集合，同时不影响全局随机流程
            episode_index = (self.total_steps // num_steps)
            _py_state = random.getstate()
            _np_state = np.random.get_state()
            try:
                random.seed(self.seed + 10007 * episode_index)
                np.random.seed(self.seed + 20011 * episode_index)
                episode_orders = generate_random_orders()
            finally:
                try:
                    random.setstate(_py_state)
                except Exception:
                    pass
                try:
                    np.random.set_state(_np_state)
                except Exception:
                    pass
            episode_tag = "随机订单"
        
        # 10-23-20-00 确定动态事件配置（所有worker统一）
        episode_equipment_failure_enabled = False
        episode_emergency_orders_enabled = False
        if self.generalization_phase_active:
            # 阶段二：启用动态事件
            dynamic_events = TRAINING_FLOW_CONFIG["generalization_phase"].get("dynamic_events", {})
            episode_equipment_failure_enabled = dynamic_events.get("equipment_failure_enabled", False)
            episode_emergency_orders_enabled = dynamic_events.get("emergency_orders_enabled", False)
        
        # --- 1. 并行运行worker收集数据 ---
        try:
            # 10-22-10-55 修复：使用模块级别包装函数（线程池模式）
            worker_args_list = []
            for i in range(self.num_workers):
                # 默认使用当前的课程配置
                worker_curriculum_config = curriculum_config.copy() if curriculum_config else {}
                worker_curriculum_config['worker_id'] = i
                
                # 10-23-20-00 核心修复：所有worker使用相同的订单配置
                # 设计理念：
                #   - 避免多MDP梯度冲突
                #   - 通过回合间轮换实现多任务混合（而非worker间）
                #   - 保证训练稳定性和收敛性
                
                # 10-23-20-00 所有worker使用本回合统一的订单和动态事件配置
                worker_curriculum_config['custom_orders'] = episode_orders
                worker_curriculum_config['randomize_env'] = (episode_tag != "BASE_ORDERS")
                worker_curriculum_config['equipment_failure_enabled'] = episode_equipment_failure_enabled
                worker_curriculum_config['emergency_orders_enabled'] = episode_emergency_orders_enabled
                # 统一步长：与采集步数一致
                worker_curriculum_config['MAX_SIM_STEPS'] = num_steps
                
                # 10-25-16-10 线程池模式优化：直接传递模型对象而非权重，避免重复构建网络
                # if self.pool_type == "ThreadPool":
                #     # 线程池：传递模型对象（线程共享内存，无需序列化）
                #     worker_curriculum_config['actor_model'] = self.shared_network.actor
                #     worker_curriculum_config['critic_model'] = self.shared_network.critic
                
                worker_args_list.append((
                    self.shared_network.actor.get_weights(),
                    self.shared_network.critic.get_weights(),
                    self.state_dim,
                    self.action_space, # 🔧 核心修复：传递action_space
                    num_steps,
                    self.seed + self.total_steps + i, # 每个worker有不同的seed
                    self.global_state_dim,
                    self.network_config,
                    worker_curriculum_config
                ))

            # 10-22-10-55 修复：使用模块级别包装函数，并行执行采集任务（线程池）
            futures = [self.pool.submit(_collect_experience_wrapper, args) for args in worker_args_list]
            results = [f.result() for f in futures]

        except Exception as e:
            print(f"❌ 并行工作进程失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, None

        # 10-27-16-30 修复：更健壮地处理worker失败返回空缓冲的情况
        if not results or all((not buffers) for (buffers, _, _, _, _) in results):
            print("⚠️ 所有worker均返回空缓冲，跳过本轮更新。")
            return 0.0, None

        total_reward = 0.0
        worker_rewards = []  # 🔧 新增：记录每个worker的奖励
        
        # 初始化用于聚合所有worker数据的列表
        all_states, all_global_states, all_actions, all_old_probs, all_advantages, all_returns = [], [], [], [], [], []

        for (buffers, ep_reward, last_values, any_terminated, _graduated) in results:
            # 汇总奖励
            total_reward += float(ep_reward)
            worker_rewards.append(float(ep_reward))

            # 将每个agent的缓冲区转换为GAE并聚合
            if not buffers:
                continue
            for agent_id, buf in buffers.items():
                if len(buf) == 0:
                    continue
                # 使用截断时的bootstrap值（如有）
                next_v = None
                if last_values is not None and agent_id in last_values:
                    next_v = float(last_values[agent_id])
                states, global_states, actions, old_probs, advantages, returns = buf.get_batch(
                    gamma=PPO_NETWORK_CONFIG.get("gamma", 0.99),
                    lam=PPO_NETWORK_CONFIG.get("lambda_gae", 0.95),
                    next_value_if_truncated=next_v,
                    advantage_clip_val=PPO_NETWORK_CONFIG.get("advantage_clip_val")
                )
                all_states.extend(states)
                all_global_states.extend(global_states)
                all_actions.extend(actions)
                all_old_probs.extend(old_probs)
                all_advantages.extend(advantages)
                all_returns.extend(returns)

        if len(all_states) == 0:
            # 返回时将完成统计编码在None批次旁边（通过总奖励的info在外层打印）
            self._last_collect_finished_workers = self.num_workers
            self._last_collect_completed_workers = 0
            self._last_collect_worker_rewards = worker_rewards  # 🔧 新增：保存worker奖励列表
            # 10-23-20-00 更新：保存回合级别的任务轮换信息（空批次返回）
            current_mixing_config = self.foundation_multi_task_config if not self.foundation_training_completed else self.generalization_multi_task_config
            mixing_enabled = current_mixing_config.get("enabled", False)
            self._last_collect_mixing_summary = {
                'enabled': bool(mixing_enabled),
                'episode_task': episode_tag,  # 本回合的任务类型
                'all_workers': int(self.num_workers),
                'avg_reward': float(np.mean(worker_rewards)) if worker_rewards else None,
            }
            
            # 10-23-20-15 保存本回合的实际环境配置（空批次情况）
            self._last_episode_config = {
                'custom_orders': episode_orders,
                'episode_tag': episode_tag,
                'equipment_failure_enabled': episode_equipment_failure_enabled,
                'emergency_orders_enabled': episode_emergency_orders_enabled,
            }
            
            avg_reward = total_reward / self.num_workers if self.num_workers > 0 else 0.0
            return avg_reward, None

        # 10-25-14-30 统计成功完成的workers（缓冲区非空即可视为成功）
        successful_workers = 0
        for (buffers, _, _, _, _) in results:
            if buffers:
                # 至少一个agent有数据
                if any(len(buf) > 0 for buf in buffers.values()):
                    successful_workers += 1

        # 将聚合后的数据列表转换为NumPy数组，形成最终的训练批次
        states_array = np.array(all_states, dtype=np.float32)
        global_states_array = np.array(all_global_states, dtype=np.float32)
        returns_array = np.array(all_returns, dtype=np.float32)
        
        # 🔧 新增：观测和回报归一化
        if self.normalize_obs and len(states_array) > 0:
            # 更新归一化器统计量
            self.obs_normalizer.update(states_array)
            self.global_obs_normalizer.update(global_states_array)
            # 归一化观测
            states_array = self.obs_normalizer.normalize(states_array)
            global_states_array = self.global_obs_normalizer.normalize(global_states_array)
        
        if self.normalize_rewards and len(returns_array) > 0:
            # 回报归一化（使用returns作为目标）
            self.reward_normalizer.update(returns_array.reshape(-1, 1))
            returns_array = self.reward_normalizer.normalize(returns_array.reshape(-1, 1)).flatten()
        
        batch = {
            "states": states_array,
            "global_states": global_states_array,
            "actions": np.array(all_actions),
            "old_probs": np.array(all_old_probs, dtype=np.float32),
            "advantages": np.array(all_advantages, dtype=np.float32),
            "returns": returns_array,
        }
        # 记录本轮采集完成worker与达成worker数量，供外层日志打印
        self._last_collect_finished_workers = self.num_workers
        self._last_collect_completed_workers = successful_workers
        self._last_collect_worker_rewards = worker_rewards  # 🔧 新增：保存worker奖励列表
        # 10-23-20-00 更新：保存回合级别的任务轮换信息
        current_mixing_config = self.foundation_multi_task_config if not self.foundation_training_completed else self.generalization_multi_task_config
        mixing_enabled = current_mixing_config.get("enabled", False)
        self._last_collect_mixing_summary = {
            'enabled': bool(mixing_enabled),
            'episode_task': episode_tag,  # 本回合的任务类型（BASE_ORDERS或随机订单）
            'all_workers': int(self.num_workers),
            'avg_reward': float(np.mean(worker_rewards)) if worker_rewards else None,
        }
        
        # 10-23-20-15 保存本回合的实际环境配置（供评估时使用，确保训练-评估一致性）
        self._last_episode_config = {
            'custom_orders': episode_orders,
            'episode_tag': episode_tag,
            'equipment_failure_enabled': episode_equipment_failure_enabled,
            'emergency_orders_enabled': episode_emergency_orders_enabled,
        }
        avg_reward = total_reward / self.num_workers if self.num_workers > 0 else 0.0
        return avg_reward, batch
    
    def update_policy(self, batch: Dict[str, np.ndarray], entropy_coeff: float) -> Dict[str, float]:
        """
        专家修复：接收已处理好的数据批次，执行标准的PPO更新流程
        - 移除了数据聚合和GAE计算逻辑，因为这些已在 `collect_and_process_experience` 中完成
        """
        # 1. 从批次中解包数据
        all_states = batch["states"]
        all_global_states = batch["global_states"]
        all_actions = batch["actions"]
        all_old_probs = batch["old_probs"]
        all_advantages = batch["advantages"]
        all_returns = batch["returns"]

        total_samples = len(all_states)
        if total_samples == 0:
            return {}

        # 初始化训练统计
        total_actor_loss, total_critic_loss, total_entropy = 0, 0, 0
        total_approx_kl, total_clip_fraction = 0, 0
        update_count = 0

        # 2. 标准PPO更新循环 (Epochs + Mini-batch)
        ppo_epochs = PPO_NETWORK_CONFIG.get("ppo_epochs", 10)
        num_minibatches = PPO_NETWORK_CONFIG.get("num_minibatches", 4)
        
        if total_samples < num_minibatches:
            num_minibatches = 1
            
        batch_size = total_samples // num_minibatches

        for epoch in range(ppo_epochs):
            # 2.1. 数据随机化 (Shuffle)
            indices = np.arange(total_samples)
            np.random.shuffle(indices)

            shuffled_states = all_states[indices]
            shuffled_global_states = all_global_states[indices]
            shuffled_actions = all_actions[indices]
            shuffled_old_probs = all_old_probs[indices]
            shuffled_advantages = all_advantages[indices]
            shuffled_returns = all_returns[indices]

            # 2.2. Mini-batch 训练
            for i in range(0, total_samples, batch_size):
                start = i
                end = i + batch_size
                
                if end > total_samples:
                    end = total_samples
                if start == end:
                    continue

                # 提取Mini-batch数据
                mini_batch_states = shuffled_states[start:end]
                mini_batch_global_states = shuffled_global_states[start:end]
                mini_batch_actions = shuffled_actions[start:end]
                mini_batch_old_probs = shuffled_old_probs[start:end]
                mini_batch_advantages = shuffled_advantages[start:end]
                mini_batch_returns = shuffled_returns[start:end]

                # 2.3. 执行网络更新
                loss_info = self.shared_network.update(
                    mini_batch_states,
                    mini_batch_global_states,
                    mini_batch_actions,
                    mini_batch_old_probs,
                    mini_batch_advantages,
                    mini_batch_returns,
                    entropy_coeff=entropy_coeff
                )

                # 累加统计信息
                if loss_info:
                    total_actor_loss += loss_info["actor_loss"]
                    total_critic_loss += loss_info["critic_loss"]
                    total_entropy += loss_info["entropy"]
                    total_approx_kl += loss_info["approx_kl"]
                    total_clip_fraction += loss_info["clip_fraction"]
                    update_count += 1
        
        # 返回平均损失
        if update_count > 0:
            return {
                "actor_loss": total_actor_loss / update_count,
                "critic_loss": total_critic_loss / update_count,
                "entropy": total_entropy / update_count,
                "approx_kl": total_approx_kl / update_count,
                "clip_fraction": total_clip_fraction / update_count,
            }
        return {}
    
    def _independent_exam_evaluation(self, env, curriculum_config, seed):
        """🔧 V33 新增：独立的考试评估，确保每轮都是全新的仿真"""
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        
        observations, infos = env.reset(seed=seed)
        episode_reward = 0
        step_count = 0
        
        while step_count < self.max_steps_for_eval:
            actions = {}
            
            # 使用确定性策略，但基于新的随机环境状态
            for agent in env.agents:
                if agent in observations:
                    state = tf.expand_dims(observations[agent], 0)
                    network = self.shared_network
                    # 10-27-16-30 评估阶段明确禁用Dropout等随机性
                    action_probs = network.actor(state, training=False)
                    if network.is_multidiscrete:
                        # 10-23-16-30 修复：评估阶段使用无放回贪心，避免系统性冲突
                        action_prob_list = action_probs if isinstance(action_probs, list) else [action_probs]
                        heads = len(action_prob_list)
                        used = set()
                        selected = []
                        for i in range(heads):
                            p = action_prob_list[i][0].numpy()
                            # 🔧 应用动作掩码（若提供）
                            mask = infos.get(agent, {}).get('action_mask', None)
                            if mask is not None and len(mask) == p.shape[0]:
                                p = p * mask
                            # 允许多个头选择 IDLE(0)，仅屏蔽已选的非零动作
                            if used:
                                for u in list(used):
                                    if u != 0:
                                        p[u] = 0.0
                            idx = int(np.argmax(p)) if p.sum() > 1e-8 else 0
                            selected.append(idx)
                            if idx != 0:
                                used.add(idx)
                        action = np.array(selected, dtype=network.action_space.dtype)
                    else:
                        action = int(tf.argmax(action_probs[0]))
                    actions[agent] = action
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            step_count += 1
            
            if any(terminations.values()) or any(truncations.values()):
                break
        
        # 获取最终统计
        final_stats = env.sim.get_final_stats()
        return {
            'mean_reward': episode_reward,
            'mean_makespan': final_stats.get('makespan', 0),
            'mean_utilization': final_stats.get('mean_utilization', 0),
            'mean_completed_parts': final_stats.get('total_parts', 0),
            'mean_tardiness': final_stats.get('total_tardiness', 0)
        }
    
    def quick_kpi_evaluation(self, num_episodes: int = 1, curriculum_config: Dict[str, Any] = None) -> Dict[str, float]:
        """10-23-20-15 修复版：快速KPI评估，确保评估环境与训练环境完全一致"""
        # 10-23-20-15 核心改进：使用上一个训练回合的实际配置进行评估
        # 这确保了评估KPI能够真实反映当前训练环境的表现
        eval_config = curriculum_config.copy() if curriculum_config else {}
        
        # 10-23-20-15 优先使用保存的实际环境配置（如果存在）
        if hasattr(self, '_last_episode_config') and self._last_episode_config:
            last_config = self._last_episode_config
            eval_config['custom_orders'] = last_config['custom_orders']
            eval_config['equipment_failure_enabled'] = last_config['equipment_failure_enabled']
            eval_config['emergency_orders_enabled'] = last_config['emergency_orders_enabled']
            # 保存订单标签供日志使用
            eval_config['episode_tag'] = last_config['episode_tag']
        
        # 评估步长对齐环境超时
        eval_config['MAX_SIM_STEPS'] = self.max_steps_for_eval
        # 训练期评估强制确定性候选，保证可复现
        eval_config['deterministic_candidates'] = True
        
        env = make_parallel_env(eval_config)
        
        total_rewards = []
        makespans = []
        utilizations = []
        completed_parts_list = []
        tardiness_list = []
        
        for episode in range(num_episodes):
            observations, infos = env.reset()
            episode_reward = 0
            step_count = 0
            
            # 使用与训练一致的步数限制
            while step_count < self.max_steps_for_eval:
                actions = {}
                
                # 10-23-16-35 修复：确定性评估采用无放回贪心，避免系统性冲突
                for agent in env.agents:
                    if agent in observations:
                        state = tf.expand_dims(observations[agent], 0)
                        # 10-27-16-30 训练期快速评估：禁用Dropout
                        action_probs = self.shared_network.actor(state, training=False)
                        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
                            action_prob_list = action_probs if isinstance(action_probs, list) else [action_probs]
                            heads = len(action_prob_list)
                            used = set()
                            selected = []
                            for i in range(heads):
                                p = action_prob_list[i][0].numpy()
                                # 🔧 应用动作掩码（若提供）
                                mask = infos.get(agent, {}).get('action_mask', None)
                                if mask is not None and len(mask) == p.shape[0]:
                                    p = p * mask
                                # 允许多个头选择 IDLE(0)，仅屏蔽已选的非零动作
                                if used:
                                    for u in list(used):
                                        if u != 0:
                                            p[u] = 0.0
                                idx = int(np.argmax(p)) if p.sum() > 1e-8 else 0
                                selected.append(idx)
                                if idx != 0:
                                    used.add(idx)
                            actions[agent] = np.array(selected, dtype=self.action_space.dtype)
                        else:
                            actions[agent] = int(tf.argmax(action_probs[0]))
                
                observations, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                step_count += 1
                
                if any(terminations.values()) or any(truncations.values()):
                    break
            
            # 获取最终统计
            final_stats = env.sim.get_final_stats()
            total_rewards.append(episode_reward)
            makespans.append(final_stats.get('makespan', 0))
            utilizations.append(final_stats.get('mean_utilization', 0))
            completed_parts_list.append(final_stats.get('total_parts', 0))
            tardiness_list.append(final_stats.get('total_tardiness', 0))
        
        # 🔧 V37 新增：检查环境重置信号
        strategy_reset_signal = getattr(env.sim, '_trigger_strategy_reset', False)
        if strategy_reset_signal:
            self._env_strategy_reset_signal = True
        
        env.close()
        
        return {
            'mean_reward': np.mean(total_rewards),
            'mean_makespan': np.mean(makespans),
            'mean_utilization': np.mean(utilizations),
            'mean_completed_parts': np.mean(completed_parts_list),
            'mean_tardiness': np.mean(tardiness_list)
        }
    
    def simple_evaluation(self, num_episodes: int = 5) -> Dict[str, float]:
        """🔧 修复版：简单评估，返回核心业务指标"""
        env, _ = self.create_environment()
        
        total_rewards = []
        total_steps = []
        makespans = []
        completed_parts = []
        utilizations = []
        tardiness_list = []
        
        for episode in range(num_episodes):
            observations, infos = env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < self.max_steps_for_eval:
                actions = {}
                
                # 10-23-14-30 修复：使用确定性策略评估（正确处理MultiDiscrete多头输出）
                for agent in env.agents:
                    if agent in observations:
                        state = tf.expand_dims(observations[agent], 0)
                        action_probs = self.shared_network.actor(state, training=False)
                        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
                            # 10-23-14-30 修复：每个头分别选择argmax，而非把多头输出当单个分布
                            action_prob_list = action_probs if isinstance(action_probs, list) else [action_probs]
                            actions_list = []
                            for i, p_head in enumerate(action_prob_list):
                                p = p_head[0].numpy()
                                mask = infos.get(agent, {}).get('action_mask', None)
                                if mask is not None and len(mask) == p.shape[0]:
                                    p = p * mask
                                actions_list.append(int(np.argmax(p)) if p.sum() > 1e-8 else 0)
                            actions[agent] = np.array(actions_list, dtype=self.action_space.dtype)
                        else:
                            # 单头：应用掩码后取argmax
                            p = action_probs[0].numpy()
                            mask = infos.get(agent, {}).get('action_mask', None)
                            if mask is not None and len(mask) == p.shape[0]:
                                p = p * mask
                            actions[agent] = int(np.argmax(p)) if p.sum() > 1e-8 else 0
                
                observations, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                step_count += 1
                
                if any(terminations.values()) or any(truncations.values()):
                    break
            
            # 🔧 修复：获取完整的业务指标
            final_stats = env.sim.get_final_stats()
            total_rewards.append(episode_reward)
            total_steps.append(step_count)
            makespans.append(final_stats.get('makespan', 0))
            completed_parts.append(final_stats.get('total_parts', 0))
            utilizations.append(final_stats.get('mean_utilization', 0))
            tardiness_list.append(final_stats.get('total_tardiness', 0))
        
        env.close()
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_steps': np.mean(total_steps),
            'mean_makespan': np.mean(makespans),
            'mean_completed_parts': np.mean(completed_parts),
            'mean_utilization': np.mean(utilizations),
            'mean_tardiness': np.mean(tardiness_list)
        }
    
    
    def train(self, max_episodes: int = 1000, steps_per_episode: int = 200, 
              eval_frequency: int = 20, adaptive_mode: bool = True):
        """ 自适应训练主循环：根据性能自动调整训练策略和轮数"""
        # 自适应模式：最大轮数作为上限，实际轮数根据性能动态决定

        if adaptive_mode:
            self.training_targets["max_episodes"] = max_episodes
        
        # 🔧 V16：显示课程学习配置
        curriculum_config = self.training_flow_config["foundation_phase"]["curriculum_learning"]
        if curriculum_config.get("enabled", False):
            print(f"📚 课程学习已启用，共{len(curriculum_config['stages'])}个阶段:")
            for i, stage in enumerate(curriculum_config["stages"]):
                print(f"   阶段{i+1}: {stage['name']} - 订单 {stage['orders_scale']*100:.0f}%")
        print("=" * 80)
        
        if not validate_config():
            print("❌ 配置验证失败")
            return
        
        # 训练开始时间记录
        training_start_time = time.time()
        training_start_datetime = datetime.now()
        print(f"🕐 训练开始时间: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 🔧 V16：课程学习管理
        curriculum_config = self.training_flow_config["foundation_phase"]["curriculum_learning"]
        curriculum_enabled = curriculum_config.get("enabled", False)
        current_stage = 0
        stage_episode_count = 0
        
        # 🔧 V8 优化: 不再需要创建主环境，只创建缓冲区
        buffers = {
            agent: ExperienceBuffer() 
            for agent in self.agent_ids
        }
        
        best_reward = float('-inf')
        best_makespan = float('inf')
        
        # 🔧 V27 核心修复：为课程学习的每个阶段独立跟踪最佳分数
        stage_best_scores = [float('-inf')] * len(curriculum_config["stages"]) if curriculum_enabled else []
        
        # 🔧 初始化用于课程学习毕业检查的性能指标，毕业检查将使用上一个回合的准确数据
        last_kpi_results = {}
        last_current_score = 0.0
        
        try:
            for episode in range(max_episodes):
                iteration_start_time = time.time()
                
                # --- 核心创新：基础训练 + 随机领域强化 逻辑 ---
                current_curriculum_config = None
                
                # 首先处理课程学习逻辑（如果启用）
                if curriculum_enabled and not self.foundation_training_completed:
                    stage_config = curriculum_config["stages"][current_stage]
                    
                    # 检查是否满足自适应毕业条件
                    if self.check_curriculum_stage_graduation(last_kpi_results, last_current_score, stage_config):
                        print(f"✅ 阶段 '{stage_config['name']}' 毕业标准达成！")
                        
                        if stage_config.get('is_final_stage', False):
                            print("🏆 课程学习完成！现在开始基础能力认证，通过后进入泛化强化阶段。")
                            # 标记课程学习部分结束，后续逻辑将接管并启动基础能力认证
                            self.foundation_training_completed = True 
                        else:
                            # 晋级到下一个课程阶段
                            current_stage += 1
                            stage_episode_count = 0
                            self.curriculum_stage_achievement_count = 0  # 为新阶段重置计数器
                            next_stage_name = curriculum_config["stages"][current_stage]['name']
                            print(f"🚀 进入下一阶段: '{next_stage_name}'")
                    
                    # 获取当前阶段配置 (阶段可能已更新)
                    stage = curriculum_config["stages"][current_stage]
                    current_curriculum_config = {
                        'orders_scale': stage.get('orders_scale', 1.0),
                        'time_scale': stage.get('time_scale', 1.0),
                        'stage_name': stage.get('name', f'Stage {current_stage}')
                    }
                    
                    # 详细的阶段切换和状态日志
                    if stage_episode_count == 0:
                        print(f"📚 [回合 {episode+1}] 🔄 课程学习阶段切换!")
                        print(f"   新阶段: {stage['name']}")
                        print(f"   订单比例: {stage['orders_scale']} (目标零件数: {int(get_total_parts_count() * stage['orders_scale'])})")
                        print(f"   时间比例: {stage['time_scale']} (时间限制: {int(SIMULATION_TIME * stage['time_scale'])}分钟)")
                        print(f"🔧 当前课程配置将传递给所有worker: orders_scale={stage['orders_scale']}, time_scale={stage['time_scale']}")
                        print("-" * 60)
                    
                    # 🔧 V17新增：每10轮显示阶段状态
                    if episode % 10 == 0:
                        print(f"📚 课程状态: {stage['name']} (第 {stage_episode_count} 回合)")
                        print(f"   当前难度: {int(get_total_parts_count() * stage['orders_scale'])}零件, {stage['time_scale']:.1f}x时间")    
                    stage_episode_count += 1
                
                # --- 核心训练阶段判断 ---
                
                # 检查课程学习是否已完成所有阶段
                curriculum_just_completed = False
                if curriculum_enabled and self.foundation_training_completed and not self.generalization_phase_active:
                    # 这是一个过渡状态，表示课程学习刚刚完成，但还未正式进入泛化阶段
                    # 在这个状态下，我们将使用基础能力认证的配置
                    curriculum_just_completed = True

                if not self.foundation_training_completed or curriculum_just_completed:
                    # 10-23-18-00 新范式：阶段1：基础能力训练（随机订单泛化训练）
                    # 核心改变：不再使用固定BASE_ORDERS，而是采用随机订单训练
                    # 注意：多任务混合逻辑会在collect_and_process_experience中应用
                    # 这里只提供主配置框架，具体的worker级别订单分配在数据采集时完成
                    if not curriculum_enabled or curriculum_just_completed:
                        foundation_config = {
                            'orders_scale': 1.0,
                            'time_scale': 1.0,
                            'stage_name': '基础能力训练-随机订单',
                            # 10-23-18-00 关键改变：不再设置custom_orders，让多任务混合逻辑处理
                            # 随机订单将在collect_and_process_experience中按worker分配
                        }
                        current_curriculum_config = foundation_config
                    
                    # 在每个回合都添加当前回合数，供环境内部使用
                        if current_curriculum_config:
                            current_curriculum_config['current_episode'] = episode
                
                elif not self.generalization_phase_active:
                    # 10-23-18-00 阶段转换：基础训练完成，进入泛化强化阶段
                    self.generalization_phase_active = True
                    print("\n" + "="*80)
                    print(f"🚀 [回合 {episode+1}] 基础训练已完成，正式进入动态事件鲁棒性训练阶段!")
                    print("="*80 + "\n")
                
                if self.generalization_phase_active:
                    # 10-23-18-00 新范式：阶段2：动态事件鲁棒性训练（动态事件鲁棒性训练）
                    # 核心改变：启用设备故障、紧急插单等动态事件
                    # 注意：多任务混合逻辑会在collect_and_process_experience中应用
                    # 动态事件（设备故障、紧急插单）也在那里启用
                    generalization_config = {
                        'randomize_env': True,  # 启用环境扰动
                        'stage_name': f'泛化强化-动态事件-R{episode}',
                        'current_episode': episode
                        # 10-23-18-00 关键改变：不再在这里设置custom_orders和动态事件配置
                        # 这些将在collect_and_process_experience中按worker分配
                    }
                    
                    current_curriculum_config = generalization_config
                    
                    if episode % 20 == 0:
                        # 10-23-18-00 新范式：信息显示调整（不再在这里生成random_orders）
                        # 动态事件状态由配置文件控制
                        generalization_criteria = self.training_flow_config["generalization_phase"]["completion_criteria"]
                        dynamic_events = TRAINING_FLOW_CONFIG["generalization_phase"].get("dynamic_events", {})
                        print(f"🎲 泛化强化阶段: 动态事件训练")
                        print(f"   设备故障: {'✓' if dynamic_events.get('equipment_failure_enabled', False) else '✗'}")
                        print(f"   紧急插单: {'✓' if dynamic_events.get('emergency_orders_enabled', False) else '✗'}")
                        print(f"   泛化阶段连续达标: {self.generalization_achievement_count}/{generalization_criteria['target_consistency']} 次")
                

                collect_start_time = time.time()
                episode_reward, batch = self.collect_and_process_experience(steps_per_episode, current_curriculum_config)
                collect_duration = time.time() - collect_start_time

                # 10-25-14-30 🔧 修复：递增总步数用于多任务混合与seed多样化
                self.total_steps += steps_per_episode
                
                # 🔧 V6 安全的策略更新（包含内存检查）
                update_start_time = time.time()
                if batch is not None:
                    losses = self.update_policy(batch, entropy_coeff=self.current_entropy_coeff)
                else:
                    # 空批次防御：提供安全的默认指标并跳过更新
                    losses = {
                        'actor_loss': 0.0,
                        'critic_loss': 0.0,
                        'entropy': float(self.current_entropy_coeff),
                        'approx_kl': 0.0,
                        'clip_fraction': 0.0,
                    }
                update_duration = time.time() - update_start_time
                
                # 记录统计
                iteration_end_time = time.time()
                iteration_duration = iteration_end_time - iteration_start_time
                self.iteration_times.append(iteration_duration)
                self.episode_rewards.append(episode_reward)

                
                # 提前进行KPI评估（按频率控制），以便整合TensorBoard日志
                if eval_frequency is None:
                    eval_frequency = 1
                if eval_frequency <= 1 or (episode % eval_frequency == 0):
                    kpi_results = self.quick_kpi_evaluation(num_episodes=1, curriculum_config=current_curriculum_config)
                    self.kpi_history.append(kpi_results)
                else:
                    # 若本回合不评估，则沿用上一次可用KPI（若无则做最小占位）
                    if self.kpi_history:
                        kpi_results = self.kpi_history[-1]
                    else:
                        kpi_results = {
                            'mean_reward': 0.0,
                            'mean_makespan': 0.0,
                            'mean_utilization': 0.0,
                            'mean_completed_parts': 0.0,
                            'mean_tardiness': 0.0
                        }
                        self.kpi_history.append(kpi_results)

                # 🔧 核心改造：计算当前回合的综合评分
                current_score = calculate_episode_score(kpi_results, config=current_curriculum_config)
                
                # 🔧 BUG修复：保存本回合的KPI结果，供下一回合的毕业检查使用
                last_kpi_results = kpi_results
                last_current_score = current_score
                
                # --- 核心创新：检查阶段转换和训练完成条件 ---
                target_parts_for_check = self._get_target_parts(current_curriculum_config)
                
                completion_rate_for_check = (kpi_results.get('mean_completed_parts', 0) / target_parts_for_check) * 100 if target_parts_for_check > 0 else 0
                
                # 🔧 修复：只有在最终阶段或课程学习完成后才检查基础训练完成
                should_check_foundation_completion = False
                if not self.foundation_training_completed:
                    if curriculum_enabled:
                        # 课程学习模式：只有在最终阶段才检查基础训练完成
                        if current_stage < len(curriculum_config["stages"]):
                            current_stage_info = curriculum_config["stages"][current_stage]
                            if current_stage_info.get('is_final_stage', False):
                                should_check_foundation_completion = True
                        # 或者课程学习已完成所有阶段
                        elif current_stage >= len(curriculum_config["stages"]):
                            should_check_foundation_completion = True
                    else:
                        # 非课程学习模式：直接检查
                        should_check_foundation_completion = True
                    
                    if should_check_foundation_completion:
                        # 🔧 BUG修复：与课程学习逻辑统一，使用上一个回合的KPI结果来判断是否毕业
                        # 10-27-17-30 修复：传入current_curriculum_config以使用正确的目标零件数
                        if self.check_foundation_training_completion(last_kpi_results, last_current_score, current_curriculum_config):
                            self.foundation_training_completed = True
                
                # 检查泛化训练是否完成（这将触发整个训练的结束）
                training_should_end = False
                if self.generalization_phase_active:
                    if self.check_generalization_training_completion(current_score, completion_rate_for_check):
                        training_should_end = True
                
                # --- 🔧 修复：自适应熵的停滞计数器仅在允许熵增加的阶段累积 ---
                # 1. 判断是否处于允许熵增加的阶段
                # 课程学习下：仅当处于最终阶段或已经进入泛化阶段才允许；
                # 非课程学习：全程允许。
                curriculum_is_final_stage = False
                if curriculum_enabled and not self.foundation_training_completed and current_stage < len(curriculum_config["stages"]):
                    curriculum_is_final_stage = bool(curriculum_config["stages"][current_stage].get("is_final_stage", False))

                allow_entropy_increase = (not curriculum_enabled) or curriculum_is_final_stage or self.generalization_phase_active
                
                # 2. 只在允许熵增加的阶段才累积停滞计数
                if allow_entropy_increase:
                    self.epochs_without_improvement += 1
                else:
                    # 非熵增加阶段，重置计数器（避免累积无意义的停滞）
                    self.epochs_without_improvement = 0
                    self.stagnation_level = 0
                
                # 3. 自适应熵调整逻辑
                adaptive_entropy_enabled = ADAPTIVE_ENTROPY_CONFIG["enabled"]
                start_episode = ADAPTIVE_ENTROPY_CONFIG["start_episode"]
                patience = ADAPTIVE_ENTROPY_CONFIG["patience"]
                boost_factor = ADAPTIVE_ENTROPY_CONFIG["boost_factor"]

                # 正确的触发点：在第 start_episode + patience 回合之后才可能触发
                if adaptive_entropy_enabled and allow_entropy_increase and episode >= (start_episode + patience):
                    # 当前的完成率，用于判断是否需要降低熵
                    target_parts_for_entropy = self._get_target_parts(current_curriculum_config)
                    completion_rate_for_entropy = kpi_results['mean_completed_parts'] / (target_parts_for_entropy + 1e-6)

                    # 检查是否停滞
                    if self.epochs_without_improvement >= patience:
                        self.stagnation_level += 1
                        boost_multiplier = 1.0 + boost_factor * self.stagnation_level
                        self.current_entropy_coeff = min(
                            self.current_entropy_coeff * boost_multiplier,
                            PPO_NETWORK_CONFIG["entropy_coeff"] * 5 # 设置一个硬上限，例如原始的5倍
                        )
                        print(f"📈 停滞等级 {self.stagnation_level}! 性能已停滞 {self.epochs_without_improvement} 回合。")
                        print(f"   采取强力措施: 将熵提升至 {self.current_entropy_coeff:.4f} (提升因子: {boost_multiplier:.2f})")
                        # 核心修复：重置计数器，给予模型适应新熵值的窗口期
                        self.epochs_without_improvement = 0
                    
                    # 🔧 缺陷四修复：使用配置化的熵衰减逻辑
                    elif completion_rate_for_entropy > ADAPTIVE_ENTROPY_CONFIG["high_completion_threshold"]:
                        self.current_entropy_coeff = max(
                            self.current_entropy_coeff * ADAPTIVE_ENTROPY_CONFIG["high_completion_decay"],
                            ADAPTIVE_ENTROPY_CONFIG["min_entropy"]
                        )
                
                # 确保熵不会低于设定的最小值
                self.current_entropy_coeff = max(self.current_entropy_coeff, ADAPTIVE_ENTROPY_CONFIG["min_entropy"])

                
                # 🔧 V36 统一TensorBoard日志记录，并根据课程阶段动态切换run
                if TENSORBOARD_AVAILABLE:
                    try:
                        # 根据课程阶段切换run，在悬停提示中显示阶段名
                        run_name = "train_default" # Fallback run name
                        if curriculum_enabled and current_curriculum_config:
                            # Get stage name and sanitize it for use as a directory name
                            run_name = current_curriculum_config['stage_name'].replace(" ", "_")
                        
                        if self.train_writer is None or self.current_tensorboard_run_name != run_name:
                            if self.train_writer is not None:
                                try:
                                    self.train_writer.flush()  # 🔧 关键修复：关闭前先刷新缓冲
                                    self.train_writer.close()
                                except Exception as e:
                                    print(f"⚠️ 关闭旧TensorBoard writer时出错: {e}")
                            
                            logdir = os.path.join(self.tensorboard_dir, run_name)
                            self.train_writer = tf.summary.create_file_writer(logdir)
                            self.current_tensorboard_run_name = run_name
                            print(f"📊 TensorBoard run已切换至: '{run_name}' (日志目录: {logdir})")

                        if self.train_writer:
                            with self.train_writer.as_default():
                                # 训练核心指标
                                tf.summary.scalar('Training/Avg_Episode_Reward', episode_reward, step=episode)
                                tf.summary.scalar('Training/Actor_Loss', losses['actor_loss'], step=episode)
                                tf.summary.scalar('Training/Critic_Loss', losses['critic_loss'], step=episode)
                                tf.summary.scalar('Training/Entropy', losses['entropy'], step=episode)
                                tf.summary.scalar('Training/KL_Divergence', losses['approx_kl'], step=episode)
                                tf.summary.scalar('Training/Clip_Fraction', losses['clip_fraction'], step=episode)
                                # 性能指标
                                tf.summary.scalar('Performance/Iteration_Duration', iteration_duration, step=episode)
                                tf.summary.scalar('Performance/CPU_Collection_Time', collect_duration, step=episode)
                                tf.summary.scalar('Performance/GPU_Update_Time', update_duration, step=episode)
                                # 业务KPI指标
                                tf.summary.scalar('KPI/Makespan', kpi_results['mean_makespan'], step=episode)
                                tf.summary.scalar('KPI/Completed_Parts', kpi_results['mean_completed_parts'], step=episode)
                                tf.summary.scalar('KPI/Utilization', kpi_results['mean_utilization'], step=episode)
                                tf.summary.scalar('KPI/Tardiness', kpi_results['mean_tardiness'], step=episode)
                                # 记录综合评分
                                tf.summary.scalar('KPI/Score', current_score, step=episode)
                                
                            # 🔧 关键修复：在with块外调用flush，确保数据立即写入磁盘
                            self.train_writer.flush()
                    except Exception as e:
                        print(f"❌ TensorBoard写入失败 (回合{episode}): {e}")
                        import traceback
                        traceback.print_exc()
                
                # --- 核心创新：新的训练结束逻辑 ---
                if training_should_end:
                    print(f"\n🎉 训练完成！模型已通过基础训练和泛化强化两个阶段的认证。")
                    break
                
                # 检查最大轮数限制
                if episode >= max_episodes - 1:
                    print(f"\n⏰ 达到最大训练轮数 {max_episodes}，训练结束。")
                    break
                
                # 🔧 V36 新增：记录当前课程阶段信息供其他方法使用
                if current_curriculum_config:
                    self._current_orders_scale = current_curriculum_config.get('orders_scale', 1.0)
                
                # 简化的性能监控，移除复杂的重启机制
                # 基础性能跟踪（用于调试和监控）
                current_performance = kpi_results.get('mean_completed_parts', 0)
                if not hasattr(self, '_performance_history'):
                    self._performance_history = []
                
                self._performance_history.append(current_performance)
                # 只保留最近20轮的历史
                if len(self._performance_history) > 20:
                    self._performance_history.pop(0)
                

                # 正确更新最佳记录（只有当makespan > 0时才更新）
                current_makespan = kpi_results['mean_makespan']
                if current_makespan > 0 and current_makespan < best_makespan:
                    best_makespan = current_makespan
                
                # ------------------- 统一日志输出开始 -------------------
                
                # 准备KPI数据用于日志显示
                makespan = kpi_results['mean_makespan']
                completed_parts = kpi_results['mean_completed_parts']
                utilization = kpi_results['mean_utilization']
                tardiness = kpi_results['mean_tardiness']
                # current_score 已经在前面通过 _calculate_score 计算过了
                
                if not hasattr(self, 'best_score'):
                    self.best_score = float('-inf')

                model_update_info = ""
                timestamp = datetime.now().strftime("%m%d_%H%M") # 获取当前时间戳
                # 🔧 核心改造：区分"全局最佳"和"最终阶段最佳"
                # 1. 更新全局最佳分数（用于日志显示）
                if current_score > self.best_score:
                    self.best_score = current_score

                # === 核心重构：模型保存逻辑 ===
                
                model_update_info = ""
                
                if curriculum_enabled:
                    # --- 启用课程学习时的保存逻辑 ---
                    if not self.foundation_training_completed:
                        # 1. 保存当前课程阶段的最佳模型
                        if current_score > stage_best_scores[current_stage]:
                            stage_best_scores[current_stage] = current_score
                            stage_name = current_curriculum_config['stage_name'].replace(" ", "_")
                            # 🔧 只在需要保存时创建时间戳目录
                            timestamp_dir = os.path.join(self.models_dir, timestamp)
                            os.makedirs(timestamp_dir, exist_ok=True)
                            model_path = self.save_model(f"{timestamp_dir}/{timestamp}_{stage_name}_best")
                            if model_path:
                                stage_display_name = current_curriculum_config['stage_name']
                                model_update_info = f"✅ {stage_display_name}阶段最佳! 模型保存至: {model_path}"
                                # 🔧 修复：只在最终阶段重置停滞计数器
                                if curriculum_is_final_stage:
                                    self.epochs_without_improvement = 0
                                    self.stagnation_level = 0
                    elif self.generalization_phase_active:
                        # 2. 泛化强化阶段的模型保存
                        if current_score > self.best_score_generalization_phase:
                            self.best_score_generalization_phase = current_score
                            self.best_kpi_generalization_phase = kpi_results.copy()
                            self.best_episode_generalization_phase = episode + 1
                            # 🔧 只在需要保存时创建时间戳目录
                            timestamp_dir = os.path.join(self.models_dir, timestamp)
                            os.makedirs(timestamp_dir, exist_ok=True)
                            model_path = self.save_model(f"{timestamp_dir}/{timestamp}general_train_best")
                            if model_path:
                                model_update_info = f"🏆 泛化强化阶段最佳! 模型保存至: {model_path}"
                                # 🔧 修复：泛化阶段保存最佳模型时重置停滞计数器
                                self.epochs_without_improvement = 0
                                self.stagnation_level = 0
                else:  # curriculum_enabled is False
                    # --- 未启用课程学习时的保存逻辑 ---
                    if not self.foundation_training_completed:
                        # 1. 基础训练阶段的模型保存
                        if current_score > self.best_score_foundation_phase:
                            self.best_score_foundation_phase = current_score
                            self.best_kpi_foundation_phase = kpi_results.copy()
                            self.best_episode_foundation_phase = episode + 1
                            # 🔧 只在需要保存时创建时间戳目录
                            timestamp_dir = os.path.join(self.models_dir, timestamp)
                            os.makedirs(timestamp_dir, exist_ok=True)
                            model_path = self.save_model(f"{timestamp_dir}/{timestamp}base_train_best")
                            if model_path:
                                model_update_info = f"✅ 基础训练阶段最佳! 模型保存至: {model_path}"
                                # 🔧 修复：非课程学习模式下，基础阶段也可以重置（因为allow_entropy_increase=True）
                                self.epochs_without_improvement = 0
                                self.stagnation_level = 0
                    elif self.generalization_phase_active:
                        # 2. 泛化强化阶段的模型保存
                        if current_score > self.best_score_generalization_phase:
                            self.best_score_generalization_phase = current_score
                            self.best_kpi_generalization_phase = kpi_results.copy()
                            self.best_episode_generalization_phase = episode + 1
                            # 🔧 只在需要保存时创建时间戳目录
                            timestamp_dir = os.path.join(self.models_dir, timestamp)
                            os.makedirs(timestamp_dir, exist_ok=True)
                            model_path = self.save_model(f"{timestamp_dir}/{timestamp}general_train_best")
                            if model_path:
                                model_update_info = f"🏆 泛化强化阶段最佳! 模型保存至: {model_path}"
                                # 🔧 修复：泛化阶段保存最佳模型时重置停滞计数器
                                self.epochs_without_improvement = 0
                                self.stagnation_level = 0
                
                # 3. 全局"双达标"最佳模型保存（独立于所有其他逻辑）
                #    首先，获取当前回合的正确目标零件数
                target_parts_for_dual_check = self._get_target_parts(current_curriculum_config)
                
                completion_rate_kpi = (kpi_results.get('mean_completed_parts', 0) / target_parts_for_dual_check) * 100 if target_parts_for_dual_check > 0 else 0
                
                # 🔧 修复：根据课程学习状态决定是否保存"双达标"模型
                save_condition_met = False
                if not curriculum_enabled:
                    # 未启用课程学习：全程允许保存
                    save_condition_met = True
                else:
                    # 启用课程学习：只在最终阶段或泛化阶段允许保存
                    is_final_curriculum_stage = False
                    if not self.foundation_training_completed and current_stage < len(curriculum_config["stages"]):
                        current_stage_info = curriculum_config["stages"][current_stage]
                        is_final_curriculum_stage = current_stage_info.get('is_final_stage', False)
                    
                    if is_final_curriculum_stage or self.generalization_phase_active or curriculum_just_completed:
                        save_condition_met = True
                
                dual_objective_model_update_info = ""
                if save_condition_met and completion_rate_kpi >= 100 and current_score > self.best_score_dual_objective:
                    self.best_score_dual_objective = current_score
                    self.best_kpi_dual_objective = kpi_results.copy()
                    self.best_episode_dual_objective = episode + 1
                    # 🔧 只在需要保存时创建时间戳目录
                    timestamp_dir = os.path.join(self.models_dir, timestamp)
                    os.makedirs(timestamp_dir, exist_ok=True)
                    dual_objective_best_path = self.save_model(f"{timestamp_dir}/{timestamp}Twin_best")
                    if dual_objective_best_path:
                        dual_objective_model_update_info = f" ⭐完成所有零件得分最佳!模型保存至: {dual_objective_best_path}"
                        
                        # 🔧 修复：双达标模型保存时重置停滞计数器（如果处于允许熵增加的阶段）
                        if allow_entropy_increase:
                            print(f"🎉 新的双达标最佳模型! 重置停滞计数。")
                            self.epochs_without_improvement = 0
                            self.stagnation_level = 0  # 创下新高，"警报"解除
                
                # ------------------- 统一日志输出开始 -------------------

                 # 第一行：回合信息和性能数据
                # 采集统计（并行worker完成与达成情况）
                finished_workers = getattr(self, '_last_collect_finished_workers', self.num_workers)
                completed_workers = getattr(self, '_last_collect_completed_workers', 0)
                worker_rewards = getattr(self, '_last_collect_worker_rewards', [])
                
                # 🔧 新增：格式化每个worker的奖励
                if worker_rewards:
                    worker_rewards_str = ", ".join([f"{r:.0f}" for r in worker_rewards])
                else:
                    worker_rewards_str = "N/A"
                
                line1 = (
                    f"🔂 训练回合 {episode + 1:3d}/{max_episodes} | 平均奖励: {episode_reward:.1f}"
                    f" (每个worker奖励: {worker_rewards_str}, 完成全部: {completed_workers}/{finished_workers})"
                    f" | Actor损失: {losses['actor_loss']:.4f}| ⏱️本轮用時: {iteration_duration:.1f}s"
                    f" (CPU采集: {collect_duration:.1f}s, GPU更新: {update_duration:.1f}s)"
                )

                # 10-23-20-00 如开启多任务混合，显示本回合的任务类型（回合级轮换）
                mixing_summary = getattr(self, '_last_collect_mixing_summary', None)
                if mixing_summary and mixing_summary.get('enabled', False):
                    episode_task = mixing_summary.get('episode_task', 'Unknown')
                    avg_reward = mixing_summary.get('avg_reward')
                    avg_reward_str = f"{avg_reward:.1f}" if avg_reward is not None else "N/A"
                    
                    # 10-23-20-00 显示本回合任务类型和动态事件配置
                    task_display = episode_task
                    if self.generalization_phase_active and episode_task != "BASE_ORDERS":
                        task_display += "+动态事件"
                    
                    line1 += (
                        f" | 本回合任务: [{task_display}]×{self.num_workers}workers(均奖:{avg_reward_str})"
                    )

                # 10-23-20-15 第二行：KPI数据和阶段信息（包含实际评估环境配置）
                target_parts_for_log = self._get_target_parts(current_curriculum_config)
                stage_info_str = ""
                if current_curriculum_config and 'stage_name' in current_curriculum_config:
                    stage_name = current_curriculum_config['stage_name']
                    # 🔧 修复：显示两级阶段信息（课程学习阶段 + 基础训练阶段）
                    if curriculum_enabled and not curriculum_just_completed:
                        curriculum_stage_name = curriculum_config["stages"][current_stage]['name']
                        foundation_phase = '基础训练' if not self.foundation_training_completed else '泛化训练'
                        stage_info_str = f"   | 课程: '{curriculum_stage_name}' | 大阶段: '{foundation_phase}'"
                    else:
                        stage_info_str = f"   | 阶段: '{stage_name}'"
                
                # 10-23-20-15 核心修复：显示实际评估环境的配置（确保训练-评估一致）
                if hasattr(self, '_last_episode_config') and self._last_episode_config:
                    eval_task = self._last_episode_config['episode_tag']
                    eval_failure = self._last_episode_config['equipment_failure_enabled']
                    eval_emergency = self._last_episode_config['emergency_orders_enabled']
                    
                    # 组合评估环境描述
                    eval_env_desc = f"评估环境:[{eval_task}"
                    if eval_failure or eval_emergency:
                        events = []
                        if eval_failure:
                            events.append("故障✓")
                        if eval_emergency:
                            events.append("插单✓")
                        eval_env_desc += f"+{'+'.join(events)}"
                    eval_env_desc += "]"
                    stage_info_str += f" | {eval_env_desc}"
                
                target_parts_str = f"/{target_parts_for_log}"
                line2 = f"📊 此回合KPI评估 - 总完工时间: {makespan:.1f}min  | 设备利用率: {utilization:.1%} | 订单延期时间: {tardiness:.1f}min |  完成零件数: {completed_parts:.0f}{target_parts_str}{stage_info_str}"

                # 第三行：评分和模型更新信息
                phase_best_str = ""
                if curriculum_enabled:
                    # 🔧 修复：启用课程学习时，显示当前课程阶段的最佳分数
                    if not self.foundation_training_completed:
                        stage_display_name = current_curriculum_config.get('stage_name', '当前阶段')
                        stage_best_str = f" ({stage_display_name}最佳: {stage_best_scores[current_stage]:.3f})"
                        line3_score = f"🚥 回合评分: {current_score:.3f} (全局最佳: {self.best_score:.3f}){stage_best_str}"
                    elif self.generalization_phase_active:
                        phase_best_str = f" (泛化阶段最佳: {self.best_score_generalization_phase:.3f})"
                        line3_score = f"🚥 回合评分: {current_score:.3f} (全局最佳: {self.best_score:.3f}){phase_best_str}"
                else:
                    # 🔧 修复：未启用课程学习时，显示基础训练阶段的最佳分数
                    if not self.foundation_training_completed:
                        phase_best_str = f" (基础阶段最佳: {self.best_score_foundation_phase:.3f})"
                    elif self.generalization_phase_active:
                        phase_best_str = f" (泛化阶段最佳: {self.best_score_generalization_phase:.3f})"
                    line3_score = f"🚥 回合评分: {current_score:.3f} (全局最佳: {self.best_score:.3f}){phase_best_str}"
                
                # 合并所有模型更新信息
                combined_model_info = model_update_info + dual_objective_model_update_info
                line3 = f"{line3_score}{combined_model_info}" if combined_model_info else line3_score

                avg_time = np.mean(self.iteration_times)
                remaining_episodes = max_episodes - (episode + 1)
                estimated_remaining = remaining_episodes * avg_time
                progress_percent = ((episode + 1) / max_episodes) * 100
                current_time = datetime.now().strftime('%H:%M:%S')
                finish_str = ""
                if remaining_episodes > 0:
                    finish_time = time.time() + estimated_remaining
                    finish_str = time.strftime('%H:%M:%S', time.localtime(finish_time))
                line4 = f"🔮 当前训练进度: {progress_percent:.1f}% | 当前时间：{current_time} | 预计完成时间: {finish_str}"

                # 打印日志
                print(line1)
                print(line2)
                print(line3)
                print(line4)
                print() # 每个回合后添加一个空行
                
                # ------------------- 统一日志输出结束 -------------------
                        
            
            # 🔧 修复版：简化的训练完成统计
            training_end_time = time.time()
            training_end_datetime = datetime.now()
            total_training_time = training_end_time - training_start_time
            
            print("\n" + "=" * 80)
            print("🎉 训练完成！")
            print(f"🕐 训练开始: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🏁 训练结束: {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️ 总训练时间: {total_training_time/60:.1f}分钟 ({total_training_time:.1f}秒)")
            
            # 训练效率统计
            if self.iteration_times:
                avg_iteration_time = np.mean(self.iteration_times)
                print(f"⚡ 平均每轮: {avg_iteration_time:.1f}s | 训练效率: {len(self.iteration_times)/total_training_time*60:.1f}轮/分钟")

            # 🔧 Bug修复：输出最终的、可靠的最佳KPI
            print("\n" + "="*40)
            print("🏆 最终最佳KPI表现 (双重标准最佳) 🏆")
            print("="*40)
            
            # 检查是否有模型达到了双重标准，并实现优雅降级
            if self.best_episode_dual_objective != -1:
                best_kpi = self.best_kpi_dual_objective
                best_episode_to_report = self.best_episode_dual_objective
            elif self.best_episode_generalization_phase != -1:
                print("⚠️ 未找到双重标准模型，将报告【泛化阶段】的最佳模型。")
                best_kpi = self.best_kpi_generalization_phase
                best_episode_to_report = self.best_episode_generalization_phase
            elif self.best_episode_foundation_phase != -1:
                print("⚠️ 未找到双重标准或泛化阶段模型，将报告【基础训练阶段】的最佳模型。")
                best_kpi = self.best_kpi_foundation_phase
                best_episode_to_report = self.best_episode_foundation_phase
            else:
                print("⚠️ 未能记录任何阶段的最佳模型。")
                # 使用一个空的KPI字典来避免错误
                best_kpi = self.best_kpi_dual_objective 
                best_episode_to_report = -1

            target_parts_final = get_total_parts_count() # 最终评估总是基于完整任务
            completion_rate_final = (best_kpi.get('mean_completed_parts', 0) / target_parts_final) * 100 if target_parts_final > 0 else 0
            
            print(f"   (在第 {best_episode_to_report} 回合取得)") # 🔧 新增
            print(f"   完成零件: {best_kpi.get('mean_completed_parts', 0):.1f} / {target_parts_final} ({completion_rate_final:.1f}%)")
            print(f"   总完工时间: {best_kpi.get('mean_makespan', 0):.1f} 分钟")
            print(f"   设备利用率: {best_kpi.get('mean_utilization', 0):.1%}")
            print(f"   订单延期时间: {best_kpi.get('mean_tardiness', 0):.1f} 分钟")
            print("="*40)
            
            # --- 核心修复：输出每个阶段的最佳KPI ---
            print("\n" + "="*40)
            print("🏆 各阶段最佳KPI表现 🏆")
            print("="*40)

            # 基础训练阶段最佳
            if self.best_episode_foundation_phase != -1:
                print("\n--- 基础训练阶段 ---")
                best_kpi = self.best_kpi_foundation_phase
                target_parts = get_total_parts_count()
                completion_rate = (best_kpi.get('mean_completed_parts', 0) / target_parts) * 100 if target_parts > 0 else 0
                print(f"   (在第 {self.best_episode_foundation_phase} 回合取得)")
                print(f"   完成零件: {best_kpi.get('mean_completed_parts', 0):.1f} / {target_parts} ({completion_rate:.1f}%)")
                print(f"   总完工时间: {best_kpi.get('mean_makespan', 0):.1f} 分钟")
                print(f"   设备利用率: {best_kpi.get('mean_utilization', 0):.1%}")
                print(f"   订单延期时间: {best_kpi.get('mean_tardiness', 0):.1f} 分钟")
                print(f"   综合评分: {self.best_score_foundation_phase:.3f}")

            # 泛化强化阶段最佳
            if self.best_episode_generalization_phase != -1:
                print("\n--- 泛化强化阶段 ---")
                best_kpi = self.best_kpi_generalization_phase
                # 注意：泛化阶段的目标零件数是动态的，此处仅为参考
                print(f"   (在第 {self.best_episode_generalization_phase} 回合取得)")
                print(f"   完成零件: {best_kpi.get('mean_completed_parts', 0):.1f}")
                print(f"   总完工时间: {best_kpi.get('mean_makespan', 0):.1f} 分钟")
                print(f"   设备利用率: {best_kpi.get('mean_utilization', 0):.1%}")
                print(f"   订单延期时间: {best_kpi.get('mean_tardiness', 0):.1f} 分钟")
                print(f"   综合评分: {self.best_score_generalization_phase:.3f}")
            
            # 新增：如果启用了课程学习，则展示每个课程阶段的最佳分数
            if curriculum_enabled:
                 print("\n--- 课程学习各阶段最佳分数 ---")
                 for i, score in enumerate(stage_best_scores):
                     if score > -np.inf:
                         stage_name = curriculum_config["stages"][i]['name']
                         print(f"   阶段 '{stage_name}': {score:.3f}")
                     else:
                         stage_name = curriculum_config["stages"][i]['name']
                         print(f"   阶段 '{stage_name}': 未记录最佳分数")


            # 最终黄金标准：双达标模型
            print("\n" + "="*40)
            print("⭐ 最终黄金标准模型 (完成所有零件且得分最高) ⭐")
            print("="*40)
            
            if self.best_episode_dual_objective != -1:
                best_kpi = self.best_kpi_dual_objective
                best_episode_to_report = self.best_episode_dual_objective
                
                # 在双达标的情况下，目标零件数是确定的
                target_parts_final = get_total_parts_count()
                completion_rate_final = (best_kpi.get('mean_completed_parts', 0) / target_parts_final) * 100 if target_parts_final > 0 else 0
            
                print(f"   (在第 {best_episode_to_report} 回合取得)") 
                print(f"   完成零件: {best_kpi.get('mean_completed_parts', 0):.1f} / {target_parts_final} ({completion_rate_final:.1f}%)")
                print(f"   总完工时间: {best_kpi.get('mean_makespan', 0):.1f} 分钟")
                print(f"   设备利用率: {best_kpi.get('mean_utilization', 0):.1%}")
                print(f"   订单延期时间: {best_kpi.get('mean_tardiness', 0):.1f} 分钟")
                print(f"   综合评分: {self.best_score_dual_objective:.3f}")
            else:
                print("   ⚠️ 本次训练未产生满足'完成所有零件'条件的最佳模型。")

            print("="*40)
            
            return {
                'training_time': total_training_time,
                'kpi_history': self.kpi_history,
                'iteration_times': self.iteration_times,
                'best_kpi': best_kpi
            }
            
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # 10-27-16-30 训练收尾：优雅关闭进程池，释放系统资源
            try:
                if hasattr(self, 'pool') and self.pool:
                    self.pool.shutdown(wait=True)
            except Exception:
                pass
    
    def save_model(self, filepath: str) -> str:
        """
        保存模型 - TensorFlow 2.15.0 兼容版本
        使用多格式冗余保存，确保跨版本兼容性
        """
        import json
        import os
        import warnings
        from datetime import datetime
        
        # 屏蔽特定的TensorFlow警告（只屏蔽compile_metrics和HDF5 legacy警告）
        warnings.filterwarnings('ignore', message='.*compile_metrics.*')
        warnings.filterwarnings('ignore', message='.*HDF5 file.*legacy.*')
        
        # 确定基础路径（移除扩展名）
        base_path = filepath.replace('.keras', '').replace('.h5', '')
        
        # 记录保存状态
        saved_formats = []
        failed_formats = []
        
        try:
            # 策略1：保存为H5格式（最稳定，兼容性最好）
            try:
                actor_h5_path = f"{base_path}_actor.h5"
                self.shared_network.actor.save(actor_h5_path, save_format='h5')
                critic_h5_path = f"{base_path}_critic.h5"
                self.shared_network.critic.save(critic_h5_path, save_format='h5')
                saved_formats.append("H5")
            except Exception as e:
                failed_formats.append(f"H5({str(e)[:30]})")
            
            # 策略2：保存权重为独立的H5文件（作为备份）
            try:
                actor_weights_path = f"{base_path}_actor_weights.h5"
                self.shared_network.actor.save_weights(actor_weights_path)
                critic_weights_path = f"{base_path}_critic_weights.h5"
                self.shared_network.critic.save_weights(critic_weights_path)
                saved_formats.append("Weights")
            except Exception as e:
                failed_formats.append(f"Weights({str(e)[:30]})")
            
            # 策略3：保存元数据为JSON（关键！用于重建模型）
            try:
                is_multidiscrete = self.shared_network.is_multidiscrete
                
                meta = {
                    'state_dim': int(self.state_dim),
                    'action_space': {
                        'type': 'MultiDiscrete' if is_multidiscrete else 'Discrete',
                        'nvec': [int(x) for x in self.action_space.nvec] if is_multidiscrete else None,
                        'n': int(self.action_space.n) if not is_multidiscrete else None
                    },
                    'global_state_dim': int(self.global_state_dim),
                    'network_config': self.shared_network.config,
                    'num_agents': len(self.agent_ids),
                    'tensorflow_version': tf.__version__,
                    'save_timestamp': datetime.now().isoformat()
                }
                
                meta_path = f"{base_path}_meta.json"
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                saved_formats.append("Meta")
            except Exception as e:
                failed_formats.append(f"Meta({str(e)[:30]})")
            
            # 策略4：尝试保存为.keras格式（TF 2.15+更稳定）
            try:
                keras_actor_path = f"{base_path}_actor.keras"
                self.shared_network.actor.save(keras_actor_path, save_format='keras')
                keras_critic_path = f"{base_path}_critic.keras"
                self.shared_network.critic.save(keras_critic_path, save_format='keras')
                saved_formats.append("Keras")
            except Exception as e:
                failed_formats.append(f"Keras({str(e)[:30]})")
            
            # 统一输出：简洁版
            if len(saved_formats) == 4:
                # 所有格式都成功 - 只输出一行
                print(f"✅ 4种格式模型已保存至: {base_path}_*")
            else:
                # 有失败的格式 - 输出详细信息
                if saved_formats:
                    print(f"✅ 已保存 {len(saved_formats)}/4 种格式: {', '.join(saved_formats)}")
                if failed_formats:
                    print(f"⚠️ 保存失败: {', '.join(failed_formats)}")
            
            # 返回H5路径作为主要加载目标
            actor_h5_path = f"{base_path}_actor.h5"
            return actor_h5_path if "H5" in saved_formats else ""
            
        except Exception as e:
            print(f"❌ 模型保存过程失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
        finally:
            # 恢复警告设置
            warnings.filterwarnings('default', message='.*compile_metrics.*')
            warnings.filterwarnings('default', message='.*HDF5 file.*legacy.*')

    def _get_target_parts(self, curriculum_config: Optional[Dict]) -> int:
        """10-23-20-15 修复版：统一获取当前回合的目标零件数，优先使用实际训练配置"""
        # 10-23-20-15 优先使用上一个训练回合的实际订单配置
        if hasattr(self, '_last_episode_config') and self._last_episode_config:
            custom_orders = self._last_episode_config.get('custom_orders')
            if custom_orders:
                return get_total_parts_count(custom_orders)
        
        # 备用逻辑：从curriculum_config获取
        if curriculum_config and 'custom_orders' in curriculum_config:
            # 泛化阶段或自定义订单
            return get_total_parts_count(curriculum_config['custom_orders'])
        elif curriculum_config and 'orders_scale' in curriculum_config:
            # 课程学习阶段
            base_parts = get_total_parts_count()
            return int(base_parts * curriculum_config['orders_scale'])
        else:
            # 默认或基础训练阶段
            return get_total_parts_count()

    def check_curriculum_stage_graduation(self, kpi_results: Dict[str, float], current_score: float, stage_config: Dict[str, Any]) -> bool:
        """检查当前课程学习阶段是否达到毕业标准"""
        criteria = stage_config.get("graduation_criteria")
        if not criteria:
            return False # 如果没有定义标准，则无法毕业

        # 获取当前阶段的目标零件数
        target_parts = int(get_total_parts_count() * stage_config.get('orders_scale', 1.0))
        completion_rate_kpi = (kpi_results.get('mean_completed_parts', 0) / target_parts) * 100 if target_parts > 0 else 0
        completion_rate_kpi = float(min(100.0, completion_rate_kpi))
        
        target_score = criteria["target_score"]
        stability_goal = criteria["target_consistency"]
        min_completion_rate = criteria["min_completion_rate"]
        # 新增：处理延期阈值
        tardiness_threshold = criteria.get("tardiness_threshold")
        current_tardiness = kpi_results.get('mean_tardiness', float('inf'))

        conditions_met = {
            f"完成率(>={min_completion_rate}%)": completion_rate_kpi >= min_completion_rate,
            f"分数(>={target_score})": current_score >= target_score,
        }
        
        if tardiness_threshold is not None:
            conditions_met[f"延期(<={tardiness_threshold}min)"] = current_tardiness <= tardiness_threshold

        if all(conditions_met.values()):
            self.curriculum_stage_achievement_count += 1
            print(f"[CURRICULUM] 阶段 '{stage_config['name']}' 达标: 完成率 {completion_rate_kpi:.1f}%, 分数 {current_score:.3f} (连续第{self.curriculum_stage_achievement_count}/{stability_goal}次)")
        else:
            if self.curriculum_stage_achievement_count > 0:
                reasons = [k for k, v in conditions_met.items() if not v]
                print(f"[CURRICULUM] 阶段 '{stage_config['name']}' 连续达标中断. 未达标项: {', '.join(reasons)}")
            self.curriculum_stage_achievement_count = 0

        return self.curriculum_stage_achievement_count >= stability_goal

