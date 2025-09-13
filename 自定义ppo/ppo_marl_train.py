"""
纯净的多智能体PPO训练脚本
专注于核心训练功能，移除复杂的评估和可视化
"""

import os
# 🔧 V10.2 终极日志清理: 在所有库导入前，强制设置日志级别
# 这能最有效地屏蔽掉CUDA和cuBLAS在子进程中的初始化错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import sys
import time
import random
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 🔧 V12 新增：TensorBoard支持
try:
    from tensorflow.python.summary.writer.writer import FileWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env, WFactoryEnv
from environments.w_factory_config import *
# 🔧 V38 新增：导入任务可行性分析函数
from environments.w_factory_config import validate_config, get_total_parts_count

class ExperienceBuffer:
    """🔧 MAPPO经验缓冲区 - 支持全局状态"""
    
    def __init__(self):
        self.states = []
        self.global_states = []  # 🔧 新增：存储全局状态
        self.actions = []
        self.rewards = []
        self.values = []
        self.action_probs = []
        self.dones = []
        self.truncateds = []
        
    def store(self, state, global_state, action, reward, value, action_prob, done, truncated=False):
        self.states.append(state)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.action_probs.append(action_prob)
        self.dones.append(done)
        self.truncateds.append(truncated)
    
    def get_batch(self, gamma=0.99, lam=0.95, next_value_if_truncated=None):
        """🔧 MAPPO改进：正确处理轨迹截断"""
        states = np.array(self.states)
        global_states = np.array(self.global_states) # 🔧 新增
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        action_probs = np.array(self.action_probs)
        dones = np.array(self.dones)
        truncateds = np.array(self.truncateds)
        
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 🔧 关键修复：正确处理最后一步
                if truncateds[t] and next_value_if_truncated is not None:
                    # 截断：使用critic预测的下一个状态价值
                    next_value = next_value_if_truncated
                elif dones[t]:
                    # 真正终止：价值为0
                    next_value = 0
                else:
                    # 🔧 修复：既不截断也不终止（正常trajectory结束）
                    # 使用bootstrap价值（如果提供）
                    next_value = next_value_if_truncated if next_value_if_truncated is not None else 0
            else:
                next_value = values[t + 1]
            
            # GAE计算
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        
        # 🔧 修复：更稳健的优势标准化
        if len(advantages) > 1:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            # 避免标准差过小导致的数值不稳定
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                advantages = advantages - adv_mean
        
        # 🔧 新增：优势裁剪，防止极端值（但保留足够的动态范围）
        advantages = np.clip(advantages, -5, 5)
        
        return states, global_states, actions, action_probs, advantages, returns
    
    def clear(self):
        self.states.clear()
        self.global_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.action_probs.clear()
        self.dones.clear()
        self.truncateds.clear()

class PPONetwork:
    """🔧 MAPPO网络实现 - 包含集中式Critic"""
    
    # 🔧 V3 修复: lr参数现在可以是学习率调度器
    def __init__(self, state_dim: int, action_dim: int, lr: Any, global_state_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim # 🔧 新增
        self.lr = lr
        
        # 构建网络
        self.actor, self.critic = self._build_networks()
        
        # 优化器 - 🔧 修复：处理lr为None的情况（worker不需要优化器）
        if lr is not None:
            self.actor_optimizer = tf.keras.optimizers.Adam(lr)
            self.critic_optimizer = tf.keras.optimizers.Adam(lr)
        else:
            self.actor_optimizer = None
            self.critic_optimizer = None
        
    def _build_networks(self):
        """🔧 MAPPO优化：使用配置文件参数构建网络"""
        # 导入配置
        from environments.w_factory_config import PPO_NETWORK_CONFIG
        hidden_sizes = PPO_NETWORK_CONFIG["hidden_sizes"]
        dropout_rate = PPO_NETWORK_CONFIG["dropout_rate"]
        
        # Actor网络 (去中心化) - 使用局部观测
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        # 🔧 修复：添加正确的权重初始化
        actor_x = tf.keras.layers.Dense(
            hidden_sizes[0], 
            activation='relu',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(state_input)
        actor_x = tf.keras.layers.Dropout(dropout_rate)(actor_x)
        actor_x = tf.keras.layers.Dense(
            hidden_sizes[1], 
            activation='relu',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(actor_x)
        # 🔧 策略输出层使用较小的初始化值
        action_probs = tf.keras.layers.Dense(
            self.action_dim, 
            activation='softmax',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(actor_x)
        actor = tf.keras.Model(inputs=state_input, outputs=action_probs)

        # Critic网络 (中心化) - 🔧 修复：网络大小应该与Actor平衡
        # 全局状态本身已经包含了更多信息，不需要过度增大网络
        global_state_input = tf.keras.layers.Input(shape=(self.global_state_dim,))
        # 🔧 修复：使用正确的权重初始化
        critic_x = tf.keras.layers.Dense(
            hidden_sizes[0],
            activation='relu',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(global_state_input)
        critic_x = tf.keras.layers.Dropout(dropout_rate)(critic_x)
        critic_x = tf.keras.layers.Dense(
            hidden_sizes[1],
            activation='relu',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(critic_x)
        critic_x = tf.keras.layers.Dropout(dropout_rate)(critic_x)
        # 🔧 Value输出层使用标准初始化
        value_output = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(critic_x)
        critic = tf.keras.Model(inputs=global_state_input, outputs=value_output)
        
        return actor, critic
    
    def get_action_and_value(self, state: np.ndarray, global_state: np.ndarray) -> Tuple[int, float, float]:
        """获取动作、价值和动作概率"""
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
        probs = self.actor(state_tensor)
        # 🔧 修复：数值稳定性
        probs = tf.clip_by_value(probs, 1e-8, 1.0)
        action = tf.random.categorical(tf.math.log(probs + 1e-8), 1)[0, 0].numpy()
        action_prob = probs[0, action].numpy()

        # 🔧 Critic使用全局状态
        value = self.critic(tf.expand_dims(tf.convert_to_tensor(global_state), 0))[0, 0].numpy()
        
        return action, float(value), float(action_prob)
    
    def get_value(self, global_state: np.ndarray) -> float:
        """获取状态价值（仅使用全局状态）"""
        global_state = tf.expand_dims(global_state, 0)
        return float(self.critic(global_state)[0, 0])
    
    def update(self, states: np.ndarray, global_states: np.ndarray, actions: np.ndarray, 
               old_probs: np.ndarray, advantages: np.ndarray, 
               returns: np.ndarray, clip_ratio: float = None, entropy_coeff: float = None) -> Dict[str, float]:
        """🔧 MAPPO更新：Critic使用全局状态"""
        # 🔧 修复：检查优化器是否存在
        if self.actor_optimizer is None or self.critic_optimizer is None:
            raise ValueError("Optimizers not initialized. Cannot update network.")
            
        # 🔧 V32 使用配置文件中的PPO参数
        if clip_ratio is None:
            clip_ratio = PPO_NETWORK_CONFIG["clip_ratio"]
        # 关键修复：优先使用传入的动态熵系数
        current_entropy_coeff = entropy_coeff if entropy_coeff is not None else PPO_NETWORK_CONFIG["entropy_coeff"]
        
        # Actor更新
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            # 🔧 修复：添加数值稳定性保护
            probs = tf.clip_by_value(probs, 1e-8, 1.0)
            dist = tf.compat.v1.distributions.Categorical(probs=probs)
            
            new_probs = dist.prob(actions)
            # 🔧 修复：防止除零和数值爆炸
            ratio = new_probs / (old_probs + 1e-8)
            ratio = tf.clip_by_value(ratio, 0.01, 100.0)  # 防止极端ratio
            
            # 🔧 修复：正确计算KL散度
            old_log_probs = tf.math.log(old_probs + 1e-8)
            new_log_probs = tf.math.log(new_probs + 1e-8)
            approx_kl = tf.reduce_mean(old_probs * (old_log_probs - new_log_probs))
            
            # 计算裁剪比例 (用于监控)
            clipped_mask = tf.greater(tf.abs(ratio - 1.0), clip_ratio)
            clip_fraction = tf.reduce_mean(tf.cast(clipped_mask, tf.float32))

            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            
            entropy = tf.reduce_mean(dist.entropy())
            actor_loss -= current_entropy_coeff * entropy
            
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # 🔧 新增：梯度裁剪以提高训练稳定性
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, 1.0)  # 增加到1.0，允许更大梯度
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Critic更新 (使用全局状态)
        with tf.GradientTape() as tape:
            values = self.critic(global_states)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # 🔧 新增：梯度裁剪
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, 1.0)  # 与actor保持一致
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return {
            "actor_loss": actor_loss.numpy(),
            "critic_loss": critic_loss.numpy(),
            "entropy": entropy.numpy(),
            "approx_kl": approx_kl.numpy(),
            "clip_fraction": clip_fraction.numpy()
        }

# 🔧 V8 新增: 多进程并行工作函数
def run_simulation_worker(network_weights: Dict[str, List[np.ndarray]],
                          state_dim: int, action_dim: int, num_steps: int, seed: int, 
                          global_state_dim: int, curriculum_config: Dict[str, Any] = None) -> Tuple[Dict[str, ExperienceBuffer], float]:
    """并行仿真工作进程 - 🔧 MAPPO改造：收集全局状态"""
    
    # 🔧 终极修复：将tf导入移至顶部，解决UnboundLocalError
    import tensorflow as tf
    import numpy as np
    import random
    
    # 1. 初始化
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用GPU
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    tf.random.set_seed(seed)
    env = make_parallel_env(curriculum_config)
    # 🔧 修复：使用动态学习率而非固定值
    # 注意：worker不需要学习率，只做推理
    network = PPONetwork(state_dim, action_dim, None, global_state_dim) # Worker不需要优化器
    network.actor.set_weights(network_weights['actor'])
    network.critic.set_weights(network_weights['critic']) # 🔧 Critic权重也需要同步
    
    buffers = {agent: ExperienceBuffer() for agent in env.agents}
    
    observations, infos = env.reset(seed=seed)
    global_state = infos[env.agents[0]]['global_state']
    
    total_reward_collected = 0.0
    collected_steps = 0
    step_count = 0
    
    while collected_steps < num_steps:
        actions = {}
        values = {}
        action_probs = {}
        
        # 🔧 修复：确保所有智能体使用同一个全局状态
        current_global_state = global_state.copy() if global_state is not None else np.zeros(global_state_dim)

        # 🔧 修复：确保智能体动作的同步性
        for agent in env.agents:  # 使用env.agents确保顺序一致
            if agent in observations:
                obs = observations[agent]
                action, value, action_prob = network.get_action_and_value(obs, current_global_state)
                actions[agent] = action
                values[agent] = value
                action_probs[agent] = action_prob
            
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1
        collected_steps += 1
        global_state = infos[env.agents[0]]['global_state']
        
        
        total_reward_collected += sum(rewards.values())

        # 🔧 修复：确保所有智能体的数据一致性
        for agent in env.agents:
            if agent in observations and agent in actions:
                terminated = terminations.get(agent, False)
                truncated = truncations.get(agent, False)
                reward = rewards.get(agent, 0)
                # 🔧 重要：存储时使用相同的全局状态
                buffers[agent].store(
                    observations[agent], 
                    current_global_state.copy(),  # 使用副本避免引用问题
                    actions[agent], 
                    reward,
                    values[agent], 
                    action_probs[agent], 
                    terminated,
                    truncated
                )

        observations = next_observations

        # 🔧 修复：与评估一致的终止条件
        if any(terminations.values()) or any(truncations.values()) or step_count >= 1500:
            
            # 🔧 MAPPO关键修复：正确处理截断时的bootstrap价值
            # 注意：这里暂时不处理，让buffer自己在get_batch时处理
            pass
            
        
            # total_reward += sum(episode_rewards.values())
            # 重置
            observations, infos = env.reset(seed=seed)
            global_state = infos[env.agents[0]]['global_state']
            step_count = 0  # 重置episode步数计数器

    env.close()
    return buffers, total_reward_collected

class SimplePPOTrainer:
    """🔧 V31 自适应PPO训练器：根据训练状态自动调整训练策略"""
    
    # 🔧 V31 新增：支持自适应训练目标和动态轮数调整
    def __init__(self, initial_lr: float, total_train_episodes: int, steps_per_episode: int, training_targets: dict = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        
        # 🔧 V32 使用配置文件的系统资源配置
        self.num_workers = SYSTEM_CONFIG["num_parallel_workers"]
        print(f"🔧 使用 {self.num_workers} 个并行环境进行数据采集")
        
        # 🔧 V32 使用配置文件的TensorFlow线程配置
        tf.config.threading.set_inter_op_parallelism_threads(SYSTEM_CONFIG["tf_inter_op_threads"])
        tf.config.threading.set_intra_op_parallelism_threads(SYSTEM_CONFIG["tf_intra_op_threads"])
        print(f"🔧 TensorFlow将使用 {SYSTEM_CONFIG['tf_inter_op_threads']}个inter线程, {SYSTEM_CONFIG['tf_intra_op_threads']}个intra线程")
        
        # 环境探测
        # 之前的代码依赖动态配置，现在我们直接创建
        temp_env = make_parallel_env()
        self.state_dim = temp_env.observation_space(temp_env.possible_agents[0]).shape[0]
        self.action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
        self.global_state_dim = temp_env.global_state_space.shape[0]
        self.agent_ids = temp_env.possible_agents
        temp_env.close()
        
        print("🔧 环境空间检测:")
        print(f"   观测维度: {self.state_dim}")
        print(f"   动作维度: {self.action_dim}")
        print(f"   智能体数量: {len(self.agent_ids)}")
        
        # 🔧 V26 终极修复：移除动态参数调整
        optimized_episodes = total_train_episodes
        optimized_steps = steps_per_episode
        
        # 🔧 V32 使用配置文件的学习率调度配置
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=LEARNING_RATE_CONFIG["initial_lr"],
            decay_steps=optimized_episodes * optimized_steps,
            end_learning_rate=LEARNING_RATE_CONFIG["end_lr"],
            power=LEARNING_RATE_CONFIG["decay_power"]
        )

        # 共享网络
        self.shared_network = PPONetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=self.lr_schedule,
            global_state_dim=self.global_state_dim
        )
        
        # 训练统计
        self.episode_rewards = []
        self.training_losses = []
        self.iteration_times = []  # 🔧 V5 新增：记录每轮训练时间
        self.kpi_history = []      # 🔧 V5 新增：记录每轮KPI历史
        self.initial_lr = initial_lr  # 🔧 V19 修复: 保存初始学习率
        self.start_time = time.time()
        self.start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🔧 核心改造：新增"最终阶段"最佳KPI跟踪器
        self.final_stage_best_kpi = {
            'mean_completed_parts': -1.0,
            'mean_makespan': float('inf'),
            'mean_utilization': 0.0,
            'mean_tardiness': float('inf')
        }
        self.final_stage_best_score = -1.0
        self.final_stage_best_episode = -1 # 🔧 新增：记录最佳KPI的回合数
        
        # 🔧 核心改造：新增"双达标"最佳KPI跟踪器
        self.best_kpi_dual_objective = self.final_stage_best_kpi.copy()
        self.best_score_dual_objective = -1.0
        self.best_episode_dual_objective = -1

        # 🔧 V32 统一：使用配置文件中的自适应训练配置
        self.training_targets = training_targets or ADAPTIVE_TRAINING_CONFIG.copy()
        
        # 🔧 V31 新增：自适应训练状态跟踪
        self.adaptive_state = {
            "target_achieved_count": 0,          # 连续达到目标的次数
            "best_performance": 0.0,             # 历史最佳性能
            "last_improvement_episode": 0,       # 上次改进的轮数
            "performance_history": [],           # 性能历史记录
            "training_phase": "exploration",     # 当前训练阶段：exploration, exploitation, fine_tuning
            "stagnation_counter": 0,             # 停滞计数器
            "last_stagnation_performance": -1.0, # 上一次停滞时的性能
        }
        # 🔧 V34 初始化动态训练参数
        self.current_entropy_coeff = PPO_NETWORK_CONFIG["entropy_coeff"] # 初始化动态熵系数
        self.current_learning_rate = LEARNING_RATE_CONFIG["initial_lr"] # 🔧 V34 修复：使用正确的学习率配置
        
        # 🔧 新增：熵系数退火计划（改进版）
        self.entropy_decay_rate = 0.999  # 更慢的衰减率
        self.min_entropy_coeff = 0.02    # 更高的最小熵系数，保持基本探索
        
        
        # 🔧 V40 新增：回合事件日志记录器
        self.episode_events = []
        
        # 创建保存目录 (V31新增：以训练开始时间创建专用文件夹)
        self.base_models_dir = "自定义ppo/ppo_models"
        self.models_dir = f"{self.base_models_dir}/{self.start_time_str}"
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"📁 模型保存目录: {self.models_dir}")
        
        # 🔧 V12 新增：TensorBoard支持
        self.tensorboard_dir = f"自定义ppo/tensorboard_logs/{self.timestamp}"
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        if TENSORBOARD_AVAILABLE:
            self.train_writer = None
            self.current_tensorboard_run_name = None
            print(f"📊 TensorBoard命令: tensorboard --logdir=\"{self.tensorboard_dir}\"")
        else:
            self.train_writer = None
            print("⚠️  TensorBoard不可用")
    
    def should_continue_training(self, episode: int, current_score: float, completion_rate: float) -> tuple:
        """🔧 核心改造：评估是否应该继续训练，基于"综合评分"""
        targets = self.training_targets
        state = self.adaptive_state
        
        # 基本限制检查 (移除min_episodes检查)
        if episode >= targets["max_episodes"]:
            return False, f"已达到最大训练轮数({targets['max_episodes']})", 0
        
        # 核心逻辑：必须同时满足100%完成率和目标分数
        target_score = targets["target_score"]
        if completion_rate >= 100 and current_score >= target_score:
            state["target_achieved_count"] += 1
            print(f"🎯 双重目标达成: 完成率 {completion_rate:.1f}% & 分数 {current_score:.3f} (连续第{state['target_achieved_count']}次)")
            
            if state["target_achieved_count"] >= targets["target_consistency"]:
                return False, f"连续{targets['target_consistency']}次达到双重目标", 0
        else:
            # 任何一个不满足，计数器就重置
            state["target_achieved_count"] = 0

        # 早停逻辑 (保持不变，基于分数)
        state["performance_history"].append(current_score)
        if len(state["performance_history"]) > targets["performance_window"]:
            state["performance_history"].pop(0)

        if current_score > state["best_performance"]:
            state["best_performance"] = current_score
            state["last_improvement_episode"] = episode
        
        improvement_gap = episode - state["last_improvement_episode"]
        if improvement_gap >= targets["early_stop_patience"]:
            if len(state["performance_history"]) >= targets["performance_window"]:
                recent_avg_score = sum(state["performance_history"]) / len(state["performance_history"])
                if recent_avg_score < target_score * 0.8:
                    return False, f"连续{improvement_gap}轮无改进，且平均分数低于{target_score*0.8:.3f}", 0
        
        return True, f"当前分数 {current_score:.3f}, 完成率 {completion_rate:.1f}%", 0
    
    def create_environment(self, curriculum_stage=None):
        """创建环境（支持课程学习）"""
        config = {}
        
        # 🔧 V16：实现课程学习的环境配置
        if curriculum_stage is not None and CURRICULUM_CONFIG.get("enabled", False):
            stage = CURRICULUM_CONFIG["stages"][curriculum_stage] if curriculum_stage < len(CURRICULUM_CONFIG["stages"]) else CURRICULUM_CONFIG["stages"][-1]
            config['curriculum_stage'] = stage
            config['orders_scale'] = stage.get('orders_scale', 1.0)
            config['time_scale'] = stage.get('time_scale', 1.0)
            print(f"📚 课程学习阶段 {curriculum_stage+1}: {stage['name']} (订单比例: {stage['orders_scale']}, 时间倍数: {stage['time_scale']})")
        
        env = make_parallel_env(config)
        buffers = {
            agent: ExperienceBuffer() 
            for agent in env.possible_agents
        }
        return env, buffers
    
    def collect_experience_parallel(self, buffers, num_steps: int, curriculum_config: Dict[str, Any] = None) -> float:
        """🔧 V17修复：使用多进程并行收集经验，支持课程学习"""
        for buffer in buffers.values():
            buffer.clear()

        network_weights = {
            'actor': self.shared_network.actor.get_weights(),
            'critic': self.shared_network.critic.get_weights()
        }
        steps_per_worker = num_steps // self.num_workers
        
        total_reward = 0

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.num_workers):
                seed = random.randint(0, 1_000_000)
                future = executor.submit(
                    run_simulation_worker,
                    network_weights,
                    self.state_dim,
                    self.action_dim,
                    steps_per_worker,
                    seed,
                    self.global_state_dim,
                    curriculum_config  # 🔧 V17修复：传递课程学习配置
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    worker_buffers, worker_reward = future.result()
                    total_reward += worker_reward
                    
                    for agent_id, worker_buffer in worker_buffers.items():
                        buffers[agent_id].states.extend(worker_buffer.states)
                        buffers[agent_id].global_states.extend(worker_buffer.global_states)
                        buffers[agent_id].actions.extend(worker_buffer.actions)
                        buffers[agent_id].rewards.extend(worker_buffer.rewards)
                        buffers[agent_id].values.extend(worker_buffer.values)
                        buffers[agent_id].action_probs.extend(worker_buffer.action_probs)
                        buffers[agent_id].dones.extend(worker_buffer.dones)
                        buffers[agent_id].truncateds.extend(worker_buffer.truncateds)
                except Exception as e:
                    print(f"❌ 一个并行工作进程失败: {e}")
                    import traceback
                    traceback.print_exc()

        return total_reward
    
    def update_policy(self, buffers, entropy_coeff: float) -> Dict[str, float]:
        """🔧 MAPPO改进：正确处理多智能体的策略更新"""
        all_states = []
        all_global_states = []
        all_actions = []
        all_action_probs = []
        all_advantages = []
        all_returns = []
        
        # 🔧 为每个智能体单独计算advantages，考虑截断
        for agent, buffer in buffers.items():
            if len(buffer.states) > 0:
                # 🔧 修复：正确获取截断时的bootstrap价值
                next_value_if_truncated = None
                if len(buffer.truncateds) > 0 and buffer.truncateds[-1]:
                    # 如果最后一步是截断，使用最后存储的全局状态估计价值
                    # 注意：这里应该使用"下一个"全局状态，但如果没有，就用最后一个
                    last_global_state = buffer.global_states[-1]
                    next_value_if_truncated = self.shared_network.get_value(last_global_state)
                elif len(buffer.states) > 0 and not buffer.dones[-1]:
                    # 如果trajectory既不终止也不截断（被steps_per_episode截断）
                    # 也需要bootstrap
                    last_global_state = buffer.global_states[-1]
                    next_value_if_truncated = self.shared_network.get_value(last_global_state)
                
                states, global_states, actions, action_probs, advantages, returns = buffer.get_batch(
                    next_value_if_truncated=next_value_if_truncated
                )
                
                all_states.extend(states)
                all_global_states.extend(global_states)
                all_actions.extend(actions)
                all_action_probs.extend(action_probs)
                all_advantages.extend(advantages)
                all_returns.extend(returns)
                
                buffer.clear()
        
        if len(all_states) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        
        # 转换为numpy数组
        all_states = np.array(all_states)
        all_global_states = np.array(all_global_states)
        all_actions = np.array(all_actions)
        all_action_probs = np.array(all_action_probs, dtype=np.float32) # 🔧 修复：确保数据类型为float32
        all_advantages = np.array(all_advantages, dtype=np.float32)     # 🔧 修复：确保数据类型为float32
        all_returns = np.array(all_returns, dtype=np.float32).reshape(-1, 1)
        
        # 🔧 新增：奖励标准化（提高训练稳定性）
        returns_mean = np.mean(all_returns)
        returns_std = np.std(all_returns) + 1e-8
        all_returns = (all_returns - returns_mean) / returns_std
        
        # 🔧 V32 使用配置文件的策略更新次数
        losses = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'approx_kl': 0, 'clip_fraction': 0}
        num_updates = PPO_NETWORK_CONFIG["num_policy_updates"]
        
        # 🔧 修复：添加早停机制，避免过度更新
        for epoch in range(num_updates):
            batch_losses = self.shared_network.update(
                states=all_states,
                global_states=all_global_states,
                actions=all_actions,
                old_probs=all_action_probs,
                advantages=all_advantages,
                returns=all_returns,
                entropy_coeff=entropy_coeff # 传递动态熵系数
            )
            
            for key in losses:
                losses[key] += batch_losses[key] / num_updates
            
            # 🔧 新增：如果KL散度过大，提前停止更新
            if batch_losses['approx_kl'] > 0.02:  # 稍微提高KL阈值
                if epoch > 0:  # 至少更新一次
                    break
        
        return losses
    
    def _independent_exam_evaluation(self, env, curriculum_config, seed):
        """🔧 V33 新增：独立的考试评估，确保每轮都是全新的仿真"""
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        
        observations, _ = env.reset(seed=seed)
        episode_reward = 0
        step_count = 0
        
        while step_count < 1200:
            actions = {}
            
            # 使用确定性策略，但基于新的随机环境状态
            for agent in env.agents:
                if agent in observations:
                    state = tf.expand_dims(observations[agent], 0)
                    action_probs = self.shared_network.actor(state)
                    # 🔧 使用确定性评估，但保留少量探索
                    if random.random() < 0.1:  # 10%概率探索，避免完全卡死
                        action = int(tf.random.categorical(tf.math.log(action_probs + 1e-8), 1)[0])
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
    
    def quick_kpi_evaluation(self, num_episodes: int = 3, curriculum_config: Dict[str, Any] = None) -> Dict[str, float]:
        """🔧 V39修复：快速KPI评估（支持课程学习配置和静默模式）"""
        # 🔧 V39修复：创建环境时传递课程配置，包括静默模式
        # 课程配置直接通过make_parallel_env传递，由环境内部处理
        if curriculum_config:
            env = make_parallel_env(curriculum_config)
        else:
            # 🔧 V39 修复一个潜在bug：正确解包create_environment的返回值
            env, _ = self.create_environment()
        
        total_rewards = []
        makespans = []
        utilizations = []
        completed_parts_list = []
        tardiness_list = []
        
        for episode in range(num_episodes):
            observations, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            # 🔧 修复：使用与训练一致的步数限制
            while step_count < 1200:
                actions = {}
                
                # 使用确定性策略评估
                for agent in env.agents:
                    if agent in observations:
                        state = tf.expand_dims(observations[agent], 0)
                        action_probs = self.shared_network.actor(state)
                        action = int(tf.argmax(action_probs[0]))
                        actions[agent] = action
                
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
            observations, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < 1200:
                actions = {}
                
                # 使用确定性策略评估
                for agent in env.agents:
                    if agent in observations:
                        state = tf.expand_dims(observations[agent], 0)
                        action_probs = self.shared_network.actor(state)
                        action = int(tf.argmax(action_probs[0]))
                        actions[agent] = action
                
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
        """🔧 V31 自适应训练主循环：根据性能自动调整训练策略和轮数"""
        # 🔧 V31 自适应模式：最大轮数作为上限，实际轮数根据性能动态决定

        if adaptive_mode:
            self.training_targets["max_episodes"] = max_episodes
        
        # 🔧 V16：显示课程学习配置
        if CURRICULUM_CONFIG.get("enabled", False):
            print(f"📚 课程学习已启用，共{len(CURRICULUM_CONFIG['stages'])}个阶段:")
            for i, stage in enumerate(CURRICULUM_CONFIG["stages"]):
                print(f"   阶段{i+1}: {stage['name']} - {stage['iterations']}轮，订单{stage['orders_scale']*100:.0f}%")
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
        curriculum_enabled = CURRICULUM_CONFIG.get("enabled", False)
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
        stage_best_scores = [-1.0] * len(CURRICULUM_CONFIG["stages"])
        
        try:
            for episode in range(max_episodes):
                iteration_start_time = time.time()
                
                # 🔧 V17关键修复：课程学习阶段管理
                current_curriculum_config = None
                if curriculum_enabled:
                    stage_config = CURRICULUM_CONFIG["stages"][current_stage]
                    
                    # 🔧 V31 强化毕业考试机制：使用新的高标准门槛，防止带病毕业
                    if stage_episode_count >= stage_config["iterations"]:
                        if current_stage < len(CURRICULUM_CONFIG["stages"]) - 1:
                            # 🔧 V33 修复：暂停训练计时，隔离考试时间
                            iteration_pause_time = time.time()
                            
                            print("\n" + "="*60)
                            print(f"🎓 阶段 '{stage_config['name']}' 训练完成，开始强化毕业考试...")
                            
                            # 🔧 V31 使用新的毕业门槛配置
                            graduation_config = CURRICULUM_CONFIG.get("graduation_config", {})
                            
                            # 🔧 修复：从当前阶段配置中获取毕业阈值
                            current_threshold = stage_config.get("graduation_thresholds", 95.0)
                            exam_episodes = graduation_config.get("exam_episodes", 5)
                            stability_requirement = graduation_config.get("stability_requirement", 3)
                            max_retries = graduation_config.get("max_retries", 3)
                            retry_extension = graduation_config.get("retry_extension", 15)
                            
                            # 🔧 V34 修复：毕业考试应检验当前阶段的掌握情况，而不是用下一阶段的标准
                            current_stage_data = CURRICULUM_CONFIG["stages"][current_stage]
                            exam_target_parts = int(get_total_parts_count() * current_stage_data['orders_scale'])
                            exam_config = {
                                'orders_scale': current_stage_data.get('orders_scale', 1.0),
                                'time_scale': current_stage_data.get('time_scale', 1.0),
                                'stage_name': f"考试: {current_stage_data.get('name', '')}"
                            }
                            
                            # 🔧 V33 修复：强化考试随机性，确保每轮考试结果独立
                            exam_results = []
                            for exam_round in range(exam_episodes):
                                # 关键修复：为每轮考试设置不同的随机种子
                                exam_seed = random.randint(0, 1000000) + exam_round * 1000
                                
                                # 创建独立的评估环境，避免状态污染
                                temp_env = make_parallel_env(exam_config)
                                temp_env.reset(seed=exam_seed)
                                
                                # 执行独立的评估轮次
                                exam_kpi = self._independent_exam_evaluation(temp_env, exam_config, exam_seed)
                                temp_env.close()
                                
                                exam_completed_parts = exam_kpi.get('mean_completed_parts', 0)
                                exam_completion_rate = (exam_completed_parts / exam_target_parts) * 100 if exam_target_parts > 0 else 0
                                exam_results.append(exam_completion_rate)
                                print(f"   第{exam_round+1}轮考试: {exam_completed_parts:.1f}/{exam_target_parts} 零件 ({exam_completion_rate:.1f}%)")
                            
                            # 计算稳定性：需要连续多次达到门槛
                            avg_completion_rate = sum(exam_results) / len(exam_results)
                            passed_exams = sum(1 for rate in exam_results if rate >= current_threshold)
                            stability_achieved = passed_exams >= stability_requirement
                            
                            print(f"   考试结果: 平均 {avg_completion_rate:.1f}% | 通过门槛: {current_threshold:.1f}% | 达标次数: {passed_exams}/{exam_episodes}")
                            print(f"   稳定性要求: {stability_requirement}次达标")
                            
                            # 🔧 V37 修复：稳定性达到即通过，无需重复检查平均分数
                            if stability_achieved:
                                # 关键修复：需要获取下一阶段的数据来打印日志
                                next_stage_data = CURRICULUM_CONFIG["stages"][current_stage + 1]
                                print(f"   ✅ 毕业考试通过！进入下一阶段: '{next_stage_data['name']}'")
                                current_stage += 1
                                stage_episode_count = 0
                                if not hasattr(self, '_stage_retry_count'):
                                    self._stage_retry_count = {}
                                self._stage_retry_count[current_stage] = 0  # 重置重考计数
                            else:
                                if not hasattr(self, '_stage_retry_count'):
                                    self._stage_retry_count = {}
                                retry_count = self._stage_retry_count.get(current_stage, 0)
                                
                                if retry_count < max_retries:
                                    self._stage_retry_count[current_stage] = retry_count + 1
                                    print(f"   ❌ 考试未通过。延长{retry_extension}轮训练后重考 (第{retry_count+1}/{max_retries}次重考)")
                                    stage_config["iterations"] += retry_extension
                                else:
                                    print(f"   ⚠️ 已达最大重考次数，强制进入下一阶段（但可能表现不佳）")
                                    current_stage += 1
                                    stage_episode_count = 0
                                    self._stage_retry_count[current_stage] = 0
                            
                            print("="*60 + "\n")
                            
                            # 🔧 V33 修复：恢复训练计时，补偿考试时间
                            exam_duration = time.time() - iteration_pause_time
                            iteration_start_time += exam_duration  # 关键修复：补偿考试时间
                            
                        # 如果是最后阶段，则不再切换
                    
                    # 获取当前阶段配置 (可能已更新)
                    stage = CURRICULUM_CONFIG["stages"][current_stage]
                    current_curriculum_config = {
                        'orders_scale': stage.get('orders_scale', 1.0),
                        'time_scale': stage.get('time_scale', 1.0),
                        'stage_name': stage.get('name', f'Stage {current_stage}')
                    }
                    
                    # 🔧 V17增强：详细的阶段切换和状态日志
                    if stage_episode_count == 0:
                        print(f"📚 [回合 {episode+1}] 🔄 课程学习阶段切换!")
                        print(f"   新阶段: {stage['name']}")
                        print(f"   订单比例: {stage['orders_scale']} (目标零件数: {int(get_total_parts_count() * stage['orders_scale'])})")
                        print(f"   时间比例: {stage['time_scale']} (时间限制: {int(1200 * stage['time_scale'])}分钟)")
                        print(f"   计划训练轮数: {stage['iterations']}")
                        
                        # 🔧 V30 关键修复：确保课程配置正确传递到所有环境
                        print(f"🔧 当前课程配置将传递给所有worker: orders_scale={stage['orders_scale']}, time_scale={stage['time_scale']}")
                        
                        print("-" * 60)
                    
                    # 🔧 V17新增：每10轮显示阶段状态
                    if episode % 10 == 0:
                        progress = stage_episode_count / stage['iterations'] * 100
                        print(f"📚 课程状态: {stage['name']} ({stage_episode_count}/{stage['iterations']}, {progress:.1f}%)")
                        print(f"   当前难度: {int(get_total_parts_count() * stage['orders_scale'])}零件, {stage['time_scale']:.1f}x时间")    
                    stage_episode_count += 1
                

                collect_start_time = time.time()
                episode_reward = self.collect_experience_parallel(buffers, steps_per_episode, current_curriculum_config)
                collect_duration = time.time() - collect_start_time
                
                # 🔧 V6 安全的策略更新（包含内存检查）
                update_start_time = time.time()
                losses = self.update_policy(buffers, entropy_coeff=self.current_entropy_coeff)
                update_duration = time.time() - update_start_time
                
                # 记录统计
                iteration_end_time = time.time()
                iteration_duration = iteration_end_time - iteration_start_time
                self.iteration_times.append(iteration_duration)
                self.episode_rewards.append(episode_reward)

                
                # 提前进行KPI评估，以便整合TensorBoard日志
                kpi_results = self.quick_kpi_evaluation(num_episodes=2, curriculum_config=current_curriculum_config)
                self.kpi_history.append(kpi_results)

                # 🔧 核心改造：计算当前回合的综合评分
                current_score = self._calculate_score(kpi_results, current_curriculum_config)
                
                # 🔧 新增：智能熵系数调整（基于性能）
                completion_rate_kpi = (kpi_results.get('mean_completed_parts', 0) / get_total_parts_count()) * 100
                if episode > 100:  # 前100轮保持高探索
                    # 如果完成率高，可以降低探索；否则保持探索
                    if completion_rate_kpi >= 95:  # 高完成率时才降低熵
                        self.current_entropy_coeff = max(
                            self.min_entropy_coeff,
                            self.current_entropy_coeff * self.entropy_decay_rate
                        )
                    elif completion_rate_kpi < 80:  # 完成率低时增加探索
                        self.current_entropy_coeff = min(
                            PPO_NETWORK_CONFIG["entropy_coeff"],
                            self.current_entropy_coeff * 1.01  # 缓慢增加
                        )

                # 🔧 V36 统一TensorBoard日志记录，并根据课程阶段动态切换run
                if TENSORBOARD_AVAILABLE:
                    # 根据课程阶段切换run，在悬停提示中显示阶段名
                    run_name = "train_default" # Fallback run name
                    if curriculum_enabled and current_curriculum_config:
                        # Get stage name and sanitize it for use as a directory name
                        run_name = current_curriculum_config['stage_name'].replace(" ", "_")
                    
                    if self.train_writer is None or self.current_tensorboard_run_name != run_name:
                        if self.train_writer is not None:
                            self.train_writer.close()
                        
                        logdir = os.path.join(self.tensorboard_dir, run_name)
                        self.train_writer = tf.summary.create_file_writer(logdir)
                        self.current_tensorboard_run_name = run_name
                        print(f"📊 TensorBoard run已切换至: '{run_name}'")

                    if self.train_writer:
                        with self.train_writer.as_default():
                            # 训练核心指标
                            tf.summary.scalar('Training/Episode_Reward', episode_reward, step=episode)
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
                            # 🌟 新增：记录综合评分
                            tf.summary.scalar('KPI/Score', current_score, step=episode)
                            
                            self.train_writer.flush()
                
                # 🔧 核心改造：动态早停逻辑 - 在完成"完整挑战"阶段的指定轮数后，才开始评估
                should_continue = True
                reason = "继续训练"
                estimated_remaining = 0
                
                # 检查是否在最终阶段（完整挑战）
                is_final_stage = curriculum_enabled and (current_stage == len(CURRICULUM_CONFIG["stages"]) - 1)
                
                if is_final_stage:
                    # 获取最终阶段必须完成的课程轮数
                    final_stage_iterations = CURRICULUM_CONFIG["stages"][-1].get("iterations", 100)
                    
                    # 只有在完成了最终阶段的指定课程轮数后，才开始早停评估
                    if stage_episode_count > final_stage_iterations:
                        completion_rate_check = (kpi_results.get('mean_completed_parts', 0) / get_total_parts_count()) * 100
                        should_continue, reason, estimated_remaining = self.should_continue_training(episode + 1, current_score, completion_rate_check)
                        
                        # 每10轮打印一次早停评估状态
                        if episode % 10 == 0:
                            print(f"📊 最终阶段早停评估: {reason}")
                    else:
                        remaining_curriculum_eps = final_stage_iterations - stage_episode_count
                        reason = f"最终阶段课程还需 {remaining_curriculum_eps} 轮"
                
                # 🔧 V31 关键：检查是否应该提前结束训练
                if not should_continue:
                    print(f"\n🏁 自适应训练提前结束: {reason}")
                    break
                
                # 🔧 V36 新增：记录当前课程阶段信息供其他方法使用
                if current_curriculum_config:
                    self._current_orders_scale = current_curriculum_config.get('orders_scale', 1.0)
                
                # 🔧 重构版：简化的性能监控，移除复杂的重启机制
                # 基础性能跟踪（用于调试和监控）
                current_performance = kpi_results.get('mean_completed_parts', 0)
                if not hasattr(self, '_performance_history'):
                    self._performance_history = []
                
                self._performance_history.append(current_performance)
                # 只保留最近20轮的历史
                if len(self._performance_history) > 20:
                    self._performance_history.pop(0)
                
                # 🔧 V31 关键：检查是否应该提前结束训练
                if not should_continue:
                    print(f"\n🏁 自适应训练提前结束: {reason}")
                    break
                
                # 🔧 V38修复：每30回合进行一次完整难度评估（静默模式，避免输出污染）
                if episode > 0 and episode % 30 == 0:
                    print("\n" + "="*60)
                    print("🎓 进行完整难度评估（100%订单，标准时间）...")
                    full_config = {
                        'orders_scale': 1.0,
                        'time_scale': 1.0,
                        'stage_name': '完整评估',
                        'silent_evaluation': True  # 🔧 V38 关键：启用静默模式
                    }
                    full_kpi = self.quick_kpi_evaluation(num_episodes=3, curriculum_config=full_config)
                    
                    # 计算真实性能指标
                    real_completion = full_kpi.get('mean_completed_parts', 0)
                    real_completion_rate = real_completion / get_total_parts_count() * 100
                    real_makespan = full_kpi.get('mean_makespan', 0)
                    real_utilization = full_kpi.get('mean_utilization', 0)
                    
                    # 🔧 V34 修复：获取完整的评估指标，修复设备利用率显示异常
                    real_tardiness = full_kpi.get('mean_tardiness', 0) / 60  # 转换为分钟
                    real_reward = full_kpi.get('mean_reward', 0)
                    
                    print(f"🎯 完整难度评估结果（3轮平均）:")
                    print(f"   平均完成零件: {real_completion:.1f}/{get_total_parts_count()} ({real_completion_rate:.1f}%)")
                    print(f"   平均总完工时间: {real_makespan:.1f}分钟")
                    print(f"   平均设备利用率: {real_utilization*100:.1f}%")
                    print(f"   平均延期时间: {real_tardiness:.1f}分钟") 
                    print(f"   平均奖励: {real_reward:.1f}")
                    
                    # 评估进展
                    if real_completion_rate > 90:
                        print(f"🏆 优秀！接近完全掌握任务!")
                    elif real_completion_rate > 60:
                        print(f"💪 良好！已具备基本能力!")
                    elif real_completion_rate > 30:
                        print(f"📈 进步中！继续努力!")
                    else:
                        print(f"📚 仍需更多训练!")
                    print("="*60 + "\n")
                
                # 🔧 V12 TensorBoard KPI记录 (V36已整合)
                
                # 🔧 修复：正确更新最佳记录（只有当makespan > 0时才更新）
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
                # 🔧 核心改造：区分"全局最佳"和"最终阶段最佳"
                # 1. 更新全局最佳分数（用于日志显示）
                if current_score > self.best_score:
                    self.best_score = current_score

                # 2. 更新课程各阶段最佳分数并保存模型
                if curriculum_enabled:
                    if current_score > stage_best_scores[current_stage]:
                        stage_best_scores[current_stage] = current_score
                        stage_name = current_curriculum_config['stage_name'].replace(" ", "_")
                        model_path = self.save_model(f"{self.models_dir}/{stage_name}_best")
                        if model_path:
                            stage_display_name = current_curriculum_config['stage_name']
                            model_update_info = f"✅ {stage_display_name}阶段最佳得分刷新，模型已保存至: {model_path}"

                    # 3. 如果是最终阶段，则更新"最终阶段最佳模型"
                    if current_stage == len(CURRICULUM_CONFIG["stages"]) - 1:
                        if current_score > self.final_stage_best_score:
                            self.final_stage_best_score = current_score
                            self.final_stage_best_kpi = kpi_results.copy()
                            self.final_stage_best_episode = episode + 1 # 🔧 记录最佳KPI的回合数
                            final_model_path = self.save_model(f"{self.models_dir}/final_challenge_best")
                            model_update_info = f" 🏆最终阶段最佳! 模型保存至: {final_model_path}"
                        
                        # 🔧 核心改造：检查并更新"双达标"最佳模型
                        completion_rate_kpi = (kpi_results.get('mean_completed_parts', 0) / get_total_parts_count()) * 100
                        if completion_rate_kpi >= 100 and current_score > self.best_score_dual_objective:
                            self.best_score_dual_objective = current_score
                            self.best_kpi_dual_objective = kpi_results.copy()
                            self.best_episode_dual_objective = episode + 1
                            dual_objective_best_path = self.save_model(f"{self.models_dir}/dual_objective_best")
                            model_update_info = f" ⭐双达标最佳!模型保存至: {dual_objective_best_path}"

                else: # 非课程学习模式
                    # 在非课程学习模式下，我们将训练视为一个单一的"最终挑战"阶段
                    # 1. 更新"最终挑战"最佳模型 (等同于全局最佳)
                    if current_score > self.final_stage_best_score:
                        self.final_stage_best_score = current_score
                        self.final_stage_best_kpi = kpi_results.copy()
                        self.final_stage_best_episode = episode + 1 # 记录最佳KPI的回合数
                        final_model_path = self.save_model(f"{self.models_dir}/final_challenge_best")
                        if final_model_path:
                            model_update_info = f" 🏆全局最佳! 模型保存至: {final_model_path}"
                    
                    # 2. 检查并更新"双达标"最佳模型
                    completion_rate_kpi = (kpi_results.get('mean_completed_parts', 0) / get_total_parts_count()) * 100
                    if completion_rate_kpi >= 100 and current_score > self.best_score_dual_objective:
                        self.best_score_dual_objective = current_score
                        self.best_kpi_dual_objective = kpi_results.copy()
                        self.best_episode_dual_objective = episode + 1
                        dual_objective_best_path = self.save_model(f"{self.models_dir}/dual_objective_best")
                        if dual_objective_best_path:
                            model_update_info = f" ⭐双达标最佳!模型保存至: {dual_objective_best_path}"
                
                # 🔧 V33 优化：严格按照用户要求的日志格式
                # 第一行：回合信息和性能数据
                line1 = f"🔂 回合 {episode + 1:3d}/{max_episodes} | 奖励: {episode_reward:.1f} | Actor损失: {losses['actor_loss']:.4f}| ⏱️本轮用时: {iteration_duration:.1f}s (CPU采集: {collect_duration:.1f}s, GPU更新: {update_duration:.1f}s)"
                
                # 第二行：KPI数据和阶段信息
                target_parts_str = f"/{int(get_total_parts_count() * current_curriculum_config['orders_scale'])}" if curriculum_enabled and current_curriculum_config else f"/{get_total_parts_count()}"
                stage_info = f"   | 阶段：'{current_curriculum_config['stage_name']}'" if curriculum_enabled and current_curriculum_config else ""
                line2 = f"📊 KPI - 总完工时间: {makespan:.1f}min  | 设备利用率: {utilization:.1%} | 延期时间: {tardiness:.1f}min |  完成零件数: {completed_parts:.0f}{target_parts_str}{stage_info}"
                
                # 第三行：评分和模型更新信息
                if curriculum_enabled:
                    stage_best_str = f" (阶段最佳: {stage_best_scores[current_stage]:.3f})"
                    line3_score = f"🚥 回合评分: {current_score:.3f} (全局最佳: {self.best_score:.3f}){stage_best_str}"
                else:
                    line3_score = f"🚥 回合评分: {current_score:.3f} (全局最佳: {self.best_score:.3f})"
                line3 = f"{line3_score}{model_update_info}" if model_update_info else line3_score

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
            
            # 检查是否有模型达到了双重标准
            if self.best_episode_dual_objective != -1:
                best_kpi = self.best_kpi_dual_objective
                best_episode_to_report = self.best_episode_dual_objective
            else:
                print("⚠️ 未找到同时满足100%完成率和目标分数的模型，将报告最终阶段的最佳分数模型。")
                best_kpi = self.final_stage_best_kpi
                best_episode_to_report = self.final_stage_best_episode

            target_parts_final = get_total_parts_count() # 最终评估总是基于完整任务
            completion_rate_final = (best_kpi.get('mean_completed_parts', 0) / target_parts_final) * 100 if target_parts_final > 0 else 0
            
            print(f"   (在第 {best_episode_to_report} 回合取得)") # 🔧 新增
            print(f"   完成零件: {best_kpi.get('mean_completed_parts', 0):.1f} / {target_parts_final} ({completion_rate_final:.1f}%)")
            print(f"   总完工时间: {best_kpi.get('mean_makespan', 0):.1f} 分钟")
            print(f"   设备利用率: {best_kpi.get('mean_utilization', 0):.1%}")
            print(f"   总延期时间: {best_kpi.get('mean_tardiness', 0):.1f} 分钟")
            print("="*40)
            
            return {
                'training_time': total_training_time,
                'kpi_history': self.kpi_history,
                'iteration_times': self.iteration_times,
                'best_kpi': self.best_kpi_dual_objective if self.best_episode_dual_objective != -1 else self.final_stage_best_kpi
            }
            
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # 🔧 V8 优化: 主循环中没有env需要关闭
            pass
    
    def save_model(self, filepath: str) -> str:
        """保存模型并返回路径"""
        actor_path = f"{filepath}_actor.keras"
        try:
            self.shared_network.actor.save(actor_path)
            self.shared_network.critic.save(f"{filepath}_critic.keras")
            return actor_path
        except Exception as e:
            print(f"⚠️ 保存模型时出错: {e}")
            return ""

    def _calculate_score(self, kpi_results: Dict[str, float], curriculum_config: Dict) -> float:
        """统一计算回合评分的辅助函数"""
        makespan = kpi_results.get('mean_makespan', 0)
        completed_parts = kpi_results.get('mean_completed_parts', 0)
        utilization = kpi_results.get('mean_utilization', 0)
        tardiness = kpi_results.get('mean_tardiness', 0)

        if completed_parts == 0:
            return 0.0
        
        makespan_score = max(0, 1 - makespan / (SIMULATION_TIME * 1.5)) # 使用1.5倍仿真时间作为基准
        utilization_score = utilization
        tardiness_score = max(0, 1 - tardiness / (SIMULATION_TIME * 2.0)) # 使用2倍仿真时间作为基准

        target_parts = get_total_parts_count()
        if curriculum_config:
            target_parts = int(get_total_parts_count() * curriculum_config.get('orders_scale', 1.0))
        
        completion_score = completed_parts / target_parts if target_parts > 0 else 0
        
        current_score = (
            completion_score * 0.5 +
            tardiness_score * 0.25 +
            makespan_score * 0.15 +
            utilization_score * 0.1
        )
        return current_score

def main():
    
    print(f"✨ 训练进程PID: {os.getpid()}")

    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    try:
        max_episodes = 1000  # 最大轮数上限，实际轮数根据性能动态决定
        steps_per_episode = 1500  # 与评估保持一致的步数
        
        # 🔧 V32 使用配置文件的自适应训练目标配置
        training_targets = ADAPTIVE_TRAINING_CONFIG.copy()
        training_targets["max_episodes"] = max_episodes  # 只覆盖最大轮数
        
        print("🚀 启动V31自适应PPO训练系统")
        print("=" * 80)
        print(f"🎯 训练目标: 综合评分达到 {training_targets['target_score']:.2f}")
        print(f"⚖️ 稳定性要求: 连续{training_targets['target_consistency']}次达到目标")
        print(f"📊 轮数上限: {training_targets['max_episodes']}轮 (完整挑战阶段完成后开始早停评估)")
        print(f"🔄 早停耐心: {training_targets['early_stop_patience']}轮无改进")
        print("=" * 80)
        print("🔧 核心配置:")
        print("  工作站:")
        for station, config in WORKSTATIONS.items():
            print(f"    - {station}: 数量={config['count']}, 容量={config['capacity']}")
        
        grad_config = CURRICULUM_CONFIG.get("graduation_config", {})
        print("  毕业考试:")
        print(f"    - 考试轮数: {grad_config.get('exam_episodes', 'N/A')}")
        print(f"    - 稳定要求: {grad_config.get('stability_requirement', 'N/A')}次通过")
        print(f"    - 最大重试: {grad_config.get('max_retries', 'N/A')}次")
        print(f"    - 补课轮数: {grad_config.get('retry_extension', 'N/A')}轮")
        
        print(f"  设备故障: {'启用' if EQUIPMENT_FAILURE.get('enabled', False) else '禁用'}")
        print(f"  紧急插单: {'启用' if EMERGENCY_ORDERS.get('enabled', False) else '禁用'}")
        print("-" * 40)
        
        trainer = SimplePPOTrainer(
            initial_lr=LEARNING_RATE_CONFIG["initial_lr"],  # 🔧 V32：使用配置文件的学习率
            total_train_episodes=max_episodes,  # 传递最大轮数
            steps_per_episode=steps_per_episode,
            training_targets=training_targets   # 🔧 V32核心：传递自适应训练目标
        )
        
        # 🔧 V31 启动自适应训练：系统将根据性能自动决定何时停止
        results = trainer.train(
            max_episodes=max_episodes,           # 最大轮数（上限）
            steps_per_episode=steps_per_episode,
            eval_frequency=20,                  # 评估频率
            adaptive_mode=True                  # 🔧 V31核心：启用自适应模式
        )
        
        if results:
            print("\n🎉 自适应训练成功完成！")
            print(f"📊 实际训练轮数: {len(trainer.iteration_times)}")
            final_completion_rate = (results['best_kpi'].get('mean_completed_parts', 0) / get_total_parts_count()) * 100 if get_total_parts_count() > 0 else 0
            print(f"🎯 最终目标达成: {trainer.adaptive_state['target_achieved_count']}次连续达标 (基于最终阶段分数)")
            
            best_episode_final = trainer.best_episode_dual_objective if trainer.best_episode_dual_objective != -1 else trainer.final_stage_best_episode
            print(f"📈 历史最佳性能 (双重标准，第 {best_episode_final} 回合): {final_completion_rate:.1f}% ({results['best_kpi'].get('mean_completed_parts', 0):.1f}个零件)")
        else:
            print("\n❌ 训练失败")
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 🔧 V10 关键修复: 设置多进程启动方法为'spawn'，避免TensorFlow的fork不安全问题
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()