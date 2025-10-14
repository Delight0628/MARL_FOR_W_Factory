"""
纯净的多智能体PPO训练脚本
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
import socket
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 🔧 V12 新增：TensorBoard支持（基于 TF2 正确检测）
TENSORBOARD_AVAILABLE = hasattr(tf.summary, "create_file_writer")

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env, WFactoryEnv
from environments.w_factory_config import *
from environments.w_factory_config import validate_config, get_total_parts_count, generate_random_orders, calculate_episode_score, ADAPTIVE_ENTROPY_CONFIG, EVALUATION_CONFIG

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
    
    def get_batch(self, gamma=0.99, lam=0.95, next_value_if_truncated=None, advantage_clip_val: Optional[float] = None):
        """🔧 MAPPO改进：正确处理轨迹截断，并支持优势裁剪"""
        states = np.array(self.states, dtype=np.float32)
        global_states = np.array(self.global_states, dtype=np.float32)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        action_probs = np.array(self.action_probs, dtype=np.float32)
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
        
        # 🔧 MAPPO修复：更稳健的优势标准化，处理边界情况
        if len(advantages) > 1:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            # 只有当标准差足够大时才进行完整标准化
            if adv_std > 1e-6:  # 提高阈值，避免数值不稳定
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                # 标准差太小时只进行去均值，不进行缩放
                advantages = advantages - adv_mean
        # 单样本情况：不进行任何标准化，保持原值
        
        # 🔧 缺陷修复：使用配置化的优势裁剪
        if advantage_clip_val is not None:
            advantages = np.clip(advantages, -advantage_clip_val, advantage_clip_val)
        
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
    def __init__(self, state_dim: int, action_dim: int, lr: Any, global_state_dim: int, network_config: Optional[Dict[str, Any]] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim # 🔧 新增
        self.lr = lr
        self.network_config_override = network_config
        
        # 构建网络
        self.actor, self.critic = self._build_networks()
        
        # 优化器 - 🔧 修复：处理lr为None的情况（worker不需要优化器）
        if lr is not None:
            # 专家修复：为Critic设置一个较低的学习率乘数
            critic_lr = lr
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                # 如果lr是调度器，我们不能直接乘，但可以创建一个新的调度器或在优化器层面处理
                # Adam优化器支持在创建时传入学习率调度器
                pass # 优化器将直接使用调度器
            
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            
            # 🔧 关键修复：为Critic创建相同类型的学习率调度器，但使用较低的乘数
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                # 如果lr是调度器，为Critic创建一个带乘数的调度器
                critic_lr_multiplier = LEARNING_RATE_CONFIG.get("critic_lr_multiplier", 0.5)
                critic_lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=LEARNING_RATE_CONFIG["initial_lr"] * critic_lr_multiplier,
                    decay_steps=lr.decay_steps,
                    end_learning_rate=LEARNING_RATE_CONFIG["end_lr"] * critic_lr_multiplier,
                    power=LEARNING_RATE_CONFIG["decay_power"]
                )
                self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr_schedule)
            else:
                # 如果lr是固定值，则使用固定值乘以乘数
                critic_lr_value = lr * LEARNING_RATE_CONFIG.get("critic_lr_multiplier", 0.5)
                self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr_value)
        else:
            self.actor_optimizer = None
            self.critic_optimizer = None
        
    def _build_networks(self):
        """🔧 MAPPO优化：使用配置文件参数构建网络"""
        # 导入配置
        if self.network_config_override:
            config = self.network_config_override
        else:
            from environments.w_factory_config import PPO_NETWORK_CONFIG
            config = PPO_NETWORK_CONFIG

        hidden_sizes = config["hidden_sizes"]
        dropout_rate = config.get("dropout_rate", 0.1) # Use .get for safety
        
        # Actor网络 (去中心化) - 使用局部观测
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        
        # 🔧 关键修复：添加层归一化，稳定训练
        actor_x = tf.keras.layers.LayerNormalization()(state_input)
        
        actor_x = tf.keras.layers.Dense(
            hidden_sizes[0], 
            activation='relu',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(actor_x)
        actor_x = tf.keras.layers.Dropout(dropout_rate)(actor_x)
        actor_x = tf.keras.layers.Dense(
            hidden_sizes[1], 
            activation='relu',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(actor_x)
        actor_x = tf.keras.layers.Dropout(dropout_rate)(actor_x)
        actor_x = tf.keras.layers.Dense(
            hidden_sizes[2], 
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

        # Critic网络 (中心化) - 使用全局状态
        global_state_input = tf.keras.layers.Input(shape=(self.global_state_dim,))
        
        # 🔧 关键修复：Critic也加层归一化
        critic_x = tf.keras.layers.LayerNormalization()(global_state_input)
        
        critic_x = tf.keras.layers.Dense(
            hidden_sizes[0],
            activation='relu',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(critic_x)
        critic_x = tf.keras.layers.Dropout(dropout_rate)(critic_x)
        critic_x = tf.keras.layers.Dense(
            hidden_sizes[1],
            activation='relu',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(critic_x)
        critic_x = tf.keras.layers.Dropout(dropout_rate)(critic_x)
        critic_x = tf.keras.layers.Dense(
            hidden_sizes[2],
            activation='relu',
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(critic_x)
        # Value输出层
        value_output = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )(critic_x)
        critic = tf.keras.Model(inputs=global_state_input, outputs=value_output)
        
        return actor, critic
    
    def get_action_and_value(self, state: np.ndarray, global_state: np.ndarray) -> Tuple[int, np.float32, np.float32]:
        """获取动作、价值和动作概率"""
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
        probs = self.actor(state_tensor)
        # 🔧 修复：数值稳定性
        probs = tf.clip_by_value(probs, 1e-8, 1.0)
        action = tf.random.categorical(tf.math.log(probs + 1e-8), 1)[0, 0].numpy()
        action_prob = probs[0, action].numpy()

        # 🔧 Critic使用全局状态
        value = self.critic(tf.expand_dims(tf.convert_to_tensor(global_state), 0))[0, 0].numpy()
        
        return action, np.float32(value), np.float32(action_prob)
    
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
            probs = self.actor(states, training=True)
            # 🔧 修复：添加数值稳定性保护
            probs = tf.clip_by_value(probs, 1e-8, 1.0)
            # 计算选择动作的概率 new_probs
            batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            indices = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            new_probs = tf.gather_nd(probs, indices)
            # 🔧 修复：防止除零和数值爆炸
            ratio = new_probs / (old_probs + 1e-8)
            
            # 🔧 修复：正确计算KL散度（基于被选动作的近似）
            old_log_probs = tf.math.log(old_probs + 1e-8)
            new_log_probs = tf.math.log(new_probs + 1e-8)
            approx_kl = tf.reduce_mean(old_log_probs - new_log_probs)
            
            # 计算裁剪比例 (用于监控)
            clipped_mask = tf.greater(tf.abs(ratio - 1.0), clip_ratio)
            clip_fraction = tf.reduce_mean(tf.cast(clipped_mask, tf.float32))

            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            
            # 计算分类熵：-sum p*log p
            entropy_per_sample = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
            entropy = tf.reduce_mean(entropy_per_sample)
            actor_loss -= current_entropy_coeff * entropy
            
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # 🔧 新增：梯度裁剪以提高训练稳定性
        grad_clip_norm = PPO_NETWORK_CONFIG.get("grad_clip_norm", 1.0)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, grad_clip_norm)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Critic更新 (使用全局状态)
        with tf.GradientTape() as tape:
            values = self.critic(global_states, training=True)
            returns_tf = tf.expand_dims(tf.convert_to_tensor(returns, dtype=tf.float32), 1)
            critic_loss = tf.reduce_mean(tf.square(returns_tf - values))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # 🔧 新增：梯度裁剪（使用配置值）
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, grad_clip_norm)
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
                          global_state_dim: int, network_config: Dict[str, Any], curriculum_config: Dict[str, Any] = None) -> Tuple[Dict[str, ExperienceBuffer], float, Optional[np.ndarray], bool, bool]:
    """并行仿真工作进程 - 🔧 MAPPO改造：收集全局状态"""
    
    # 🔧 终极修复：将tf导入移至顶部，解决UnboundLocalError
    import tensorflow as tf
    import numpy as np
    import random
    
    # 1. 初始化
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用GPU
    
    tf.random.set_seed(seed)
    env = make_parallel_env(curriculum_config)
    # 🔧 修复：使用动态学习率而非固定值
    # 注意：worker不需要学习率，只做推理
    network = PPONetwork(state_dim, action_dim, None, global_state_dim, network_config=network_config) # Worker不需要优化器
    network.actor.set_weights(network_weights['actor'])
    network.critic.set_weights(network_weights['critic']) # 🔧 Critic权重也需要同步
    
    buffers = {agent: ExperienceBuffer() for agent in env.agents}
    
    observations, infos = env.reset(seed=seed)
    global_state = infos[env.agents[0]]['global_state']
    # 🔧 智能体条件化：构建one-hot映射
    agent_list = list(env.agents)
    agent_index = {agent_id: idx for idx, agent_id in enumerate(agent_list)}
    # 专家修复：修正base_global_dim的计算方式，避免硬编码或依赖不一致的环境实例
    base_global_dim = global_state_dim - len(agent_list)
    
    total_reward_collected = 0.0
    collected_steps = 0
    step_count = 0
    
    while collected_steps < num_steps:
        actions = {}
        values = {}
        action_probs = {}
        augmented_global_states = {} # 修复缺陷：为每个智能体分别存储增强全局状态
        
        # 🔧 修复：基础全局状态（不含one-hot）
        if global_state is not None:
            base_global_state = global_state.copy()
        else:
            base_global_state = np.zeros(base_global_dim, dtype=np.float32)

        # 🔧 修复：确保智能体动作的同步性
        for agent in env.agents:  # 使用env.agents确保顺序一致
            if agent in observations:
                obs = observations[agent]
                # 🔧 拼接agent one-hot到全局状态
                one_hot = np.zeros(len(agent_list), dtype=np.float32)
                one_hot[agent_index[agent]] = 1.0
                # 注意：global_state_dim 已经包含one-hot长度
                augmented_global_state = np.concatenate([base_global_state, one_hot]).astype(np.float32)
                augmented_global_states[agent] = augmented_global_state # 修复缺陷：存储
                action, value, action_prob = network.get_action_and_value(obs, augmented_global_state)
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
                # 🔧 重要：存储时使用agent条件化的全局状态
                agent_specific_global_state = augmented_global_states.get(agent)
                if agent_specific_global_state is not None:
                    buffers[agent].store(
                        observations[agent], 
                        agent_specific_global_state.copy(),  # 修复缺陷：使用正确的增强全局状态
                        actions[agent], 
                        reward,
                        values[agent], 
                        action_probs[agent], 
                        terminated,
                        truncated
                    )

        observations = next_observations

        # 🔧 修复：与评估一致的终止条件
        if any(terminations.values()) or any(truncations.values()):
            
            # 🔧 MAPPO关键修复：正确处理截断时的bootstrap价值
            # 注意：这里暂时不处理，让buffer自己在get_batch时处理
            pass
            
            # 🔧 关键修复：episode结束时应该break，而不是reset继续收集
            # 一个worker调用应该只收集单个trajectory的数据
            break

    # 🔧 核心修复：返回最后一个全局状态和截断标志，用于价值引导
    # 只要trajectory未真正终止（即存在截断或仅因采样步数达到上限而退出），就提供bootstrap价值
    was_truncated = any(truncations.values()) or not any(terminations.values())
    # 返回基础全局状态（不含one-hot），主进程将为各agent添加one-hot后计算bootstrap
    next_global_state_for_bootstrap = global_state if was_truncated else None
    
    # 统计本worker是否完成了全部零件（用于日志与终局奖励核验）
    try:
        total_required_worker = sum(o.quantity for o in env.sim.orders)
        completed_all_worker = (len(env.sim.completed_parts) >= total_required_worker)
    except Exception:
        completed_all_worker = False
    
    env.close()
    return buffers, total_reward_collected, next_global_state_for_bootstrap, was_truncated, completed_all_worker

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
        self.agent_ids = temp_env.possible_agents
        self.num_agents = len(self.agent_ids)
        # 🔧 Critic智能体条件化：将智能体one-hot并入全局状态输入维度
        self.global_state_dim = temp_env.global_state_space.shape[0] + self.num_agents
        temp_env.close()
        
        print("🔧 环境空间检测:")
        print(f"   观测维度: {self.state_dim}")
        print(f"   动作维度: {self.action_dim}")
        print(f"   智能体数量: {len(self.agent_ids)}")
        print(f"   全局状态维度(含agent one-hot): {self.global_state_dim}")
        
        # 🔧 V26 终极修复：移除动态参数调整
        optimized_episodes = total_train_episodes
        optimized_steps = steps_per_episode
        # 统一评估/采集最大步数
        self.max_steps_for_eval = int(optimized_steps)
        
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
        self.final_stage_best_score = float('-inf')
        self.final_stage_best_episode = -1 # 🔧 新增：记录最佳KPI的回合数
        
        # 🔧 核心改造：新增"双达标"最佳KPI跟踪器
        self.best_kpi_dual_objective = {
            'mean_completed_parts': -1.0,
            'mean_makespan': float('inf'),
            'mean_utilization': 0.0,
            'mean_tardiness': float('inf')
        }
        self.best_score_dual_objective = float('-inf')
        self.best_episode_dual_objective = -1

        # 🔧 核心重构：训练流程由配置文件驱动
        self.training_flow_config = TRAINING_FLOW_CONFIG
        self.training_targets = self.training_flow_config["general_params"] # 通用参数
        
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
        
        # 🔧 V34 初始化动态训练参数
        self.current_entropy_coeff = PPO_NETWORK_CONFIG["entropy_coeff"] # 初始化动态熵系数
        self.current_learning_rate = LEARNING_RATE_CONFIG["initial_lr"] # 🔧 V34 修复：使用正确的学习率配置
        
        # 🔧 新增：熵系数退火计划（改进版）
        self.entropy_decay_rate = 0.9995  # 🔧 更慢的衰减率，保持更长时间的探索
        self.min_entropy_coeff = 0.05     # 🔧 更高的最小熵系数，避免过早收敛
        
        
        # 🔧 V40 新增：回合事件日志记录器
        self.episode_events = []
        
        # 创建保存目录 (V31新增：以训练开始时间创建专用文件夹)
        self.base_models_dir = "mappo/ppo_models"
        self.models_dir = f"{self.base_models_dir}/{self.start_time_str}"
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"📁 模型保存目录: {self.models_dir}")
        
        # 🔧 V12 新增：TensorBoard支持
        self.tensorboard_dir = f"mappo/tensorboard_logs/{self.timestamp}"
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
    
    def should_continue_training(self, episode: int, current_score: float, completion_rate: float) -> tuple:
        """🔧 修复：基于TRAINING_FLOW_CONFIG的阶段标准评估是否继续训练"""
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
    
    def check_foundation_training_completion(self, kpi_results: Dict[str, float], current_score: float) -> bool:
        """检查基础训练是否达到毕业标准，由配置文件驱动"""
        criteria = self.training_flow_config["foundation_phase"]["graduation_criteria"]
        
        total_parts_target = get_total_parts_count()
        completion_rate_kpi = (kpi_results.get('mean_completed_parts', 0) / total_parts_target) * 100 if total_parts_target > 0 else 0
        
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
        🔧 核心修复：并行收集经验，并在主进程中统一处理价值引导和GAE计算
        - 返回一个处理完成、可以直接用于更新的训练批次
        """
        from environments.w_factory_config import PPO_NETWORK_CONFIG

        network_weights = {
            'actor': self.shared_network.actor.get_weights(),
            'critic': self.shared_network.critic.get_weights()
        }
        # 🔧 关键修复：使用入参作为每个 worker 的最大步数
        steps_per_worker = int(num_steps)
        
        total_reward = 0
        
        # 初始化用于聚合所有worker数据的列表
        all_states, all_global_states, all_actions, all_old_probs, all_advantages, all_returns = [], [], [], [], [], []

        # 🔧 使用 spawn 上下文，避免 TF 在 fork 下的不安全
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = []
            for i in range(self.num_workers):
                seed = random.randint(0, 1_000_000)
                # 为 worker 传入统一的 MAX_SIM_STEPS
                worker_config = (curriculum_config.copy() if curriculum_config else {})
                worker_config['MAX_SIM_STEPS'] = steps_per_worker
                future = executor.submit(
                    run_simulation_worker,
                    network_weights,
                    self.state_dim,
                    self.action_dim,
                    steps_per_worker,
                    seed,
                    self.global_state_dim,
                    PPO_NETWORK_CONFIG.copy(),
                    worker_config
                )
                futures.append(future)

            completed_workers = 0
            finished_workers = 0
            for future in as_completed(futures):
                try:
                    # 接收worker返回的原始经验、下一个全局状态和截断标志
                    worker_buffers, worker_reward, next_global_state, was_truncated, worker_completed_all = future.result()
                    total_reward += worker_reward
                    completed_workers += 1 if worker_completed_all else 0
                    finished_workers += 1
                    
                    # 在主进程中为该worker的每个智能体计算GAE和回报
                    for agent_id in self.agent_ids:
                        if agent_id in worker_buffers:
                            buffer = worker_buffers[agent_id]
                            if not buffer.states:  # 跳过空缓冲区
                                continue
                            
                            # 🔧 使用正确的引导价值（逐智能体 one-hot 条件化）
                            if was_truncated and next_global_state is not None:
                                # 专家修复：使用在主训练器中定义的agent_ids和num_agents，确保索引一致性
                                one_hot = np.zeros(self.num_agents, dtype=np.float32)
                                one_hot[self.agent_ids.index(agent_id)] = 1.0
                                augmented_next_state = np.concatenate([next_global_state, one_hot]).astype(np.float32)
                                bootstrap_value = self.shared_network.get_value(augmented_next_state)
                            else:
                                bootstrap_value = None

                            # 🔧 缺陷修复：将配置中的优势裁剪值传递给get_batch
                            advantage_clip_val = PPO_NETWORK_CONFIG.get("advantage_clip_val")
                            states, global_states, actions, old_probs, advantages, returns = buffer.get_batch(
                                next_value_if_truncated=bootstrap_value,
                                advantage_clip_val=advantage_clip_val
                            )
                            
                            # 将处理好的数据聚合到总批次中
                            all_states.extend(states)
                            all_global_states.extend(global_states)
                            all_actions.extend(actions)
                            all_old_probs.extend(old_probs)
                            all_advantages.extend(advantages)
                            all_returns.extend(returns)
                            
                except Exception as e:
                    print(f"❌ 一个并行工作进程失败: {e}")
                    import traceback
                    traceback.print_exc()

        if not all_states:
            # 返回时将完成统计编码在None批次旁边（通过总奖励的info在外层打印）
            self._last_collect_finished_workers = finished_workers
            self._last_collect_completed_workers = completed_workers
            avg_reward = total_reward / finished_workers if finished_workers > 0 else 0.0
            return avg_reward, None

        # 将聚合后的数据列表转换为NumPy数组，形成最终的训练批次
        batch = {
            "states": np.array(all_states),
            "global_states": np.array(all_global_states),
            "actions": np.array(all_actions),
            "old_probs": np.array(all_old_probs),
            "advantages": np.array(all_advantages),
            "returns": np.array(all_returns),
        }
        # 记录本轮采集完成worker与达成worker数量，供外层日志打印
        self._last_collect_finished_workers = finished_workers
        self._last_collect_completed_workers = completed_workers
        avg_reward = total_reward / finished_workers if finished_workers > 0 else 0.0
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
        
        observations, _ = env.reset(seed=seed)
        episode_reward = 0
        step_count = 0
        
        while step_count < self.max_steps_for_eval:
            actions = {}
            
            # 使用确定性策略，但基于新的随机环境状态
            for agent in env.agents:
                if agent in observations:
                    state = tf.expand_dims(observations[agent], 0)
                    action_probs = self.shared_network.actor(state)
                    # 🔧 使用确定性评估，但保留少量探索
                    if random.random() < EVALUATION_CONFIG["exploration_rate"]:
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
            curriculum_config = curriculum_config.copy()
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
            while step_count < self.max_steps_for_eval:
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
            
            while step_count < self.max_steps_for_eval:
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
                    # 阶段1：基础能力训练阶段
                    # 如果课程学习未启用，或刚刚完成，则使用标准的基础订单进行训练
                    if not curriculum_enabled or curriculum_just_completed:
                        foundation_config = {
                            'orders_scale': 1.0,
                            'time_scale': 1.0,
                            'stage_name': '基础能力认证',
                            'custom_orders': BASE_ORDERS
                        }
                        current_curriculum_config = foundation_config
                    
                    # 在每个回合都添加当前回合数，供环境内部使用
                        if current_curriculum_config:
                            current_curriculum_config['current_episode'] = episode
                    
                    if episode % 20 == 0:
                        phase_name = "课程学习中" if curriculum_enabled and not curriculum_just_completed else "基础能力认证中"
                        foundation_criteria = self.training_flow_config["foundation_phase"]["graduation_criteria"]
                        print(f"📚 {phase_name}: 连续达标 {self.foundation_achievement_count}/{foundation_criteria['target_consistency']} 次")
                
                elif not self.generalization_phase_active:
                    # 基础训练刚完成，准备进入泛化阶段
                    self.generalization_phase_active = True
                    print("\n" + "="*80)
                    print(f"🚀 [回合 {episode+1}] 基础训练已完成，正式进入随机领域强化阶段!")
                    print("   每轮将使用全新的随机订单配置，并启用环境扰动。")
                    print("   这将全面锻炼模型的泛化能力和鲁棒性。")
                    print("="*80 + "\n")
                
                if self.generalization_phase_active:
                    # 阶段2：随机领域强化阶段
                    # 每轮生成全新的随机订单配置
                    random_orders = generate_random_orders()
                    generalization_config = {
                        'custom_orders': random_orders,
                        'randomize_env': True,  # 启用环境扰动
                        'stage_name': f'随机领域强化-R{episode}',
                        'current_episode': episode
                    }
                    
                    current_curriculum_config = generalization_config
                    
                    if episode % 20 == 0:
                        total_parts = sum(order["quantity"] for order in random_orders)
                        generalization_criteria = self.training_flow_config["generalization_phase"]["completion_criteria"]
                        print(f"🎲 随机领域强化: 本轮{len(random_orders)}个订单，共{total_parts}个零件")
                        print(f"   泛化阶段连续达标: {self.generalization_achievement_count}/{generalization_criteria['target_consistency']} 次")
                

                collect_start_time = time.time()
                episode_reward, batch = self.collect_and_process_experience(steps_per_episode, current_curriculum_config)
                collect_duration = time.time() - collect_start_time
                
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

                
                # 提前进行KPI评估，以便整合TensorBoard日志
                kpi_results = self.quick_kpi_evaluation(num_episodes=1, curriculum_config=current_curriculum_config)
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
                        if self.check_foundation_training_completion(last_kpi_results, last_current_score):
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
                            
                            self.train_writer.flush()
                
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
                
                # 🔧 重构版：简化的性能监控，移除复杂的重启机制
                # 基础性能跟踪（用于调试和监控）
                current_performance = kpi_results.get('mean_completed_parts', 0)
                if not hasattr(self, '_performance_history'):
                    self._performance_history = []
                
                self._performance_history.append(current_performance)
                # 只保留最近20轮的历史
                if len(self._performance_history) > 20:
                    self._performance_history.pop(0)
                
                
                # 🔧 V38修复：每30回合进行一次完整难度评估（静默模式，避免输出污染）
                if curriculum_enabled and episode > 0 and episode % 30 == 0:
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
                    real_tardiness = full_kpi.get('mean_tardiness', 0)  
                    real_reward = full_kpi.get('mean_reward', 0)
                    
                    print(f"🎯 完整难度评估结果（3轮平均）:")
                    print(f"   平均完成零件: {real_completion:.1f}/{get_total_parts_count()} ({real_completion_rate:.1f}%)")
                    print(f"   平均总完工时间: {real_makespan:.1f}分钟")
                    print(f"   平均设备利用率: {real_utilization*100:.1f}%")
                    print(f"   平均订单延期时间: {real_tardiness:.1f}分钟") 
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
                            model_path = self.save_model(f"{self.models_dir}/{timestamp}_{stage_name}_best")
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
                            model_path = self.save_model(f"{self.models_dir}/{timestamp}general_train_best")
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
                            model_path = self.save_model(f"{self.models_dir}/{timestamp}base_train_best")
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
                            model_path = self.save_model(f"{self.models_dir}/{timestamp}general_train_best")
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
                    dual_objective_best_path = self.save_model(f"{self.models_dir}/{timestamp}Twin_best")
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
                per_worker_avg_reward = (episode_reward / finished_workers) if finished_workers > 0 else episode_reward
                line1 = (
                    f"🔂 训练回合 {episode + 1:3d}/{max_episodes} | 平均奖励: {episode_reward:.1f}"
                    f" (均值/worker: {per_worker_avg_reward:.1f}, 完成全部: {completed_workers}/{finished_workers})"
                    f" | Actor损失: {losses['actor_loss']:.4f}| ⏱️本轮用時: {iteration_duration:.1f}s"
                    f" (CPU采集: {collect_duration:.1f}s, GPU更新: {update_duration:.1f}s)"
                )

                # 第二行：KPI数据和阶段信息 (核心修复：动态显示目标零件数)
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

    def _get_target_parts(self, curriculum_config: Optional[Dict]) -> int:
        """统一获取当前回合的目标零件数"""
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

def main():
    
    print(f"✨ 训练进程PID: {os.getpid()}")

    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    try:
        # 核心重构：从TRAINING_FLOW_CONFIG获取训练参数
        max_episodes = TRAINING_FLOW_CONFIG["general_params"]["max_episodes"]
        steps_per_episode = TRAINING_FLOW_CONFIG["general_params"]["steps_per_episode"]
        eval_frequency = TRAINING_FLOW_CONFIG["general_params"]["eval_frequency"]
        
        # 训练目标现在分散在TRAINING_FLOW_CONFIG中，不再需要独立的training_targets字典
        

        print("=" * 80)
        foundation_criteria = TRAINING_FLOW_CONFIG["foundation_phase"]["graduation_criteria"]
        generalization_criteria = TRAINING_FLOW_CONFIG["generalization_phase"]["completion_criteria"]
        
        print(f"🎯 基础训练目标: 综合评分 > {foundation_criteria['target_score']:.2f}, "
              f"完成率 > {foundation_criteria['min_completion_rate']:.0f}%, "
              f"延期 < {foundation_criteria['tardiness_threshold']:.0f}min, "
              f"连续{foundation_criteria['target_consistency']}次")
              
        print(f"🎯 泛化训练目标: 综合评分 > {generalization_criteria['target_score']:.2f}, "
              f"完成率 > {generalization_criteria['min_completion_rate']:.0f}%, "
              f"连续{generalization_criteria['target_consistency']}次")

        print(f"📊 轮数上限: {max_episodes}轮")
        print("=" * 80)
        print("🔧 核心配置:")
        print("  工作站:")
        for station, config in WORKSTATIONS.items():
            print(f"    - {station}: 数量={config['count']}, 容量={config['capacity']}")

        print("  奖励系统:")
        for key, value in REWARD_CONFIG.items():
            print(f"    - {key}: {value}")
        
        cl_config = TRAINING_FLOW_CONFIG["foundation_phase"]["curriculum_learning"]
        
        print("  启用/禁用模块:")
        print(f"    - 课程学习: {'启用' if cl_config.get('enabled', False) else '禁用'}")
        print(f"    - 设备故障: {'启用' if EQUIPMENT_FAILURE.get('enabled', False) else '禁用'}")
        print(f"    - 紧急插单: {'启用' if EMERGENCY_ORDERS.get('enabled', False) else '禁用'}")
        print("-" * 40)
        
        trainer = SimplePPOTrainer(
            initial_lr=LEARNING_RATE_CONFIG["initial_lr"],
            total_train_episodes=max_episodes,
            steps_per_episode=steps_per_episode,
            training_targets=None  # 不再需要，由内部读取配置文件
        )
        
        # 🔧 V31 启动自适应训练：系统将根据性能自动决定何时停止
        results = trainer.train(
            max_episodes=max_episodes,
            steps_per_episode=steps_per_episode,
            eval_frequency=eval_frequency,
            adaptive_mode=True
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