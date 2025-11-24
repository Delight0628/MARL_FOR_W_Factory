"""
MAPPO神经网络模块
==================
实现Actor和Critic网络架构，支持集中式训练分布式执行(CTDE)范式
"""

import os
import numpy as np
import tensorflow as tf
import gymnasium as gym
from typing import Dict, Tuple, Any, Optional

# 导入配置
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from environments.w_factory_config import PPO_NETWORK_CONFIG, LEARNING_RATE_CONFIG, WORKSTATIONS


class PPONetwork:
    """
    MAPPO神经网络架构
    -------------------
    实现集中式训练分布式执行 (CTDE) 范式：
    - Actor: 基于局部观测生成动作分布（分布式执行）
    - Critic: 基于全局状态评估价值（集中式训练）
    
    特性：
    - 支持MultiDiscrete动作空间（多头输出）
    - 支持学习率调度
    - 梯度裁剪和优势裁剪
    """
    
    def __init__(self, state_dim: int, action_space: gym.spaces.Space, lr: Any, global_state_dim: int, network_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            state_dim: 局部观测维度
            action_space: 动作空间（支持Discrete和MultiDiscrete）
            lr: 学习率或学习率调度器
            global_state_dim: 全局状态维度（包含智能体one-hot）
            network_config: 网络架构配置
        """
        self.state_dim = int(state_dim)
        self.action_space = action_space
        self.global_state_dim = int(global_state_dim)
        self.config = network_config or PPO_NETWORK_CONFIG
        
        # ====== 代理与设备数映射（用于无效头掩码）======
        # 与环境中 agents 的构造顺序保持一致：list(WORKSTATIONS.keys())
        self._station_names_order = list(WORKSTATIONS.keys())
        self._agent_machine_counts = tf.constant(
            [int(WORKSTATIONS[name]['count']) for name in self._station_names_order], dtype=tf.int32
        )
        self._num_agents = int(len(self._station_names_order))

        # 解析动作空间类型
        self.is_multidiscrete = isinstance(self.action_space, gym.spaces.MultiDiscrete)
        if self.is_multidiscrete:
            self.action_dims = self.action_space.nvec
        else:
            self.action_dims = [self.action_space.n]
            
        # 确定性初始化标志（评估模式使用，避免随机数问题）
        self._deterministic_init = (lr is None) or bool(int(os.environ.get("DETERMINISTIC_INIT", "1")))

        self.actor, self.critic = self._build_networks()
        
        # 初始化优化器（支持学习率调度）
        if lr is not None:
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            
            # Critic使用较低学习率以稳定价值学习
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                critic_lr_multiplier = LEARNING_RATE_CONFIG.get("critic_lr_multiplier", 0.5)
                critic_lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=LEARNING_RATE_CONFIG["initial_lr"] * critic_lr_multiplier,
                    decay_steps=lr.decay_steps,
                    end_learning_rate=LEARNING_RATE_CONFIG["end_lr"] * critic_lr_multiplier,
                    power=LEARNING_RATE_CONFIG["decay_power"]
                )
                self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr_schedule)
            else:
                critic_lr_value = lr * LEARNING_RATE_CONFIG.get("critic_lr_multiplier", 0.5)
                self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr_value)
        else:
            self.actor_optimizer = None
            self.critic_optimizer = None
        
    def _build_networks(self):
        """
        构建Actor和Critic神经网络
        
        Actor架构：
        - 输入：局部观测
        - 主干：多层全连接网络 + ReLU + Dropout
        - 输出：动作概率分布（MultiDiscrete时为多头输出）
        
        Critic架构：
        - 输入：全局状态（包含智能体one-hot）
        - 主干：多层全连接网络 + ReLU + Dropout
        - 输出：状态价值估计
        """
        config = self.config or PPO_NETWORK_CONFIG
        hidden_sizes = [int(max(1, hs)) for hs in config["hidden_sizes"]]
        dropout_rate = config.get("dropout_rate", 0.1)
        
        # ==================== Actor网络 ====================
        state_input = tf.keras.layers.Input(
            shape=(int(self.state_dim),), 
            dtype=tf.float32, 
            name="actor_input"
        )
        
        # Actor主干网络
        actor_x = state_input
        for i, hidden_size in enumerate(hidden_sizes):
            kernel_init = (tf.keras.initializers.Zeros() if self._deterministic_init 
                          else tf.keras.initializers.GlorotUniform(seed=1234 + i))
            actor_x = tf.keras.layers.Dense(
                hidden_size,
                activation="relu",
                kernel_initializer=kernel_init,
                bias_initializer=tf.keras.initializers.Zeros(),
                name=f"actor_dense_{i}"
            )(actor_x)
            if dropout_rate > 0:
                actor_x = tf.keras.layers.Dropout(dropout_rate, name=f"actor_dropout_{i}")(actor_x)

        # Actor输出层（支持MultiDiscrete多头）
        if self.is_multidiscrete:
            num_heads = len(self.action_dims)
            action_dim_per_head = self.action_dims[0]
            actor_outputs = []
            for i in range(num_heads):
                kernel_init = (tf.keras.initializers.Zeros() if self._deterministic_init 
                              else tf.keras.initializers.GlorotUniform(seed=2000 + i))
                head_output = tf.keras.layers.Dense(
                    action_dim_per_head,
                    activation="softmax",
                    kernel_initializer=kernel_init,
                    bias_initializer=tf.keras.initializers.Zeros(),
                    name=f"actor_output_{i}"
                )(actor_x)
                actor_outputs.append(head_output)
            self.actor = tf.keras.Model(inputs=state_input, outputs=actor_outputs)
        else:
            kernel_init = (tf.keras.initializers.Zeros() if self._deterministic_init 
                          else tf.keras.initializers.GlorotUniform(seed=3000))
            actor_output = tf.keras.layers.Dense(
                self.action_dims[0],
                activation="softmax",
                kernel_initializer=kernel_init,
                bias_initializer=tf.keras.initializers.Zeros(),
                name="actor_output"
            )(actor_x)
            self.actor = tf.keras.Model(inputs=state_input, outputs=actor_output)

        # ==================== Critic网络（集中式）====================
        global_state_input = tf.keras.layers.Input(
            shape=(int(self.global_state_dim),), 
            dtype=tf.float32, 
            name="critic_input"
        )
        
        # Critic主干网络
        critic_x = global_state_input
        for i, hidden_size in enumerate(hidden_sizes):
            kernel_init = (tf.keras.initializers.Zeros() if self._deterministic_init 
                          else tf.keras.initializers.GlorotUniform(seed=4000 + i))
            critic_x = tf.keras.layers.Dense(
                hidden_size,
                activation="relu",
                kernel_initializer=kernel_init,
                bias_initializer=tf.keras.initializers.Zeros(),
                name=f"critic_dense_{i}"
            )(critic_x)
            if dropout_rate > 0:
                critic_x = tf.keras.layers.Dropout(dropout_rate, name=f"critic_dropout_{i}")(critic_x)

        # Critic价值输出层
        value_kernel_init = (tf.keras.initializers.Zeros() if self._deterministic_init 
                            else tf.keras.initializers.Orthogonal(gain=1.0, seed=5000))
        value_output = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=value_kernel_init,
            bias_initializer=tf.keras.initializers.Constant(0.0),
            name="critic_value"
        )(critic_x)
        critic = tf.keras.Model(inputs=global_state_input, outputs=value_output)
        
        return self.actor, critic
    
    def get_action_and_value(self, state: np.ndarray, global_state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> Tuple[int, np.float32, np.float32]:
        """
        采样动作并计算状态价值
        
        实现要点：
        - MultiDiscrete使用无放回采样，避免多智能体选择重复候选
        - 推理模式（training=False），禁用Dropout等正则化
        
        Args:
            state: 局部观测状态
            global_state: 全局状态
            
        Returns:
            tuple: (action, value, log_prob)
                - action: 采样的动作（MultiDiscrete时为向量）
                - value: 状态价值估计
                - log_prob: 动作的对数概率
        """
        action_probs = self.actor(state, training=False)
        value = self.critic(global_state, training=False)

        # MultiDiscrete：多头无放回采样（带无效头掩码）
        if self.is_multidiscrete:
            action_prob_list = action_probs if isinstance(action_probs, list) else [action_probs]
            
            # 标准化各头张量形状为 (B, action_dim)
            normalized_heads = []
            for head_probs in action_prob_list:
                hp = tf.convert_to_tensor(head_probs, dtype=tf.float32)
                if tf.rank(hp) == 1:
                    hp = tf.expand_dims(hp, 0)
                hp = tf.reshape(hp, (-1, int(self.action_dims[0])))
                normalized_heads.append(hp)
            action_prob_list = normalized_heads
            
            num_heads = len(self.action_dims)
            action_dim = int(self.action_dims[0])
            chosen_actions = []
            log_prob_list = []
            mask = tf.zeros_like(action_prob_list[0], dtype=tf.bool)
            
            # 从全局状态中解析 agent 索引与有效头数量 k（该站点设备数）
            gs = tf.convert_to_tensor(global_state, dtype=tf.float32)
            if len(gs.shape) == 1:
                gs = tf.expand_dims(gs, 0)
            agent_one_hot = gs[:, -self._num_agents:]
            agent_idx = tf.argmax(agent_one_hot, axis=1, output_type=tf.int32)  # (B,)
            k_valid_heads = tf.gather(self._agent_machine_counts, agent_idx)      # (B,)
            
            # 逐头采样，屏蔽已选动作
            for i in range(num_heads):
                probs_i = tf.convert_to_tensor(action_prob_list[i], dtype=tf.float32)
                probs_i = tf.reshape(probs_i, (-1, action_dim))
                
                # 🔧 新增：应用动作掩码（如果提供）
                if action_mask is not None:
                    action_mask_tf = tf.convert_to_tensor(action_mask, dtype=tf.float32)
                    if len(action_mask_tf.shape) == 1:
                        action_mask_tf = tf.expand_dims(action_mask_tf, 0)  # (1, action_dim)
                    # 将掩码应用到概率分布
                    probs_i = probs_i * action_mask_tf
                
                # 当前头在每个样本中是否有效：i < k_valid_heads
                valid = tf.less(tf.cast(i, tf.int32), k_valid_heads)  # (B,)
                valid_f = tf.cast(valid, tf.float32)                  # (B,)
                valid_f_ex = tf.expand_dims(valid_f, axis=1)          # (B,1)

                # 屏蔽已选动作并重归一化（仅对有效样本生效）
                masked_probs_full = tf.where(mask, tf.zeros_like(probs_i), probs_i)
                norm_full = tf.reduce_sum(masked_probs_full, axis=1, keepdims=True) + 1e-8
                masked_probs_full = masked_probs_full / norm_full
                logits_full = tf.math.log(masked_probs_full + 1e-8)

                # 采样（有效样本用真实分布；无效样本固定为0=IDLE）
                sampled_valid = tf.squeeze(tf.random.categorical(logits_full, 1), axis=1)  # (B,)
                sampled = tf.where(valid, sampled_valid, tf.zeros_like(sampled_valid))
                chosen_actions.append(sampled)

                # 对数概率（无效样本记为0贡献）
                action_one_hot = tf.one_hot(sampled, action_dim)
                log_prob_for_head_full = tf.reduce_sum(logits_full * action_one_hot, axis=1)  # (B,)
                log_prob_for_head = log_prob_for_head_full * valid_f
                log_prob_list.append(log_prob_for_head)

                # 更新掩码（仅对有效样本更新），且允许多个头选择 IDLE(0)
                is_idle = tf.equal(sampled, tf.zeros_like(sampled))  # (B,)
                not_idle_ex = tf.expand_dims(tf.cast(tf.logical_not(is_idle), tf.bool), axis=1)  # (B,1)
                mask_update = tf.logical_and(tf.cast(action_one_hot > 0, tf.bool), tf.cast(valid_f_ex > 0, tf.bool))
                mask_update = tf.logical_and(mask_update, not_idle_ex)
                mask = tf.logical_or(mask, mask_update)

            action = tf.stack(chosen_actions, axis=1)
            joint_log_prob = tf.add_n(log_prob_list)
            
            # 转换为numpy并处理批次维度
            action_np = action.numpy()
            if action_np.ndim == 2 and action_np.shape[0] >= 1:
                action_out = action_np[0]
            elif action_np.ndim == 1:
                action_out = action_np
            else:
                action_out = np.zeros((num_heads,), dtype=self.action_space.dtype)

            joint_log_prob_np = joint_log_prob.numpy()
            if np.ndim(joint_log_prob_np) == 0:
                jlp_out = float(joint_log_prob_np)
            elif len(joint_log_prob_np) >= 1:
                jlp_out = float(joint_log_prob_np[0])
            else:
                jlp_out = 0.0

            return action_out, float(tf.squeeze(value).numpy()), jlp_out
        
        # Discrete：单头采样
        else:
            # 🔧 新增：应用动作掩码（如果提供）
            probs_to_use = action_probs
            if action_mask is not None:
                action_mask_tf = tf.convert_to_tensor(action_mask, dtype=tf.float32)
                if len(action_mask_tf.shape) == 1:
                    action_mask_tf = tf.expand_dims(action_mask_tf, 0)  # (1, action_dim)
                probs_to_use = action_probs * action_mask_tf
                # 重归一化
                norm = tf.reduce_sum(probs_to_use, axis=1, keepdims=True) + 1e-8
                probs_to_use = probs_to_use / norm
            
            sampled = tf.random.categorical(tf.math.log(probs_to_use + 1e-8), 1)
            action_one_hot = tf.one_hot(tf.squeeze(sampled, axis=-1), self.action_dims[0])
            action_prob = tf.math.log(tf.reduce_sum(probs_to_use * action_one_hot, axis=1) + 1e-8)
            action = tf.squeeze(sampled, axis=-1)
            action_np = action.numpy()
            return action_np, float(tf.squeeze(value).numpy()), float(tf.squeeze(action_prob).numpy())
    
    def get_value(self, global_state: np.ndarray) -> float:
        """
        获取全局状态的价值估计
        
        Args:
            global_state: 全局状态（支持批次或单样本）
            
        Returns:
            状态价值估计（标量）
        """
        gs = tf.convert_to_tensor(global_state)
        if len(gs.shape) == 1:
            gs = tf.expand_dims(gs, 0)
        return float(self.critic(gs)[0, 0])
    
    def update(self, states: np.ndarray, global_states: np.ndarray, actions: np.ndarray, 
               old_probs: np.ndarray, advantages: np.ndarray, 
               returns: np.ndarray, clip_ratio: float = None, entropy_coeff: float = None) -> Dict[str, float]:
        """
        执行一次PPO策略更新
        
        实现标准PPO-Clip算法，包含：
        1. Actor更新：Clip目标 + 熵奖励
        2. Critic更新：均方误差损失
        3. 梯度裁剪：提升训练稳定性
        
        Args:
            states: 局部观测批次
            global_states: 全局状态批次
            actions: 执行的动作
            old_probs: 旧策略的对数概率
            advantages: GAE优势函数
            returns: 目标回报
            clip_ratio: PPO裁剪比例 (默认0.2)
            entropy_coeff: 熵正则化系数 (默认0.01)
            
        Returns:
            dict: 损失统计 {actor_loss, critic_loss, entropy, approx_kl, clip_fraction}
        """
        clip_ratio = clip_ratio or self.config.get("clip_ratio", 0.2)
        entropy_coeff = entropy_coeff or self.config.get("entropy_coeff", 0.01)

        # 统一dtype，避免TF运算的float32/float64混用错误
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        global_states = tf.convert_to_tensor(global_states, dtype=tf.float32)
        # 动作用于one_hot索引，需整型
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # ========== Actor损失计算 ==========
            # 前向传播：与采样一致，禁用Dropout等正则化
            action_probs = self.actor(states, training=False)
            
            # 处理MultiDiscrete多头输出
            if self.is_multidiscrete:
                action_prob_list = action_probs if isinstance(action_probs, list) else [action_probs]
                num_heads = len(self.action_dims)
                log_probs_list = []
                entropy_list = []

                # 标准化各头张量形状为 (B, action_dim)
                normalized_heads = []
                for head_probs in action_prob_list:
                    hp = tf.convert_to_tensor(head_probs, dtype=tf.float32)
                    if tf.rank(hp) == 1:
                        hp = tf.expand_dims(hp, 0)
                    hp = tf.reshape(hp, (-1, int(self.action_dims[0])))
                    normalized_heads.append(hp)
                action_prob_list = normalized_heads

                # 解析 agent 索引与有效头数量 k（该站点设备数）
                gs = tf.convert_to_tensor(global_states, dtype=tf.float32)
                agent_one_hot = gs[:, -self._num_agents:]
                agent_idx = tf.argmax(agent_one_hot, axis=1, output_type=tf.int32)  # (B,)
                k_valid_heads = tf.gather(self._agent_machine_counts, agent_idx)      # (B,)

                # 按采样阶段的一致顺序进行“无放回”掩码与重归一化（含无效头掩码）
                mask = tf.zeros_like(action_prob_list[0], dtype=tf.bool)
                for i in range(num_heads):
                    head_probs = action_prob_list[i]
                    action_slice = actions[:, i]
                    action_one_hot = tf.one_hot(action_slice, int(self.action_dims[0]))

                    # 当前头是否有效
                    valid = tf.less(tf.cast(i, tf.int32), k_valid_heads)  # (B,)
                    valid_f = tf.cast(valid, tf.float32)
                    valid_f_ex = tf.expand_dims(valid_f, axis=1)

                    # 应用掩码并重归一化
                    masked_probs = tf.where(mask, tf.zeros_like(head_probs), head_probs)
                    norm = tf.reduce_sum(masked_probs, axis=1, keepdims=True) + 1e-8
                    masked_probs = masked_probs / norm

                    # 选中动作的对数概率（在掩码后的分布下，且仅对有效头计入）
                    log_prob_for_head_full = tf.math.log(tf.reduce_sum(masked_probs * action_one_hot, axis=1) + 1e-8)
                    log_prob_for_head = log_prob_for_head_full * valid_f
                    log_probs_list.append(log_prob_for_head)

                    # 使用掩码后的分布计算熵，保持与采样一致（无效头贡献为0）
                    entropy_for_head_full = -tf.reduce_sum(masked_probs * tf.math.log(masked_probs + 1e-8), axis=1)
                    entropy_for_head = entropy_for_head_full * valid_f
                    entropy_list.append(entropy_for_head)

                    # 更新掩码（仅对有效样本更新），且允许多个头选择 IDLE(0)
                    is_idle = tf.equal(action_slice, 0)
                    not_idle_ex = tf.expand_dims(tf.cast(tf.logical_not(is_idle), tf.bool), axis=1)
                    mask_update = tf.logical_and(tf.cast(action_one_hot > 0, tf.bool), tf.cast(valid_f_ex > 0, tf.bool))
                    mask_update = tf.logical_and(mask_update, not_idle_ex)
                    mask = tf.logical_or(mask, mask_update)

                # 联合对数概率与总熵
                current_probs = tf.add_n(log_probs_list)
                entropy_loss = tf.add_n(entropy_list)
            else:
                # Discrete单头输出
                action_one_hot = tf.one_hot(tf.squeeze(actions), self.action_dims[0])
                log_prob = tf.math.log(tf.reduce_sum(action_probs * action_one_hot, axis=1) + 1e-8)
                current_probs = log_prob
                entropy_loss = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)

            # PPO裁剪损失
            ratio = tf.exp(current_probs - old_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            
            # 计算裁剪比例（用于监控）
            clipped_mask = tf.greater(tf.abs(ratio - 1.0), clip_ratio)
            clip_fraction = tf.reduce_mean(tf.cast(clipped_mask, tf.float32))

            # Actor总损失 = 策略损失 - 熵奖励
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            entropy_mean = tf.reduce_mean(entropy_loss)
            actor_loss -= entropy_coeff * entropy_mean
            
        # Actor梯度更新
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        grad_clip_norm = self.config.get("grad_clip_norm", 1.0)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, grad_clip_norm)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # ========== Critic损失计算 ==========
        with tf.GradientTape() as tape:
            values = self.critic(global_states, training=False)
            # 显式统一 dtype，避免 float64/float32 混用
            values = tf.cast(values, tf.float32)
            returns_expanded = tf.expand_dims(returns, 1) if len(returns.shape) == 1 else returns
            returns_expanded = tf.cast(returns_expanded, tf.float32)
            critic_loss = tf.reduce_mean(tf.square(returns_expanded - values))
        
        # Critic梯度更新
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, grad_clip_norm)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # 近似KL：E[old_logp - new_logp]
        approx_kl = tf.reduce_mean(old_probs - current_probs)
        return {
            "actor_loss": actor_loss.numpy(),
            "critic_loss": critic_loss.numpy(),
            "entropy": entropy_mean.numpy(),
            "approx_kl": float(approx_kl.numpy()),
            "clip_fraction": clip_fraction.numpy()
        }

