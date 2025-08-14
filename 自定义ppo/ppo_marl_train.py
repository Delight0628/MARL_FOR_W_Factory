"""
纯净的多智能体PPO训练脚本
专注于核心训练功能，移除复杂的评估和可视化
"""

import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any
from datetime import datetime

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

class PPONetwork:
    """PPO网络实现"""
    
    # 🔧 V3 修复: lr参数现在可以是学习率调度器
    def __init__(self, state_dim: int, action_dim: int, lr: Any):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # 构建网络
        self.actor, self.critic = self._build_networks()
        
        # 优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr)
        
    def _build_networks(self):
        """构建Actor-Critic网络"""
        # Actor网络 - 增强版
        actor_input = tf.keras.layers.Input(shape=(self.state_dim,))
        actor_hidden1 = tf.keras.layers.Dense(512, activation='relu')(actor_input)
        actor_hidden2 = tf.keras.layers.Dense(256, activation='relu')(actor_hidden1)
        actor_output = tf.keras.layers.Dense(self.action_dim, activation='softmax')(actor_hidden2)
        actor = tf.keras.Model(inputs=actor_input, outputs=actor_output)
        
        # Critic网络 - 增强版
        critic_input = tf.keras.layers.Input(shape=(self.state_dim,))
        critic_hidden1 = tf.keras.layers.Dense(512, activation='relu')(critic_input)
        critic_hidden2 = tf.keras.layers.Dense(256, activation='relu')(critic_hidden1)
        critic_output = tf.keras.layers.Dense(1)(critic_hidden2)
        critic = tf.keras.Model(inputs=critic_input, outputs=critic_output)
        
        return actor, critic
    
    def get_action_and_value(self, state: np.ndarray) -> Tuple[int, float, float]:
        """获取动作、动作概率和状态价值"""
        state = tf.expand_dims(state, 0)
        
        action_probs = self.actor(state)
        action_dist = tf.random.categorical(tf.math.log(action_probs + 1e-8), 1)
        action = int(action_dist[0, 0])
        
        action_prob = float(action_probs[0, action])
        value = float(self.critic(state)[0, 0])
        
        return action, action_prob, value
    
    def get_value(self, state: np.ndarray) -> float:
        """获取状态价值"""
        state = tf.expand_dims(state, 0)
        return float(self.critic(state)[0, 0])
    
    def update(self, states: np.ndarray, actions: np.ndarray, 
               old_probs: np.ndarray, advantages: np.ndarray, 
               returns: np.ndarray, clip_ratio: float = 0.15) -> Dict[str, float]:  # 🔧 降低裁剪范围
        """PPO更新"""
        
        # Actor更新
        with tf.GradientTape() as tape:
            action_probs = self.actor(states)
            action_probs_selected = tf.reduce_sum(
                action_probs * tf.one_hot(actions, self.action_dim), axis=1
            )
            
            ratio = action_probs_selected / (old_probs + 1e-8)
            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )
            
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            actor_loss -= 0.01 * tf.reduce_mean(entropy)
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Critic更新
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.critic(states))
            critic_loss = tf.reduce_mean(tf.square(returns - values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return {
            'actor_loss': float(actor_loss),
            'critic_loss': float(critic_loss),
            'entropy': float(tf.reduce_mean(entropy))
        }

class ExperienceBuffer:
    """经验缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.action_probs = []
        self.dones = []
        
    def store(self, state, action, reward, value, action_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.action_probs.append(action_prob)
        self.dones.append(done)
    
    def get_batch(self, gamma=0.99, lam=0.95):
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        action_probs = np.array(self.action_probs)
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return states, actions, action_probs, advantages, returns
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.action_probs.clear()
        self.dones.clear()

class SimplePPOTrainer:
    """简化的PPO训练器"""
    
    # 🔧 V5 系统资源优化: 根据配置调整训练参数
    def __init__(self, initial_lr: float, total_train_episodes: int, steps_per_episode: int):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🔧 V5 性能优化：检测系统资源
        self.system_info = self._detect_system_resources()
        self._optimize_tensorflow_settings()
        
        # 环境探测
        temp_env, _ = self.create_environment()
        self.state_dim = temp_env.observation_space(temp_env.possible_agents[0]).shape[0]
        self.action_dim = temp_env.action_space(temp_env.possible_agents[0]).n
        self.agent_ids = temp_env.possible_agents
        temp_env.close()
        
        print("🔧 环境空间检测:")
        print(f"   观测维度: {self.state_dim}")
        print(f"   动作维度: {self.action_dim}")
        print(f"   智能体数量: {len(self.agent_ids)}")
        
        # 🔧 V5 资源优化：根据内存调整训练参数
        optimized_episodes, optimized_steps = self._optimize_training_params(
            total_train_episodes, steps_per_episode
        )
        
        # 🔧 V3 修复: 创建学习率衰减调度器
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=optimized_episodes * optimized_steps,
            end_learning_rate=1e-5,  # 衰减到较低的值
            power=1.0  # 线性衰减
        )

        # 共享网络
        self.shared_network = PPONetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=self.lr_schedule
        )
        
        # 训练统计
        self.episode_rewards = []
        self.training_losses = []
        self.iteration_times = []  # 🔧 V5 新增：记录每轮训练时间
        self.kpi_history = []      # 🔧 V5 新增：记录每轮KPI历史
        
        # 创建保存目录
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _detect_system_resources(self) -> Dict[str, Any]:
        """🔧 V5 新增：检测系统资源"""
        try:
            import psutil  # type: ignore
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.total / (1024**3)
            available_gb = memory_info.available / (1024**3)
            
            # 检测GPU
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            gpu_memory = 0
            if gpu_available:
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                    for gpu in gpus:
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        gpu_memory = gpu_details.get('device_name', 'Unknown')
                except:
                    gpu_available = False
            
            system_info = {
                'cpu_count': cpu_count,
                'memory_gb': memory_gb,
                'available_gb': available_gb,
                'gpu_available': gpu_available,
                'gpu_memory': gpu_memory
            }
            
            print("💻 系统资源检测:")
            print(f"   CPU核心数: {cpu_count}")
            print(f"   总内存: {memory_gb:.1f}GB")
            print(f"   可用内存: {available_gb:.1f}GB")
            print(f"   GPU可用: {'✅' if gpu_available else '❌'}")
            if gpu_available:
                print(f"   GPU信息: {gpu_memory}")
            
            return system_info
            
        except ImportError:
            # 如果没有psutil，使用保守估计
            print("⚠️  无法检测系统资源，使用默认配置")
            return {
                'cpu_count': 4,
                'memory_gb': 8.0,
                'available_gb': 4.0,
                'gpu_available': False,
                'gpu_memory': None
            }
    
    def _optimize_tensorflow_settings(self):
        """🔧 V5 新增：优化TensorFlow设置"""
        # 内存增长设置
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU内存增长模式已启用")
            except RuntimeError as e:
                print(f"⚠️  GPU设置失败: {e}")
        
        # 根据内存情况设置TensorFlow
        available_gb = self.system_info.get('available_gb', 4.0)
        if available_gb < 6.0:
            # 低内存模式
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
            print("🔧 低内存模式: 限制TensorFlow并行度")
        else:
            # 正常模式
            cpu_count = self.system_info.get('cpu_count', 4)
            tf.config.threading.set_inter_op_parallelism_threads(min(cpu_count, 4))
            tf.config.threading.set_intra_op_parallelism_threads(min(cpu_count, 8))
            print("🔧 正常模式: TensorFlow并行度优化")
    
    def _optimize_training_params(self, num_episodes: int, steps_per_episode: int) -> Tuple[int, int]:
        """🔧 V5 新增：根据系统资源优化训练参数"""
        available_gb = self.system_info.get('available_gb', 4.0)
        
        # 根据可用内存调整训练规模
        if available_gb < 4.0:
            # 极低内存：大幅降低参数
            optimized_episodes = min(num_episodes, 60)
            optimized_steps = min(steps_per_episode, 800)
            print("🚨 极低内存模式: 训练规模大幅缩减")
        elif available_gb < 6.0:
            # 低内存：适度降低参数
            optimized_episodes = min(num_episodes, 80)
            optimized_steps = min(steps_per_episode, 1000)
            print("⚠️  低内存模式: 训练规模适度缩减")
        elif available_gb < 8.0:
            # 中等内存：略微降低参数
            optimized_episodes = min(num_episodes, 100)
            optimized_steps = min(steps_per_episode, 1200)
            print("🔧 中等内存模式: 训练规模略微调整")
        else:
            # 充足内存：使用原始参数
            optimized_episodes = num_episodes
            optimized_steps = steps_per_episode
            print("✅ 充足内存模式: 使用完整训练规模")
        
        if optimized_episodes != num_episodes or optimized_steps != steps_per_episode:
            print(f"🔧 参数调整: {num_episodes}回合×{steps_per_episode}步 → {optimized_episodes}回合×{optimized_steps}步")
        
        return optimized_episodes, optimized_steps

    def create_environment(self):
        """创建环境"""
        env = make_parallel_env()
        buffers = {
            agent: ExperienceBuffer() 
            for agent in env.possible_agents
        }
        return env, buffers
    
    def collect_experience(self, env, buffers, num_steps: int = 200) -> float:
        """收集经验"""
        observations, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        step_count = 0
        
        for step in range(num_steps):
            actions = {}
            values = {}
            action_probs = {}
            
            # 为每个智能体获取动作
            for agent in env.agents:
                if agent in observations:
                    action, action_prob, value = self.shared_network.get_action_and_value(
                        observations[agent]
                    )
                    actions[agent] = action
                    values[agent] = value
                    action_probs[agent] = action_prob
            
            # 执行动作
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 存储经验
            for agent in env.agents:
                if agent in observations and agent in actions:
                    done = terminations.get(agent, False) or truncations.get(agent, False)
                    reward = rewards.get(agent, 0)
                    
                    buffers[agent].store(
                        state=observations[agent],
                        action=actions[agent],
                        reward=reward,
                        value=values[agent],
                        action_prob=action_probs[agent],
                        done=done
                    )
                    
                    episode_rewards[agent] += reward
            
            observations = next_observations
            step_count += 1
            
            # 检查结束条件
            if any(terminations.values()) or any(truncations.values()):
                observations, _ = env.reset()
        
        return sum(episode_rewards.values())
    
    def update_policy(self, buffers) -> Dict[str, float]:
        """更新策略"""
        all_states = []
        all_actions = []
        all_action_probs = []
        all_advantages = []
        all_returns = []
        
        # 合并所有智能体的经验
        for agent, buffer in buffers.items():
            if len(buffer.states) > 0:
                states, actions, action_probs, advantages, returns = buffer.get_batch()
                
                all_states.extend(states)
                all_actions.extend(actions)
                all_action_probs.extend(action_probs)
                all_advantages.extend(advantages)
                all_returns.extend(returns)
                
                buffer.clear()
        
        if len(all_states) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        
        # 转换为numpy数组
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_action_probs = np.array(all_action_probs)
        all_advantages = np.array(all_advantages)
        all_returns = np.array(all_returns)
        
        # 多次更新 - 🔧 增加迭代次数提升学习充分性
        losses = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        num_updates = 10  # 从5增加到10
        
        for _ in range(num_updates):
            batch_losses = self.shared_network.update(
                states=all_states,
                actions=all_actions,
                old_probs=all_action_probs,
                advantages=all_advantages,
                returns=all_returns
            )
            
            for key in losses:
                losses[key] += batch_losses[key] / num_updates
        
        return losses
    
    def quick_kpi_evaluation(self, num_episodes: int = 3) -> Dict[str, float]:
        """🔧 V5 新增：快速KPI评估（用于每轮监控）"""
        env, _ = self.create_environment()
        
        total_rewards = []
        makespans = []
        utilizations = []
        completed_parts_list = []
        
        for episode in range(num_episodes):
            observations, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < 800:  # 快速评估，步数较少
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
            makespans.append(final_stats.get('makespan', step_count))
            utilizations.append(final_stats.get('mean_utilization', 0))
            completed_parts_list.append(final_stats.get('total_parts', 0))
        
        env.close()
        
        return {
            'mean_reward': np.mean(total_rewards),
            'mean_makespan': np.mean(makespans),
            'mean_utilization': np.mean(utilizations),
            'mean_completed_parts': np.mean(completed_parts_list)
        }
    
    def simple_evaluation(self, num_episodes: int = 5) -> Dict[str, float]:
        """简单评估（仅用于训练期间的快速检查）"""
        env, _ = self.create_environment()
        
        total_rewards = []
        total_steps = []
        
        for episode in range(num_episodes):
            observations, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < 1200:  # 提升最大步数限制，增加看到正向奖励概率
                actions = {}
                
                # 使用确定性策略评估
                for agent in env.agents:
                    if agent in observations:
                        state = tf.expand_dims(observations[agent], 0)
                        action_probs = self.shared_network.actor(state)
                        action = int(tf.argmax(action_probs[0]))  # 选择概率最高的动作
                        actions[agent] = action
                
                observations, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                step_count += 1
                
                if any(terminations.values()) or any(truncations.values()):
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(step_count)
        
        env.close()
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_steps': np.mean(total_steps)
        }
    
    def comprehensive_evaluation(self, num_episodes: int = 10) -> Dict[str, Any]:
        """🔧 V3 修复: 完整的业务指标评估, 修复KPI统计缺陷"""
        print(f"\n📊 完整业务指标评估 ({num_episodes} 回合)")
        print("=" * 60)
        
        env, _ = self.create_environment()
        
        eval_results = {
            'episode_rewards': [],
            'makespans': [],
            'total_tardiness': [],
            'max_tardiness': [],
            'completed_parts': [],
            'utilizations': [],
        }
        
        for episode in range(num_episodes):
            observations, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < 1500:  # 进一步提升步数上限，确保有充分时间完成订单
                actions = {}
                for agent in env.agents:
                    if agent in observations:
                        state = tf.expand_dims(observations[agent], 0)
                        action_probs = self.shared_network.actor(state)
                        action = int(tf.argmax(action_probs[0]))  # 确定性策略
                        actions[agent] = action
                
                observations, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values())
                step_count += 1
                
                if any(terminations.values()) or any(truncations.values()):
                    break # 仿真自然结束，退出循环
            
            # --- 🔧 V3 关键修复 ---
            # 无论循环如何结束 (自然完成或超时)，都直接从环境中获取最终统计数据
            # 这是获取真实KPI的唯一可靠方法
            final_stats = env.sim.get_final_stats()
            
            eval_results['episode_rewards'].append(episode_reward)
            eval_results['makespans'].append(final_stats.get('makespan', 0))
            eval_results['total_tardiness'].append(final_stats.get('total_tardiness', 0))
            eval_results['max_tardiness'].append(final_stats.get('max_tardiness', 0))
            eval_results['completed_parts'].append(final_stats.get('total_parts', 0))
            eval_results['utilizations'].append(final_stats.get('mean_utilization', 0))
            
            print(f"    ✅ 回合{episode+1}: Makespan={final_stats.get('makespan', 0):.1f}, 完成={final_stats.get('total_parts', 0)}, 利用率={final_stats.get('mean_utilization', 0):.1%}")

        # 计算统计指标
        summary_stats = {
            'mean_reward': np.mean(eval_results['episode_rewards']),
            'std_reward': np.std(eval_results['episode_rewards']),
            'mean_makespan': np.mean(eval_results['makespans']) if eval_results['makespans'] else 0,
            'mean_tardiness': np.mean(eval_results['total_tardiness']) if eval_results['total_tardiness'] else 0,
            'mean_utilization': np.mean(eval_results['utilizations']) if eval_results['utilizations'] else 0,
            'mean_completed_parts': np.mean(eval_results['completed_parts']) if eval_results['completed_parts'] else 0,
        }
        
        eval_results['summary'] = summary_stats
        
        print(f"\n📊 业务指标汇总:")
        print(f"  平均奖励: {summary_stats['mean_reward']:.2f} ± {summary_stats['std_reward']:.2f}")
        print(f"  平均Makespan: {summary_stats['mean_makespan']:.1f} 分钟")
        print(f"  平均延期时间: {summary_stats['mean_tardiness']:.1f} 分钟")
        print(f"  平均设备利用率: {summary_stats['mean_utilization']:.1%}")
        print(f"  平均完成零件数: {summary_stats['mean_completed_parts']:.1f}")
        
        env.close()
        return eval_results
    
    def train(self, num_episodes: int = 100, steps_per_episode: int = 200, 
              eval_frequency: int = 20):
        """🔧 V5 增强版训练主循环 - 详细日志和KPI监控"""
        # 🔧 V5 应用系统优化的参数
        optimized_episodes, optimized_steps = self._optimize_training_params(num_episodes, steps_per_episode)
        
        print(f"🚀 开始PPO训练 (V5 系统优化版)")
        print(f"📊 训练参数: {optimized_episodes}回合, 每回合{optimized_steps}步")
        print(f"💻 系统配置: {self.system_info['memory_gb']:.1f}GB内存, GPU={'✅' if self.system_info['gpu_available'] else '❌'}")
        print("=" * 80)
        
        if not validate_config():
            print("❌ 配置验证失败")
            return
        
        # 训练开始时间记录
        training_start_time = time.time()
        training_start_datetime = datetime.now()
        print(f"🕐 训练开始时间: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 创建环境
        env, buffers = self.create_environment()
        
        best_reward = float('-inf')
        best_makespan = float('inf')
        
        try:
            for episode in range(optimized_episodes):
                iteration_start_time = time.time()
                
                # 收集经验
                episode_reward = self.collect_experience(env, buffers, optimized_steps)
                
                # 更新策略
                losses = self.update_policy(buffers)
                
                # 记录统计
                iteration_end_time = time.time()
                iteration_duration = iteration_end_time - iteration_start_time
                self.iteration_times.append(iteration_duration)
                self.episode_rewards.append(episode_reward)
                self.training_losses.append(losses)
                
                # 🔧 V5 核心：每轮进行快速KPI评估
                if (episode + 1) % 5 == 0 or episode == 0:  # 每5轮或第一轮评估KPI
                    kpi_results = self.quick_kpi_evaluation(num_episodes=2)
                    self.kpi_history.append(kpi_results)
                    
                    # 更新最佳记录
                    current_makespan = kpi_results['mean_makespan']
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                    
                    print(f"\n📊 回合 {episode + 1:3d}/{optimized_episodes} | "
                          f"奖励: {episode_reward:8.2f} | "
                          f"Actor损失: {losses['actor_loss']:7.4f}")
                    print(f"   ⏱️  用时: {iteration_duration:.1f}s | "
                          f"KPI - Makespan: {current_makespan:.1f}min | "
                          f"利用率: {kpi_results['mean_utilization']:.1%} | "
                          f"完成: {kpi_results['mean_completed_parts']:.0f}/33")
                    
                    # 🔧 V5 时间预测（参考WSL脚本）
                    if len(self.iteration_times) > 1:
                        avg_time = np.mean(self.iteration_times)
                        remaining_episodes = optimized_episodes - (episode + 1)
                        estimated_remaining = remaining_episodes * avg_time
                        
                        if remaining_episodes > 0:
                            finish_time = time.time() + estimated_remaining
                            finish_str = time.strftime('%H:%M:%S', time.localtime(finish_time))
                            print(f"   🔮 预计剩余: {estimated_remaining/60:.1f}min | "
                                  f"完成时间: {finish_str}")
                else:
                    # 简化输出
                    if (episode + 1) % 10 == 0:
                        recent_rewards = self.episode_rewards[-10:]
                        avg_reward = np.mean(recent_rewards)
                        
                        print(f"回合 {episode + 1:3d}/{optimized_episodes} | "
                              f"奖励: {episode_reward:8.2f} | "
                              f"平均: {avg_reward:8.2f} | "
                              f"Actor损失: {losses['actor_loss']:7.4f} | "
                              f"用时: {iteration_duration:.1f}s")
                
                # 定期详细评估和模型保存
                if (episode + 1) % eval_frequency == 0:
                    print(f"\n🔍 第{episode + 1}回合详细评估...")
                    eval_results = self.simple_evaluation()
                    print(f"   评估奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                    print(f"   平均步数: {eval_results['mean_steps']:.1f}")
                    
                    # 🔧 V3.1 修复: 正确获取和打印当前学习率的值
                    optimizer_step = self.shared_network.actor_optimizer.iterations
                    current_lr_value = self.shared_network.actor_optimizer.learning_rate(optimizer_step)
                    print(f"   当前学习率: {current_lr_value.numpy():.6f}")
                    
                    # 保存最佳模型
                    if eval_results['mean_reward'] > best_reward:
                        best_reward = eval_results['mean_reward']
                        self.save_model(f"{self.models_dir}/best_ppo_model_{self.timestamp}")
                        print(f"   ✅ 新的最佳模型已保存 (奖励: {best_reward:.2f})")
                    print()
            
            # 🔧 V5 训练完成统计（参考WSL脚本）
            training_end_time = time.time()
            training_end_datetime = datetime.now()
            total_training_time = training_end_time - training_start_time
            
            print("\n" + "=" * 80)
            print("🎉 训练完成！")
            print(f"🕐 训练开始: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🏁 训练结束: {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  总训练时间: {total_training_time/60:.1f}分钟 ({total_training_time:.1f}秒)")
            print(f"📈 最佳评估奖励: {best_reward:.2f}")
            print(f"🎯 最佳Makespan: {best_makespan:.1f}分钟")
            
            # 训练效率统计
            if self.iteration_times:
                avg_iteration_time = np.mean(self.iteration_times)
                min_iteration_time = np.min(self.iteration_times)
                max_iteration_time = np.max(self.iteration_times)
                print(f"⚡ 平均每轮: {avg_iteration_time:.1f}s | "
                      f"最快: {min_iteration_time:.1f}s | "
                      f"最慢: {max_iteration_time:.1f}s")
            
            # KPI趋势分析
            if self.kpi_history:
                initial_makespan = self.kpi_history[0]['mean_makespan']
                final_makespan = self.kpi_history[-1]['mean_makespan']
                makespan_improvement = (initial_makespan - final_makespan) / initial_makespan * 100
                
                initial_utilization = self.kpi_history[0]['mean_utilization']
                final_utilization = self.kpi_history[-1]['mean_utilization']
                utilization_improvement = (final_utilization - initial_utilization) * 100
                
                print(f"📊 KPI改进:")
                print(f"   Makespan: {initial_makespan:.1f}→{final_makespan:.1f}min "
                      f"({'改进' if makespan_improvement > 0 else '退化'}{abs(makespan_improvement):.1f}%)")
                print(f"   利用率: {initial_utilization:.1%}→{final_utilization:.1%} "
                      f"({'提升' if utilization_improvement > 0 else '降低'}{abs(utilization_improvement):.1f}%)")
            
            # 🔧 最终完整评估（包含真实业务指标）
            print("\n📊 最终完整评估...")
            final_eval = self.comprehensive_evaluation(num_episodes=10)
            
            # 保存最终模型
            self.save_model(f"{self.models_dir}/final_ppo_model_{self.timestamp}")
            
            return {
                'training_time': total_training_time,
                'best_reward': best_reward,
                'best_makespan': best_makespan,
                'final_eval': final_eval,
                'episode_rewards': self.episode_rewards,
                'kpi_history': self.kpi_history,
                'iteration_times': self.iteration_times
            }
            
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            env.close()
    
    def save_model(self, filepath: str):
        """保存模型"""
        try:
            self.shared_network.actor.save(f"{filepath}_actor.keras")
            self.shared_network.critic.save(f"{filepath}_critic.keras")
            print(f"✅ 模型已保存: {filepath}_actor.keras")
        except Exception as e:
            print(f"⚠️ 保存模型时出错: {e}")

def main():
    """主函数"""
    print("🏭 W工厂订单思维革命PPO训练系统 V5")
    print("🎯 奖励革命：从零件思维到订单思维的根本性转变")
    print("🔧 V5新特性: 系统资源优化 + GPU加速 + 详细训练日志 + 实时KPI监控")
    print("🔧 革命项: 订单奖励5000 vs 零件奖励1 (5000:1压倒性优势) + 严厉遗弃惩罚")
    print("=" * 80)
    
    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    try:
        # 🔧 V5 系统优化: 根据硬件配置动态调整训练参数
        num_episodes = 120
        steps_per_episode = 1200
        
        trainer = SimplePPOTrainer(
            initial_lr=1e-4,
            total_train_episodes=num_episodes,
            steps_per_episode=steps_per_episode
        )
        
        # 开始训练（系统会自动根据资源调整参数）
        results = trainer.train(
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            eval_frequency=20       # 评估频率
        )
        
        if results:
            print("\n🎉 训练成功完成！")
            
            # 🔧 V5 增强版结果分析
            final_summary = results['final_eval']['summary']
            print(f"\n📊 最终业务表现:")
            print(f"  奖励: {final_summary['mean_reward']:.2f} ± {final_summary['std_reward']:.2f}")
            print(f"  Makespan: {final_summary['mean_makespan']:.1f} 分钟")
            print(f"  延期时间: {final_summary['mean_tardiness']:.1f} 分钟")
            print(f"  设备利用率: {final_summary['mean_utilization']:.1%}")
            print(f"  完成零件数: {final_summary['mean_completed_parts']:.1f}")
            print(f"  最佳训练Makespan: {results['best_makespan']:.1f} 分钟")
            
            # 🔧 V5 训练效率分析
            training_time_min = results['training_time'] / 60
            if 'iteration_times' in results and results['iteration_times']:
                total_iterations = len(results['iteration_times'])
                avg_per_iteration = results['training_time'] / total_iterations
                print(f"\n⚡ 训练效率分析:")
                print(f"  总训练时长: {training_time_min:.1f}分钟")
                print(f"  平均每轮时间: {avg_per_iteration:.1f}秒")
                print(f"  训练总轮数: {total_iterations}轮")
                print(f"  训练效率: {total_iterations/training_time_min:.1f}轮/分钟")
            
            # 🔧 V5 KPI趋势分析
            if 'kpi_history' in results and results['kpi_history']:
                kpi_history = results['kpi_history']
                print(f"\n📈 KPI训练趋势:")
                print(f"  初始Makespan: {kpi_history[0]['mean_makespan']:.1f}min")
                print(f"  最终Makespan: {kpi_history[-1]['mean_makespan']:.1f}min")
                print(f"  初始利用率: {kpi_history[0]['mean_utilization']:.1%}")
                print(f"  最终利用率: {kpi_history[-1]['mean_utilization']:.1%}")
                print(f"  KPI监控点数: {len(kpi_history)}个")
            
            # 稳定性分析
            rewards_history = results['episode_rewards']
            if len(rewards_history) >= 20:
                early_avg = np.mean(rewards_history[:20])
                late_avg = np.mean(rewards_history[-20:])
                stability = abs(late_avg - early_avg) / (abs(early_avg) + 1e-8) * 100
                print(f"\n🔍 学习稳定性分析:")
                print(f"  前20回合平均奖励: {early_avg:.2f}")
                print(f"  后20回合平均奖励: {late_avg:.2f}")
                print(f"  波动幅度: {stability:.1f}%")
                if stability < 10:
                    print("  ✅ 学习过程较为稳定")
                else:
                    print("  ⚠️ 学习过程存在较大波动，建议进一步调整超参数")
        else:
            print("\n❌ 训练失败")
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
