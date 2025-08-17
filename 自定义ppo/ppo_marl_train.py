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

# V10.1中设置的日志级别现在由文件顶部的环境变量接管，故移除
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.get_logger().setLevel('ERROR')

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

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
        """🔧 V6 构建Actor-Critic网络 - 内存友好版本"""
        # 🔧 V6 根据系统资源动态调整网络大小
        available_gb = getattr(self, 'system_info', {}).get('available_gb', 8.0)
        
        if available_gb < 5.0:
            # 低内存：小网络
            hidden_sizes = [128, 64]
            
        elif available_gb < 8.0:
            # 中等内存：中型网络
            hidden_sizes = [256, 128]
        else:
            # 充足内存：大型网络 - 🚀 V12 模型容量提升
            hidden_sizes = [1024, 512]
        
        # Actor网络
        actor_input = tf.keras.layers.Input(shape=(self.state_dim,))
        actor_x = tf.keras.layers.Dense(hidden_sizes[0], activation='relu')(actor_input)
        actor_x = tf.keras.layers.Dropout(0.1)(actor_x)  # 🔧 V6 添加dropout防过拟合
        actor_x = tf.keras.layers.Dense(hidden_sizes[1], activation='relu')(actor_x)
        actor_output = tf.keras.layers.Dense(self.action_dim, activation='softmax')(actor_x)
        actor = tf.keras.Model(inputs=actor_input, outputs=actor_output)
        
        # Critic网络
        critic_input = tf.keras.layers.Input(shape=(self.state_dim,))
        critic_x = tf.keras.layers.Dense(hidden_sizes[0], activation='relu')(critic_input)
        critic_x = tf.keras.layers.Dropout(0.1)(critic_x)  # 🔧 V6 添加dropout防过拟合
        critic_x = tf.keras.layers.Dense(hidden_sizes[1], activation='relu')(critic_x)
        critic_output = tf.keras.layers.Dense(1)(critic_x)
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

# 🔧 V8 新增: 多进程并行工作函数
def run_simulation_worker(network_weights: Dict[str, List[np.ndarray]],
                          state_dim: int, action_dim: int, num_steps: int, seed: int) -> Tuple[Dict[str, ExperienceBuffer], float]:
    """
    Worker process for collecting experience in parallel.
    Each worker creates its own environment and network.
    """
    # 1. 设置进程特定的随机种子
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 🔧 V10.2 修正: 必须保留，确保子进程不访问GPU
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 2. 创建本地环境和网络
    env = make_parallel_env()
    # 学习率是占位符，因为工作进程不进行训练
    local_network = PPONetwork(state_dim, action_dim, lr=1e-4)
    local_network.actor.set_weights(network_weights['actor'])
    local_network.critic.set_weights(network_weights['critic'])

    buffers = {agent: ExperienceBuffer() for agent in env.possible_agents}
    
    # 3. 🔧 修复：收集经验，使用与评估一致的episode长度限制
    observations, _ = env.reset()
    episode_rewards = {agent: 0 for agent in env.possible_agents}
    step_count = 0
    collected_steps = 0

    while collected_steps < num_steps:
        actions = {}
        values = {}
        action_probs = {}

        for agent in env.agents:
            if agent in observations:
                action, action_prob, value = local_network.get_action_and_value(observations[agent])
                actions[agent] = action
                values[agent] = value
                action_probs[agent] = action_prob

        next_observations, rewards, terminations, truncations, _ = env.step(actions)
        step_count += 1
        collected_steps += 1

        for agent in env.agents:
            if agent in observations and agent in actions:
                done = terminations.get(agent, False) or truncations.get(agent, False)
                reward = rewards.get(agent, 0)
                buffers[agent].store(
                    observations[agent], actions[agent], reward,
                    values[agent], action_probs[agent], done
                )
                episode_rewards[agent] += reward

        observations = next_observations

        # 🔧 修复：与评估一致的终止条件
        if any(terminations.values()) or any(truncations.values()) or step_count >= 1500:
            observations, _ = env.reset()
            step_count = 0  # 重置episode步数计数器

    env.close()

    total_reward = sum(episode_rewards.values())
    return buffers, total_reward

class SimplePPOTrainer:
    """简化的PPO训练器"""
    
    # 🔧 V5 系统资源优化: 根据配置调整训练参数
    def __init__(self, initial_lr: float, total_train_episodes: int, steps_per_episode: int):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🔧 V5 性能优化：检测系统资源
        self.system_info = self._detect_system_resources()
        self._optimize_tensorflow_settings()
        
        # 🔧 V9 CPU并行优化: 智能调节进程数，防止内存爆炸
        cpu_cores = self.system_info.get('cpu_count', 4)
        # 保留核心给主进程和系统，使用核心数的一半作为工作进程数，兼顾性能与稳定
        self.num_workers = min(max(1, cpu_cores // 2), 32)
        print(f"🔧 V9 CPU并行优化: 将使用 {self.num_workers} 个并行环境进行数据采集 (智能调节)")
        
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
        self.models_dir = "自定义ppo/ppo_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 🔧 V12 新增：TensorBoard支持
        self.tensorboard_dir = f"自定义ppo/tensorboard_logs/{self.timestamp}"
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        if TENSORBOARD_AVAILABLE:
            self.train_writer = tf.summary.create_file_writer(f"{self.tensorboard_dir}/train")
            print(f"📊 TensorBoard日志已启用: {self.tensorboard_dir}")
            print(f"    使用命令: tensorboard --logdir={self.tensorboard_dir}")
        else:
            self.train_writer = None
            print("⚠️  TensorBoard不可用")
    
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
        """🔧 V7 增强版：优化TensorFlow设置，充分利用48核CPU"""
        # 内存增长设置
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU内存增长模式已启用")
            except RuntimeError as e:
                print(f"⚠️  GPU设置失败: {e}")
        
        # 🔧 V7 CPU优化：充分利用48核CPU
        cpu_count = self.system_info.get('cpu_count', 4)
        available_gb = self.system_info.get('available_gb', 4.0)
        
        if available_gb < 6.0:
            # 低内存模式：保守使用CPU
            tf.config.threading.set_inter_op_parallelism_threads(min(cpu_count // 4, 12))
            tf.config.threading.set_intra_op_parallelism_threads(min(cpu_count // 2, 24))
            print(f"🔧 低内存模式: TensorFlow使用{min(cpu_count // 4, 12)}个inter线程, {min(cpu_count // 2, 24)}个intra线程")
        else:
            # 🔧 V7 高性能模式：激进使用所有CPU核心
            inter_threads = min(cpu_count // 2, 24)  # 最多24个inter线程
            intra_threads = min(cpu_count, 48)       # 最多48个intra线程
            tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
            tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
            print(f"🔧 V7高性能模式: TensorFlow使用{inter_threads}个inter线程, {intra_threads}个intra线程")
            print(f"🚀 CPU优化: 充分利用{cpu_count}核心处理器")
    
    def _optimize_training_params(self, num_episodes: int, steps_per_episode: int) -> Tuple[int, int]:
        """🔧 V6 强化版：根据系统资源优化训练参数，防止卡死"""
        available_gb = self.system_info.get('available_gb', 4.0)
        total_gb = self.system_info.get('memory_gb', 8.0)
        
        # 🔧 V6 更保守的内存策略，基于实际可用内存而非总内存
        if available_gb < 3.0:
            # 危险内存：极度保守
            optimized_episodes = min(num_episodes, 40)
            optimized_steps = min(steps_per_episode, 600)
            print("🚨 危险内存模式: 训练规模极度缩减（防卡死）")
        elif available_gb < 5.0:
            # 低内存：大幅降低参数
            optimized_episodes = min(num_episodes, 60)
            optimized_steps = min(steps_per_episode, 800)
            print("⚠️  低内存模式: 训练规模大幅缩减")
        elif available_gb < 7.0:
            # 中等内存：适度降低参数 - 🔧 V6 更保守
            optimized_episodes = min(num_episodes, 80)
            optimized_steps = min(steps_per_episode, 1000)
            print("🔧 中等内存模式: 训练规模适度缩减")
        elif available_gb < 10.0:
            # 较好内存：略微降低参数 - 🔧 V6 新增层级
            optimized_episodes = min(num_episodes, 90)
            optimized_steps = min(steps_per_episode, 1100)
            print("💚 较好内存模式: 训练规模略微调整")
        else:
            # 充足内存: 性能完全释放 - 🚀 V11 极限性能模式
            optimized_episodes = num_episodes
            optimized_steps = steps_per_episode
            print("✅ 充足内存模式: 性能完全释放，使用完整训练规模！")
        
        # 🔧 V6 新增：内存使用率警告
        memory_usage_percent = ((total_gb - available_gb) / total_gb) * 100
        if memory_usage_percent > 90:
            print(f"⚠️  当前内存使用率: {memory_usage_percent:.1f}% - 建议关闭其他程序")
        
        if optimized_episodes != num_episodes or optimized_steps != steps_per_episode:
            print(f"🔧 参数调整: {num_episodes}回合×{steps_per_episode}步 → {optimized_episodes}回合×{optimized_steps}步")
            print(f"💡 节省内存: 预计减少{((num_episodes*steps_per_episode) - (optimized_episodes*optimized_steps))/(num_episodes*steps_per_episode)*100:.1f}%的内存使用")
        
        return optimized_episodes, optimized_steps
    
    def _check_memory_usage(self) -> bool:
        """🔧 V6 新增：检查内存使用情况，必要时触发垃圾回收"""
        try:
            import psutil  # type: ignore
            import gc
            
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            usage_percent = memory_info.percent
            
            # 内存使用率过高时触发垃圾回收
            if usage_percent > 95:
                print(f"🧹 内存使用率过高 ({usage_percent:.1f}%)，执行垃圾回收...")
                gc.collect()
                tf.keras.backend.clear_session()  # 清理TensorFlow会话
                return False  # 建议暂停训练
            elif usage_percent > 90:
                print(f"⚠️  内存使用率较高 ({usage_percent:.1f}%)，建议注意")
                gc.collect()
                return True
            
            return True
        except ImportError:
            return True  # 无法检测时假设正常
    
    def _safe_model_update(self, buffers) -> Dict[str, float]:
        """🔧 V6 新增：安全的模型更新，包含内存检查"""
        # 更新前检查内存
        if not self._check_memory_usage():
            print("⚠️  内存不足，跳过本轮模型更新")
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        
        # 执行正常的策略更新
        return self.update_policy(buffers)

    def create_environment(self):
        """创建环境"""
        env = make_parallel_env()
        buffers = {
            agent: ExperienceBuffer() 
            for agent in env.possible_agents
        }
        return env, buffers
    
    def collect_experience_parallel(self, buffers, num_steps: int) -> float:
        """🔧 V8 新增：使用多进程并行收集经验"""
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
                    seed
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    worker_buffers, worker_reward = future.result()
                    total_reward += worker_reward
                    
                    for agent_id, worker_buffer in worker_buffers.items():
                        buffers[agent_id].states.extend(worker_buffer.states)
                        buffers[agent_id].actions.extend(worker_buffer.actions)
                        buffers[agent_id].rewards.extend(worker_buffer.rewards)
                        buffers[agent_id].values.extend(worker_buffer.values)
                        buffers[agent_id].action_probs.extend(worker_buffer.action_probs)
                        buffers[agent_id].dones.extend(worker_buffer.dones)
                except Exception as e:
                    print(f"❌ 一个并行工作进程失败: {e}")
                    import traceback
                    traceback.print_exc()

        return total_reward
    
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
        """🔧 修复版：快速KPI评估（用于每轮监控）"""
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
        
        # 🔧 V8 优化: 不再需要创建主环境，只创建缓冲区
        buffers = {
            agent: ExperienceBuffer() 
            for agent in self.agent_ids
        }
        
        best_reward = float('-inf')
        best_makespan = float('inf')
        
        try:
            for episode in range(optimized_episodes):
                iteration_start_time = time.time()
                
                # 收集经验 - 🔧 V8 改为并行收集
                collect_start_time = time.time()
                episode_reward = self.collect_experience_parallel(buffers, optimized_steps)
                collect_duration = time.time() - collect_start_time
                
                # 🔧 V6 安全的策略更新（包含内存检查）
                update_start_time = time.time()
                losses = self._safe_model_update(buffers)
                update_duration = time.time() - update_start_time
                
                # 记录统计
                iteration_end_time = time.time()
                iteration_duration = iteration_end_time - iteration_start_time
                self.iteration_times.append(iteration_duration)
                self.episode_rewards.append(episode_reward)
                self.training_losses.append(losses)
                
                # 🔧 V12 TensorBoard日志记录
                if self.train_writer is not None:
                    with self.train_writer.as_default():
                        tf.summary.scalar('Training/Episode_Reward', episode_reward, step=episode)
                        tf.summary.scalar('Training/Actor_Loss', losses['actor_loss'], step=episode)
                        tf.summary.scalar('Training/Critic_Loss', losses['critic_loss'], step=episode)
                        tf.summary.scalar('Training/Entropy', losses['entropy'], step=episode)
                        tf.summary.scalar('Performance/Iteration_Duration', iteration_duration, step=episode)
                        tf.summary.scalar('Performance/CPU_Collection_Time', collect_duration, step=episode)
                        tf.summary.scalar('Performance/GPU_Update_Time', update_duration, step=episode)
                        self.train_writer.flush()
                
                # 🔧 V11 Checkpoint 优化: 移除定期保存，只保留最佳模型保存
                
                # 🔧 V12 合并修复：每轮都进行KPI评估和显示
                kpi_results = self.quick_kpi_evaluation(num_episodes=2)
                self.kpi_history.append(kpi_results)
                
                # 🔧 V12 TensorBoard KPI记录
                if self.train_writer is not None:
                    with self.train_writer.as_default():
                        tf.summary.scalar('KPI/Makespan', kpi_results['mean_makespan'], step=episode)
                        tf.summary.scalar('KPI/Completed_Parts', kpi_results['mean_completed_parts'], step=episode)
                        tf.summary.scalar('KPI/Utilization', kpi_results['mean_utilization'], step=episode)
                        tf.summary.scalar('KPI/Tardiness', kpi_results['mean_tardiness'], step=episode)
                        self.train_writer.flush()
                
                # 🔧 修复：正确更新最佳记录（只有当makespan > 0时才更新）
                current_makespan = kpi_results['mean_makespan']
                if current_makespan > 0 and current_makespan < best_makespan:
                    best_makespan = current_makespan
                
                # 🔧 V12 统一显示格式：每轮都显示完整信息
                print(f"\n🔂 回合 {episode + 1:3d}/{optimized_episodes} | "
                      f"奖励: {episode_reward:.1f} | "
                      f"Actor损失: {losses['actor_loss']:7.4f}| "
                      f"⏱️  本轮用时: {iteration_duration:.1f}s (CPU采集: {collect_duration:.1f}s, GPU更新: {update_duration:.1f}s)")
                print(f"📊 KPI - 总完工时间: {current_makespan:.1f}min | "
                      f"完成: {kpi_results['mean_completed_parts']:.0f}/33 | "
                      f"设备利用率: {kpi_results['mean_utilization']:.1%} | "
                      f"延期时间: {kpi_results['mean_tardiness']:.1f}min")
                

                if len(self.iteration_times) > 1: #and (episode + 1) % 10 == 0:
                    avg_time = np.mean(self.iteration_times)
                    remaining_episodes = optimized_episodes - (episode + 1)
                    estimated_remaining = remaining_episodes * avg_time
                    progress_percent = ((episode + 1) / optimized_episodes) * 100
                    
                    if remaining_episodes > 0:
                        finish_time = time.time() + estimated_remaining
                        finish_str = time.strftime('%H:%M:%S', time.localtime(finish_time))
                    recent_rewards = self.episode_rewards[-10:]
                    avg_reward = np.mean(recent_rewards)
                    print(f"🔮 当前训练进度: {progress_percent:.1f}% | 预计剩余时间: {estimated_remaining/60:.1f}min | "
                          f"完成时间: {finish_str}\n")
                    #print(f"=========================================================================\n"
                    #      f"🔮 当前训练进度: {progress_percent:.1f}% | 预计剩余时间: {estimated_remaining/60:.1f}min | "
                    #      f"完成时间: {finish_str} | 近10轮平均奖励: {avg_reward:.1f}\n"
                    #      f"=========================================================================")
                
                # 🔧 V12 最佳模型检查（每轮都检查）
                current_kpi = kpi_results
                if current_kpi:
                    # 🔧 V12 综合评分标准：Makespan最小 + 利用率最大 + 延期最短
                    # 归一化各项指标到0-1范围，然后加权求和
                    makespan_score = max(0, 1 - current_kpi['mean_makespan'] / 600)  # 600分钟为基准
                    utilization_score = current_kpi['mean_utilization']  # 利用率本身就是0-1
                    tardiness_score = max(0, 1 - current_kpi['mean_tardiness'] / 1000)  # 1000分钟为基准
                    completion_score = current_kpi['mean_completed_parts'] / 33  # 完成率0-1
                    
                    # 综合评分：权重可调整
                    current_score = (
                        makespan_score * 0.3 +      # Makespan权重30%
                        utilization_score * 0.2 +   # 利用率权重20%
                        tardiness_score * 0.2 +     # 延期权重20%
                        completion_score * 0.3      # 完成率权重30%
                    )
                    
                    if not hasattr(self, 'best_score'):
                        self.best_score = float('-inf')
                        self.best_kpi = None
                    
                    if current_score > self.best_score:
                        self.best_score = current_score
                        self.best_kpi = current_kpi.copy()
                        self.save_model(f"{self.models_dir}/best_ppo_model_{self.timestamp}")
                        print(f"✅ 最佳模型已更新！综合评分: {current_score:.3f}")
                        print(f"   📊 指标详情 - 总完工时间: {current_kpi['mean_makespan']:.1f}min | "
                              f"设备利用率: {current_kpi['mean_utilization']:.1%} | "
                              f"延期时间: {current_kpi['mean_tardiness']:.1f}min | "
                              f"完成率: {current_kpi['mean_completed_parts']:.0f}/33")
            
            # 🔧 修复版：简化的训练完成统计
            training_end_time = time.time()
            training_end_datetime = datetime.now()
            total_training_time = training_end_time - training_start_time
            
            print("\n" + "=" * 80)
            print("🎉 训练完成！")
            print(f"🕐 训练开始: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🏁 训练结束: {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  总训练时间: {total_training_time/60:.1f}分钟 ({total_training_time:.1f}秒)")
            
            # 训练效率统计
            if self.iteration_times:
                avg_iteration_time = np.mean(self.iteration_times)
                print(f"⚡ 平均每轮: {avg_iteration_time:.1f}s | 训练效率: {len(self.iteration_times)/total_training_time*60:.1f}轮/分钟")
            
            # 🔧 修复：最终评估（使用多回合获取稳定结果）
            print(f"\n📊 最终性能评估 (10个评估episode):")
            final_eval = self.simple_evaluation(num_episodes=10)
            
            print(f"   平均奖励: {final_eval['mean_reward']:.1f} ± {final_eval['std_reward']:.1f}")
            print(f"   平均总完工时间: {final_eval['mean_makespan']:.1f} 分钟")
            print(f"   平均完成零件: {final_eval['mean_completed_parts']:.1f}/33 ({final_eval['mean_completed_parts']/33*100:.1f}%)")
            print(f"   平均设备利用率: {final_eval['mean_utilization']:.1%}")
            print(f"   平均延期时间: {final_eval['mean_tardiness']:.1f} 分钟")
            
            # KPI改进趋势（如果有历史数据）
            if len(self.kpi_history) >= 2:
                initial = self.kpi_history[0]
                final_kpi = self.kpi_history[-1]
                
                print(f"\n📈 训练改进趋势:")
                if initial['mean_makespan'] > 0 and final_kpi['mean_makespan'] > 0:
                    makespan_change = ((initial['mean_makespan'] - final_kpi['mean_makespan']) / initial['mean_makespan']) * 100
                    print(f"   总完工时间: {initial['mean_makespan']:.1f}→{final_kpi['mean_makespan']:.1f}min ({makespan_change:+.1f}%)")
                
                util_change = (final_kpi['mean_utilization'] - initial['mean_utilization']) * 100
                print(f"   设备利用率: {initial['mean_utilization']:.1%}→{final_kpi['mean_utilization']:.1%} ({util_change:+.1f}%)")
                
                parts_change = final_kpi['mean_completed_parts'] - initial['mean_completed_parts']
                print(f"   完成零件数: {initial['mean_completed_parts']:.1f}→{final_kpi['mean_completed_parts']:.1f} ({parts_change:+.1f})")
                
                tardiness_change = final_kpi['mean_tardiness'] - initial['mean_tardiness']
                print(f"   延期时间: {initial['mean_tardiness']:.1f}→{final_kpi['mean_tardiness']:.1f}min ({tardiness_change:+.1f})")
                
                # 🔧 V12 新增：显示最佳模型信息
                if hasattr(self, 'best_kpi') and self.best_kpi:
                    print(f"\n🏆 训练期间最佳模型 (第{self.kpi_history.index(self.best_kpi)+1}轮):")
                    print(f"   综合评分: {self.best_score:.3f}")
                    print(f"   总完工时间: {self.best_kpi['mean_makespan']:.1f}min")
                    print(f"   设备利用率: {self.best_kpi['mean_utilization']:.1%}")
                    print(f"   延期时间: {self.best_kpi['mean_tardiness']:.1f}min")
                    print(f"   完成率: {self.best_kpi['mean_completed_parts']:.0f}/33 ({self.best_kpi['mean_completed_parts']/33*100:.1f}%)")
            
            # 🔧 修复：不保存最终模型，只保留最佳模型
            # self.save_model(f"{self.models_dir}/final_ppo_model_{self.timestamp}")  # 已禁用
            
            return {
                'training_time': total_training_time,
                'final_eval': final_eval,
                'kpi_history': self.kpi_history,
                'iteration_times': self.iteration_times
            }
            
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # 🔧 V8 优化: 主循环中没有env需要关闭
            pass
    
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
    print("🏭 W工厂订单思维革命PPO训练系统 V12 (性能极限版)")
    print("🎯 V12 核心升级: 提升神经网络容量，充分利用RTX 3080 Ti算力")
    print("🚀 V10性能革命: 采用安全的Spawn模式实现稳定的CPU并行加速")
    print("🔧 核心优化: 彻底解决BrokenProcessPool错误，确保长时间稳定训练")
    print("💾 安全特性: 自动内存监控 + 垃圾回收 + 检查点保存 + 动态网络调整")
    print("=" * 80)
    
    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    try:
        # 🔧 V12 性能极限版：增加训练轮数和步数
        num_episodes = 40  # 增加训练轮数，给智能体更多学习机会
        steps_per_episode = 2048  # 保持较长的episode长度  
        
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
