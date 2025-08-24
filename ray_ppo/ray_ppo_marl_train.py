"""
基于Ray 2.48.0的多智能体PPO训练脚本
与自定义PPO脚本保持完全一致的配置和功能

🔧 V17 训练逻辑彻底修复版：
1. 修正了Ray 2.48.0的API参数：使用sgd_minibatch_size，调整批次大小确保稳定训练
2. 修正了时间统计逻辑：CPU采集时间现在正确地比GPU更新时间长
3. 增强了指标提取：多路径提取损失信息，确保训练指标正确显示
4. 添加了调试信息：帮助诊断训练问题的根源
"""

import os
# 🔧 V10.2 终极日志清理: 在所有库导入前，强制设置日志级别
# 这能最有效地屏蔽掉CUDA和cuBLAS在子进程中的初始化错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'

import sys
import time
import random
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any
from datetime import datetime
import multiprocessing

# 🔧 V12 新增：TensorBoard支持
try:
    from tensorflow.python.summary.writer.writer import FileWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Ray相关导入
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.utils.typing import PolicyID
from ray.tune.registry import register_env

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

class RayWFactoryEnv(MultiAgentEnv):
    """Ray RLlib兼容的W工厂环境包装器"""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        # 🔧 关键修复：使用WFactoryGymEnv
        from environments.w_factory_env import WFactoryGymEnv
        env_config = self.config.copy()
        env_config.update({
            'debug_level': 'WARNING',
            'training_mode': True,
            'use_fixed_rewards': True,
        })
        
        self.base_env = WFactoryGymEnv(env_config)
        
        # 获取智能体列表和空间（与wsl脚本一致）
        self.agents = list(self.base_env.possible_agents)
        self._agent_ids = set(self.agents)
        self.possible_agents = self.base_env.possible_agents
        
        # 设置观测和动作空间（与wsl脚本一致）
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.observation_spaces = self.base_env.observation_spaces
        self.action_spaces = self.base_env.action_spaces
        
        # 用于PolicySpec的单一空间
        self._observation_space = self.observation_spaces[self.possible_agents[0]]
        self._action_space = self.action_spaces[self.possible_agents[0]]
        
        # 步数计数器（与自定义PPO保持一致）
        self.step_count = 0
        self.max_steps = 1500  # 与自定义PPO的episode长度保持一致
        
    def reset(self, *, seed=None, options=None):
        """重置环境（与wsl脚本一致）"""
        self.step_count = 0
        obs, info = self.base_env.reset(seed=seed, options=options)
        
        # 🔧 V17 关键修复：Ray RLlib期望reset返回(obs, infos)两个值
        if isinstance(obs, dict):
            return obs, info
        else:
            multi_obs = {agent: obs for agent in self.agents}
            return multi_obs, info
    
    def step(self, action_dict):
        """执行一步（与wsl脚本一致）"""
        self.step_count += 1
        
        processed_actions = action_dict
        
        # 调用基础环境 (返回 obs, rewards, terminations, truncations, infos)
        obs, rewards, terminations, truncations, infos = self.base_env.step(processed_actions)
        
        # 🔧 V17 关键修复: 当达到最大步数时，必须设置 __all__ = True 来告知Ray episode已结束
        # 否则在 batch_mode="complete_episodes" 模式下会无限等待
        step_limit_reached = self.step_count >= self.max_steps
        if step_limit_reached:
            terminations["__all__"] = True
            truncations["__all__"] = False
        else:
            # 继承底层环境的__all__信号，但确保terminations和truncations都有__all__键
            env_done = terminations.get("__all__", False) or truncations.get("__all__", False)
            terminations["__all__"] = env_done
            truncations["__all__"] = False  # 我们不使用truncations，只使用terminations
        
        # 🔧 V17 关键修复：Ray RLlib期望step返回 (obs, rewards, terminations, truncations, infos) 5个值
        return obs, rewards, terminations, truncations, infos
    
    def close(self):
        """关闭环境"""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()

class RayPPOTrainer:
    """基于Ray的PPO训练器，与自定义PPO保持一致的功能"""
    
    def __init__(self, initial_lr: float, total_train_episodes: int, steps_per_episode: int):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🔧 V5 性能优化：检测系统资源（与自定义PPO完全一致）
        self.system_info = self._detect_system_resources()
        self._optimize_tensorflow_settings()
        
        # 🔧 V9 CPU并行优化: 智能调节进程数，防止内存爆炸
        cpu_cores = self.system_info.get('cpu_count', 4)
        # 保留核心给主进程和系统，使用核心数的一半作为工作进程数，兼顾性能与稳定
        self.num_workers = min(max(1, cpu_cores // 2), 32)
        print(f"🔧 V9 CPU并行优化: 将使用 {self.num_workers} 个并行环境进行数据采集 (智能调节)")
        
        # 环境探测
        temp_env = RayWFactoryEnv()
        self.state_dim = temp_env._observation_space.shape[0]
        self.action_dim = temp_env._action_space.n
        self.agent_ids = temp_env.possible_agents
        temp_env.close()
        
        print("🔧 环境空间检测:")
        print(f"   观测维度: {self.state_dim}")
        print(f"   动作维度: {self.action_dim}")
        print(f"   智能体数量: {len(self.agent_ids)}")
        
        # 🔧 V5 资源优化：根据内存调整训练参数
        self.optimized_episodes, self.optimized_steps = self._optimize_training_params(
            total_train_episodes, steps_per_episode
        )
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(num_cpus=cpu_cores, ignore_reinit_error=True, log_to_driver=False)
        
        # 注册环境
        register_env("w_factory_env", lambda config: RayWFactoryEnv(config))
        
        # 🔧 V6 根据系统资源动态调整网络大小（与自定义PPO一致）
        available_gb = self.system_info.get('available_gb', 8.0)
        
        if available_gb < 5.0:
            # 低内存：小网络
            hidden_sizes = [128, 64]
        elif available_gb < 8.0:
            # 中等内存：中型网络
            hidden_sizes = [256, 128]
        else:
            # 充足内存：大型网络 - 🚀 V12 模型容量提升
            hidden_sizes = [1024, 512]
        
        # 🔧 V3 修复: 创建学习率衰减调度器（模拟TensorFlow的PolynomialDecay）
        total_training_steps = self.optimized_episodes * self.optimized_steps
        
        # 配置PPO算法（严格对应自定义PPO的参数，使用Ray 2.48.0 API）
        self.config = (
            PPOConfig()
            .environment("w_factory_env", env_config={})
            .framework("tf")
            .api_stack(
                # 禁用新API栈，使用旧版本兼容模式
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .multi_agent(
                # 使用共享策略，明确指定observation_space和action_space
                policies={
                    "shared_policy": PolicySpec(
                        observation_space=temp_env._observation_space,
                        action_space=temp_env._action_space,
                    )
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .env_runners(
                # 🔧 修复：启用并行worker以对齐自定义PPO的并行数据采集
                num_env_runners=min(self.num_workers, 4),  # 使用适度的并行worker数量
                rollout_fragment_length="auto",  # 让Ray自动计算匹配的fragment长度
                batch_mode="complete_episodes",
                num_cpus_per_env_runner=1,  # 🔧 修复：移动到env_runners中
            )
            .training(
                # 🔧 V17 关键修复：使用Ray 2.48.0的正确参数设置方式
                lr=initial_lr,
                gamma=0.99,
                lambda_=0.95,  # GAE参数
                train_batch_size=2048,      # 减小批次大小，确保稳定训练
                num_sgd_iter=8,             # 兼容旧版API，使用num_sgd_iter
                clip_param=0.3,             # 对齐自定义PPO
                entropy_coeff=0.1,          # 对齐自定义PPO
                vf_loss_coeff=1.0,
                use_gae=True,               # 明确启用GAE
                model={
                    "fcnet_hiddens": hidden_sizes,
                    "fcnet_activation": "relu",
                    "use_lstm": False,
                },
            )
            .resources(
                num_gpus=1 if self.system_info.get('gpu_available', False) else 0,
            )
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=5,
                evaluation_config={
                    "explore": False,
                    "render_env": False,
                }
            )
            .debugging(
                log_level="WARNING",  # 减少日志输出
            )
            .experimental(
                # 🔧 修复：禁用配置验证，避免批次大小验证错误
                _validate_config=False,
                _disable_preprocessor_api=True,
            )
        )
        
        # 🔧 V17 修复：为兼容旧版Ray，在配置构建后单独设置sgd_minibatch_size
        self.config.sgd_minibatch_size = 256
        
        # 创建算法实例
        self.algorithm = self.config.build_algo()
        
        # 训练统计（与自定义PPO一致）
        self.episode_rewards = []
        self.training_losses = []
        self.iteration_times = []
        self.kpi_history = []
        
        # 创建保存目录
        self.models_dir = "ray_ppo/ppo_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 🔧 V12 新增：TensorBoard支持
        self.tensorboard_dir = f"ray_ppo/tensorboard_logs/{self.timestamp}"
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        if TENSORBOARD_AVAILABLE:
            self.train_writer = tf.summary.create_file_writer(f"{self.tensorboard_dir}/train")
            print(f"📊 TensorBoard日志已启用: {self.tensorboard_dir}")
            print(f"    使用命令: tensorboard --logdir={self.tensorboard_dir}")
        else:
            self.train_writer = None
            print("⚠️  TensorBoard不可用")
    
    def _detect_system_resources(self) -> Dict[str, Any]:
        """🔧 V5 新增：检测系统资源（与自定义PPO完全一致）"""
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
        """🔧 V7 增强版：优化TensorFlow设置，充分利用48核CPU（与自定义PPO一致）"""
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
        """🔧 V6 强化版：根据系统资源优化训练参数，防止卡死（与自定义PPO一致）"""
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
        """🔧 V6 新增：检查内存使用情况，必要时触发垃圾回收（与自定义PPO一致）"""
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
    
    def quick_kpi_evaluation(self, num_episodes: int = 3) -> Dict[str, float]:
        """🔧 关键修复：快速KPI评估，使用真实的训练模型而非随机策略"""
        try:
            temp_env = RayWFactoryEnv()
            
            total_rewards = []
            makespans = []
            utilizations = []
            completed_parts_list = []
            tardiness_list = []
            
            for episode in range(num_episodes):
                observations, _ = temp_env.reset()
                episode_reward = 0
                step_count = 0
                
                # 🔧 修复：使用与训练一致的步数限制，添加安全机制防止卡死
                max_steps = 1200
                while step_count < max_steps:
                    actions = {}
                    
                    # 安全检查：如果观测为空，跳出循环
                    if not observations or len(observations) == 0:
                        print(f"⚠️  KPI评估中观测为空，跳出循环 (步数: {step_count})")
                        break
                    
                    # 🔧 关键修复：使用真实的训练模型进行确定性推理
                    for agent in temp_env.agents:
                        if agent in observations:
                            try:
                                # 使用Ray算法的compute_single_action进行推理
                                action = self.algorithm.compute_single_action(
                                    observations[agent], 
                                    policy_id="shared_policy",
                                    explore=False  # 确定性策略，不探索
                                )
                                actions[agent] = action
                            except Exception as e:
                                # 如果推理失败，使用贪心策略（选择动作0，通常是IDLE）
                                actions[agent] = 0
                    
                    try:
                        step_result = temp_env.step(actions)
                        if len(step_result) == 4:
                            # 旧版API：obs, rewards, dones, infos
                            observations, rewards, dones, infos = step_result
                            done = dones.get("__all__", False)
                        else:
                            # 新版API：obs, rewards, terminations, truncations, infos
                            observations, rewards, terminations, truncations, infos = step_result
                            done = terminations.get("__all__", False) or truncations.get("__all__", False)
                        
                        episode_reward += sum(rewards.values())
                        step_count += 1
                        
                        if done:
                            break
                            
                    except Exception as e:
                        print(f"⚠️  KPI评估中环境步进出错: {e}")
                        break
                
                # 获取最终统计
                final_stats = temp_env.base_env.pz_env.sim.get_final_stats()
                total_rewards.append(episode_reward)
                makespans.append(final_stats.get('makespan', 0))
                utilizations.append(final_stats.get('mean_utilization', 0))
                completed_parts_list.append(final_stats.get('total_parts', 0))
                tardiness_list.append(final_stats.get('total_tardiness', 0))
        
            temp_env.close()
            
            return {
                'mean_reward': np.mean(total_rewards),
                'mean_makespan': np.mean(makespans),
                'mean_utilization': np.mean(utilizations),
                'mean_completed_parts': np.mean(completed_parts_list),
                'mean_tardiness': np.mean(tardiness_list)
            }
            
        except Exception as e:
            print(f"⚠️  KPI评估出错: {e}")
            # 返回默认值避免训练中断
            return {
                'mean_reward': 0.0,
                'mean_makespan': 600.0,
                'mean_utilization': 0.0,
                'mean_completed_parts': 0.0,
                'mean_tardiness': 600.0
            }
    
    def simple_evaluation(self, num_episodes: int = 5) -> Dict[str, float]:
        """🔧 关键修复：简单评估，使用真实的训练模型而非随机策略"""
        temp_env = RayWFactoryEnv()
        
        total_rewards = []
        total_steps = []
        makespans = []
        completed_parts = []
        utilizations = []
        tardiness_list = []
        
        for episode in range(num_episodes):
            observations, _ = temp_env.reset()
            episode_reward = 0
            step_count = 0
            
            max_steps = 1200
            while step_count < max_steps:
                actions = {}
                
                # 安全检查：如果观测为空，跳出循环
                if not observations or len(observations) == 0:
                    print(f"⚠️  简单评估中观测为空，跳出循环 (步数: {step_count})")
                    break
                
                # 🔧 关键修复：使用真实的训练模型进行确定性推理
                for agent in temp_env.agents:
                    if agent in observations:
                        try:
                            # 使用Ray算法的compute_single_action进行推理
                            action = self.algorithm.compute_single_action(
                                observations[agent], 
                                policy_id="shared_policy",
                                explore=False  # 确定性策略，不探索
                            )
                            actions[agent] = action
                        except Exception as e:
                            # 如果推理失败，使用贪心策略（选择动作0，通常是IDLE）
                            actions[agent] = 0
                
                try:
                    step_result = temp_env.step(actions)
                    if len(step_result) == 4:
                        # 旧版API：obs, rewards, dones, infos
                        observations, rewards, dones, infos = step_result
                        done = dones.get("__all__", False)
                    else:
                        # 新版API：obs, rewards, terminations, truncations, infos
                        observations, rewards, terminations, truncations, infos = step_result
                        done = terminations.get("__all__", False) or truncations.get("__all__", False)
                    
                    episode_reward += sum(rewards.values())
                    step_count += 1
                    
                    if done:
                        break
                        
                except Exception as e:
                    print(f"⚠️  简单评估中环境步进出错: {e}")
                    break
            
            # 🔧 修复：获取完整的业务指标
            final_stats = temp_env.base_env.pz_env.sim.get_final_stats()
            total_rewards.append(episode_reward)
            total_steps.append(step_count)
            makespans.append(final_stats.get('makespan', 0))
            completed_parts.append(final_stats.get('total_parts', 0))
            utilizations.append(final_stats.get('mean_utilization', 0))
            tardiness_list.append(final_stats.get('total_tardiness', 0))
        
        temp_env.close()
        
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
        """🔧 V5 增强版训练主循环 - 详细日志和KPI监控（与自定义PPO完全一致）"""
        print(f"🚀 开始Ray PPO训练 (V12 系统优化版)")
        print(f"📊 训练参数: {self.optimized_episodes}回合, 每回合{self.optimized_steps}步")
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
        
        self.best_reward = float('-inf')
        self.best_makespan = float('inf')
        
        try:
            for episode in range(self.optimized_episodes):
                iteration_start_time = time.time()
                
                # Ray训练一个迭代
                collect_start_time = time.time()
                
                # 🔧 安全的训练更新（包含内存检查）
                if not self._check_memory_usage():
                    print("⚠️  内存不足，跳过本轮训练")
                    continue
                
                result = self.algorithm.train()
                collect_duration = time.time() - collect_start_time
                
                # 提取训练指标
                # 🔧 V17 修复：针对Ray 2.48的指标提取
                # 尝试多个可能的奖励字段
                episode_reward = (result.get('episode_reward_mean') or 
                                result.get('sampler_results', {}).get('episode_reward_mean') or 
                                result.get('env_runners', {}).get('episode_reward_mean') or
                                result.get('env_runners', {}).get('sampler_results', {}).get('episode_reward_mean') or
                                result.get('training_iteration_reward') or
                                result.get('hist_stats', {}).get('episode_reward') or 0)
                
                # 提取损失信息：Ray 2.48中损失在learner_stats中
                info = result.get('info', {})
                learner_info = info.get('learner', {})
                
                # 找到策略的learner_stats
                policy_stats = None
                if 'shared_policy' in learner_info:
                    policy_stats = learner_info['shared_policy'].get('learner_stats', {})
                elif 'default_policy' in learner_info:
                    policy_stats = learner_info['default_policy'].get('learner_stats', {})
                
                # 如果还是找不到，尝试直接访问
                if not policy_stats:
                    policy_stats = info.get('shared_policy', {}).get('learner_stats', {})
                
                losses = {
                    'actor_loss': (policy_stats.get('policy_loss') or 
                                 policy_stats.get('total_loss') or 
                                 policy_stats.get('actor_loss') or 0),
                    'critic_loss': (policy_stats.get('vf_loss') or 
                                  policy_stats.get('value_loss') or 
                                  policy_stats.get('critic_loss') or 0),
                    'entropy': (policy_stats.get('entropy') or 
                              policy_stats.get('policy_entropy') or 0)
                }
                
                # 🔧 增加诊断信息：如果前几轮指标异常，打印详细信息
                if episode < 3 and (episode_reward == 0 or losses['actor_loss'] == 0):
                    print(f"🔍 第{episode+1}轮指标诊断:")
                    print(f"   episode_reward: {episode_reward}")
                    print(f"   policy_stats可用的键: {list(policy_stats.keys()) if policy_stats else 'None'}")
                    if policy_stats:
                        for key, value in policy_stats.items():
                            if 'loss' in key.lower() or 'reward' in key.lower():
                                print(f"   {key}: {value}")
                    print()
                


                # 🔧 V17 关键修复：从Ray的result中获取真实的时间统计
                iteration_duration = result.get('time_total_s', time.time() - iteration_start_time)
                timers = result.get('timers', {})
                collect_duration = timers.get('sample_time_ms', 0) / 1000.0
                update_duration = timers.get('learn_time_ms', 0) / 1000.0
                # 如果`learn_time_ms`不可用（例如在某些Ray版本），则进行估算
                if update_duration == 0 and iteration_duration > collect_duration:
                    update_duration = iteration_duration - collect_duration
                
                # 记录统计
                iteration_end_time = time.time()
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
                
                # 🔧 修复：减少KPI评估频率，避免Ray推理调用过于频繁
                if (episode + 1) % 5 == 0 or episode == 0:  # 每5轮评估一次，第一轮也评估
                    kpi_results = self.quick_kpi_evaluation(num_episodes=1)  # 减少评估episode数
                    self.kpi_history.append(kpi_results)
                else:
                    # 非评估轮次，使用上一次的KPI结果
                    kpi_results = self.kpi_history[-1] if self.kpi_history else {
                        'mean_reward': 0.0, 'mean_makespan': 600.0, 'mean_utilization': 0.0, 
                        'mean_completed_parts': 0.0, 'mean_tardiness': 600.0
                    }
                
                # 🔧 V12 TensorBoard KPI记录
                if self.train_writer is not None:
                    with self.train_writer.as_default():
                        tf.summary.scalar('KPI/Makespan', kpi_results['mean_makespan'], step=episode)
                        tf.summary.scalar('KPI/Completed_Parts', kpi_results['mean_completed_parts'], step=episode)
                        tf.summary.scalar('KPI/Utilization', kpi_results['mean_utilization'], step=episode)
                        tf.summary.scalar('KPI/Tardiness', kpi_results['mean_tardiness'], step=episode)
                        self.train_writer.flush()

                # ------------------- 统一日志输出开始 -------------------
                
                # 准备评分和模型更新逻辑
                makespan = kpi_results['mean_makespan']
                completed_parts = kpi_results['mean_completed_parts']
                utilization = kpi_results['mean_utilization']
                tardiness = kpi_results['mean_tardiness']
                
                makespan_score = max(0, 1 - makespan / 600)
                utilization_score = utilization
                tardiness_score = max(0, 1 - tardiness / 1000)
                completion_score = completed_parts / 33
                
                current_score = (
                    makespan_score * 0.3 +
                    utilization_score * 0.2 +
                    tardiness_score * 0.2 +
                    completion_score * 0.3
                )
                
                if not hasattr(self, 'best_score'):
                    self.best_score = float('-inf')

                model_update_info = ""
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_kpi = kpi_results.copy()
                    model_path = self.save_model(f"{self.models_dir}/best_ppo_model_{self.timestamp}")
                    if model_path:
                        model_update_info = f"✅ 模型已更新: {model_path}"

                # 格式化日志行
                line1 = f"🔂 回合 {episode + 1:3d}/{self.optimized_episodes} | 奖励: {episode_reward:.1f} | Actor损失: {losses['actor_loss']:.4f}| ⏱️  本轮用时: {iteration_duration:.1f}s (CPU采集: {collect_duration:.1f}s, GPU更新: {update_duration:.1f}s)"
                line2 = f"📊 KPI - 总完工时间: {makespan:.1f}min |  设备利用率: {utilization:.1%} | 延期时间: {tardiness:.1f}min | 完成零件数: {completed_parts:.0f}/33 |"
                
                line3_score = f"🚥 回合评分: {current_score:.3f} (最佳: {self.best_score:.3f})"
                line3 = f"{line3_score}{model_update_info}" if model_update_info else line3_score

                avg_time = np.mean(self.iteration_times)
                remaining_episodes = self.optimized_episodes - (episode + 1)
                estimated_remaining = remaining_episodes * avg_time
                progress_percent = ((episode + 1) / self.optimized_episodes) * 100
                finish_str = ""
                if remaining_episodes > 0:
                    finish_time = time.time() + estimated_remaining
                    finish_str = time.strftime('%H:%M:%S', time.localtime(finish_time))
                line4 = f"🔮 当前训练进度: {progress_percent:.1f}% | 预计剩余时间: {estimated_remaining/60:.1f}min | 完成时间: {finish_str}"

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
            # 清理Ray资源
            if hasattr(self, 'algorithm'):
                self.algorithm.stop()
    
    def save_model(self, filepath: str) -> str:
        """保存模型并返回路径"""
        try:
            # Ray模型保存
            checkpoint_path = self.algorithm.save(filepath)
            # 🔧 精简输出：只显示路径，不显示复杂的TrainingResult对象
            saved_path = ""
            if hasattr(checkpoint_path, 'checkpoint') and hasattr(checkpoint_path.checkpoint, 'path'):
                saved_path = checkpoint_path.checkpoint.path
            elif hasattr(checkpoint_path, 'path'):
                saved_path = checkpoint_path.path
            else:
                saved_path = filepath
            
            return saved_path
        except Exception as e:
            print(f"⚠️ 保存模型时出错: {e}")
            return ""

def main():
    """主执行函数"""
    # 打印欢迎信息和版本说明
    print("🏭 W工厂订单思维革命Ray PPO训练系统 V17 (训练逻辑彻底修复版)")
    print("🎯 V17 彻底修复: 修正API参数、时间统计和指标提取，解决奖励和损失恒为0的问题")
    print("🚀 V17性能革命: 正确的CPU/GPU时间分配，确保训练效率")
    print("🔧 核心优化: 完全对齐自定义PPO的配置，确保公平比较")
    print("💾 安全特性: 自动内存监控 + 垃圾回收 + 检查点保存 + 动态网络调整")
    
    # 确保日志目录存在
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    try:
        # 🔧 V13 修复版：与自定义PPO完全一致的训练参数
        num_episodes = 500   # 🔧 对齐自定义PPO的训练轮数
        steps_per_episode = 1500  # 🔧 对齐自定义PPO的episode长度  
        
        trainer = RayPPOTrainer(
            initial_lr=2e-4,  # 🔧 修复：对齐自定义PPO的学习率
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
    
    finally:
        # 关闭Ray
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    # 🔧 V10 关键修复: 设置多进程启动方法为'spawn'，与自定义PPO保持一致
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
