"""
WSL Ubuntu专用的Ray RLlib多智能体强化学习训练脚本
针对WSL环境优化，解决Windows Ray兼容性问题
"""

import os
import sys
import time
import json
import tempfile
import subprocess
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# 🕐 脚本真实启动时间（第一行代码执行）
SCRIPT_START_TIME = time.time()
SCRIPT_START_DATETIME = datetime.now()
print(f"🕐 脚本启动时间: {SCRIPT_START_DATETIME.strftime('%Y-%m-%d %H:%M:%S')}")

# WSL环境优化设置
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'
os.environ['RAY_DISABLE_PYARROW_TENSOR_EXTENSION'] = '1'
# 禁用一些可能在WSL中有问题的功能
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
os.environ['RAY_USAGE_STATS_ENABLED'] = '0'

print("🐧 WSL Ubuntu环境检测...")

# 检查是否在WSL环境中
def check_wsl_environment():
    """检查WSL环境"""
    try:
        # 检查/proc/version文件
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            if 'microsoft' in version_info or 'wsl' in version_info:
                print("✅ 检测到WSL环境")
                return True
        
        # 检查环境变量
        if 'WSL_DISTRO_NAME' in os.environ:
            print("✅ 检测到WSL环境 (通过环境变量)")
            return True
            
        print("⚠️  未检测到WSL环境，但继续执行...")
        return False
        
    except Exception as e:
        print(f"⚠️  WSL检测失败: {e}")
        return False

# 检查WSL环境
is_wsl = check_wsl_environment()

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.env import PettingZooEnv, MultiAgentEnv  # 🔧 添加MultiAgentEnv导入
    from ray.tune.registry import register_env
    import numpy as np
    import gymnasium as gym
    print("✅ Ray库导入成功")
except ImportError as e:
    print(f"❌ Ray库导入失败: {e}")
    print("请在WSL中安装Ray:")
    print("pip install ray[rllib] gymnasium pettingzoo")
    sys.exit(1)

# 添加环境路径 - WSL路径处理
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent  # 上一级目录，包含environments
sys.path.append(str(current_dir))
sys.path.append(str(parent_dir))

print(f"🔍 脚本目录: {current_dir}")
print(f"🔍 项目根目录: {parent_dir}")
print(f"🔍 查找environments目录: {parent_dir / 'environments'}")

try:
    from environments.w_factory_env import WFactoryGymEnv
    from environments.w_factory_config import *
    print("✅ 工厂环境导入成功")
except ImportError as e:
    print(f"❌ 工厂环境导入失败: {e}")
    print(f"请确保environments目录存在于: {parent_dir}")
    sys.exit(1)

# 🔧 集成V3版本的成功包装器 + 继承修复
class OptimizedWFactoryWrapper(MultiAgentEnv):
    """优化版工厂环境包装器 - 基于V3成功版本 + 正确继承MultiAgentEnv"""
    
    def __init__(self, config=None):
        super().__init__()  # 🔧 关键修复：调用MultiAgentEnv的初始化
        self.config = config or {}
        
        # 🔧 确保使用修复后的奖励配置
        env_config = self.config.copy()
        env_config.update({
            'debug_level': 'INFO',         # 🔧 显示关键信息用于诊断
            'training_mode': False,        # 🔧 修复：改为False以显示详细统计
            'use_fixed_rewards': True,     # 🔧 新增：使用修复后的奖励系统
            'show_completion_stats': True  # 🔧 V5新增：显示完成统计
        })
        
        self.base_env = WFactoryGymEnv(env_config)
        
        # 获取智能体列表
        self.agents = list(self.base_env.possible_agents)
        self._agent_ids = set(self.agents)
        
        # 设置观测和动作空间
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.observation_spaces = self.base_env.observation_spaces
        self.action_spaces = self.base_env.action_spaces
        
        # 🔧 修复：恢复与仿真时间匹配的episode长度
        self.max_episode_steps = 480   # 🔧 恢复到480，匹配SIMULATION_TIME
        self.current_step = 0
        
        # 🔧 V5新增：自然终止优先标志
        self.prefer_natural_termination = True
        
        # 🔧 完全移除人为奖励阈值 - 让智能体面对真实挑战
        self.episode_reward_threshold = None  # 不设置任何人为成功标准
        self.cumulative_reward = 0.0
        
        # 🔧 大幅放宽无进展检测 - 真实学习需要更多探索时间
        self.consecutive_no_progress_steps = 0
        self.max_no_progress_steps = 500  # 大幅增加，给智能体充分探索机会
        
        # 训练模式控制（减少输出）
        self.training_mode = self.config.get('training_mode', True)
        
        # 🔧 显示修复状态（仅在非训练模式下）
        if not self.training_mode:
            print(f"🔧 优化版环境包装器初始化:")
            print(f"   继承类: {self.__class__.__bases__}")
            print(f"   智能体数量: {len(self.agents)}")
            print(f"   最大episode步数: {self.max_episode_steps}")
            print(f"   无进展检测: {self.max_no_progress_steps}步")
            print(f"   奖励阈值: 已完全移除 (面对真实挑战)")
            print(f"   🎯 使用修复后奖励系统: completion_reward={REWARD_CONFIG['completion_reward']}")
            print(f"   🎯 奖励分配: 完成奖励只给包装台智能体")
        
    def reset(self, *, seed=None, options=None):
        """重置环境"""
        try:
            # 重置计数器
            self.current_step = 0
            self.cumulative_reward = 0.0
            self.consecutive_no_progress_steps = 0
            
            obs, info = self.base_env.reset(seed=seed, options=options)
            
            # 确保返回正确格式
            if isinstance(obs, dict):
                return obs, info
            else:
                multi_obs = {agent: obs for agent in self.agents}
                return multi_obs, info
                
        except Exception as e:
            if not self.training_mode:
                print(f"❌ 环境重置失败: {e}")
            # 返回默认观测
            default_obs = np.zeros(2, dtype=np.float32)
            multi_obs = {agent: default_obs for agent in self.agents}
            return multi_obs, {agent: {} for agent in self.agents}
    
    def step(self, actions):
        """执行动作 - 修复版"""
        try:
            self.current_step += 1
            
            # 检查动作格式
            if isinstance(actions, dict):
                processed_actions = actions
            else:
                processed_actions = {agent: actions for agent in self.agents}
            
            # 调用基础环境
            obs, rewards, terminated, truncated, info = self.base_env.step(processed_actions)
            
            # 使用环境原生奖励，但添加进度检测
            if isinstance(rewards, dict):
                step_reward = sum(rewards.values())
                self.cumulative_reward += step_reward
                
                # 检查是否有进度
                if step_reward > 0.2:
                    self.consecutive_no_progress_steps = 0
                else:
                    self.consecutive_no_progress_steps += 1
            
            # 🔧 V5修复：优先检查自然终止，而不是步数限制
            natural_done = False
            if hasattr(self.base_env, 'pz_env') and hasattr(self.base_env.pz_env, 'sim'):
                sim = self.base_env.pz_env.sim
                if sim:
                    natural_done = sim.is_done()
            elif hasattr(self.base_env, 'sim') and self.base_env.sim:
                natural_done = self.base_env.sim.is_done()
            
            step_limit_reached = self.current_step >= self.max_episode_steps
            
            # 终止条件：优先自然终止
            if natural_done:
                terminated = {agent: True for agent in self.agents}
                terminated['__all__'] = True
                if not self.training_mode:
                    print(f"   🏁 Episode自然终止于第{self.current_step}步 (任务完成)!")
            elif step_limit_reached:
                terminated = {agent: True for agent in self.agents}
                terminated['__all__'] = True
                if not self.training_mode:
                    print(f"   ⏰ Episode步数限制终止于第{self.current_step}步!")
            else:
                terminated = {agent: False for agent in self.agents}
                terminated['__all__'] = False
            
            # 确保其他返回值格式正确
            if not isinstance(obs, dict):
                obs = {agent: obs for agent in self.agents}
            if not isinstance(rewards, dict):
                rewards = {agent: rewards for agent in self.agents}
            if not isinstance(info, dict):
                info = {agent: info for agent in self.agents}
            
            return obs, rewards, terminated, truncated, info
            
        except Exception as e:
            if not self.training_mode:
                print(f"❌ 环境步进失败: {e}")
            
            # 返回默认值，强制终止
            default_obs = np.zeros(2, dtype=np.float32)
            terminated_dict = {agent: True for agent in self.agents}
            terminated_dict['__all__'] = True
            truncated_dict = {agent: False for agent in self.agents}
            truncated_dict['__all__'] = False
            
            return (
                {agent: default_obs for agent in self.agents},
                {agent: 0.0 for agent in self.agents},
                terminated_dict,
                truncated_dict,
                {agent: {} for agent in self.agents}
            )
    
    # Ray RLlib 兼容性方法
    def get_agent_ids(self):
        return self._agent_ids
    
    def get_observation_space(self, agent_id=None):
        if agent_id is None:
            return self.observation_spaces
        return self.observation_spaces.get(agent_id)
    
    def get_action_space(self, agent_id=None):
        if agent_id is None:
            return self.action_spaces
        return self.action_spaces.get(agent_id)

def env_creator(config):
    """环境创建函数 - 使用优化版包装器"""
    return OptimizedWFactoryWrapper(config)

# 注册环境
register_env("w_factory", env_creator)

def get_wsl_system_info():
    """获取WSL系统信息"""
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.total // (1024 * 1024)
        
        return {
            "cpu_count": cpu_count,
            "memory_mb": memory_mb,
            "available_memory_mb": memory_info.available // (1024 * 1024)
        }
    except ImportError:
        # 如果没有psutil，使用默认值
        import os
        cpu_count = os.cpu_count() or 4
        return {
            "cpu_count": cpu_count,
            "memory_mb": 4096,  # 默认4GB
            "available_memory_mb": 2048
        }

def get_wsl_ray_config():
    """获取WSL优化的Ray配置"""
    system_info = get_wsl_system_info()
    
    # 根据系统资源动态调整
    cpu_count = system_info.get("cpu_count", 4)
    memory_mb = system_info.get("memory_mb", 4096)
    
    # WSL环境下的保守配置
    num_cpus = min(cpu_count, 6)  # 限制CPU使用
    object_store_memory = min(memory_mb * 1024 * 1024 // 4, 500_000_000)  # 1/4内存或500MB
    
    print(f"🔧 WSL系统配置:")
    print(f"   CPU核心: {cpu_count} (使用: {num_cpus})")
    print(f"   内存: {memory_mb}MB (对象存储: {object_store_memory//1024//1024}MB)")
    
    # 创建WSL友好的临时目录
    temp_dir = tempfile.mkdtemp(prefix="ray_wsl_")
    
    return {
        "local_mode": False,
        "ignore_reinit_error": True,
        "include_dashboard": False,  # WSL中禁用dashboard
        "_temp_dir": temp_dir,
        "object_store_memory": object_store_memory,
        "num_cpus": num_cpus,
        # WSL特定配置
        "log_to_driver": True,
        "configure_logging": True,
        "logging_level": "ERROR",
    }

def create_ray_config():
    """创建Ray RLlib配置 - Ray 2.48.0兼容版本 + 奖励修复集成"""
    config = (
        PPOConfig()
        .environment(
            env="w_factory",
            env_config={
                'debug_level': 'WARNING',      # 减少环境输出
                'training_mode': True,         # 启用训练模式
                'use_fixed_rewards': True,     # 🔧 使用修复后的奖励系统
                'show_completion_stats': True  # 🔧 V5新增：显示完成统计
            }
        )
        .framework("torch")
        .api_stack(
            # 禁用新API栈，使用旧版本兼容模式
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            train_batch_size=1000,  # 🔧 修复：减少批次大小，避免过度训练
            minibatch_size=64,      # 🔧 修复：减少小批次大小
            num_epochs=3,           # 🔧 修复：减少epoch数
            lr=3e-4,               # 🔧 修复：降低学习率
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
        )
        .env_runners(
            # 🔧 V3修复：使用本地模式避免序列化问题
            num_env_runners=0,  # 本地模式，避免Ray worker序列化问题
            rollout_fragment_length=TRAINING_CONFIG["rollout_fragment_length"],  # 使用配置文件的200
            batch_mode="complete_episodes",  # 使用完整episode
        )
        .resources(
            num_gpus=0,
        )
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=5,
        )
        .debugging(
            log_level="WARNING",  # 减少Ray日志
        )
    )
    
    print(f"🔧 Ray配置已更新:")
    print(f"   🎯 奖励系统: 修复版 (completion_reward={REWARD_CONFIG['completion_reward']})")
    print(f"   🎯 奖励分配: 智能分配机制 (完成奖励只给包装台)")
    print(f"   📊 训练批次: {config.train_batch_size}")
    print(f"   🎮 Episode模式: complete_episodes")
    
    return config

def get_wsl_training_config():
    """获取WSL优化的训练配置"""
    system_info = get_wsl_system_info()
    cpu_count = system_info.get("cpu_count", 4)
    
    # 根据CPU数量调整worker数量
    num_workers = max(1, min(cpu_count - 1, 4))  # 保留1个CPU给主进程
    
    config = (
        PPOConfig()
        .environment(
            env="w_factory",  # 🔧 使用修复版环境
            env_config={},
            disable_env_checking=True
        )
        .framework("torch") 
        .env_runners(
            # 本地模式配置 (避免环境注册问题)
            num_env_runners=0,  # 本地模式不使用远程runner
            rollout_fragment_length=500,  # 增加rollout长度
            batch_mode="complete_episodes",  # 🔧 修复：改为完整episode模式，避免强制截断
        )
        .training(
            # PPO训练参数 (Ray 2.48 API)
            train_batch_size=4000,  # 增加训练批次大小
            lr=3e-4,
            gamma=0.99,
        )
        .multi_agent(
            # 多智能体配置 (按照main.py模式)
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: "shared_policy"),
        )
        .resources(
            # WSL资源配置 (Ray 2.48 API)
            num_gpus=0  # WSL通常不支持GPU，移除过时参数
        )
        .evaluation(
            # 评估配置 (简化版本)
            evaluation_interval=25,
            evaluation_duration=10,
            evaluation_config={
                "explore": False,
                "render_env": False,
            }
        )
        .debugging(
            # WSL调试配置
            log_level="INFO",  # WSL中可以显示更多日志
        )
        .experimental(
            # 实验性配置
            _disable_preprocessor_api=True,
        )
    )
    
    # 设置PPO特定的超参数 (Ray 2.48方式)
    config.lambda_ = 0.95
    config.clip_param = 0.2
    config.vf_loss_coeff = 0.5
    config.entropy_coeff = 0.01
    config.minibatch_size = 128  # Ray 2.48.0中的正确参数名
    config.num_sgd_iter = 10
    config.horizon = 2000  # 🔧 匹配环境的最大步数，让完整生产周期有机会完成
    
    print(f"🔧 训练配置:")
    print(f"   模式: 本地模式 (避免环境注册问题)")
    print(f"   Env Runners: 0 (本地模式)")
    print(f"   训练批次大小: 4000")
    print(f"   Episode长度: 2000步 (让完整生产周期完成)")
    print(f"   Rollout长度: 500步")
    print(f"   SGD迭代次数: 10")
    print(f"   SGD小批次大小: 128")
    
    return config

def run_wsl_ray_training(num_iterations=20):
    """运行WSL Ray RLlib训练"""
    print("🚀 开始WSL Ray RLlib训练...")
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    try:
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=4,
                num_gpus=0,
                object_store_memory=1000000000,  # 1GB
                ignore_reinit_error=True,
                log_to_driver=False,  # 减少日志输出
            )
        
        # 注册环境 - 使用优化版包装器
        register_env("w_factory", env_creator)
        
        # 创建配置
        config = create_ray_config()
        
        # 创建算法
        algo = config.build()
        
        # 创建检查点目录 - WSL路径格式
        checkpoint_dir = "/mnt/d/MPU/毕业论文/MARL_FOR_W_Factory/wsl/ray_result"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练循环
        print(f"🎯 开始训练 {num_iterations} 轮...")
        print(f"⏰ 训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")
        print("=" * 70)
        
        best_reward = float('-inf')
        best_checkpoint = None
        total_episodes_completed = 0
        iteration_times = []  # 记录每轮训练时间
        
        for i in range(num_iterations):
            iteration_start_time = time.time()
            
            print(f"\n📊 训练轮次 {i+1}/{num_iterations}")
            print("-" * 50)
            
            result = algo.train()
            
            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_times.append(iteration_duration)
            
            # 获取关键指标 - Ray 2.48.0兼容
            episode_reward_mean = result.get("episode_reward_mean", 0)
            episodes_this_iter = result.get("episodes_this_iter", 0)
            episode_len_mean = result.get("episode_len_mean", 0)
            
            # Ray 2.48.0中统计数据可能在env_runners中
            if episodes_this_iter == 0 and 'env_runners' in result:
                env_stats = result['env_runners']
                episode_reward_mean = env_stats.get("episode_reward_mean", episode_reward_mean)
                episodes_this_iter = env_stats.get("episodes_this_iter", episodes_this_iter)
                episode_len_mean = env_stats.get("episode_len_mean", episode_len_mean)
            
            total_episodes_completed += episodes_this_iter
            
            # 显示训练结果
            print(f"   平均奖励: {episode_reward_mean:.2f}")
            print(f"   完成episode数: {episodes_this_iter}")
            print(f"   平均episode长度: {episode_len_mean:.1f}")
            print(f"   累计完成episode: {total_episodes_completed}")
            
            # 检查是否有改进
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                print(f"   🎉 新的最佳奖励: {best_reward:.2f}")
                
                # 保存检查点 - 只显示简洁信息
                best_checkpoint = algo.save(checkpoint_dir)
                if hasattr(best_checkpoint, 'path'):
                    checkpoint_path = best_checkpoint.path
                else:
                    # 从字符串中提取路径
                    checkpoint_str = str(best_checkpoint)
                    if 'path=' in checkpoint_str:
                        path_start = checkpoint_str.find('path=') + 5
                        path_end = checkpoint_str.find(')', path_start)
                        if path_end == -1:
                            path_end = checkpoint_str.find(',', path_start)
                        checkpoint_path = checkpoint_str[path_start:path_end]
                    else:
                        checkpoint_path = "wsl_ray_results/checkpoints"
                
                print(f"   💾 检查点已保存: {checkpoint_path}")
            else:
                print(f"   📊 当前奖励: {episode_reward_mean:.2f} (最佳: {best_reward:.2f})")
            
            # 显示学习进度
            if "info" in result and "learner" in result["info"]:
                learner_info = result["info"]["learner"]["shared_policy"]
                if "learner_stats" in learner_info:
                    stats = learner_info["learner_stats"]
                    policy_loss = stats.get("policy_loss", 0)
                    vf_loss = stats.get("vf_loss", 0)
                    print(f"   策略损失: {policy_loss:.4f}, 价值损失: {vf_loss:.4f}")
            
            # 🔧 核心KPI监控：简化版本，避免Ray API兼容性问题
            try:
                # 显示基本训练统计
                print(f"   📈 训练进度: {i+1}/{num_iterations} ({(i+1)/num_iterations*100:.1f}%)")
                
                # 尝试从result中获取基础KPI信息
                if hasattr(result, 'info') and result.info:
                    episode_info = result.info.get('episode', {})
                    if episode_info:
                        episode_len = episode_info.get('len', episode_len_mean)
                        episode_reward = episode_info.get('reward', episode_reward_mean)
                        print(f"   📊 Episode信息: 长度={episode_len:.1f}, 奖励={episode_reward:.1f}")
                        
                        # 计算自然终止率
                        if episode_len < 480:
                            natural_rate = ((480 - episode_len) / 480) * 100
                            print(f"   🎯 自然终止率: {natural_rate:.1f}% (提前{480-episode_len:.1f}步完成)")
                        else:
                            print(f"   ⏰ 达到最大步数限制 (480步)")
                
                # 显示学习效果指标
                if episode_reward_mean > 1800:
                    print(f"   ✅ 学习效果: 良好 (奖励>{episode_reward_mean:.0f})")
                elif episode_reward_mean > 1500:
                    print(f"   ⚠️  学习效果: 一般 (奖励={episode_reward_mean:.0f})")
                else:
                    print(f"   ❌ 学习效果: 需改进 (奖励={episode_reward_mean:.0f})")
                
            except Exception as e:
                print(f"   ⚠️  KPI显示失败: {e}")
                print(f"   📈 训练进度: {i+1}/{num_iterations} ({(i+1)/num_iterations*100:.1f}%)")
            
            # 时间统计和预测
            elapsed_time = time.time() - training_start_time
            avg_iteration_time = sum(iteration_times) / len(iteration_times)
            remaining_iterations = num_iterations - (i + 1)
            estimated_remaining_time = remaining_iterations * avg_iteration_time
            
            print(f"   ⏱️  本轮用时: {iteration_duration:.1f}秒")
            
            if remaining_iterations > 0:
                print(f"   🔮 预计剩余: {estimated_remaining_time/60:.1f}分钟")
                estimated_finish_time = time.time() + estimated_remaining_time
                finish_time_str = time.strftime('%H:%M:%S', time.localtime(estimated_finish_time))
                print(f"   🏁 预计完成: {finish_time_str}")
            
            # 如果没有完成任何episode，给出提示
            if episodes_this_iter == 0:
                print("   ⏳ 本轮未完成episode，继续训练...")
        
        # 训练完成统计
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        print("\n" + "=" * 70)
        print(f"🏁 训练完成！")
        print(f"   最佳平均奖励: {best_reward:.2f}")
        print(f"   总完成episode数: {total_episodes_completed}")
        print(f"   总训练时间: {total_training_time/60:.1f}分钟 ({total_training_time:.1f}秒)")
        print(f"   平均每轮时间: {total_training_time/num_iterations:.1f}秒")
        print(f"   最快单轮: {min(iteration_times):.1f}秒")
        print(f"   最慢单轮: {max(iteration_times):.1f}秒")
        
        if total_episodes_completed == 0:
            print("⚠️  警告: 训练期间没有完成任何episode")
            print("💡 建议: 增加训练轮次或调整环境参数")
        
        # 创建最佳结果对象
        class BestResult:
            def __init__(self, reward, checkpoint, training_time, iteration_times):
                self.metrics = {
                    "episode_reward_mean": reward, 
                    "training_iteration": num_iterations,
                    "total_training_time": training_time,
                    "avg_iteration_time": sum(iteration_times) / len(iteration_times),
                    "total_episodes": total_episodes_completed
                }
                self.checkpoint = checkpoint
        
        # 如果没有保存过检查点，保存最后一个
        if best_checkpoint is None:
            best_checkpoint = algo.save(checkpoint_dir)
            print(f"💾 保存最终检查点: {best_checkpoint}")
        
        best_result = BestResult(best_reward, best_checkpoint, total_training_time, iteration_times)
        
        # 清理
        algo.stop()
        
        return best_result
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_wsl_setup_script():
    """创建WSL环境设置脚本"""
    setup_script = """#!/bin/bash
# WSL Ubuntu环境设置脚本

echo "🐧 设置WSL Ubuntu环境用于Ray MARL训练"

# 更新系统
echo "📦 更新系统包..."
sudo apt update && sudo apt upgrade -y

# 安装Python和pip
echo "🐍 安装Python环境..."
sudo apt install -y python3 python3-pip python3-venv

# 创建虚拟环境
echo "🔧 创建Python虚拟环境..."
python3 -m venv marl_env
source marl_env/bin/activate

# 安装依赖
echo "📚 安装Python依赖..."
pip install --upgrade pip
pip install ray[rllib]
pip install gymnasium
pip install pettingzoo
pip install simpy
pip install numpy
pip install tensorflow

echo "✅ WSL环境设置完成！"
echo "使用方法:"
echo "1. 激活环境: source marl_env/bin/activate"
echo "2. 运行训练: python3 wsl_ray_marl_train.py"
"""
    
    setup_file = Path("setup_wsl_env.sh")
    with open(setup_file, 'w') as f:
        f.write(setup_script)
    
    # 设置执行权限
    os.chmod(setup_file, 0o755)
    
    print(f"📄 WSL设置脚本已创建: {setup_file}")
    return setup_file

def main():
    """主函数"""
    # 记录脚本开始时间
    print("🐧 W工厂多智能体强化学习系统 - WSL版本")
    print("=" * 70)
    
    # 检查WSL环境
    if not is_wsl:
        print("⚠️  建议在WSL环境中运行此脚本以获得最佳性能")
    
    # 创建设置脚本
    setup_file = create_wsl_setup_script()
    
    try:
        # 运行Ray训练 - 🔧 修复：使用合理轮次避免过度训练
        ray_result = run_wsl_ray_training(num_iterations=20)  # 🔧 修复：减少到20轮，避免过度训练
        
        # 计算脚本总运行时间（使用全局启动时间）
        script_end_time = time.time()
        script_end_datetime = datetime.now()
        total_script_time = script_end_time - SCRIPT_START_TIME
        
        if ray_result:
            print("\n🎉 WSL Ray RLlib训练成功完成！")
            
            # 显示时间统计
            print(f"\n⏰ 时间统计:")
            print(f"   脚本开始: {SCRIPT_START_DATETIME.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   脚本结束: {script_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   脚本总运行时间: {total_script_time/60:.1f}分钟 ({total_script_time:.1f}秒)")
            
            # 🔧 修复：更真实的时间分析
            # 从运行日志可以看出：
            # - Ray初始化约30秒（14:45:10到14:45:41）
            # - 算法构建约1.5分钟（到14:47:29）
            # - 纯训练约9.8分钟（14:47:29到14:54:58）
            
            # 估算各阶段时间（基于实际运行观察）
            estimated_import_time = 30  # 导入和环境检测
            estimated_ray_init_time = 90  # Ray初始化和算法构建
            estimated_training_time = total_script_time - estimated_import_time - estimated_ray_init_time
            
            print(f"   导入和环境检测: ~{estimated_import_time}秒")
            print(f"   Ray初始化和算法构建: ~{estimated_ray_init_time}秒")
            print(f"   纯训练时间: ~{estimated_training_time:.1f}秒 ({estimated_training_time/60:.1f}分钟)")
            print(f"   训练效率: {estimated_training_time/total_script_time*100:.1f}%")
            
            # 🔧 自动验证已禁用，避免超时问题
            print(f"\n💡 训练完成，建议手动运行验证:")
            print(f"   🔍 推理测试: python wsl/test_trained_model_inference.py")
            print(f"   📊 性能基准: python wsl/test_performance_benchmark.py")
            
            # 显示后续步骤
            print("\n📋 后续步骤:")
            print("1. 查看训练结果: ls /mnt/d/MPU/毕业论文/MARL_FOR_W_Factory/wsl/ray_result/")
            print("2. 运行性能基准测试: python wsl/test_performance_benchmark.py")
            print("3. 手动运行推理测试: python wsl/test_trained_model_inference.py")
            print("4. 可视化训练曲线")
            
        else:
            print("\n❌ WSL Ray训练失败")
            print(f"💡 请先运行设置脚本: bash {setup_file}")
            print(f"⏰ 脚本运行时间: {total_script_time/60:.1f}分钟")
            
    except Exception as e:
        script_end_time = time.time()
        total_script_time = script_end_time - SCRIPT_START_TIME
        print(f"❌ 主程序执行失败: {e}")
        print(f"⏰ 脚本运行时间: {total_script_time/60:.1f}分钟")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 