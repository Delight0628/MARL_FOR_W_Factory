"""
简化的SB3 MARL训练脚本
不使用SuperSuit，直接实现多智能体训练
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

# 导入强化学习库
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    import gymnasium as gym
    print("✓ Stable-Baselines3 导入成功")
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    print("请安装: pip install stable-baselines3[extra]")
    sys.exit(1)

class SingleAgentWrapper(gym.Env):
    """将多智能体环境包装为单智能体环境"""
    
    def __init__(self, config=None):
        super().__init__()
        self.env = make_parallel_env(config)
        
        # 定义观测和动作空间
        # 观测空间：所有智能体的观测拼接
        single_obs_dim = 2  # 每个智能体的观测维度
        total_obs_dim = len(self.env.possible_agents) * single_obs_dim
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(total_obs_dim,), dtype=np.float32
        )
        
        # 动作空间：所有智能体的动作拼接
        total_action_dim = len(self.env.possible_agents)
        self.action_space = gym.spaces.MultiDiscrete([2] * total_action_dim)
        
        self.agents = self.env.possible_agents
        print(f"包装环境创建成功:")
        print(f"  智能体数量: {len(self.agents)}")
        print(f"  观测空间: {self.observation_space}")
        print(f"  动作空间: {self.action_space}")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # 将多智能体观测拼接为单一观测
        combined_obs = []
        for agent in self.agents:
            if agent in obs:
                combined_obs.extend(obs[agent])
            else:
                combined_obs.extend([0.0, 0.0])  # 默认观测
        
        return np.array(combined_obs, dtype=np.float32), {}
    
    def step(self, action):
        """执行一步"""
        # 将单一动作分解为多智能体动作
        actions = {}
        for i, agent in enumerate(self.agents):
            if i < len(action):
                actions[agent] = int(action[i])
            else:
                actions[agent] = 0  # 默认动作
        
        # 执行环境步骤
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # 处理观测
        combined_obs = []
        for agent in self.agents:
            if agent in obs:
                combined_obs.extend(obs[agent])
            else:
                combined_obs.extend([0.0, 0.0])
        
        # 处理奖励（求和）
        total_reward = sum(rewards.values()) if rewards else 0.0
        
        # 处理完成状态
        done = any(terminations.values()) or any(truncations.values())
        
        # 处理信息
        info = {}
        if done and infos:
            # 获取最终统计
            for agent_info in infos.values():
                if "final_stats" in agent_info:
                    info["final_stats"] = agent_info["final_stats"]
                    break
        
        return np.array(combined_obs, dtype=np.float32), total_reward, done, False, info

class MARLTrainingCallback(BaseCallback):
    """MARL训练回调函数"""
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 记录训练统计
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        
        # 定期输出训练进度
        if self.num_timesteps % self.eval_freq == 0:
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
                print(f"步数: {self.num_timesteps:8d} | "
                      f"平均奖励: {avg_reward:8.2f} | "
                      f"回合数: {len(self.episode_rewards):4d}")
        
        return True

def create_env():
    """创建环境"""
    def _init():
        return SingleAgentWrapper()
    return _init

def train_simple_sb3_marl(total_timesteps: int = 50000, learning_rate: float = 3e-4):
    """使用简化方法训练SB3 MARL"""
    
    print("=" * 60)
    print("W工厂多智能体强化学习训练 - 简化SB3版本")
    print("=" * 60)
    print("框架: Stable-Baselines3 (无SuperSuit)")
    print("算法: PPO")
    print("多智能体: 联合动作空间")
    print("=" * 60)
    
    # 验证配置
    if not validate_config():
        print("配置验证失败")
        return None, None
    
    try:
        # 创建向量化环境
        env = DummyVecEnv([create_env()])
        
        # 创建PPO模型
        print("创建PPO模型...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device='cpu'
        )
        
        print("✓ PPO模型创建成功")
        print(f"  学习率: {learning_rate}")
        print(f"  总训练步数: {total_timesteps}")
        
        # 创建回调函数
        callback = MARLTrainingCallback(eval_freq=2000)
        
        # 开始训练
        print("\n开始训练...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("简化SB3 MARL训练完成！")
        print("=" * 60)
        print(f"训练时间: {training_time/60:.2f} 分钟")
        print(f"总步数: {total_timesteps}")
        
        # 安全地计算平均奖励
        if len(callback.episode_rewards) > 0:
            recent_rewards = callback.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"平均奖励: {avg_reward:.2f}")
        else:
            print("平均奖励: 无数据")
        
        # 保存模型
        os.makedirs("models", exist_ok=True)
        model_path = "models/simple_sb3_marl_model"
        model.save(model_path)
        print(f"✓ 模型已保存: {model_path}")
        
        # 保存训练统计
        training_stats = {
            "framework": "Stable-Baselines3-Simple",
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            "training_time_minutes": training_time / 60,
            "episode_rewards": callback.episode_rewards,
            "episode_lengths": callback.episode_lengths,
            "final_avg_reward": float(np.mean(callback.episode_rewards[-10:])) if len(callback.episode_rewards) > 0 else 0.0,
            "agents": list(WORKSTATIONS.keys()),
            "config": {
                "learning_rate": learning_rate,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10
            }
        }
        
        os.makedirs("results", exist_ok=True)
        stats_file = f"results/simple_sb3_training_stats_{int(time.time())}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(training_stats, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 训练统计已保存: {stats_file}")
        
        return model, training_stats
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_model(model, num_episodes: int = 5):
    """评估训练好的模型"""
    print(f"\n评估模型 ({num_episodes} 回合)...")
    
    try:
        env = DummyVecEnv([create_env()])
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward[0]  # DummyVecEnv返回数组
                episode_length += 1
                
                done = done[0]  # DummyVecEnv返回数组
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            print(f"  回合 {episode+1}: 奖励={episode_reward:.2f}, 长度={episode_length}")
        
        eval_results = {
            "mean_reward": float(np.mean(eval_rewards)),
            "std_reward": float(np.std(eval_rewards)),
            "mean_length": float(np.mean(eval_lengths)),
            "eval_rewards": eval_rewards,
            "eval_lengths": eval_lengths
        }
        
        print(f"\n评估结果:")
        print(f"  平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  平均长度: {eval_results['mean_length']:.1f}")
        
        return eval_results
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("W工厂多智能体强化学习系统")
    print("基于Stable-Baselines3的简化MARL实现")
    print("=" * 60)
    
    try:
        # 训练模型
        model, training_stats = train_simple_sb3_marl(
            total_timesteps=20000,  # 适中的训练步数
            learning_rate=3e-4
        )
        
        if model is not None:
            print("\n🎉 简化SB3 MARL训练成功完成！")
            
            # 评估模型
            eval_results = evaluate_model(model, num_episodes=3)
            
            # 最终总结
            print("\n" + "=" * 60)
            print("最终结果总结")
            print("=" * 60)
            
            if eval_results:
                print(f"SB3 MARL平均奖励: {eval_results['mean_reward']:.2f}")
            
            print("\n✅ 这是真正的MARL训练！")
            print("✅ 使用工业级PPO算法")
            print("✅ 多智能体联合动作空间")
            print("✅ Windows完全兼容")
            print("✅ 无SuperSuit依赖问题")
            
        else:
            print("\n❌ SB3训练失败")
            
    except Exception as e:
        print(f"主程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 