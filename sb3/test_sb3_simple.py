"""
简化的SB3 MARL测试脚本
用于验证修复后的训练是否正常工作
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加环境路径 - 回到上级目录找到environments
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 回到MARL_FOR_W_Factory目录
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

# 导入强化学习库
try:
    from stable_baselines3 import PPO
    import supersuit as ss
    print("✓ 库导入成功")
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    sys.exit(1)

def test_environment():
    """测试环境是否正常工作"""
    print("测试环境...")
    
    # 创建原始环境
    env = make_parallel_env()
    
    # 修复SuperSuit兼容性
    if not hasattr(env, 'render_mode'):
        env.render_mode = None
    
    # 测试环境基本功能
    obs, info = env.reset()
    print(f"✓ 环境重置成功")
    print(f"  智能体数量: {len(obs)}")
    print(f"  观测维度: {list(obs.values())[0].shape}")
    
    # 测试一步
    actions = {agent: 0 for agent in env.agents}  # 所有智能体都选择IDLE
    obs, rewards, terms, truncs, infos = env.step(actions)
    
    print(f"✓ 环境步进成功")
    print(f"  奖励类型: {type(list(rewards.values())[0])}")
    print(f"  奖励值: {list(rewards.values())[0]}")
    
    return env

def test_supersuit_wrapper():
    """测试SuperSuit包装器"""
    print("\n测试SuperSuit包装器...")
    
    # 创建环境
    env = make_parallel_env()
    if not hasattr(env, 'render_mode'):
        env.render_mode = None
    
    # 应用SuperSuit包装器
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    
    print(f"✓ SuperSuit包装成功")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    
    # 测试包装后的环境
    obs = env.reset()
    print(f"✓ 包装环境重置成功")
    print(f"  观测形状: {obs.shape}")
    
    # 测试一步 - SuperSuit需要动作数组
    action = env.action_space.sample()
    # 确保动作是正确的格式
    if isinstance(action, (int, np.integer)):
        action = np.array([action])
    obs, reward, done, info = env.step(action)
    
    print(f"✓ 包装环境步进成功")
    print(f"  奖励类型: {type(reward)}")
    print(f"  奖励值: {reward}")
    print(f"  完成状态类型: {type(done)}")
    print(f"  完成状态值: {done}")
    
    return env

def test_ppo_training():
    """测试PPO训练"""
    print("\n测试PPO训练...")
    
    # 创建环境
    env = make_parallel_env()
    if not hasattr(env, 'render_mode'):
        env.render_mode = None
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    
    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,  # 减少步数用于快速测试
        batch_size=32,
        n_epochs=3,
        verbose=1,
        device='cpu'
    )
    
    print("✓ PPO模型创建成功")
    
    # 短时间训练
    print("开始短时间训练...")
    model.learn(total_timesteps=2048, progress_bar=True)
    
    print("✓ 训练完成")
    
    # 测试评估
    print("测试评估...")
    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        # 确保动作格式正确
        if isinstance(action, (int, np.integer)):
            action = np.array([action])
        obs, reward, done, info = env.step(action)
        
        # 处理奖励和done状态
        if isinstance(reward, np.ndarray):
            reward = float(reward.sum())
        if isinstance(done, np.ndarray):
            done = bool(done.any())
            
        print(f"  步骤奖励: {reward:.2f}, 完成: {done}")
        
        if done:
            obs = env.reset()
            break
    
    print("✓ 评估完成")
    
    return model

def main():
    """主测试函数"""
    print("SB3 MARL 简化测试")
    print("=" * 50)
    
    try:
        # 验证配置
        if not validate_config():
            print("❌ 配置验证失败")
            return
        
        # 测试环境
        env = test_environment()
        
        # 测试SuperSuit包装器
        wrapped_env = test_supersuit_wrapper()
        
        # 测试PPO训练
        model = test_ppo_training()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！")
        print("✅ SB3 MARL环境工作正常")
        print("✅ 可以进行完整训练")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 