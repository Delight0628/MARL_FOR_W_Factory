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
    from ray.rllib.env import PettingZooEnv
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
sys.path.append(str(current_dir))

try:
    from environments.w_factory_env import make_parallel_env
    from environments.w_factory_config import *
    print("✅ 工厂环境导入成功")
except ImportError as e:
    print(f"❌ 工厂环境导入失败: {e}")
    print("请确保environments目录在当前路径下")
    sys.exit(1)

def get_wsl_system_info():
    """获取WSL系统信息"""
    info = {
        "platform": "WSL",
        "python_version": sys.version,
        "ray_version": ray.__version__,
    }
    
    try:
        # 获取CPU信息
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.read()
            cpu_count = cpu_info.count('processor')
            info["cpu_count"] = cpu_count
        
        # 获取内存信息
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.read()
            for line in mem_info.split('\n'):
                if 'MemTotal' in line:
                    mem_total = int(line.split()[1]) // 1024  # Convert to MB
                    info["memory_mb"] = mem_total
                    break
        
        # 获取WSL版本
        if 'WSL_DISTRO_NAME' in os.environ:
            info["wsl_distro"] = os.environ['WSL_DISTRO_NAME']
            
    except Exception as e:
        print(f"⚠️  系统信息获取失败: {e}")
    
    return info

def env_creator(config):
    """环境创建函数"""
    return PettingZooEnv(make_parallel_env(config))

# 注册环境
register_env("w_factory", env_creator)

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

def get_wsl_training_config():
    """获取WSL优化的训练配置"""
    system_info = get_wsl_system_info()
    cpu_count = system_info.get("cpu_count", 4)
    
    # 根据CPU数量调整worker数量
    num_workers = max(1, min(cpu_count - 1, 4))  # 保留1个CPU给主进程
    
    config = (
        PPOConfig()
        .environment(
            env="w_factory",
            env_config={},
            disable_env_checking=True
        )
        .framework("tf2")
        .rollouts(
            # WSL优化的rollout配置
            num_rollout_workers=num_workers,
            num_envs_per_worker=1,
            rollout_fragment_length=200,
            batch_mode="complete_episodes",
            # WSL环境下的超时设置
            sample_timeout_s=60.0,
        )
        .training(
            # PPO训练参数
            train_batch_size=2000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
        )
        .multi_agent(
            # 多智能体配置
            policies={
                "shared_policy": (
                    None,
                    gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                    gym.spaces.Discrete(2),
                    {}
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"]
        )
        .resources(
            # WSL资源配置
            num_gpus=0,  # WSL通常不支持GPU
            num_cpus_per_worker=1,
            num_gpus_per_worker=0
        )
        .evaluation(
            # 评估配置
            evaluation_interval=25,
            evaluation_duration=10,
            evaluation_num_workers=1,
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
    
    print(f"🔧 训练配置:")
    print(f"   Rollout Workers: {num_workers}")
    print(f"   训练批次大小: 2000")
    print(f"   SGD迭代次数: 10")
    
    return config

def run_wsl_ray_training(num_iterations: int = 100):
    """在WSL中运行Ray RLlib训练"""
    print("=" * 70)
    print("🐧 W工厂多智能体强化学习训练 - WSL Ubuntu版本")
    print("=" * 70)
    print("环境: WSL Ubuntu")
    print("框架: Ray RLlib")
    print("算法: PPO (Proximal Policy Optimization)")
    print("多智能体: 策略共享MAPPO")
    print("=" * 70)
    
    # 显示系统信息
    system_info = get_wsl_system_info()
    for key, value in system_info.items():
        print(f"系统信息 - {key}: {value}")
    print("=" * 70)
    
    # 验证配置
    if not validate_config():
        print("❌ 配置验证失败")
        return None
    
    try:
        # 初始化Ray
        ray_config = get_wsl_ray_config()
        print("🚀 初始化Ray (WSL模式)...")
        
        if ray.is_initialized():
            ray.shutdown()
        
        ray.init(**ray_config)
        print("✅ Ray初始化成功")
        
        # 获取训练配置
        training_config = get_wsl_training_config()
        
        # 设置停止条件
        stop_config = {
            "training_iteration": num_iterations,
            "timesteps_total": num_iterations * 2000,
            "time_total_s": 3600,  # 最大1小时
        }
        
        # 创建结果目录
        results_dir = Path.cwd() / "wsl_ray_results"
        results_dir.mkdir(exist_ok=True)
        
        print(f"📁 结果目录: {results_dir}")
        print(f"🎯 开始训练 ({num_iterations} 轮)...")
        start_time = time.time()
        
        # 运行训练
        tuner = tune.Tuner(
            "PPO",
            param_space=training_config.to_dict(),
            run_config=tune.RunConfig(
                name="w_factory_wsl_marl",
                local_dir=str(results_dir),
                stop=stop_config,
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=20,
                    num_to_keep=5
                ),
                verbose=2  # WSL中显示更多信息
            )
        )
        
        results = tuner.fit()
        
        # 获取最佳结果
        best_result = results.get_best_result(
            metric="episode_reward_mean", 
            mode="max"
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("🎉 WSL Ray RLlib训练完成！")
        print("=" * 70)
        print(f"⏱️  训练时间: {training_time/60:.2f} 分钟")
        print(f"🏆 最佳平均奖励: {best_result.metrics['episode_reward_mean']:.2f}")
        print(f"📊 训练轮数: {best_result.metrics['training_iteration']}")
        print(f"📁 最佳检查点: {best_result.checkpoint}")
        
        # 保存WSL专用结果摘要
        summary = {
            "environment": "WSL Ubuntu",
            "framework": "Ray RLlib",
            "algorithm": "PPO/MAPPO",
            "system_info": system_info,
            "training_time_minutes": training_time / 60,
            "best_reward": best_result.metrics['episode_reward_mean'],
            "training_iterations": best_result.metrics['training_iteration'],
            "checkpoint_path": str(best_result.checkpoint),
            "total_iterations": num_iterations,
            "agents": list(WORKSTATIONS.keys()),
            "wsl_optimizations": {
                "ray_config": ray_config,
                "disabled_features": ["dashboard", "gpu_support"],
                "enabled_features": ["multi_worker", "checkpointing"]
            }
        }
        
        summary_file = results_dir / f"wsl_ray_training_summary_{int(time.time())}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📄 训练摘要已保存: {summary_file}")
        
        return best_result
        
    except Exception as e:
        print(f"❌ WSL Ray训练过程中出现错误: {e}")
        print("错误详情:")
        import traceback
        traceback.print_exc()
        
        # WSL特定的故障排除建议
        print("\n🔧 WSL故障排除建议:")
        print("1. 确保WSL2已启用: wsl --set-version <distro> 2")
        print("2. 增加WSL内存限制: 编辑 ~/.wslconfig")
        print("3. 重启WSL: wsl --shutdown && wsl")
        print("4. 检查Python环境: which python3 && python3 --version")
        
        return None
    
    finally:
        # 清理Ray
        if ray.is_initialized():
            print("🧹 清理Ray资源...")
            ray.shutdown()

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
    print("🐧 W工厂多智能体强化学习系统 - WSL版本")
    print("=" * 70)
    
    # 检查WSL环境
    if not is_wsl:
        print("⚠️  建议在WSL环境中运行此脚本以获得最佳性能")
    
    # 创建设置脚本
    setup_file = create_wsl_setup_script()
    
    try:
        # 运行Ray训练
        ray_result = run_wsl_ray_training(num_iterations=50)
        
        if ray_result:
            print("\n🎉 WSL Ray RLlib训练成功完成！")
            print("✅ 这是在WSL中运行的真正MARL训练！")
            print("✅ 使用Ray RLlib框架")
            print("✅ PPO/MAPPO算法")
            print("✅ 多智能体策略共享")
            print("✅ Linux原生性能")
            
            # 显示后续步骤
            print("\n📋 后续步骤:")
            print("1. 查看训练结果: ls wsl_ray_results/")
            print("2. 加载模型进行推理")
            print("3. 可视化训练曲线")
            
        else:
            print("\n❌ WSL Ray训练失败")
            print(f"💡 请先运行设置脚本: bash {setup_file}")
            
    except Exception as e:
        print(f"❌ 主程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 