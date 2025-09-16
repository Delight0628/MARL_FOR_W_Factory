import os
import sys
import numpy as np
import tensorflow as tf
from collections import Counter
import argparse

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from environments.w_factory_env import WFactoryEnv
from evaluation import (
    STATIC_EVAL_CONFIG, 
    GENERALIZATION_CONFIG_1, GENERALIZATION_CONFIG_2, GENERALIZATION_CONFIG_3
)

def debug_marl_actions(model_path: str, config: dict, max_steps: int = 600, deterministic: bool = False):
    """
    调试MARL模型的动作输出模式。
    
    新增功能:
    - 可选择确定性策略或与evaluation.py对齐的随机策略。
    - 更具体的模型加载异常处理。
    """
    print(f"🔍 开始调试MARL模型行为")
    print(f"📋 配置: {config.get('stage_name', '未知')}")
    print(f"🕹️  策略: {'确定性 (Greedy)' if deterministic else '随机 (与evaluation.py对齐)'}")
    
    # 加载模型
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在于路径: {model_path}")
        actor_model = tf.keras.models.load_model(model_path)
        print(f"✅ 成功加载模型: {model_path}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    except (IOError, tf.errors.OpError) as e:
        print(f"❌ 加载模型失败，文件可能已损坏或格式不正确: {e}")
        return
    except Exception as e:
        print(f"❌ 加载模型时发生未知错误: {e}")
        return

    # 创建环境
    env = WFactoryEnv(config=config)
    obs, info = env.reset(seed=42)
    
    print(f"🏭 环境信息:")
    print(f"   智能体数量: {len(env.agents)}")
    print(f"   智能体列表: {env.agents}")
    
    # 记录动作统计
    action_stats = {agent: Counter() for agent in env.agents}
    step_count = 0
    
    print(f"\n🎯 开始记录前{max_steps}步的动作模式...")
    
    while step_count < max_steps:
        # MARL策略
        actions = {}
        for agent in env.agents:
            if agent in obs:
                state = tf.expand_dims(obs[agent], 0)
                action_probs = actor_model(state, training=False)
                
                # 显示前几步的详细信息
                if step_count < 5:
                    print(f"   步骤{step_count+1} {agent}: 概率分布 {action_probs[0].numpy()}")
                
                if deterministic:
                    # 确定性策略：总是选择概率最高的动作
                    action = int(tf.argmax(action_probs[0]))
                else:
                    # 随机策略：80%概率选最优，20%根据概率分布采样 (与evaluation.py对齐)
                    if np.random.random() < 0.2:
                        action = tf.random.categorical(tf.math.log(action_probs + 1e-8), 1)[0, 0].numpy()
                    else:
                        action = int(tf.argmax(action_probs[0]))

                actions[agent] = action
                action_stats[agent][action] += 1
        
        # 执行动作
        obs, rewards, terminations, truncations, info = env.step(actions)
        step_count += 1
        
        # 检查是否结束
        if any(terminations.values()) or any(truncations.values()):
            print(f"🏁 环境在第{step_count}步结束")
            break
    
    # 分析动作统计
    print(f"\n📊 动作统计分析 (总共{step_count}步):")
    print("-" * 60)
    
    for agent in env.agents:
        print(f"{agent}:")
        total_actions = sum(action_stats[agent].values())
        for action, count in sorted(action_stats[agent].items()):
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            action_name = "IDLE" if action == 0 else f"处理零件{action}"
            print(f"   动作{action} ({action_name}): {count}次 ({percentage:.1f}%)")
        print()
    
    # 获取最终统计
    final_stats = env.sim.get_final_stats()
    print(f"📈 最终KPI:")
    print(f"   完成零件: {final_stats['total_parts']}")
    print(f"   总工期: {final_stats['makespan']:.1f}")
    print(f"   延期时间: {final_stats['total_tardiness']:.1f}")
    print(f"   设备利用率: {final_stats['mean_utilization']:.1%}")
    
    env.close()

def main():
    """主函数，用于解析命令行参数并运行调试脚本"""
    parser = argparse.ArgumentParser(
        description="调试和分析MARL模型的行为，检查其在不同配置下的动作模式和性能。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="指向已训练好的MARL actor模型文件 (.keras) 的路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        choices=["static", "gen1", "gen2", "gen3", "all"],
        help="要运行的测试配置名称。'all'会运行所有可用的配置。"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=600,
        help="每个环境回合的最大仿真步数。"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="如果设置此标志，将使用确定性策略（总是选择最优动作）。否则，使用与评估脚本一致的随机策略（80%最优，20%采样）。"
    )
    args = parser.parse_args()

    # 配置名称到对象的映射
    config_map = {
        "static": ("基准配置", STATIC_EVAL_CONFIG),
        "gen1": ("泛化测试1-高压力短交期", GENERALIZATION_CONFIG_1),
        "gen2": ("泛化测试2-混合优先级", GENERALIZATION_CONFIG_2),
        "gen3": ("泛化测试3-大批量长周期", GENERALIZATION_CONFIG_3),
    }

    if args.config == "all":
        configs_to_run = list(config_map.values())
    else:
        configs_to_run = [config_map[args.config]]

    print("=" * 80)
    print("🔬 MARL模型行为分析")
    print(f"模型路径: {args.model_path}")
    print(f"策略模式: {'确定性 (Greedy)' if args.deterministic else '随机 (与evaluation.py对齐)'}")
    print(f"最大步数: {args.max_steps}")
    print("=" * 80)

    for name, config in configs_to_run:
        print(f"\n{'='*20} 开始测试: {name} {'='*20}")
        debug_marl_actions(
            model_path=args.model_path,
            config=config,
            max_steps=args.max_steps,
            deterministic=args.deterministic
        )
        print()

if __name__ == "__main__":
    main()
