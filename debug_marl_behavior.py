import os
import sys

# 关键修复：强制调试脚本在CPU上运行，避免与训练进程争夺GPU资源
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TensorFlow的INFO级别日志

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse
import random # 统一随机种子
# 10201530 新增：导入gym以检测MultiDiscrete动作空间
import gymnasium as gym

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    # 假设debug脚本在项目根目录的子目录中
    sys.path.append(os.path.dirname(current_dir))

from environments.w_factory_env import WFactoryEnv, calculate_slack_time
from evaluation import (
    STATIC_EVAL_CONFIG, 
    GENERALIZATION_CONFIG_1, GENERALIZATION_CONFIG_2, GENERALIZATION_CONFIG_3
)


def decode_observation(obs_vector: np.ndarray, agent_id: str, obs_meta: dict) -> str:
    """
    动态解码观测向量，完全依赖环境提供的元信息(obs_meta)
    """
    decoded_lines = []
    current_idx = 0

    try:
        # --- 1. Agent自身特征 ---
        num_stations = obs_meta.get('num_stations', 5)
        agent_feature_names = obs_meta.get('agent_feature_names', [])
        
        decoded_lines.append(f"  --- 1. 智能体自身特征 ({num_stations + 3}维) ---")
        
        # one-hot
        one_hot_len = min(num_stations, len(obs_vector) - current_idx)
        agent_id_one_hot = obs_vector[current_idx : current_idx + one_hot_len]
        station_idx = int(np.argmax(agent_id_one_hot)) if one_hot_len > 0 else -1
        decoded_lines.append(f"    - {agent_feature_names[0]}: {station_idx}")
        current_idx += one_hot_len

        # capacity
        capacity = obs_vector[current_idx] * 5.0
        decoded_lines.append(f"    - {agent_feature_names[1]}: {capacity:.1f}")
        current_idx += 1
        
        # busy_ratio & is_failed
        busy_ratio = obs_vector[current_idx]
        is_failed = obs_vector[current_idx + 1] > 0.5
        decoded_lines.append(f"    - {agent_feature_names[2]}: {busy_ratio:.1%}")
        decoded_lines.append(f"    - {agent_feature_names[3]}: {'是' if is_failed else '否'}")
        current_idx += 2

        # --- 2. 全局宏观特征 ---
        global_feature_names = obs_meta.get('global_feature_names', [])
        decoded_lines.append(f"  --- 2. 全局宏观特征 ({len(global_feature_names)}维) ---")
        for i, name in enumerate(global_feature_names):
            value = obs_vector[current_idx + i]
            decoded_lines.append(f"    - {name}: {value:.3f}")
        current_idx += len(global_feature_names)

        # --- 3. 队列摘要统计 ---
        queue_feature_names = obs_meta.get('queue_summary_feature_names', [])
        stat_names = obs_meta.get('queue_summary_stat_names', [])
        num_stats = len(stat_names)
        queue_summary_dim = len(queue_feature_names) * num_stats
        decoded_lines.append(f"  --- 3. 队列摘要统计 ({queue_summary_dim}维) ---")
        
        queue_summary_vec = obs_vector[current_idx : current_idx + queue_summary_dim]
        current_idx += queue_summary_dim
        
        for i, feature_name in enumerate(queue_feature_names):
            stats_str_parts = []
            for j, stat_name in enumerate(stat_names):
                value = queue_summary_vec[i * num_stats + j]
                stats_str_parts.append(f"{stat_name}={value:.2f}")
            decoded_lines.append(f"    - {feature_name}: [{', '.join(stats_str_parts)}]")

        # --- 4. 候选工件详细特征 ---
        candidate_feature_names = obs_meta.get('candidate_feature_names', [])
        candidate_feature_dim = len(candidate_feature_names)
        num_candidates = obs_meta.get('num_candidate_workpieces', 10)
        decoded_lines.append(f"  --- 4. 候选工件详细特征 ({candidate_feature_dim * num_candidates}维) ---")
        
        for i in range(num_candidates):
            candidate_features_raw = obs_vector[current_idx : current_idx + candidate_feature_dim]
            current_idx += candidate_feature_dim
            
            # 检查工件是否存在
            if candidate_features_raw[0] < 0.5:
                decoded_lines.append(f"    - [候选 {i+1}]: (空)")
                continue
            
            decoded_lines.append(f"    - [候选 {i+1}]:")
            for j, feature_name in enumerate(candidate_feature_names):
                value = candidate_features_raw[j]
                # 根据特征名称进行一些可读性处理
                if "norm" in feature_name:
                    formatted_value = f"{value:.3f}"
                elif "ratio" in feature_name:
                    formatted_value = f"{value:.1%}"
                else:
                    norm_constants = obs_meta.get('normalization_constants', {})
                    # 简单的反向归一化，仅为可读性
                    if feature_name == 'remaining_ops':
                        value *= norm_constants.get('max_bom_ops_norm', 1)
                        formatted_value = f"{value:.1f}"
                    elif feature_name == 'total_remaining_time':
                        value *= norm_constants.get('total_remaining_time_norm', 1)
                        formatted_value = f"{value:.1f} min"
                    elif feature_name == 'current_op_duration':
                         value *= norm_constants.get('max_op_duration_norm', 1)
                         formatted_value = f"{value:.1f} min"
                    else:
                        formatted_value = f"{value:.3f}"
                decoded_lines.append(f"      - {feature_name}: {formatted_value}")

    except IndexError:
        decoded_lines.append("  --- !错误: 观测向量维度与解码逻辑不匹配! ---")
    except Exception as e:
        decoded_lines.append(f"  --- !解码时发生未知错误: {e}! ---")

    return "\n".join(decoded_lines)

def get_policy_details(policy, state, obs_meta: dict):
    """获取策略分布和选择的动作"""
    action_probs = policy(tf.expand_dims(state, 0))[0].numpy()
    chosen_action = int(np.argmax(action_probs))
    
    # 获取动作名称
    action_names = obs_meta.get('action_names', [])
    
    # 确保action_names的长度至少和action_probs一样长
    if len(action_names) < len(action_probs):
        action_names.extend([f"未知动作_{i}" for i in range(len(action_names), len(action_probs))])

    policy_dist_str = ", ".join([
        f"{action_names[i]}={prob:.2%}" for i, prob in enumerate(action_probs)
    ])
    chosen_action_name = action_names[chosen_action] if chosen_action < len(action_names) else f"未知动作_{chosen_action}"
    
    return policy_dist_str, chosen_action_name


def debug_marl_actions(model_path: str, config: dict, max_steps: int = 600, deterministic: bool = False, snapshot_interval: int = 100, seed: int = 42):
    """
    调试MARL模型的动作输出模式。
    
    新增功能:
    - 可选择确定性策略或与evaluation.py对齐的随机策略。
    - 更具体的模型加载异常处理。
    - 可视化智能体观测向量(视野)。
    - 定期输出KPI快照。
    - 统一随机种子。
    """
    print(f"🔍 开始调试MARL模型行为")
    print(f"📋 配置: {config.get('stage_name', '未知')}")
    print(f"🕹️  策略: {'确定性 (Greedy)' if deterministic else '随机 (与evaluation.py对齐)'}")
    print(f"🌱 随机种子: {seed}")
    
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

    # 创建环境（默认启用确定性候选，保证调试可复现）
    config_for_debug = dict(config) if isinstance(config, dict) else {}
    config_for_debug.setdefault('deterministic_candidates', True)
    env = WFactoryEnv(config=config_for_debug)
    obs, info = env.reset(seed=seed)
    
    print(f"🏭 环境信息:")
    print(f"   智能体数量: {len(env.agents)}")
    print(f"   智能体列表: {env.agents}")
    
    # 10201530 修复：为MultiDiscrete建立按设备维度的动作统计
    heads_map = {}
    for agent in env.agents:
        space = env.action_space(agent)
        if isinstance(space, gym.spaces.MultiDiscrete):
            heads_map[agent] = len(space.nvec)
        else:
            heads_map[agent] = 1

    # 10201530 修复：动作统计改为每个agent的每个设备一份Counter
    action_stats = {agent: [Counter() for _ in range(heads_map[agent])] for agent in env.agents}
    step_count = 0
    
    print(f"\n🎯 开始记录前{max_steps}步的动作模式...")
    
    # 10201530 新增：从概率分布生成并行动作的工具函数（去重、可选采样）
    def choose_parallel_actions_from_probs(probs: np.ndarray, num_heads: int, greedy: bool = True) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum()

        # 贪心：从大到小选前K个，避免重复
        if greedy:
            sorted_idx = np.argsort(probs)[::-1]
            chosen = []
            for idx in sorted_idx:
                # 允许IDLE(0)被选择；去重相同动作
                if idx not in chosen:
                    chosen.append(int(idx))
                if len(chosen) >= num_heads:
                    break
            # 如果不足K，补0
            while len(chosen) < num_heads:
                chosen.append(0)
            return np.array(chosen, dtype=np.int32)

        # 随机：无放回抽样num_heads个动作
        n = probs.shape[0]
        if num_heads >= n:
            # 边界：动作数不足时，允许部分重复
            sampled = np.random.choice(np.arange(n), size=num_heads, replace=True, p=probs)
        else:
            sampled = np.random.choice(np.arange(n), size=num_heads, replace=False, p=probs)
        return sampled.astype(np.int32)

    while step_count < max_steps:
        # MARL策略
        actions = {}
        for agent in env.agents:
            if agent in obs:
                state = tf.expand_dims(obs[agent], 0)
                action_probs_tensor = actor_model(state, training=False)
                action_probs = action_probs_tensor[0].numpy()
                space = env.action_space(agent)
                is_multi = isinstance(space, gym.spaces.MultiDiscrete)
                num_heads = heads_map.get(agent, 1)
                
                # 显示前几步的详细信息
                if step_count < 5:
                    print(f"\n--- 步骤 {step_count+1}: {agent} ---")
                    # 解码并打印观测向量
                    # 10201530 修复：向decode传入obs_meta
                    decoded_obs_str = decode_observation(obs[agent], agent, info[agent].get('obs_meta', {}))
                    print(decoded_obs_str)
                    # 打印动作概率
                    action_probs = action_probs
                    
                    # 从info中获取动作名称
                    action_names = info[agent].get('obs_meta', {}).get('action_names', [])
                    
                    if action_names:
                        policy_dist_str = ", ".join([f"{name}={prob:.2%}" for name, prob in zip(action_names, action_probs)])
                    else:
                        # Fallback if action_names is not available
                        policy_dist_str = ", ".join([f"Action{i}={prob:.2%}" for i, prob in enumerate(action_probs)])
                    print(f"  - 策略分布: [{policy_dist_str}]")

                # 10201530 修复：根据动作空间类型生成标量或并行动作数组
                if is_multi:
                    # 80/20策略：主要贪心，少量采样
                    if deterministic:
                        action = choose_parallel_actions_from_probs(action_probs, num_heads, greedy=True)
                    else:
                        if np.random.random() < 0.2:
                            action = choose_parallel_actions_from_probs(action_probs, num_heads, greedy=False)
                        else:
                            action = choose_parallel_actions_from_probs(action_probs, num_heads, greedy=True)
                else:
                    if deterministic:
                        action = int(np.argmax(action_probs))
                    else:
                        if np.random.random() < 0.2:
                            action = int(np.random.choice(np.arange(len(action_probs)), p=action_probs))
                        else:
                            action = int(np.argmax(action_probs))

                # 10201530 修复：在详细阶段打印选择结果
                if step_count < 5:
                    if is_multi:
                        decoded = [
                            (info[agent].get('obs_meta', {}).get('action_names', [])[a]
                             if a < len(info[agent].get('obs_meta', {}).get('action_names', [])) else f"Action{a}")
                            for a in list(action)
                        ]
                        print(f"  - 最终选择动作(并行): {list(action)} -> {decoded}")
                    else:
                        anames = info[agent].get('obs_meta', {}).get('action_names', [])
                        print(f"  - 最终选择动作: {action} ({anames[action] if action < len(anames) else '未知'})")

                actions[agent] = action
                # 10201530 修复：统计并行动作（按设备维度）
                if isinstance(action, (list, np.ndarray)):
                    action_list = list(action)
                else:
                    action_list = [int(action)]
                # 统一长度
                if len(action_list) < heads_map[agent]:
                    action_list += [0] * (heads_map[agent] - len(action_list))
                for k in range(heads_map[agent]):
                    action_stats[agent][k][int(action_list[k])] += 1
        
        # 执行动作
        obs, rewards, terminations, truncations, info = env.step(actions)
        step_count += 1
        
        # KPI快照
        if step_count > 0 and snapshot_interval > 0 and step_count % snapshot_interval == 0:
            print(f"\n--- 📈 KPI 快照 (第 {step_count} 步) ---")
            current_stats = env.sim.get_final_stats()
            print(f"   完成零件: {current_stats.get('total_parts', 0)}")
            print(f"   在制品(WIP): {len(env.sim.active_parts)}")
            print(f"   累计延期: {current_stats.get('total_tardiness', 0):.1f}")
            print(f"   当前利用率: {current_stats.get('mean_utilization', 0):.1%}")
            print("-" * 35)

        # 检查是否结束
        if any(terminations.values()) or any(truncations.values()):
            print(f"🏁 环境在第{step_count}步结束")
            break
    
    # 分析动作统计
    print(f"\n📊 动作统计分析 (总共{step_count}步):")
    print("-" * 60)
    
    # 10201530 修复：按设备维度输出统计
    for agent in env.agents:
        print(f"{agent}:")
        for k, counter in enumerate(action_stats[agent]):
            total = sum(counter.values())
            print(f"  设备#{k}:")
            for action, count in sorted(counter.items()):
                pct = (count / total) * 100 if total > 0 else 0
                action_names = info[agent].get('obs_meta', {}).get('action_names', [])
                action_name = action_names[action] if action < len(action_names) else f"未知动作{action}"
                print(f"    - 动作{action} ({action_name}): {count}次 ({pct:.1f}%)")
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
    parser.add_argument(
        "--snapshot_interval",
        type=int,
        default=100,
        help="每隔多少步打印一次KPI快照。设置为0则禁用。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="设置随机种子以保证可复现性。"
    )
    args = parser.parse_args()

    # 统一设置随机种子
    print(f"🌱 使用随机种子: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

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
            deterministic=args.deterministic,
            snapshot_interval=args.snapshot_interval,
            seed=args.seed
        )
        print()

if __name__ == "__main__":
    main()
