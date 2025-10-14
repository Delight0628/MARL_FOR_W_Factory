import os
import sys

# 关键修复：强制调试脚本在CPU上运行，避免与训练进程争夺GPU资源
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TensorFlow的INFO级别日志

import numpy as np
import tensorflow as tf
from collections import Counter
import argparse
import random # 统一随机种子

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from environments.w_factory_env import WFactoryEnv
from evaluation import (
    STATIC_EVAL_CONFIG, 
    GENERALIZATION_CONFIG_1, GENERALIZATION_CONFIG_2, GENERALIZATION_CONFIG_3
)
# 导入配置以解码观测向量和动作
from environments.w_factory_config import (
    WORKSTATIONS,
    PRODUCT_ROUTES,
    ENHANCED_OBS_CONFIG,
    ACTION_CONFIG_ENHANCED,
    RANDOM_SEED
)


def decode_observation(obs_vector: np.ndarray, agent_id: str) -> str:
    """
    🔧 动态适配方案B：将扁平的观测向量解码为人类可读的格式
    自动从配置中读取观测空间结构，无需硬编码
    """
    if obs_vector is None or obs_vector.size == 0:
        return "  - 观测向量为空"

    # 🔧 动态计算各部分维度（适配方案A：移除启发式后的观测空间）
    station_types = list(WORKSTATIONS.keys())
    product_types = list(PRODUCT_ROUTES.keys())
    num_stations = len(station_types)
    num_candidates = ENHANCED_OBS_CONFIG["num_candidate_workpieces"]
    candidate_feature_dim = ENHANCED_OBS_CONFIG["candidate_feature_dim"]
    queue_summary_dim = ENHANCED_OBS_CONFIG["queue_summary_features"] * ENHANCED_OBS_CONFIG["queue_summary_stats"]
    
    # 🔧 方案A修改：全局特征从7维减少到4维（移除松弛度、延期率）
    global_feature_dim = 4
    
    # 计算期望的总维度
    expected_dim = 8 + global_feature_dim + queue_summary_dim + (candidate_feature_dim * num_candidates)
    
    decoded_lines = [
        f"[Observation Vector - 总维度: {len(obs_vector)} (期望: {expected_dim})]",
        f"  结构: 8(Agent) + {global_feature_dim}(Global) + {queue_summary_dim}(Queue) + {candidate_feature_dim}×{num_candidates}(Candidates)"
    ]

    current_idx = 0
    try:
        # --- 1. Agent自身特征 (8维) ---
        decoded_lines.append("  --- 1. 智能体自身特征 (8维) ---")
        
        # Agent ID (one-hot, 5维)
        agent_id_one_hot = obs_vector[current_idx : current_idx + num_stations]
        station_idx = np.argmax(agent_id_one_hot)
        decoded_lines.append(f"    - 智能体身份: {station_types[station_idx]}")
        current_idx += num_stations

        # Capacity
        capacity = obs_vector[current_idx] * 5.0
        decoded_lines.append(f"    - 工作站容量: {capacity:.1f}")
        current_idx += 1
        
        # Status
        busy_ratio = obs_vector[current_idx]
        is_failed = obs_vector[current_idx + 1] > 0.5
        decoded_lines.append(f"    - 设备状态: [繁忙率: {busy_ratio:.1%}, 故障: {'是' if is_failed else '否'}]")
        current_idx += 2
        
        # --- 2. 🔧 方案A：移除启发式的全局宏观特征 (4维) ---
        decoded_lines.append(f"  --- 2. 全局宏观特征 ({global_feature_dim}维，已移除启发式信息) ---")
        time_prog = obs_vector[current_idx]
        wip_ratio = obs_vector[current_idx + 1]
        bottleneck_cong = obs_vector[current_idx + 2]
        queue_len_norm = obs_vector[current_idx + 3]
        
        decoded_lines.append(f"    - 时间进度: {time_prog:.1%}")
        decoded_lines.append(f"    - WIP率: {wip_ratio:.1%}")
        decoded_lines.append(f"    - 瓶颈拥堵度: {bottleneck_cong:.1%}")
        decoded_lines.append(f"    - 当前队列长度(归一化): {queue_len_norm:.2f}")
        current_idx += global_feature_dim
        
        # --- 3. 🔧 当前队列摘要 (30维 = 6特征 × 5统计量，已移除启发式) ---
        decoded_lines.append(f"  --- 3. 当前队列摘要统计 ({queue_summary_dim}维，已移除松弛度和延期统计) ---")
        decoded_lines.append("    (6种中性特征的min/max/mean/std/median统计，此处简化显示)")
        current_idx += queue_summary_dim
        
        # --- 4. 🔧 彻底移除启发式的候选工件详细特征 (8维 × num_candidates) ---
        decoded_lines.append(f"  --- 4. 候选工件详细特征 ({candidate_feature_dim}维 × {num_candidates}工件，已移除启发式) ---")
        for i in range(num_candidates):
            part_vec = obs_vector[current_idx : current_idx + candidate_feature_dim]
            exists = part_vec[0]

            if exists > 0.5:
                # 🔧 彻底移除启发式：解析8维特征（已移除松弛度、是否延期、全局紧急度对比、瓶颈感知）
                norm_rem_ops = part_vec[1]
                norm_rem_time = part_vec[2]
                norm_op_dur = part_vec[3]
                downstream_cong = part_vec[4]
                priority = part_vec[5]
                is_final = part_vec[6]
                prod_type_enc = part_vec[7]

                
                # 解码产品类型
                prod_idx = int(prod_type_enc * len(product_types))
                product_name = product_types[prod_idx] if 0 <= prod_idx < len(product_types) else "未知"
                
                # 反归一化
                rem_ops = int(norm_rem_ops * ENHANCED_OBS_CONFIG["max_bom_ops_norm"])
                rem_time = norm_rem_time * ENHANCED_OBS_CONFIG["total_remaining_time_norm"]
                op_dur = norm_op_dur * ENHANCED_OBS_CONFIG["max_op_duration_norm"]
                
                decoded_lines.append(
                    f"    候选工件 {i+1} ({product_name}):\n"
                    f"      - 剩余工序: {rem_ops}, 剩余时间: {rem_time:.1f}min, 当前工序: {op_dur:.1f}min\n"
                    f"      - 优先级: {priority*5.0:.1f}, 下游拥堵: {downstream_cong:.1%}, 最终工序: {'是' if is_final > 0.5 else '否'}"
                )
            else:
                decoded_lines.append(f"    候选工件 {i+1}: (空)")
            
            current_idx += candidate_feature_dim

    except IndexError as e:
        decoded_lines.append(f"  - (!! 观测向量维度不匹配: 期望{current_idx}维，实际{len(obs_vector)}维 !!)")
        decoded_lines.append(f"  - 错误详情: {e}")
    except Exception as e:
        decoded_lines.append(f"  - (!! 解析时发生未知错误: {e} !!)")

    return "\n".join(decoded_lines)

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

    # 创建环境
    env = WFactoryEnv(config=config)
    obs, info = env.reset(seed=seed)
    
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
                    print(f"\n--- 步骤 {step_count+1}: {agent} ---")
                    # 解码并打印观测向量
                    decoded_obs_str = decode_observation(obs[agent], agent)
                    print(decoded_obs_str)
                    # 打印动作概率
                    print(f"[Action Probs]")
                    # 🔧 修复：动态适配动作数量（模型输出可能与当前配置不同）
                    action_probs_array = action_probs[0].numpy()
                    action_names = ACTION_CONFIG_ENHANCED['action_names']
                    max_actions = min(len(action_probs_array), len(action_names))
                    prob_str = ", ".join([f"{action_names[i]}: {action_probs_array[i]:.2%}" for i in range(max_actions)])
                    if len(action_probs_array) > len(action_names):
                        prob_str += f" (+{len(action_probs_array) - len(action_names)}个额外动作)"
                    print(f"  - {prob_str}")

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
    
    for agent in env.agents:
        print(f"{agent}:")
        total_actions = sum(action_stats[agent].values())
        for action, count in sorted(action_stats[agent].items()):
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            # 🔧 修复：使用配置中的动作名称，防止越界
            action_names = ACTION_CONFIG_ENHANCED["action_names"]
            action_name = action_names[action] if action < len(action_names) else f"未知动作{action}"
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
    parser.add_argument(
        "--snapshot_interval",
        type=int,
        default=100,
        help="每隔多少步打印一次KPI快照。设置为0则禁用。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
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
