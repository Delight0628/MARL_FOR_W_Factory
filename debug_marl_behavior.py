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
    """将扁平的观测向量解码为人类可读的格式"""
    if obs_vector is None or obs_vector.size == 0:
        return "  - 观测向量为空"

    decoded_lines = ["[Observation Vector]"]
    
    # 从配置中获取维度信息
    station_types = list(WORKSTATIONS.keys())
    product_types = list(PRODUCT_ROUTES.keys())
    num_stations = len(station_types)
    obs_slot_size = ENHANCED_OBS_CONFIG["obs_slot_size"]
    workpiece_feature_count = 10  # V3版工件特征数量为10

    current_idx = 0
    try:
        # --- 1. Agent Features ---
        decoded_lines.append("  --- 1. 智能体自身特征 ---")
        
        # Agent ID (one-hot)
        agent_id_one_hot = obs_vector[current_idx : current_idx + num_stations]
        station_idx = np.argmax(agent_id_one_hot)
        decoded_lines.append(f"    - 智能体身份: {station_types[station_idx]} (one-hot)")
        current_idx += num_stations

        # Capacity
        capacity = obs_vector[current_idx] * 5.0
        decoded_lines.append(f"    - 工作站容量: {capacity:.1f}")
        current_idx += 1
        
        # Status
        busy_ratio = obs_vector[current_idx]
        is_failed = obs_vector[current_idx + 1] > 0.5
        decoded_lines.append(f"    - 设备状态: [繁忙率: {busy_ratio:.1%}, 是否故障: {'是' if is_failed else '否'}]")
        current_idx += 2
        
        # --- 2. Global Features ---
        decoded_lines.append("  --- 2. 全局宏观特征 ---")
        time_prog = obs_vector[current_idx]
        wip_ratio = obs_vector[current_idx + 1]
        decoded_lines.append(f"    - 全局信息: [时间进度: {time_prog:.1%}, WIP率: {wip_ratio:.1%}]")
        current_idx += 2
        
        # --- 3. Workpiece Features ---
        decoded_lines.append("  --- 3. 队列中工件的详细特征 ---")
        for i in range(obs_slot_size):
            part_vec = obs_vector[current_idx : current_idx + workpiece_feature_count]
            exists = part_vec[0]

            if exists > 0.5:
                # Unpack all 10 features from V3 state space
                (exists, norm_slack, norm_rem_ops, norm_rem_time, 
                 norm_op_dur, is_late, downstream_cong, priority, 
                 is_final, prod_type_enc) = part_vec
                
                # Decode product type
                prod_idx = int(round(prod_type_enc * len(product_types))) - 1
                product_name = product_types[prod_idx] if 0 <= prod_idx < len(product_types) else "未知"
                
                # Un-normalize values for readability
                time_slack = norm_slack * ENHANCED_OBS_CONFIG["time_slack_norm"]
                
                decoded_lines.append(
                    f"    槽位 {i+1} ({product_name}):\n"
                    f"      - 状态: [松弛时间: {time_slack:.1f}, 将延期: {'是' if is_late > 0.5 else '否'}, 最终工序: {'是' if is_final > 0.5 else '否'}]\n"
                    f"      - 属性: [优先级: {priority*5.0:.1f}, 下游拥堵: {downstream_cong:.1%}]"
                )
            else:
                decoded_lines.append(f"    槽位 {i+1}: (空)")
            
            current_idx += workpiece_feature_count

    except IndexError:
        decoded_lines.append("  - (!! 观测向量维度不匹配，部分信息无法解析 !!)")
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
                    prob_str = ", ".join([f"{ACTION_CONFIG_ENHANCED['action_names'][i]}: {p:.2%}" for i, p in enumerate(action_probs[0].numpy())])
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
            # 使用配置中的动作名称
            action_name = ACTION_CONFIG_ENHANCED["action_names"][action] if action < len(ACTION_CONFIG_ENHANCED["action_names"]) else f"未知动作{action}"
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
