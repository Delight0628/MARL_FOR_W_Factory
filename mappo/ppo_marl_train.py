"""
MAPPO多智能体强化学习训练入口
=================================
基于MAPPO算法的工厂调度系统训练入口

模块组织（自底向上）：
┌─────────────────────────────────────┐
│  ppo_marl_train.py (本文件)         │  ← 训练入口
├─────────────────────────────────────┤
│  ppo_trainer.py                     │  ← 训练器主类
├──────────┬──────────┬───────────────┤
│ ppo_     │ ppo_     │ ppo_          │
│ buffer.py│ network  │ worker.py     │  ← 核心组件
└──────────┴──────────┴───────────────┘

使用方式：
    python mappo/ppo_marl_train.py [--models-dir DIR] [--logs-dir DIR]
"""

import os
# 训练模式默认使用随机初始化，提高探索能力
os.environ.setdefault('DETERMINISTIC_INIT', '0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 🔧 强制worker子进程使用CPU，避免多进程GPU资源竞争导致BrokenProcessPool
# 主进程（训练器）仍使用GPU进行模型更新，子进程（采样）用CPU
os.environ['FORCE_WORKER_CPU'] = '1'

import sys
import random
import numpy as np
import tensorflow as tf
import argparse
import multiprocessing

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_config import *
from mappo.ppo_trainer import SimplePPOTrainer


def main():
    """
    训练主入口
    
    执行流程：
    1. 解析命令行参数（模型/日志目录）
    2. 设置随机种子
    3. 加载训练配置
    4. 创建训练器实例
    5. 启动自适应训练循环
    6. 输出训练结果
    """
    print(f"✨ 训练进程PID: {os.getpid()}")

    # 解析外部传入的目录参数（由 auto_train.py 传入）
    parser = argparse.ArgumentParser(description="MAPPO 训练入口")
    parser.add_argument("--models-dir", type=str, default=None, help="用于保存训练模型的根目录（由auto_train传入）")
    parser.add_argument("--logs-dir", type=str, default=None, help="用于保存TensorBoard日志的根目录（由auto_train传入）")
    cli_args, _ = parser.parse_known_args()

    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    try:
        # 从配置文件获取训练参数
        max_episodes = TRAINING_FLOW_CONFIG["general_params"]["max_episodes"]
        steps_per_episode = TRAINING_FLOW_CONFIG["general_params"]["steps_per_episode"]
        eval_frequency = TRAINING_FLOW_CONFIG["general_params"]["eval_frequency"]
        
        print("=" * 80)
        foundation_criteria = TRAINING_FLOW_CONFIG["foundation_phase"]["graduation_criteria"]
        generalization_criteria = TRAINING_FLOW_CONFIG["generalization_phase"]["completion_criteria"]
        foundation_mixing = TRAINING_FLOW_CONFIG["foundation_phase"]["multi_task_mixing"]
        generalization_mixing = TRAINING_FLOW_CONFIG["generalization_phase"]["multi_task_mixing"]
        dynamic_events = TRAINING_FLOW_CONFIG["generalization_phase"].get("dynamic_events", {})
        
        print(f"\n📚阶段一：基础能力训练（随机订单泛化）")
        print(f"   策略: 随机订单 + {int(foundation_mixing.get('base_worker_fraction', 0)*100)}% worker使用BASE_ORDERS")
        print(f"   目标: 综合评分 > {foundation_criteria['target_score']:.2f}, "
              f"完成率 > {foundation_criteria['min_completion_rate']:.0f}%, "
              f"延期 < {foundation_criteria['tardiness_threshold']:.0f}min, "
              f"连续{foundation_criteria['target_consistency']}次")
        
        print(f"\n🚀阶段二：动态事件鲁棒性训练（动态事件鲁棒性）")
        print(f"   策略: 随机订单 + 动态事件（设备故障{'✓' if dynamic_events.get('equipment_failure_enabled') else '✗'}、紧急插单{'✓' if dynamic_events.get('emergency_orders_enabled') else '✗'}）")
        print(f"        + {int(generalization_mixing.get('base_worker_fraction', 0)*100)}% worker使用BASE_ORDERS")
        print(f"   目标: 综合评分 > {generalization_criteria['target_score']:.2f}, "
              f"完成率 > {generalization_criteria['min_completion_rate']:.0f}%, "
              f"连续{generalization_criteria['target_consistency']}次")

        print(f"📊 轮数上限: {max_episodes}轮")
        print("=" * 80)
        print("🔧 核心配置:")
        print("  工作站:")
        for station, config in WORKSTATIONS.items():
            print(f"    - {station}: 数量={config['count']}, 容量={config['capacity']}")

        print("  奖励系统:")
        for key, value in REWARD_CONFIG.items():
            print(f"    - {key}: {value}")
        
        cl_config = TRAINING_FLOW_CONFIG["foundation_phase"]["curriculum_learning"]
        dynamic_events_cfg = TRAINING_FLOW_CONFIG["generalization_phase"].get("dynamic_events", {})
        
        print("  启用/禁用模块:")
        print(f"    - 课程学习: {'启用' if cl_config.get('enabled', False) else '禁用'}")
        print(f"    - 设备故障: {'启用' if dynamic_events_cfg.get('equipment_failure_enabled', False) else '禁用'}")
        print(f"    - 紧急插单: {'启用' if dynamic_events_cfg.get('emergency_orders_enabled', False) else '禁用'}")
        print("-" * 40)
        
        trainer = SimplePPOTrainer(
            initial_lr=LEARNING_RATE_CONFIG["initial_lr"],
            total_train_episodes=max_episodes,
            steps_per_episode=steps_per_episode,
            training_targets=None, 
            models_root_dir=cli_args.models_dir,
            logs_root_dir=cli_args.logs_dir
        )
        
        # 启动自适应训练：系统将根据性能自动决定何时停止
        results = trainer.train(
            max_episodes=max_episodes,
            steps_per_episode=steps_per_episode,
            eval_frequency=eval_frequency,
            adaptive_mode=True
        )
        
        if results:
            print("\n🎉 自适应训练成功完成！")
            print(f"📊 实际训练轮数: {len(trainer.iteration_times)}")
            final_completion_rate = (results['best_kpi'].get('mean_completed_parts', 0) / get_total_parts_count()) * 100 if get_total_parts_count() > 0 else 0
            print(f"🎯 最终目标达成: {trainer.adaptive_state['target_achieved_count']}次连续达标 (基于最终阶段分数)")
            
            best_episode_final = trainer.best_episode_dual_objective if trainer.best_episode_dual_objective != -1 else trainer.final_stage_best_episode
            print(f"📈 历史最佳性能 (双重标准，第 {best_episode_final} 回合): {final_completion_rate:.1f}% ({results['best_kpi'].get('mean_completed_parts', 0):.1f}个零件)")
        else:
            print("\n❌ 训练失败")
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置多进程启动方法为'spawn'，避免TensorFlow的fork不安全问题
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()

