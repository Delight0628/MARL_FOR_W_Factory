"""
MARL生产调度系统全面自动化优化框架
不仅调优超参数，还自动优化环境设计、奖励函数、网络架构等关键组件
"""

import os
import sys
import copy
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from dataclasses import dataclass
from enum import Enum

# 添加环境路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入现有模块
from environments import w_factory_config

class OptimizationLevel(Enum):
    """优化级别枚举"""
    HYPERPARAMS_ONLY = "hyperparams"          # 仅优化超参数
    ENVIRONMENT_DESIGN = "environment"        # 环境设计优化
    REWARD_ENGINEERING = "reward"            # 奖励工程优化
    ARCHITECTURE_SEARCH = "architecture"     # 架构搜索
    FULL_SYSTEM = "full"                     # 全系统优化

@dataclass
class OptimizationConfig:
    """优化配置"""
    n_trials: int = 100
    n_eval_episodes: int = 15
    max_train_episodes: int = 300
    optimization_level: OptimizationLevel = OptimizationLevel.FULL_SYSTEM
    enable_generalization_test: bool = True
    parallel_trials: int = 1
    output_dir: str = "advanced_optimization_results"

class AdvancedMARL_Optimizer:
    """高级MARL系统优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"{config.output_dir}/{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 优化历史记录
        self.optimization_history = []
        self.best_configurations = {}
        
        print(f"🚀 高级MARL优化器初始化完成")
        print(f"📊 优化级别: {config.optimization_level.value}")
        print(f"🔬 试验次数: {config.n_trials}")
        print(f"🎯 训练轮数上限: {config.max_train_episodes}")
        print(f"📁 结果目录: {self.results_dir}")
        
        # 根据优化级别设置搜索空间
        self.search_space_functions = {
            OptimizationLevel.HYPERPARAMS_ONLY: self._suggest_hyperparameters_only,
            OptimizationLevel.ENVIRONMENT_DESIGN: self._suggest_environment_design,
            OptimizationLevel.REWARD_ENGINEERING: self._suggest_reward_engineering,
            OptimizationLevel.ARCHITECTURE_SEARCH: self._suggest_architecture_search,
            OptimizationLevel.FULL_SYSTEM: self._suggest_full_system
        }

    def _suggest_hyperparameters_only(self, trial: optuna.Trial) -> Dict[str, Any]:
        """基础超参数优化（与原脚本类似）"""
        return {
            'learning_rate_config': {
                'initial_lr': trial.suggest_float('initial_lr', 1e-5, 1e-3, log=True),
                'end_lr': trial.suggest_float('end_lr', 1e-6, 1e-4, log=True),
                'decay_power': trial.suggest_float('decay_power', 0.5, 1.0, step=0.1)
            },
            'network_config': {
                'hidden_sizes': [
                    trial.suggest_categorical('hidden_size_1', [256, 512, 768, 1024]),
                    trial.suggest_categorical('hidden_size_2', [128, 256, 384, 512])
                ],
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.3, step=0.05),
                'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.3, step=0.05),
                'entropy_coeff': trial.suggest_float('entropy_coeff', 0.01, 0.2, step=0.01),
                'num_policy_updates': trial.suggest_int('num_policy_updates', 5, 20)
            }
        }

    def _suggest_environment_design(self, trial: optuna.Trial) -> Dict[str, Any]:
        """环境设计优化"""
        base_config = self._suggest_hyperparameters_only(trial)
        
        # 观察空间设计优化
        base_config['obs_config'] = {
            'top_n_parts': trial.suggest_int('top_n_parts', 2, 6),
            'include_downstream_info': trial.suggest_categorical('include_downstream_info', [True, False]),
            'time_feature_normalization': trial.suggest_categorical('time_norm', [50.0, 100.0, 200.0]),
            
            # 🌟 新增：状态特征工程
            'include_urgency_features': trial.suggest_categorical('include_urgency', [True, False]),
            'include_workload_balance': trial.suggest_categorical('include_workload', [True, False]),
            'state_history_length': trial.suggest_int('history_length', 1, 5),
            
            # 🌟 新增：动作空间优化
            'action_masking': trial.suggest_categorical('action_masking', [True, False]),
            'priority_based_action_space': trial.suggest_categorical('priority_actions', [True, False])
        }
        
        return base_config

    def _suggest_reward_engineering(self, trial: optuna.Trial) -> Dict[str, Any]:
        """奖励函数工程优化"""
        base_config = self._suggest_environment_design(trial)
        
        # 🌟 高级奖励设计
        base_config['reward_config'] = {
            # 基础奖励组件
            'part_completion_reward': trial.suggest_float('part_reward', 5.0, 50.0),
            'order_completion_reward': trial.suggest_float('order_reward', 25.0, 200.0),
            
            # 时间相关奖励
            'early_completion_bonus': trial.suggest_float('early_bonus', 0.0, 10.0),
            'lateness_penalty_type': trial.suggest_categorical('penalty_type', ['linear', 'quadratic', 'exponential']),
            'continuous_lateness_penalty': trial.suggest_float('lateness_penalty', -2.0, -0.1),
            
            # 效率奖励
            'idle_penalty': trial.suggest_float('idle_penalty', -5.0, -0.5),
            'work_bonus': trial.suggest_float('work_bonus', 0.5, 5.0),
            'utilization_bonus_weight': trial.suggest_float('util_bonus', 0.0, 5.0),
            
            # 🌟 协调奖励（多智能体特有）
            'coordination_bonus': trial.suggest_float('coordination_bonus', 0.0, 3.0),
            'bottleneck_awareness_bonus': trial.suggest_float('bottleneck_bonus', 0.0, 5.0),
            
            # 🌟 稀疏 vs 密集奖励
            'reward_frequency': trial.suggest_categorical('reward_freq', ['step', 'completion', 'mixed']),
            'shaped_reward_weight': trial.suggest_float('shaped_weight', 0.0, 1.0),
            
            # 🌟 多目标权重自动调优
            'completion_weight': trial.suggest_float('completion_w', 0.3, 0.7),
            'tardiness_weight': trial.suggest_float('tardiness_w', 0.1, 0.4),
            'makespan_weight': trial.suggest_float('makespan_w', 0.05, 0.3),
            'utilization_weight': trial.suggest_float('utilization_w', 0.05, 0.2)
        }
        
        return base_config

    def _suggest_architecture_search(self, trial: optuna.Trial) -> Dict[str, Any]:
        """神经网络架构搜索"""
        base_config = self._suggest_reward_engineering(trial)
        
        # 🌟 高级网络架构设计
        architecture_type = trial.suggest_categorical('architecture_type', 
                                                    ['standard', 'attention', 'residual', 'hierarchical'])
        
        if architecture_type == 'standard':
            # 标准全连接网络
            base_config['network_config'].update({
                'architecture_type': 'standard',
                'hidden_sizes': [
                    trial.suggest_categorical('hidden_1', [128, 256, 512, 768, 1024]),
                    trial.suggest_categorical('hidden_2', [64, 128, 256, 384, 512]),
                    trial.suggest_categorical('hidden_3', [0, 64, 128, 256])  # 0表示不使用第三层
                ]
            })
        
        elif architecture_type == 'attention':
            # 注意力机制网络
            base_config['network_config'].update({
                'architecture_type': 'attention',
                'attention_heads': trial.suggest_int('attention_heads', 2, 8),
                'attention_dim': trial.suggest_categorical('attention_dim', [64, 128, 256]),
                'use_self_attention': trial.suggest_categorical('self_attention', [True, False]),
                'use_cross_attention': trial.suggest_categorical('cross_attention', [True, False])
            })
        
        elif architecture_type == 'residual':
            # 残差网络
            base_config['network_config'].update({
                'architecture_type': 'residual',
                'residual_blocks': trial.suggest_int('residual_blocks', 2, 6),
                'block_size': trial.suggest_categorical('block_size', [128, 256, 512])
            })
        
        elif architecture_type == 'hierarchical':
            # 分层网络结构
            base_config['network_config'].update({
                'architecture_type': 'hierarchical',
                'low_level_dim': trial.suggest_categorical('low_level_dim', [64, 128, 256]),
                'high_level_dim': trial.suggest_categorical('high_level_dim', [128, 256, 512]),
                'hierarchy_levels': trial.suggest_int('hierarchy_levels', 2, 4)
            })
        
        # 🌟 激活函数和正则化优化
        base_config['network_config'].update({
            'activation_function': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish', 'leaky_relu']),
            'batch_normalization': trial.suggest_categorical('batch_norm', [True, False]),
            'layer_normalization': trial.suggest_categorical('layer_norm', [True, False]),
            'gradient_clipping': trial.suggest_float('grad_clip', 0.5, 2.0),
            'weight_decay': trial.suggest_float('weight_decay', 1e-8, 0.01, log=True)
        })
        
        return base_config

    def _suggest_full_system(self, trial: optuna.Trial) -> Dict[str, Any]:
        """全系统优化（包含所有维度）"""
        base_config = self._suggest_architecture_search(trial)
        
        # 🔧 最终修复: 将高级配置解包到顶层，确保数据结构一致性
        # 算法级别的优化
        base_config.update({
            # 多智能体协调机制
            'coordination_mechanism': trial.suggest_categorical('coordination', 
                                                               ['independent', 'parameter_sharing', 'attention_based', 'communication']),
            
            # 经验回放和学习策略
            'experience_replay': trial.suggest_categorical('exp_replay', [True, False]),
            'prioritized_replay': trial.suggest_categorical('prioritized', [True, False]),
            
            # 探索策略
            'exploration_strategy': trial.suggest_categorical('exploration', 
                                                            ['epsilon_greedy', 'boltzmann', 'noisy_networks', 'parameter_noise']),
            'exploration_decay': trial.suggest_float('exploration_decay', 0.99, 0.999),
            
            # 混合策略（RL + 启发式）
            'hybrid_strategy': trial.suggest_categorical('hybrid', [True, False]),
            'heuristic_weight': trial.suggest_float('heuristic_weight', 0.0, 0.5) if base_config.get('hybrid_strategy') else 0.0,
            'heuristic_type': trial.suggest_categorical('heuristic_type', ['SPT', 'EDD', 'FIFO']) if base_config.get('hybrid_strategy') else 'SPT',
            
            # 课程学习优化
            'adaptive_curriculum': trial.suggest_categorical('adaptive_curriculum', [True, False]),
            'curriculum_stages': trial.suggest_int('curriculum_stages', 2, 5),
            'stage_transition_threshold': trial.suggest_float('transition_threshold', 0.7, 0.95),
            'curriculum_strategy': trial.suggest_categorical('curriculum_strategy', 
                                                           ['difficulty_based', 'diversity_based', 'performance_based'])
        })
        
        return base_config

    def comprehensive_evaluation(self, model_path: str, trial_config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """综合评估：在多个测试配置上评估模型"""
        from evaluation import evaluate_marl_model, STATIC_EVAL_CONFIG, GENERALIZATION_CONFIG_1, GENERALIZATION_CONFIG_2, GENERALIZATION_CONFIG_3
        if not self.config.enable_generalization_test:
            # 仅基准测试
            all_kpis, all_scores = evaluate_marl_model(
                model_path, 
                config=STATIC_EVAL_CONFIG,
                env_config_overrides=trial_config, # 🔧 修复: 传递当前试验的环境配置
                generate_gantt=False
            )
            if all_scores is None:
                return -1.0, {}
            
            return np.mean(all_scores), {
                'static_score': np.mean(all_scores),
                'static_std': np.std(all_scores)
            }
        
        # 多配置泛化测试
        test_configs = [
            ("static", STATIC_EVAL_CONFIG),
            ("gen1", GENERALIZATION_CONFIG_1),
            ("gen2", GENERALIZATION_CONFIG_2),
            ("gen3", GENERALIZATION_CONFIG_3)
        ]
        
        all_results = {}
        total_score = 0.0
        valid_tests = 0
        
        for test_name, test_config in test_configs:
            try:
                all_kpis, all_scores = evaluate_marl_model(
                    model_path, 
                    config=test_config,
                    env_config_overrides=trial_config, # 🔧 修复: 传递当前试验的环境配置
                    generate_gantt=False
                )
                
                if all_scores is not None and len(all_scores) > 0:
                    mean_score = np.mean(all_scores)
                    all_results[f'{test_name}_score'] = mean_score
                    all_results[f'{test_name}_std'] = np.std(all_scores)
                    total_score += mean_score
                    valid_tests += 1
                else:
                    all_results[f'{test_name}_score'] = -1.0
                    all_results[f'{test_name}_std'] = 0.0
                    
            except Exception as e:
                print(f"⚠️ 测试配置 {test_name} 失败: {e}")
                all_results[f'{test_name}_score'] = -1.0
                all_results[f'{test_name}_std'] = 0.0
        
        # 计算综合分数（所有有效测试的平均分）
        final_score = total_score / valid_tests if valid_tests > 0 else -1.0
        
        # 🌟 泛化能力评估：奖励在不同配置间表现稳定的模型
        if valid_tests > 1:
            scores = [all_results[f'{name}_score'] for name, _ in test_configs if all_results.get(f'{name}_score', -1) > 0]
            if len(scores) > 1:
                score_std = np.std(scores)
                # 稳定性奖励：分数越稳定（标准差越小），获得额外奖励
                stability_bonus = max(0, 0.1 - score_std)  # 最多10%的稳定性奖励
                final_score += stability_bonus
                all_results['stability_bonus'] = stability_bonus
                all_results['score_stability'] = score_std
        
        return final_score, all_results

    def objective_function(self, trial: optuna.Trial) -> float:
        """Optuna目标函数"""
        from mappo.ppo_marl_train import SimplePPOTrainer
        trial_start_time = time.time()
        
        # 1. 根据优化级别生成配置
        search_function = self.search_space_functions[self.config.optimization_level]
        optimization_delta = search_function(trial)
        
        print(f"\n🔬 Trial {trial.number}: 开始优化 ({self.config.optimization_level.value})")
        print(f"📋 当前配置摘要: {len(optimization_delta)} 个配置类别")
        
        try:
            # 🔧 关键修复 V2: 构建完整的、自包含的配置包，不再修改全局状态
            trial_config = {
                'PPO_NETWORK_CONFIG': copy.deepcopy(w_factory_config.PPO_NETWORK_CONFIG),
                'LEARNING_RATE_CONFIG': copy.deepcopy(w_factory_config.LEARNING_RATE_CONFIG),
                'ADAPTIVE_TRAINING_CONFIG': copy.deepcopy(w_factory_config.ADAPTIVE_TRAINING_CONFIG),
                'REWARD_CONFIG': copy.deepcopy(w_factory_config.REWARD_CONFIG),
                'ENHANCED_OBS_CONFIG': copy.deepcopy(w_factory_config.ENHANCED_OBS_CONFIG),
                'ACTION_CONFIG_ENHANCED': copy.deepcopy(w_factory_config.ACTION_CONFIG_ENHANCED)
            }
            
            # 将Optuna建议的增量配置合并到完整配置中
            if 'learning_rate_config' in optimization_delta:
                trial_config['LEARNING_RATE_CONFIG'].update(optimization_delta['learning_rate_config'])
            if 'network_config' in optimization_delta:
                trial_config['PPO_NETWORK_CONFIG'].update(optimization_delta['network_config'])
            if 'obs_config' in optimization_delta:
                trial_config['ENHANCED_OBS_CONFIG'].update(optimization_delta['obs_config'])
                # 同步动作空间
                if 'top_n_parts' in optimization_delta['obs_config']:
                    trial_config['ACTION_CONFIG_ENHANCED']['action_space_size'] = optimization_delta['obs_config']['top_n_parts'] + 1
            if 'reward_config' in optimization_delta:
                trial_config['REWARD_CONFIG'].update(optimization_delta['reward_config'])

            # 🔧 最终修复: 将算法和课程学习的顶层配置也合并进来
            # 这些键直接存在于 optimization_delta 的顶层
            top_level_keys_to_copy = [
                'coordination_mechanism', 'experience_replay', 'prioritized_replay',
                'exploration_strategy', 'exploration_decay', 'hybrid_strategy',
                'heuristic_weight', 'heuristic_type', 'adaptive_curriculum',
                'curriculum_stages', 'stage_transition_threshold', 'curriculum_strategy'
            ]
            for key in top_level_keys_to_copy:
                if key in optimization_delta:
                    trial_config[key] = optimization_delta[key]
                    
            # 3. 创建和训练模型
            trainer = SimplePPOTrainer(
                # 从构建好的配置包中获取学习率
                initial_lr=trial_config['LEARNING_RATE_CONFIG'].get('initial_lr', 3e-4),
                total_train_episodes=self.config.max_train_episodes,
                steps_per_episode=800,
                training_targets=trial_config['ADAPTIVE_TRAINING_CONFIG'],
                env_config=trial_config  # 🔧 修复：传入完整的、自包含的配置包
            )
            
            # 训练
            results = trainer.train(
                max_episodes=self.config.max_train_episodes,
                steps_per_episode=800,
                eval_frequency=20,
                adaptive_mode=True
            )
            
            if results is None:
                print(f"❌ Trial {trial.number}: 训练失败")
                return -1.0
            
            # 4. 保存模型
            trial_model_dir = f"{self.results_dir}/trial_{trial.number}"
            os.makedirs(trial_model_dir, exist_ok=True)
            model_path = trainer.save_model(f"{trial_model_dir}/best_model")
            
            if not model_path:
                print(f"❌ Trial {trial.number}: 模型保存失败")
                return -1.0
            
            # 5. 综合评估
            final_score, detailed_results = self.comprehensive_evaluation(model_path, trial_config)
            
            # 6. 记录试验信息
            trial_duration = time.time() - trial_start_time
            trial_info = {
                'trial_number': trial.number,
                'optimization_level': self.config.optimization_level.value,
                'configuration': optimization_delta, # 只记录增量部分
                'final_score': float(final_score),
                'detailed_results': detailed_results,
                'training_episodes': len(trainer.iteration_times) if hasattr(trainer, 'iteration_times') else 0,
                'trial_duration': trial_duration,
                'model_path': model_path
            }
            
            # 保存详细信息
            with open(f"{trial_model_dir}/trial_info.json", 'w', encoding='utf-8') as f:
                json.dump(trial_info, f, indent=2, ensure_ascii=False)
            
            self.optimization_history.append(trial_info)
            
            print(f"✅ Trial {trial.number}: 完成")
            print(f"📊 综合分数: {final_score:.4f}")
            print(f"⏱️ 用时: {trial_duration/60:.1f}分钟")
            
            return final_score
            
        except Exception as e:
            print(f"❌ Trial {trial.number}: 发生错误 - {str(e)}")
            import traceback
            traceback.print_exc()
            return -1.0
            
        finally:
            # 恢复配置
            pass # 不再需要恢复配置，因为每次试验都是自包含的

    def run_optimization(self):
        """启动优化过程"""
        print("="*80)
        print("🚀 启动MARL系统全面自动化优化")
        print(f"🎯 优化级别: {self.config.optimization_level.value}")
        print(f"🔬 计划试验: {self.config.n_trials} 次")
        print(f"🌐 泛化测试: {'启用' if self.config.enable_generalization_test else '禁用'}")
        print("="*80)
        
        # 创建Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=10),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=15)
        )
        
        start_time = time.time()
        
        try:
            # 开始优化
            study.optimize(
                self.objective_function,
                n_trials=self.config.n_trials,
                callbacks=[self._trial_callback]
            )
            
            # 分析和保存结果
            total_time = time.time() - start_time
            self._generate_comprehensive_report(study, total_time)
            
            return study
            
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断了优化过程")
            if len(study.trials) > 0:
                self._generate_comprehensive_report(study, time.time() - start_time)
            return study
        
        except Exception as e:
            print(f"\n❌ 优化过程发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _trial_callback(self, study, trial):
        """试验回调"""
        if trial.value is not None:
            current_best = study.best_value if study.best_trial else -float('inf')
            print(f"🔄 试验进度: {len(study.trials)}/{self.config.n_trials} " +
                  f"| 当前最佳: {current_best:.4f} " +
                  f"| 本次: {trial.value:.4f}")

    def _generate_comprehensive_report(self, study, total_time):
        """生成综合优化报告"""
        print("\n" + "="*80)
        print("🎉 MARL系统全面优化完成!")
        print(f"⏱️ 总用时: {total_time/3600:.2f}小时")
        print(f"🔬 完成试验: {len(study.trials)}")
        print("="*80)
        
        if len(study.trials) == 0:
            print("⚠️ 没有可用的试验结果")
            return
        
        best_trial = study.best_trial
        
        print(f"\n🏆 最佳优化结果:")
        print(f"📊 最佳综合分数: {best_trial.value:.4f}")
        print(f"🔢 最佳试验编号: {best_trial.number}")
        print(f"🎛️ 优化级别: {self.config.optimization_level.value}")
        
        # 保存最终结果
        comprehensive_results = {
            'optimization_config': {
                'level': self.config.optimization_level.value,
                'n_trials': self.config.n_trials,
                'enable_generalization': self.config.enable_generalization_test,
                'max_train_episodes': self.config.max_train_episodes
            },
            'best_trial': {
                'number': best_trial.number,
                'score': best_trial.value,
                'params': best_trial.params
            },
            'optimization_history': self.optimization_history,
            'study_statistics': {
                'total_time_hours': total_time / 3600,
                'completed_trials': len(study.trials),
                'successful_trials': len([t for t in study.trials if t.value is not None and t.value > 0])
            },
            'timestamp': self.timestamp
        }
        
        results_path = f"{self.results_dir}/comprehensive_optimization_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 详细结果已保存至: {results_path}")
        
        # 生成优化建议
        self._generate_optimization_insights(study)

    def _generate_optimization_insights(self, study):
        """生成优化见解和建议"""
        print(f"\n💡 优化见解和建议:")
        print("="*60)
        
        # 参数重要性分析
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"📈 最重要的参数 (Top 5):")
            for i, (param_name, importance_value) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]):
                print(f"  {i+1}. {param_name}: {importance_value:.4f}")
        except:
            print("⚠️ 参数重要性分析不可用")
        
        # 性能趋势分析
        successful_trials = [t for t in study.trials if t.value is not None and t.value > 0]
        if len(successful_trials) >= 5:
            scores = [t.value for t in successful_trials]
            recent_scores = scores[-5:]  # 最近5次试验
            early_scores = scores[:5]    # 前5次试验
            
            recent_avg = np.mean(recent_scores)
            early_avg = np.mean(early_scores)
            
            print(f"\n📊 性能趋势分析:")
            print(f"  早期平均分数: {early_avg:.4f}")
            print(f"  近期平均分数: {recent_avg:.4f}")
            print(f"  改进幅度: {((recent_avg - early_avg) / early_avg * 100):+.1f}%")
        
        # 配置建议
        print(f"\n🔧 下一步优化建议:")
        if self.config.optimization_level != OptimizationLevel.FULL_SYSTEM:
            print(f"  1. 尝试更高级别的优化: {OptimizationLevel.FULL_SYSTEM.value}")
        
        if self.config.n_trials < 100:
            print(f"  2. 增加试验次数以获得更稳定的结果")
        
        if not self.config.enable_generalization_test:
            print(f"  3. 启用泛化测试以评估模型鲁棒性")
        
        print(f"  4. 考虑针对特定领域知识进行定制化优化")
        print(f"  5. 尝试集成其他MARL算法（如MADDPG、QMIX等）")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MARL系统全面自动化优化")
    parser.add_argument("--level", type=str, 
                       choices=['hyperparams', 'environment', 'reward', 'architecture', 'full'],
                       default='full',
                       help="优化级别")
    parser.add_argument("--trials", type=int, default=100, help="优化试验次数")
    parser.add_argument("--train_episodes", type=int, default=300, help="每次试验的最大训练轮数")
    parser.add_argument("--eval_episodes", type=int, default=15, help="评估回合数")
    parser.add_argument("--no_generalization", action="store_true", help="禁用泛化测试")
    parser.add_argument("--output_dir", type=str, default="advanced_optimization_results", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建优化配置
    config = OptimizationConfig(
        n_trials=args.trials,
        n_eval_episodes=args.eval_episodes,
        max_train_episodes=args.train_episodes,
        optimization_level=OptimizationLevel(args.level),
        enable_generalization_test=not args.no_generalization,
        output_dir=args.output_dir
    )
    
    # 创建并启动优化器
    optimizer = AdvancedMARL_Optimizer(config)
    study = optimizer.run_optimization()
    
    if study:
        print("\n✅ 全面优化任务完成!")
        print(f"🏆 最佳分数: {study.best_value:.4f}")
    else:
        print("\n❌ 优化任务失败!")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
