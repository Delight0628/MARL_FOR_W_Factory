"""
MAPPO并行Worker模块
====================
实现多进程并行经验采集

功能说明：
- 每个Worker在独立进程中运行独立的环境实例
- 智能体条件化全局状态（拼接one-hot编码）
- 支持GPU/CPU设备管理和内存优化
- 处理episode截断和bootstrap价值估计
"""

import os
import random
import numpy as np
import tensorflow as tf
import gymnasium as gym
from typing import Dict, List, Tuple, Any, Optional

# 导入配置和依赖
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environments.w_factory_env import make_parallel_env
from mappo.ppo_buffer import ExperienceBuffer
from mappo.ppo_network import PPONetwork


def run_simulation_worker(network_weights: Dict[str, List[np.ndarray]],
                          state_dim: int, action_space: gym.spaces.Space, num_steps: int, seed: int, 
                          global_state_dim: int, network_config: Dict[str, Any], curriculum_config: Dict[str, Any] = None) -> Tuple[Dict[str, ExperienceBuffer], float, Optional[np.ndarray], bool, bool]:
    """
    并行Worker进程：运行独立环境实例并采集经验
    
    执行流程：
    1. 配置GPU/CPU设备
    2. 创建环境和网络实例
    3. 加载网络权重
    4. 运行num_steps步仿真
    5. 返回经验缓冲区和统计数据
    
    Args:
        network_weights: 网络权重 {'actor': [...], 'critic': [...]}
        state_dim: 局部观测维度
        action_space: 动作空间（支持Discrete/MultiDiscrete）
        num_steps: 单次仿真最大步数
        seed: 随机种子
        global_state_dim: 全局状态维度（含智能体one-hot）
        network_config: 网络架构配置字典
        curriculum_config: 课程学习/任务配置（可选）
        
    Returns:
        tuple: (buffers, total_reward, last_values, terminated, graduated)
            - buffers: 各智能体的经验缓冲区
            - total_reward: 总奖励
            - last_values: 最终状态价值（用于bootstrap）
            - terminated: 是否正常终止
            - graduated: 是否达到课程毕业标准
    """
    try:
        # ========== GPU/CPU设备配置 ==========
        try:
            import os as _os
            if _os.environ.get('FORCE_WORKER_CPU', '0') == '1':
                # 强制CPU模式
                _os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                try:
                    _gpus = tf.config.list_physical_devices('GPU')
                    if _gpus:
                        tf.config.set_visible_devices([], 'GPU')
                except Exception:
                    pass
            else:
                # 启用GPU内存增长
                try:
                    _gpus = tf.config.list_physical_devices('GPU')
                    for _g in _gpus:
                        tf.config.experimental.set_memory_growth(_g, True)
                except Exception:
                    pass
            
            # 限制子进程线程数，提升稳定性
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
            
        # ========== 环境和随机种子初始化 ==========
        if curriculum_config:
            worker_seed = seed + curriculum_config.get('worker_id', 0)
            env_config = curriculum_config.copy()
        else:
            worker_seed = seed
            env_config = {}
        
        # 设置worker独立的随机种子
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        tf.random.set_seed(worker_seed)

        # 创建仿真环境
        env_config['training_mode'] = True
        env = make_parallel_env(env_config)

        # ========== 智能体条件化全局状态 ==========
        # 为集中式Critic构造包含智能体身份的全局状态
        agent_list = list(env.possible_agents)
        agent_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_list)}
        num_agents = len(agent_list)

        def _condition_global_state(raw_global_state: np.ndarray, agent_id: str) -> np.ndarray:
            """将智能体one-hot编码拼接到全局状态"""
            one_hot = np.zeros((num_agents,), dtype=np.float32)
            idx = agent_to_index.get(agent_id, 0)
            one_hot[idx] = 1.0
            return np.concatenate([raw_global_state.astype(np.float32), one_hot], axis=0)

        # ========== 网络实例化和权重加载 ==========
        tf.keras.backend.clear_session()
        
        # 构建网络（仅用于推理，不需要优化器）
        try:
            network = PPONetwork(
                state_dim=state_dim,
                action_space=action_space,
                lr=None,  # 推理模式：不构建优化器，节省资源
                global_state_dim=global_state_dim,
                network_config=network_config
            )
        except Exception as _e_build:
            # CPU初始化失败时，尝试GPU构建
            if 'vector::_M_range_check' in str(_e_build):
                try:
                    _gpus = tf.config.list_physical_devices('GPU')
                    if _gpus and _os.environ.get('FORCE_WORKER_CPU', '0') != '1':
                        with tf.device('/GPU:0'):
                            network = PPONetwork(
                                state_dim=state_dim,
                                action_space=action_space,
                                lr=None,
                                global_state_dim=global_state_dim,
                                network_config=network_config
                            )
                    else:
                        raise
                except Exception:
                    raise
            else:
                raise
        
        # 加载网络权重
        if network_weights:
            try:
                network.actor.set_weights(network_weights['actor'])
                network.critic.set_weights(network_weights['critic'])
            except (ValueError, RuntimeError) as e:
                print(f"⚠️ Worker {curriculum_config.get('worker_id', 'N/A')} 权重加载警告: {e}")
                print(f"   尝试重建网络...")
                # 重建网络作为fallback
                tf.keras.backend.clear_session()
                network = PPONetwork(
                    state_dim=state_dim,
                    action_space=action_space,
                    lr=None,
                    global_state_dim=global_state_dim,
                    network_config=network_config
                )
                # 再次尝试加载
                network.actor.set_weights(network_weights['actor'])
                network.critic.set_weights(network_weights['critic'])

            # 加载后做健壮性校验：若仍为近零权重，放弃该worker采样以避免噪声
            try:
                actor_sum = float(np.sum([np.sum(np.abs(w)) for w in network.actor.get_weights()]))
                critic_sum = float(np.sum([np.sum(np.abs(w)) for w in network.critic.get_weights()]))
                if not np.isfinite(actor_sum) or not np.isfinite(critic_sum) or (actor_sum + critic_sum) < 1e-8:
                    print(f"⚠️ Worker {curriculum_config.get('worker_id', 'N/A')} 权重校验失败（近零或非数），跳过本worker采样。")
                    env.close()
                    return {}, 0.0, None, True, False
            except Exception:
                # 校验异常时安全退出
                env.close()
                return {}, 0.0, None, True, False

        # ========== 仿真循环 ==========
        buffers = {agent: ExperienceBuffer() for agent in env.possible_agents}
        observations, infos = env.reset(seed=worker_seed)
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        total_episode_reward = 0
        terminated_by_graduation = False

        for step in range(num_steps):
            actions = {}
            values = {}
            action_probs = {}
            
            # 为每个智能体获取动作
            active_agents_in_step = list(env.agents)
            for agent in active_agents_in_step:
                if agent in observations:
                    state = tf.expand_dims(observations[agent], 0)
                    # 使用智能体条件化的全局状态
                    conditioned_global = _condition_global_state(infos[agent]['global_state'], agent)
                    global_state = tf.expand_dims(conditioned_global, 0)
                    
                    action, value, action_prob = network.get_action_and_value(state, global_state)
                    
                    actions[agent] = action
                    values[agent] = value
                    action_probs[agent] = action_prob
            
            # 为未观测到的智能体提供默认动作（兼容Discrete/MultiDiscrete）
            for agent in env.possible_agents:
                if agent not in actions:
                    sp = network.action_space
                    if isinstance(sp, gym.spaces.MultiDiscrete):
                        actions[agent] = np.zeros(len(sp.nvec), dtype=sp.dtype)
                    else:
                        actions[agent] = 0
            
            # 执行环境步进
            next_observations, rewards, terminations, truncations, next_infos = env.step(actions)
            
            # 存储经验数据
            for agent in active_agents_in_step:
                if agent in observations:
                    buffers[agent].store(
                        observations[agent], 
                        _condition_global_state(infos[agent]['global_state'], agent), 
                        actions[agent], 
                        rewards[agent], 
                        values[agent],
                        action_probs[agent],
                        terminations[agent],
                        truncations[agent]
                    )
                    episode_rewards[agent] += rewards.get(agent, 0)

            observations = next_observations
            infos = next_infos

            # 检查episode终止条件
            if any(terminations.values()) or any(truncations.values()):
                # 检查课程学习毕业标志
                if 'final_stats' in infos[active_agents_in_step[0]] and \
                   infos[active_agents_in_step[0]].get('final_stats', {}).get('graduated', False):
                    terminated_by_graduation = True

                total_episode_reward = sum(episode_rewards.values())
                
                # 未满缓冲区时重置环境继续收集
                if step < num_steps - 1:
                    observations, infos = env.reset(seed=worker_seed + step + 1)
                    episode_rewards = {agent: 0 for agent in env.possible_agents}
                else:
                    # 缓冲区已满，计算bootstrap价值
                    last_values = {}
                    for agent in active_agents_in_step:
                        if agent in observations:
                            conditioned_global = _condition_global_state(infos[agent]['global_state'], agent)
                            global_state = tf.expand_dims(conditioned_global, 0)
                            # 截断时使用critic估计，否则为0
                            if truncations[agent]:
                                last_values[agent] = network.get_value(global_state)
                            else:
                                last_values[agent] = 0.0
                    
                    env.close()
                    return buffers, total_episode_reward, last_values, any(terminations.values()), terminated_by_graduation

        # 正常结束：计算最终价值估计
        last_values = {}
        active_agents_in_step = list(env.agents)
        for agent in active_agents_in_step:
            if agent in observations:
                conditioned_global = _condition_global_state(infos[agent]['global_state'], agent)
                global_state = tf.expand_dims(conditioned_global, 0)
                last_values[agent] = network.get_value(global_state)
        
        env.close()
        return buffers, total_episode_reward, last_values, False, False

    except Exception as e:
        import traceback
        print(f"Worker {curriculum_config.get('worker_id', 'N/A')} failed with error: {e}")
        traceback.print_exc()
        # 返回空数据以防主进程崩溃
        return {}, 0.0, None, True, False


def _collect_experience_wrapper(args):
    """
    进程池参数解包包装函数
    
    用于兼容ProcessPoolExecutor的submit方法，将元组参数解包后
    调用run_simulation_worker函数
    """
    # 解包参数元组
    actor_weights, critic_weights, state_dim, action_space, num_steps, seed, global_state_dim, network_config, curriculum_config = args
    
    # 构建网络权重字典
    network_weights = {
        'actor': actor_weights,
        'critic': critic_weights
    }
    
    # 调用实际的worker函数
    return run_simulation_worker(
        network_weights=network_weights,
        state_dim=state_dim,
        action_space=action_space,
        num_steps=num_steps,
        seed=seed,
        global_state_dim=global_state_dim,
        network_config=network_config,
        curriculum_config=curriculum_config
    )

