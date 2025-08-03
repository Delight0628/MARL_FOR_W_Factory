"""
真正的多智能体强化学习训练脚本
使用简化的PPO实现，避免Ray的Windows兼容性问题
实现真正的协同学习和策略共享
"""

import os
import sys
import time
import json
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 添加环境路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.w_factory_env import make_parallel_env
from environments.w_factory_config import *

class PPONetwork:
    """简化的PPO网络实现"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # 构建网络
        self.actor, self.critic = self._build_networks()
        
        # 优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr)
        
    def _build_networks(self):
        """构建Actor-Critic网络"""
        # Actor网络 (策略网络)
        actor_input = tf.keras.layers.Input(shape=(self.state_dim,))
        actor_hidden1 = tf.keras.layers.Dense(256, activation='relu')(actor_input)
        actor_hidden2 = tf.keras.layers.Dense(256, activation='relu')(actor_hidden1)
        actor_output = tf.keras.layers.Dense(self.action_dim, activation='softmax')(actor_hidden2)
        actor = tf.keras.Model(inputs=actor_input, outputs=actor_output)
        
        # Critic网络 (价值网络)
        critic_input = tf.keras.layers.Input(shape=(self.state_dim,))
        critic_hidden1 = tf.keras.layers.Dense(256, activation='relu')(critic_input)
        critic_hidden2 = tf.keras.layers.Dense(256, activation='relu')(critic_hidden1)
        critic_output = tf.keras.layers.Dense(1)(critic_hidden2)
        critic = tf.keras.Model(inputs=critic_input, outputs=critic_output)
        
        return actor, critic
    
    def get_action_and_value(self, state: np.ndarray) -> Tuple[int, float, float]:
        """获取动作、动作概率和状态价值"""
        state = tf.expand_dims(state, 0)
        
        # 获取动作概率分布
        action_probs = self.actor(state)
        action_dist = tf.random.categorical(tf.math.log(action_probs), 1)
        action = int(action_dist[0, 0])
        
        # 获取动作概率
        action_prob = float(action_probs[0, action])
        
        # 获取状态价值
        value = float(self.critic(state)[0, 0])
        
        return action, action_prob, value
    
    def get_value(self, state: np.ndarray) -> float:
        """获取状态价值"""
        state = tf.expand_dims(state, 0)
        return float(self.critic(state)[0, 0])
    
    def update(self, states: np.ndarray, actions: np.ndarray, 
               old_probs: np.ndarray, advantages: np.ndarray, 
               returns: np.ndarray, clip_ratio: float = 0.2) -> Dict[str, float]:
        """PPO更新"""
        
        # Actor更新
        with tf.GradientTape() as tape:
            action_probs = self.actor(states)
            action_probs_selected = tf.reduce_sum(
                action_probs * tf.one_hot(actions, self.action_dim), axis=1
            )
            
            # 计算比率
            ratio = action_probs_selected / (old_probs + 1e-8)
            
            # PPO裁剪目标
            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )
            
            # 熵正则化
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            actor_loss -= 0.01 * tf.reduce_mean(entropy)
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Critic更新
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.critic(states))
            critic_loss = tf.reduce_mean(tf.square(returns - values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return {
            'actor_loss': float(actor_loss),
            'critic_loss': float(critic_loss),
            'entropy': float(tf.reduce_mean(entropy))
        }

class ExperienceBuffer:
    """经验缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.action_probs = []
        self.dones = []
        
    def store(self, state, action, reward, value, action_prob, done):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.action_probs.append(action_prob)
        self.dones.append(done)
    
    def get_batch(self, gamma=0.99, lam=0.95):
        """获取批次数据并计算优势函数"""
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        action_probs = np.array(self.action_probs)
        dones = np.array(self.dones)
        
        # 计算GAE优势函数
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # 计算回报
        returns = advantages + values
        
        # 标准化优势函数
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return states, actions, action_probs, advantages, returns
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.action_probs.clear()
        self.dones.clear()

class MARLTrainer:
    """多智能体强化学习训练器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.env = make_parallel_env()
        
        # 共享策略网络 (所有智能体共享同一个网络)
        self.shared_network = PPONetwork(
            state_dim=2,  # [队列长度, 设备状态]
            action_dim=2,  # [IDLE, PROCESS]
            lr=self.config.get('lr', 3e-4)
        )
        
        # 经验缓冲区 (每个智能体一个)
        self.buffers = {
            agent: ExperienceBuffer() 
            for agent in self.env.possible_agents
        }
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
    def collect_experience(self, num_steps: int = 200) -> Dict[str, float]:
        """收集经验"""
        observations, _ = self.env.reset()
        episode_rewards = {agent: 0 for agent in self.env.possible_agents}
        step_count = 0
        
        for step in range(num_steps):
            # 获取所有智能体的动作
            actions = {}
            values = {}
            action_probs = {}
            
            for agent in self.env.agents:
                if agent in observations:
                    action, action_prob, value = self.shared_network.get_action_and_value(
                        observations[agent]
                    )
                    actions[agent] = action
                    values[agent] = value
                    action_probs[agent] = action_prob
            
            # 执行动作
            next_observations, rewards, terminations, truncations, _ = self.env.step(actions)
            
            # 存储经验
            for agent in self.env.agents:
                if agent in observations and agent in actions:
                    done = terminations.get(agent, False) or truncations.get(agent, False)
                    reward = rewards.get(agent, 0)
                    
                    self.buffers[agent].store(
                        state=observations[agent],
                        action=actions[agent],
                        reward=reward,
                        value=values[agent],
                        action_prob=action_probs[agent],
                        done=done
                    )
                    
                    episode_rewards[agent] += reward
            
            observations = next_observations
            step_count += 1
            
            # 检查是否结束
            if any(terminations.values()) or any(truncations.values()):
                observations, _ = self.env.reset()
        
        return episode_rewards
    
    def update_policy(self) -> Dict[str, float]:
        """更新策略"""
        # 合并所有智能体的经验
        all_states = []
        all_actions = []
        all_action_probs = []
        all_advantages = []
        all_returns = []
        
        for agent, buffer in self.buffers.items():
            if len(buffer.states) > 0:
                states, actions, action_probs, advantages, returns = buffer.get_batch()
                
                all_states.extend(states)
                all_actions.extend(actions)
                all_action_probs.extend(action_probs)
                all_advantages.extend(advantages)
                all_returns.extend(returns)
                
                buffer.clear()
        
        if len(all_states) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        
        # 转换为numpy数组
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_action_probs = np.array(all_action_probs)
        all_advantages = np.array(all_advantages)
        all_returns = np.array(all_returns)
        
        # 多次更新
        losses = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        num_updates = 5
        
        for _ in range(num_updates):
            batch_losses = self.shared_network.update(
                states=all_states,
                actions=all_actions,
                old_probs=all_action_probs,
                advantages=all_advantages,
                returns=all_returns
            )
            
            for key in losses:
                losses[key] += batch_losses[key] / num_updates
        
        return losses
    
    def train(self, num_episodes: int = 100, steps_per_episode: int = 200):
        """训练主循环"""
        print("=" * 60)
        print("W工厂多智能体强化学习训练 (真正的MARL)")
        print("=" * 60)
        print(f"算法: PPO (Proximal Policy Optimization)")
        print(f"网络: 共享Actor-Critic网络")
        print(f"智能体数量: {len(self.env.possible_agents)}")
        print(f"训练回合: {num_episodes}")
        print(f"每回合步数: {steps_per_episode}")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # 收集经验
            episode_rewards = self.collect_experience(steps_per_episode)
            
            # 更新策略
            losses = self.update_policy()
            
            # 记录统计
            total_reward = sum(episode_rewards.values())
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps_per_episode)
            self.training_losses.append(losses)
            
            episode_time = time.time() - episode_start
            
            # 输出日志
            if (episode + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
                
                print(f"回合 {episode + 1:4d}/{num_episodes} | "
                      f"奖励: {total_reward:8.2f} | "
                      f"平均奖励: {avg_reward:8.2f} | "
                      f"Actor损失: {losses['actor_loss']:.4f} | "
                      f"Critic损失: {losses['critic_loss']:.4f} | "
                      f"时间: {episode_time:.2f}s")
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("MARL训练完成！")
        print("=" * 60)
        print(f"训练时间: {training_time/60:.2f} 分钟")
        print(f"平均奖励: {np.mean(self.episode_rewards):.2f}")
        print(f"最佳奖励: {max(self.episode_rewards):.2f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'training_losses': self.training_losses,
            'training_time': training_time
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        self.shared_network.actor.save(f"{filepath}_actor.h5")
        self.shared_network.critic.save(f"{filepath}_critic.h5")
        print(f"模型已保存到: {filepath}_actor.h5 和 {filepath}_critic.h5")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估模型"""
        print(f"评估模型 ({num_episodes} 回合)...")
        
        eval_rewards = []
        eval_stats = []
        
        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < 480:  # 最大仿真时间
                actions = {}
                for agent in self.env.agents:
                    if agent in observations:
                        # 使用确定性策略（不探索）
                        state = tf.expand_dims(observations[agent], 0)
                        action_probs = self.shared_network.actor(state)
                        action = int(tf.argmax(action_probs[0]))
                        actions[agent] = action
                
                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                episode_reward += sum(rewards.values())
                step_count += 1
                
                if any(terminations.values()) or any(truncations.values()):
                    # 获取最终统计
                    if any(infos.values()) and "final_stats" in list(infos.values())[0]:
                        eval_stats.append(list(infos.values())[0]["final_stats"])
                    break
            
            eval_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'eval_rewards': eval_rewards,
            'eval_stats': eval_stats
        }

def main():
    """主函数"""
    print("W工厂多智能体强化学习系统 - 真正的MARL训练")
    print("=" * 60)
    
    # 验证配置
    if not validate_config():
        print("配置验证失败")
        return
    
    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    try:
        # 创建训练器
        trainer = MARLTrainer({
            'lr': 3e-4,
        })
        
        # 开始训练
        results = trainer.train(num_episodes=100, steps_per_episode=200)
        
        # 保存模型
        os.makedirs("models", exist_ok=True)
        trainer.save_model("models/marl_model")
        
        # 评估模型
        eval_results = trainer.evaluate(num_episodes=10)
        
        # 保存结果
        final_results = {
            'training_results': results,
            'evaluation_results': eval_results,
            'config': {
                'algorithm': 'PPO',
                'network': 'Shared Actor-Critic',
                'agents': trainer.env.possible_agents,
                'state_dim': 2,
                'action_dim': 2
            }
        }
        
        os.makedirs("results", exist_ok=True)
        results_file = f"results/marl_training_results_{int(time.time())}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {results_file}")
        
        # 显示最终结果
        print("\n" + "=" * 60)
        print("最终评估结果")
        print("=" * 60)
        print(f"平均奖励: {eval_results['mean_reward']:.2f}")
        print(f"奖励标准差: {eval_results['std_reward']:.2f}")
        
        if eval_results['eval_stats']:
            avg_makespan = np.mean([s.get('makespan', 0) for s in eval_results['eval_stats']])
            avg_tardiness = np.mean([s.get('total_tardiness', 0) for s in eval_results['eval_stats']])
            avg_completed = np.mean([s.get('total_parts', 0) for s in eval_results['eval_stats']])
            
            print(f"平均Makespan: {avg_makespan:.1f}")
            print(f"平均延期时间: {avg_tardiness:.1f}")
            print(f"平均完成零件: {avg_completed:.1f}")
        
        print("\n🎉 真正的MARL训练完成！")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 