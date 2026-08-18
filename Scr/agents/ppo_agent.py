# =============================================================================
# FILE: Scr/agents/ppo_agent.py (VERSION FIXED - LIKE DQN)
# =============================================================================
# PURPOSE: PPO Agent with memory optimization
# =============================================================================

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.constants import PPO_CONFIG

logger = logging.getLogger(__name__)


# =============================================================================
# SIMPLE ACTOR-CRITIC NETWORK (Memory Optimized)
# =============================================================================

class SimpleActorCritic(nn.Module):
    """Simple Actor-Critic Network with reduced parameters"""
    
    def __init__(self, state_dim=27, action_dim=5, hidden_dim=128):
        super(SimpleActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if x.size(-1) != self.state_dim:
            if x.size(-1) < self.state_dim:
                padding = torch.zeros(x.size(0), self.state_dim - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[:, :self.state_dim]
        
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        
        return logits, value


# =============================================================================
# PPO AGENT (Memory Optimized)
# =============================================================================

class PPOAgent(BaseAgent):
    """
    Memory Optimized PPO Agent
    """
    
    def __init__(
        self,
        env,
        config: Optional[Dict] = None,
        name: str = "PPO_Agent",
        seed: Optional[int] = 42,
        device: str = "cpu",
        track_user: bool = True
    ):
        super().__init__(env, config, name, seed, device, track_user)
        
        # Get config
        self.config = config or {}
        
        # PPO parameters - OPTIMIZED
        self.learning_rate = 0.0003
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.hidden_dim = 128  # Reduced
        self.batch_size = 64
        self.epochs_per_update = 10
        self.max_grad_norm = 0.5
        self.target_kl = 0.01
        
        # Get dimensions - FIXED
        self.state_dim = 27
        self.action_dim = 5
        
        print(f"   PPO State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"   Hidden dim: {self.hidden_dim}")
        
        # Network
        self.network = SimpleActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )
        
        # Memory for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        # Training history
        self.training_history = {
            'rewards': [],
            'losses': [],
            'episode_lengths': []
        }
        
        self.start_time = time.time()
        self.is_trained = False
        self.total_steps = 0
        
        logger.info(f"✅ PPO Agent initialized (Memory Optimized)")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Select action and compute log probability and value"""
        # Fix state dimension
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1).item()
                log_prob = torch.log(probs[0, action] + 1e-8).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob, value.item()
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict]:
        """Predict action for evaluation"""
        action, _, _ = self.select_action(observation, deterministic)
        return action, {'value': None, 'log_prob': None}
    
    def collect_trajectory(self, max_steps: int = 2048) -> Dict:
        """Collect trajectory data"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for _ in range(max_steps):
            action, log_prob, value = self.select_action(obs, deterministic=False)
            
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            self.states.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done or truncated)
            self.log_probs.append(log_prob)
            self.values.append(value)
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            if done or truncated:
                obs, _ = self.env.reset()
                break
        
        # Store last value for bootstrapping
        with torch.no_grad():
            last_state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _, last_value = self.network(last_state)
            last_value = last_value.item()
        
        advantages, returns = self._compute_gae(last_value)
        
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'advantages': advantages,
            'returns': returns,
            'episode_reward': episode_reward,
            'episode_steps': episode_steps
        }
    
    def _compute_gae(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns"""
        if len(self.rewards) == 0:
            return np.array([]), np.array([])
        
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
        
        if len(advantages) > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, trajectory: Dict) -> Dict[str, float]:
        """Update policy using PPO"""
        states = torch.FloatTensor(trajectory['states']).to(self.device)
        actions = torch.LongTensor(trajectory['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(trajectory['log_probs']).to(self.device)
        advantages = torch.FloatTensor(trajectory['advantages']).to(self.device)
        returns = torch.FloatTensor(trajectory['returns']).to(self.device)
        old_values = torch.FloatTensor(trajectory['values']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch training
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        
        for _ in range(self.epochs_per_update):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                logits, values = self.network(batch_states)
                
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                entropy_loss = -self.entropy_coef * entropy
                
                loss = policy_loss + 0.5 * value_loss + entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
        
        return {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0,
            'value_loss': np.mean(value_losses) if value_losses else 0,
            'entropy_loss': np.mean(entropy_losses) if entropy_losses else 0,
            'total_loss': np.mean(total_losses) if total_losses else 0
        }
    
    def train(
        self,
        total_timesteps: int = 150000,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Train the PPO agent"""
        print(f"🚀 Starting PPO training for {total_timesteps} steps")
        
        self.start_time = time.time()
        total_steps = 0
        episode_count = 0
        
        while total_steps < total_timesteps:
            trajectory = self.collect_trajectory(max_steps=2048)
            
            update_stats = self.update(trajectory)
            
            total_steps += len(trajectory['states'])
            episode_count += 1
            
            self.training_history['rewards'].append(trajectory['episode_reward'])
            self.training_history['episode_lengths'].append(trajectory['episode_steps'])
            if 'total_loss' in update_stats:
                self.training_history['losses'].append(update_stats['total_loss'])
            
            if episode_count % 10 == 0:
                avg_reward = np.mean(self.training_history['rewards'][-10:])
                print(f"Episode {episode_count}: Reward = {trajectory['episode_reward']:.1f}, Avg = {avg_reward:.1f}")
            
            if callback and episode_count % 10 == 0:
                callback(self)
        
        self.is_trained = True
        self.total_steps = total_steps
        
        mean_reward = np.mean(self.training_history.get('rewards', [0]))
        
        return {
            'mean_reward': mean_reward,
            'total_steps': total_steps,
            'total_episodes': episode_count,
            'training_time_seconds': time.time() - self.start_time,
            'parameters': sum(p.numel() for p in self.network.parameters())
        }
    
    def save(self, path: str) -> None:
        """Save the agent"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }, path)
        logger.info(f"✅ PPO Agent saved to: {path}")
    
    def load(self, path: str) -> None:
        """Load the agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.is_trained = checkpoint.get('is_trained', False)
        self.training_history = checkpoint.get('training_history', {})
        logger.info(f"✅ PPO Agent loaded from: {path}")
    
    def get_parameter_count(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
    
    def save_training_history(self, path: str) -> None:
        """Save training history to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        logger.info(f"Training history saved to: {path}")