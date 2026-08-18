# =============================================================================
# FILE: Scr/agents/dqn_agent.py (VERSION - MEMORY OPTIMIZED)
# =============================================================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import json
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.constants import DQN_CONFIG, NETWORK_TYPES

logger = logging.getLogger(__name__)


# =============================================================================
# SIMPLE DQN NETWORK (Memory Optimized)
# =============================================================================

class SimpleDQN(nn.Module):
    """Simple DQN Network with reduced parameters"""
    
    def __init__(self, state_dim=27, action_dim=5, hidden_dim=128):
        super(SimpleDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if x.size(-1) != self.state_dim:
            if x.size(-1) < self.state_dim:
                padding = torch.zeros(x.size(0), self.state_dim - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[:, :self.state_dim]
        
        return self.network(x)


# =============================================================================
# REPLAY BUFFER (Memory Optimized)
# =============================================================================

class ReplayBuffer:
    """Simple Replay Buffer with limited capacity"""
    
    def __init__(self, capacity: int = 20000):  # Reduced capacity
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int = 32):  # Reduced batch size
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# DQN AGENT (Memory Optimized)
# =============================================================================

class DQNAgent(BaseAgent):
    """
    Memory Optimized DQN Agent
    """
    
    def __init__(
        self,
        env,
        config: Optional[Dict] = None,
        name: str = "DQN_Agent",
        seed: Optional[int] = 42,
        device: str = "cpu",
        track_user: bool = True
    ):
        # Initialize base
        super().__init__(env, config, name, seed, device, track_user)
        
        # Get config
        self.config = config or DQN_CONFIG
        
        # DQN parameters - OPTIMIZED FOR MEMORY
        self.learning_rate = 0.0005  # Fixed
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 32  # Reduced from 64
        self.buffer_size = 20000  # Reduced from 100,000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.hidden_dim = 128  # Reduced from 256
        self.grad_clip = 0.5
        
        # Get dimensions
        self.state_dim = 27
        self.action_dim = 5
        
        print(f"   DQN State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"   Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")
        
        # Networks
        self.policy_net = SimpleDQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.target_net = SimpleDQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        
        # Replay buffer - memory optimized
        self.buffer = ReplayBuffer(capacity=self.buffer_size)
        
        # Epsilon
        self.epsilon = self.epsilon_start
        
        # Counters
        self.update_counter = 0
        self.target_update_freq = 1000
        self.total_steps = 0
        
        # Training history
        self.training_history = {
            'rewards': [],
            'losses': [],
            'episode_lengths': []
        }
        
        self.start_time = time.time()
        self.is_trained = False
        
        logger.info(f"✅ DQN Agent initialized (Memory Optimized)")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action using epsilon-greedy"""
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        
        if state_tensor.size(-1) != self.state_dim:
            if state_tensor.size(-1) < self.state_dim:
                padding = torch.zeros(1, self.state_dim - state_tensor.size(-1), device=self.device)
                state_tensor = torch.cat([state_tensor, padding], dim=-1)
            else:
                state_tensor = state_tensor[:, :self.state_dim]
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict]:
        """Predict action for evaluation"""
        action = self.select_action(observation, deterministic)
        return action, {'q_values': None}
    
    def train(
        self,
        total_timesteps: int = 150000,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Train the DQN agent"""
        print(f"🚀 Starting DQN training for {total_timesteps} steps")
        
        self.start_time = time.time()
        obs, _ = self.env.reset()
        
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        
        episode_reward = 0
        episode_steps = 0
        episode_count = 0
        
        for step in range(total_timesteps):
            # Select action
            action = self.select_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            if isinstance(next_obs, np.ndarray):
                next_obs = next_obs.astype(np.float32)
            
            # Store in replay buffer
            self.buffer.push(obs, action, reward, next_obs, done)
            
            # Update
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            # Train if enough samples
            if len(self.buffer) >= self.batch_size:
                loss = self.update()
                if loss is not None:
                    self.training_history['losses'].append(loss)
            
            # Update target network
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self._soft_update_target()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # End of episode
            if done or truncated:
                self.training_history['rewards'].append(episode_reward)
                self.training_history['episode_lengths'].append(episode_steps)
                episode_count += 1
                
                if episode_count % 10 == 0:
                    avg_reward = np.mean(self.training_history['rewards'][-10:])
                    print(f"Episode {episode_count}: Reward = {episode_reward:.1f}, Avg = {avg_reward:.1f}")
                
                # Reset
                obs, _ = self.env.reset()
                if isinstance(obs, np.ndarray):
                    obs = obs.astype(np.float32)
                episode_reward = 0
                episode_steps = 0
                
                if callback and episode_count % 10 == 0:
                    callback(self, step, episode_reward, episode_steps)
            
            self.total_steps = step + 1
        
        self.is_trained = True
        
        # Calculate mean reward
        mean_reward = np.mean(self.training_history.get('rewards', [0]))
        
        return {
            'mean_reward': mean_reward,
            'total_steps': total_timesteps,
            'total_episodes': episode_count,
            'training_time_seconds': time.time() - self.start_time,
            'parameters': sum(p.numel() for p in self.policy_net.parameters())
        }
    
    def update(self) -> Optional[float]:
        """Update the agent using batch"""
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Fix dimensions
        if states.size(-1) != self.state_dim:
            if states.size(-1) < self.state_dim:
                padding = torch.zeros(states.size(0), self.state_dim - states.size(-1), device=self.device)
                states = torch.cat([states, padding], dim=-1)
            else:
                states = states[:, :self.state_dim]
        
        if next_states.size(-1) != self.state_dim:
            if next_states.size(-1) < self.state_dim:
                padding = torch.zeros(next_states.size(0), self.state_dim - next_states.size(-1), device=self.device)
                next_states = torch.cat([next_states, padding], dim=-1)
            else:
                next_states = next_states[:, :self.state_dim]
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Calculate loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()
        
        return loss.item()
    
    def _soft_update_target(self):
        """Soft update target network"""
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str) -> None:
        """Save the agent"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }, path)
        logger.info(f"✅ DQN Agent saved to: {path}")
    
    def load(self, path: str) -> None:
        """Load the agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.is_trained = checkpoint.get('is_trained', False)
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        logger.info(f"✅ DQN Agent loaded from: {path}")
    
    def save_training_history(self, path: str) -> None:
        """Save training history to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        logger.info(f"Training history saved to: {path}")