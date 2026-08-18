# =============================================================================
# FILE: Scr/agents/lstm_agent.py (VERSION 7.0 - WORKING)
# =============================================================================

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from typing import Dict, Any, Optional, List, Tuple
import time
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# =============================================================================
# SIMPLE LSTM NETWORK (NO SEQUENCE RESHAPING ISSUES)
# =============================================================================

class SimpleLSTMNet(nn.Module):
    def __init__(self, state_dim=27, action_dim=5, hidden_dim=64):
        super(SimpleLSTMNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, state_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Fix dimension if needed
        if x.size(-1) != self.state_dim:
            if x.size(-1) < self.state_dim:
                padding = torch.zeros(x.size(0), x.size(1), 
                                    self.state_dim - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[:, :, :self.state_dim]
        
        lstm_out, hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        
        logits = self.actor(last_out)
        value = self.critic(last_out)
        
        return logits, value, hidden


# =============================================================================
# LSTM AGENT (FIXED)
# =============================================================================

class LSTMAgent(BaseAgent):
    
    def __init__(self, env, config=None, name="LSTM_Agent", seed=42, device="cpu", track_user=True):
        super().__init__(env, config, name, seed, device, track_user)
        
        self.config = config or {}
        
        # Hyperparameters - REDUCED for memory
        self.lr = 0.0003
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.hidden_dim = 64  # Reduced
        self.batch_size = 16  # Reduced
        self.epochs = 2       # Reduced
        self.grad_clip = 0.5
        
        self.state_dim = 27
        self.action_dim = 5
        
        print(f"   LSTM State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"   Hidden dim: {self.hidden_dim}, Batch size: {self.batch_size}")
        
        self.policy = SimpleLSTMNet(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.hidden = None
        self.memory = []
        self.history = {'rewards': [], 'losses': []}
        
        self.total_steps = 0
        self.is_trained = False
        
        logger.info(f"✅ LSTM Agent initialized")
    
    def _init_hidden(self, batch=1):
        h = torch.zeros(1, batch, self.hidden_dim).to(self.device)
        c = torch.zeros(1, batch, self.hidden_dim).to(self.device)
        return (h, c)
    
    def _fix_state(self, state):
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        return state
    
    def select_action(self, state, deterministic=False):
        state = self._fix_state(state)
        state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        
        if self.hidden is None:
            self.hidden = self._init_hidden(1)
        
        with torch.no_grad():
            logits, value, self.hidden = self.policy(state_t, self.hidden)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1).item()
                log_prob = 0.0
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob, value.item()
    
    def predict(self, obs, deterministic=True):
        if self.hidden is None:
            self.hidden = self._init_hidden(1)
        action, _, _ = self.select_action(obs, deterministic)
        return action, {}
    
    def collect_trajectory(self, max_steps=500):  # Reduced
        self.memory = []
        
        obs, _ = self.env.reset()
        self.hidden = self._init_hidden(1)
        ep_reward = 0
        steps = 0
        
        for _ in range(max_steps):
            action, log_prob, value = self.select_action(obs)
            next_obs, reward, done, truncated, _ = self.env.step(action)
            
            self.memory.append({
                'state': obs,
                'action': action,
                'reward': reward,
                'done': done or truncated,
                'log_prob': log_prob,
                'value': value
            })
            
            obs = next_obs
            ep_reward += reward
            steps += 1
            
            if done or truncated:
                obs, _ = self.env.reset()
                self.hidden = self._init_hidden(1)
                break
        
        # Compute returns and advantages
        rewards = [m['reward'] for m in self.memory]
        values = [m['value'] for m in self.memory]
        dones = [m['done'] for m in self.memory]
        
        # Last value
        last_val = 0
        if len(self.memory) > 0:
            last_s = torch.FloatTensor(self.memory[-1]['state']).unsqueeze(0).unsqueeze(0).to(self.device)
            _, last_val, _ = self.policy(last_s, self.hidden)
            last_val = last_val.item()
        
        returns = np.zeros(len(rewards))
        advantages = np.zeros(len(rewards))
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = last_val
                non_terminal = 1.0 - dones[t]
            else:
                next_val = values[t + 1]
                non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        if len(advantages) > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store in memory
        for i, m in enumerate(self.memory):
            m['advantage'] = advantages[i]
            m['return'] = returns[i]
        
        return ep_reward, steps
    
    def update(self):
        if len(self.memory) < 2:
            return 0.0
        
        # Extract data
        states = torch.FloatTensor([m['state'] for m in self.memory]).to(self.device)
        actions = torch.LongTensor([m['action'] for m in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([m['log_prob'] for m in self.memory]).to(self.device)
        advantages = torch.FloatTensor([m['advantage'] for m in self.memory]).to(self.device)
        returns = torch.FloatTensor([m['return'] for m in self.memory]).to(self.device)
        
        n = len(self.memory)
        idx = np.arange(n)
        total_loss = 0
        count = 0
        
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                b_idx = idx[start:end]
                
                if len(b_idx) < 2:
                    continue
                
                # ===== KEY FIX: Add time dimension =====
                b_states = states[b_idx].unsqueeze(1)  # (batch, 1, state_dim)
                b_actions = actions[b_idx]
                b_old_log_probs = old_log_probs[b_idx]
                b_adv = advantages[b_idx]
                b_ret = returns[b_idx]
                
                # Forward pass
                logits, values, _ = self.policy(b_states, self._init_hidden(len(b_idx)))
                
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(b_actions)
                
                # PPO loss
                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.squeeze(), b_ret)
                entropy_loss = -self.entropy_coef * dist.entropy().mean()
                
                loss = policy_loss + 0.5 * value_loss + entropy_loss
                total_loss += loss.item()
                count += 1
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.optimizer.step()
        
        return total_loss / max(1, count)
    
    def train(self, total_timesteps=150000, callback=None):
        print(f"🚀 Starting LSTM training for {total_timesteps} steps")
        
        start = time.time()
        total_steps = 0
        episodes = 0
        
        while total_steps < total_timesteps:
            ep_reward, steps = self.collect_trajectory(max_steps=500)
            loss = self.update()
            
            total_steps += steps
            episodes += 1
            
            self.history['rewards'].append(ep_reward)
            if loss > 0:
                self.history['losses'].append(loss)
            
            if episodes % 10 == 0:
                avg_reward = np.mean(self.history['rewards'][-10:])
                print(f"Episode {episodes}: Reward = {ep_reward:.1f}, Avg = {avg_reward:.1f}")
            
            if callback and episodes % 10 == 0:
                try:
                    callback(total_steps, ep_reward, steps)
                except:
                    pass
            
            # Clear memory
            gc.collect()
        
        self.is_trained = True
        self.total_steps = total_steps
        
        return {
            'mean_reward': np.mean(self.history.get('rewards', [0])),
            'total_steps': total_steps,
            'total_episodes': episodes,
            'training_time_seconds': time.time() - start,
            'parameters': sum(p.numel() for p in self.policy.parameters())
        }
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'is_trained': self.is_trained,
            'history': self.history
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.total_steps = ckpt.get('total_steps', 0)
        self.is_trained = ckpt.get('is_trained', False)
        self.history = ckpt.get('history', {})
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
    
    def reset_hidden_state(self):
        self.hidden = None
    
    def save_training_history(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, default=str)