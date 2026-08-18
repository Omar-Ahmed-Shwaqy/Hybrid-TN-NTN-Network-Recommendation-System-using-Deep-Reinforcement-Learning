# =============================================================================
# FILE: Scr/environment/hybrid_network_env.py (VERSION 4.0 - PROFESSIONAL)
# =============================================================================

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
import warnings
from typing import List
warnings.filterwarnings('ignore')

# ===== NEW IMPORTS =====
from environment.state_builder import StateBuilder
from environment.reward_calculator import RewardCalculator

class HybridNetworkEnv(gym.Env):
    """
    Hybrid Network Environment for RL Agents.
    
    Now integrated with:
    - StateBuilder for state construction
    - RewardCalculator for multi-objective rewards
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        state_type: str = 'classical',
        sequence_length: int = 15,
        seed: int = 42,
        track_user: bool = True,
        use_state_builder: bool = True,      # NEW
        use_reward_calculator: bool = True    # NEW
    ):
        super(HybridNetworkEnv, self).__init__()
        
        # ===== DATA PREPARATION =====
        self.data = data.reset_index(drop=True)
        
        # Handle infinite values
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        for col in self.data.select_dtypes(include=[np.number]).columns:
            self.data[col] = self.data[col].fillna(self.data[col].median() or 0)
        
        # Validate network_type
        valid_networks = ['NR_5G', 'WiFi', 'SAT (LEO)', 'HAPS', 'UAV']
        if 'network_type' in self.data.columns:
            self.data['network_type'] = self.data['network_type'].apply(
                lambda x: x if x in valid_networks else valid_networks[0]
            )
        
        # Normalize column names
        if 'available_networks' in self.data.columns and 'Available_Networks' not in self.data.columns:
            self.data = self.data.rename(columns={'available_networks': 'Available_Networks'})
        if 'area' in self.data.columns and 'Area' not in self.data.columns:
            self.data = self.data.rename(columns={'area': 'Area'})
        
        # Validate available networks
        if 'Available_Networks' in self.data.columns:
            if isinstance(self.data['Available_Networks'].iloc[0], str):
                self.data['Available_Networks'] = self.data['Available_Networks'].apply(
                    lambda x: x.split(',') if isinstance(x, str) and x else ['NR_5G']
                )
            elif pd.isna(self.data['Available_Networks'].iloc[0]):
                self.data['Available_Networks'] = self.data['Available_Networks'].apply(
                    lambda x: ['NR_5G'] if pd.isna(x) else x
                )
        
        # ===== ENVIRONMENT PARAMETERS =====
        self.state_type = state_type
        self.sequence_length = sequence_length
        self.seed = seed
        self.track_user = track_user
        self.use_state_builder = use_state_builder
        self.use_reward_calculator = use_reward_calculator
        
        # ===== INITIALIZE STATE BUILDER =====
        if use_state_builder:
            self.state_builder = StateBuilder(
                state_type=state_type,
                sequence_length=sequence_length,
                normalize_state=True,
                include_ntn_features=False
            )
            # Fit state builder on data
            self.state_builder.fit(self.data.head(2000))
            state_dim = self.state_builder.get_state_dimension()
        else:
            state_dim = 27  # Default POMDP dimension
        
        # ===== INITIALIZE REWARD CALCULATOR =====
        if use_reward_calculator:
            self.reward_calculator = RewardCalculator(
                normalize=True,
                normalize_type='robust',
                use_qos_penalty=True,
                reward_shaping=True,
                area_specific_weights=True
            )
            # Fit reward calculator bounds
            self.reward_calculator.fit_bounds(self.data.head(5000))
        
        # ===== SPACES =====
        # Observation space
        if state_type in ['lstm', 'gru']:
            # LSTM/GRU states are sequences
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(state_dim,),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(state_dim,),
                dtype=np.float32
            )
        
        # Action space (5 networks)
        self.action_space = spaces.Discrete(5)
        
        # ===== STATE =====
        self.current_step = 0
        self.episode_data = []
        self.history = []  # For LSTM/GRU
        self.previous_action = None
        self.previous_network = None
        self.n_networks = 5
        
        # ===== NETWORK NAMES =====
        self.valid_networks = ['NR_5G', 'WiFi', 'SAT (LEO)', 'HAPS', 'UAV']
        self.valid_areas = ['Urban', 'Indoor', 'Rural', 'Highway', 'Maritime', 'Desert']
        
        # ===== LIMIT DATA SIZE =====
        if len(self.data) > 10000:
            self.data = self.data.iloc[:10000].copy()
        
        print(f"✅ Environment initialized: {len(self.data)} rows")
        print(f"   State type: {state_type}, Dimension: {state_dim}")
        print(f"   State Builder: {use_state_builder}")
        print(f"   Reward Calculator: {use_reward_calculator}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            self.seed = seed
        
        self.current_step = 0
        self.episode_data = []
        self.history = []
        self.previous_action = None
        self.previous_network = None
        
        # Get initial state
        obs = self._get_observation(0)
        
        info = {
            'user_id': 'test_user',
            'area': self._get_area(0),
            'step': 0
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Validate action
        if action >= self.action_space.n or action < 0:
            action = 0
        
        self.current_step += 1
        
        # Get actual network
        actual_network = self._get_actual_network(self.current_step)
        predicted_network = self.valid_networks[action]
        area = self._get_area(self.current_step)
        
        # ===== CALCULATE REWARD =====
        if self.use_reward_calculator:
            # Get current row
            row = self.data.iloc[self.current_step] if self.current_step < len(self.data) else self.data.iloc[-1]
            
            # Calculate reward using RewardCalculator
            reward = self.reward_calculator.calculate_reward(
                row=row,
                selected_network=predicted_network,
                previous_network=self.previous_network,
                area=area,
                step=self.current_step
            )
        else:
            # Simple reward (fallback)
            is_correct = predicted_network == actual_network
            reward = 1.0 if is_correct else -0.1
            
            # Handover penalty (simple)
            if self.previous_network is not None and self.previous_network != predicted_network:
                reward -= 0.2
        
        # Check handover
        is_handover = (self.previous_network is not None and 
                      self.previous_network != predicted_network)
        
        # Check correctness
        is_correct = predicted_network == actual_network
        
        # Update history
        if self.current_step < len(self.data):
            row_data = self.data.iloc[self.current_step]
            self.history.append(row_data)
            if len(self.history) > self.sequence_length:
                self.history.pop(0)
        
        # Get next observation
        obs = self._get_observation(self.current_step)
        
        # Update previous
        self.previous_action = action
        self.previous_network = predicted_network
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Build info
        info = {
            'is_correct': is_correct,
            'is_handover': is_handover,
            'actual_network': actual_network,
            'network': predicted_network,
            'area': area,
            'reward': reward,
            'decision_time_ms': 0.1,
            'step': self.current_step
        }
        
        return obs, float(reward), done, truncated, info
    
    def _get_observation(self, idx: int) -> np.ndarray:
        """Get observation at index."""
        if idx >= len(self.data):
            idx = len(self.data) - 1
        
        row = self.data.iloc[idx]
        
        # ===== USE STATE BUILDER =====
        if self.use_state_builder:
            if self.state_type in ['lstm', 'gru']:
                # Use history for LSTM/GRU
                if len(self.history) >= self.sequence_length:
                    state = self.state_builder.build_lstm_state(self.history[-self.sequence_length:])
                else:
                    # Pad with current row
                    padded_history = [row] * (self.sequence_length - len(self.history)) + self.history
                    state = self.state_builder.build_lstm_state(padded_history[-self.sequence_length:])
            else:
                # Classical state
                state = self.state_builder.build_classical_state(row)
            
            return state.astype(np.float32)
        
        # ===== FALLBACK: Manual observation =====
        # Get numeric features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_values = []
        
        for col in numeric_cols[:20]:
            val = row.get(col, 0)
            if pd.isna(val):
                val = 0
            numeric_values.append(float(val))
        
        while len(numeric_values) < 20:
            numeric_values.append(0.0)
        
        # Get available networks
        avail_networks = self._encode_available_networks(idx)
        
        return np.array(numeric_values[:20] + avail_networks, dtype=np.float32)
    
    def _encode_available_networks(self, idx: int) -> List[int]:
        """Encode available networks as one-hot."""
        try:
            if idx >= len(self.data):
                return [1, 0, 0, 0, 0]
            
            row = self.data.iloc[idx]
            
            if 'Available_Networks' not in self.data.columns:
                return [1, 0, 0, 0, 0]
            
            networks = row.get('Available_Networks')
            if networks is None or (isinstance(networks, float) and pd.isna(networks)):
                return [1, 0, 0, 0, 0]
            
            if isinstance(networks, str):
                networks = networks.split(',')
            
            if not isinstance(networks, (list, tuple)):
                networks = [str(networks)]
            
            result = [1 if net in networks else 0 for net in self.valid_networks]
            
            if sum(result) == 0:
                result[0] = 1
            
            return result
            
        except Exception:
            return [1, 0, 0, 0, 0]
    
    def _get_actual_network(self, idx: int) -> str:
        """Get actual network at index."""
        try:
            if idx >= len(self.data):
                idx = len(self.data) - 1
            
            row = self.data.iloc[idx]
            network = row.get('network_type', 'NR_5G')
            
            if network not in self.valid_networks:
                network = 'NR_5G'
            
            return network
            
        except Exception:
            return 'NR_5G'
    
    def _get_area(self, idx: int) -> str:
        """Get area at index."""
        try:
            if idx >= len(self.data):
                idx = len(self.data) - 1
            
            row = self.data.iloc[idx]
            area = row.get('Area', 'Urban')
            
            if area not in self.valid_areas:
                area = 'Urban'
            
            return area
            
        except Exception:
            return 'Urban'
    
    def close(self) -> None:
        """Close the environment."""
        pass