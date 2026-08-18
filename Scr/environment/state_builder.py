# =============================================================================
# FILE: Scr/environment/state_builder.py (VERSION 4.0 - PROFESSIONAL)
# =============================================================================
# PURPOSE: Build different types of states for RL agents.
#          Separates state construction logic from environment.
# =============================================================================
# This module handles:
# 1. Building Classical (POMDP) states - 27 features
# 2. Building LSTM states - sequence of 27 features
# 3. Building GRU states - sequence of 27 features
# 4. Building Quantum states (Phase 2 preparation)
# 5. Feature extraction and encoding with proper error handling
# 6. State dimension management
# 7. State normalization and scaling
# 8. Feature importance tracking
# 9. Batch state building for efficiency
# =============================================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import deque
import logging
import warnings
warnings.filterwarnings('ignore')

# Import constants
from utils.constants import (
    NETWORK_TYPES, AREA_TYPES, NUM_NETWORKS, NUM_AREAS,
    POMDP_STATE_DIM, LSTM_STATE_DIM,
    CORE_FEATURES, NTN_ONLY_FEATURES, UE_FEATURES,
    FAIR_COMPARISON_FEATURES, NTN_COMPARISON_FEATURES,
    AREA_COLUMN, NETWORK_COLUMN, AVAILABLE_COLUMN,
    USER_ID_COLUMN
)

logger = logging.getLogger(__name__)
# تقليل مستوى الـ logging
logger.setLevel(logging.WARNING)


class StateBuilder:
    """
    State Builder for Hybrid Network Recommendation System.
    
    This class handles building different types of states for RL agents.
    It separates state construction logic from the environment.
    
    Attributes:
        state_type (str): Type of state ('classical', 'lstm', 'gru', 'quantum')
        sequence_length (int): History length for LSTM/GRU
        include_ntn_features (bool): Whether to include NTN-only features
        normalize_state (bool): Whether to normalize state values
        feature_importance (Dict): Feature importance weights
    """
    
    def __init__(
        self,
        state_type: str = 'classical',
        sequence_length: int = 10,
        include_ntn_features: bool = False,
        normalize_state: bool = True,
        use_feature_selection: bool = False,
        feature_importance: Optional[Dict[str, float]] = None,
        fallback_value: float = 0.0
    ):
        """
        Initialize the state builder.
        
        Args:
            state_type (str): Type of state ('classical', 'lstm', 'gru', 'quantum')
            sequence_length (int): History length for LSTM/GRU
            include_ntn_features (bool): Include NTN-only features in state
            normalize_state (bool): Whether to normalize state values
            use_feature_selection (bool): Whether to use feature selection
            feature_importance (Dict): Feature importance weights
            fallback_value (float): Value to use when feature is missing
        """
        self.state_type = state_type
        self.sequence_length = sequence_length
        self.include_ntn_features = include_ntn_features
        self.normalize_state = normalize_state
        self.use_feature_selection = use_feature_selection
        self.fallback_value = fallback_value
        
        # Feature importance weights (default: equal importance)
        self.feature_importance = feature_importance or {}
        
        # Cache for processed states
        self._state_cache: Dict[int, np.ndarray] = {}
        self._history_cache: Dict[int, deque] = {}
        self._max_cache_size: int = 10000
        
        # Feature dimensions
        self._feature_dims = self._calculate_feature_dims()
        
        # State statistics (for normalization)
        self._state_stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0
        }
        self._is_fitted = False
        
        # Feature names cache
        self._feature_names_cache: Optional[List[str]] = None
        
        logger.info(f"✅ StateBuilder initialized")
        logger.info(f"   State type: {state_type}")
        logger.info(f"   Sequence length: {sequence_length}")
        logger.info(f"   Include NTN features: {include_ntn_features}")
        logger.info(f"   Normalize state: {normalize_state}")
        logger.info(f"   Feature selection: {use_feature_selection}")
        logger.info(f"   Feature dimensions: {self._feature_dims}")
    
    def _calculate_feature_dims(self) -> Dict[str, int]:
        """
        Calculate feature dimensions for each component.
        
        Returns:
            Dict[str, int]: Feature dimensions
        """
        base_dims = {
            'area': NUM_AREAS,                      # 6
            'available_networks': NUM_NETWORKS,     # 5
            'measurements': NUM_NETWORKS * 3,       # 15
            'ue_features': len(UE_FEATURES)         # 1 (speed only)
        }
        
        if self.include_ntn_features:
            base_dims['ntn_features'] = len(NTN_ONLY_FEATURES)  # 5
        else:
            base_dims['ntn_features'] = 0
        
        base_dims['core_features'] = len(CORE_FEATURES)  # 9
        
        base_dims['total_classical'] = sum([
            base_dims['area'],
            base_dims['available_networks'],
            base_dims['measurements'],
            base_dims['ue_features'],
            base_dims['ntn_features']
        ])
        
        return base_dims
    
    def fit(self, data: pd.DataFrame, sample_size: Optional[int] = None) -> None:
        """
        Fit state statistics for normalization.
        
        Args:
            data (pd.DataFrame): Data to fit statistics from
            sample_size (int): Number of samples to use (None for all)
        """
        logger.info("🔄 Fitting state statistics...")
        
        # Use sample if specified
        if sample_size is not None and len(data) > sample_size:
            sample_data = data.sample(n=sample_size, random_state=42)
        else:
            sample_data = data
        
        # Build states from data
        states = []
        for _, row in sample_data.iterrows():
            state = self.build_classical_state(row, normalize=False)
            states.append(state)
        
        if not states:
            logger.warning("No states built for fitting")
            return
        
        states = np.array(states)
        
        # Calculate statistics
        self._state_stats['mean'] = np.mean(states, axis=0)
        self._state_stats['std'] = np.std(states, axis=0) + 1e-8
        self._state_stats['min'] = np.min(states, axis=0)
        self._state_stats['max'] = np.max(states, axis=0)
        self._state_stats['count'] = len(states)
        
        self._is_fitted = True
        
        logger.info(f"✅ State statistics fitted: {states.shape[1]} features from {len(states)} samples")
    
    def build_classical_state(
        self, 
        row: pd.Series,
        normalize: Optional[bool] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Build a classical (POMDP) state from a single row.
        
        State composition:
        - Area (One-Hot): 6 dimensions
        - Available Networks (One-Hot): 5 dimensions
        - Network Measurements (SNR, SINR, RSSI): 15 dimensions
        - UE Features: 1 dimension (speed only)
        - NTN Features (optional): 5 dimensions
        
        Args:
            row (pd.Series): Data row
            normalize (bool): Override default normalization
            use_cache (bool): Whether to use cache
            
        Returns:
            np.ndarray: State vector (27 dimensions)
        """
        # Check cache
        if use_cache:
            row_hash = self._get_row_hash(row)
            if row_hash in self._state_cache:
                return self._state_cache[row_hash].copy()
        
        try:
            # 1. Area (One-Hot)
            area_one_hot = self._one_hot_area(self._safe_get(row, AREA_COLUMN, 'Unknown'))
            
            # 2. Available Networks (One-Hot)
            available_one_hot = self._one_hot_available(self._safe_get(row, AVAILABLE_COLUMN, ''))
            
            # 3. Network measurements
            measurements = self._get_measurements(row)
            
            # 4. UE features
            ue_state = self._get_ue_state(row)
            
            # 5. NTN features (optional)
            if self.include_ntn_features:
                ntn_state = self._get_ntn_state(row)
            else:
                ntn_state = np.array([], dtype=np.float32)
            
            # Combine all features
            state = np.concatenate([
                area_one_hot,       # 6
                available_one_hot,  # 5
                measurements,       # 15
                ue_state,          # 1
                ntn_state          # 0 or 5
            ])
            
            # Ensure correct length
            expected_len = self._feature_dims['total_classical']
            if len(state) != expected_len:
                if len(state) < expected_len:
                    state = np.pad(state, (0, expected_len - len(state)), constant_values=self.fallback_value)
                else:
                    state = state[:expected_len]
            
            # Apply normalization
            should_normalize = normalize if normalize is not None else self.normalize_state
            if should_normalize and self._is_fitted:
                state = self._normalize_state_vector(state)
            
            # Apply feature selection
            if self.use_feature_selection:
                state = self._apply_feature_selection(state)
            
            state = state.astype(np.float32)
            
            # Store in cache
            if use_cache and row_hash is not None:
                self._add_to_cache(row_hash, state)
            
            return state
            
        except Exception as e:
            logger.debug(f"Error building state: {e}")
            # Return zero state as fallback
            return np.zeros(self._feature_dims['total_classical'], dtype=np.float32)
    
    def build_lstm_state(
        self, 
        history: List[pd.Series],
        normalize: Optional[bool] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Build LSTM state from a sequence of rows.
        
        Args:
            history (List[pd.Series]): List of historical data rows
            normalize (bool): Override default normalization
            use_cache (bool): Whether to use cache
            
        Returns:
            np.ndarray: LSTM state of shape (sequence_length, feature_dim)
        """
        if not history:
            return np.zeros(
                (self.sequence_length, self._feature_dims['total_classical']),
                dtype=np.float32
            )
        
        states = []
        for row in history:
            state = self.build_classical_state(row, normalize, use_cache)
            states.append(state)
        
        # Pad if necessary
        while len(states) < self.sequence_length:
            # Use first state or zeros
            if states:
                pad_state = np.zeros_like(states[0])
            else:
                pad_state = np.zeros(self._feature_dims['total_classical'], dtype=np.float32)
            states.insert(0, pad_state)
        
        # Trim if too long
        if len(states) > self.sequence_length:
            states = states[-self.sequence_length:]
        
        return np.array(states, dtype=np.float32)
    
    def build_gru_state(
        self, 
        history: List[pd.Series],
        normalize: Optional[bool] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Build GRU state from a sequence of rows.
        
        Args:
            history (List[pd.Series]): List of historical data rows
            normalize (bool): Override default normalization
            use_cache (bool): Whether to use cache
            
        Returns:
            np.ndarray: GRU state of shape (sequence_length, feature_dim)
        """
        return self.build_lstm_state(history, normalize, use_cache)
    
    def build_quantum_state(self, row: pd.Series, normalize: Optional[bool] = None) -> np.ndarray:
        """
        Build quantum state (Phase 2 preparation).
        
        Args:
            row (pd.Series): Data row
            normalize (bool): Override default normalization
            
        Returns:
            np.ndarray: Quantum state placeholder
        """
        # Get classical state first
        classical_state = self.build_classical_state(row, normalize)
        
        # In Phase 2, this will be replaced with actual quantum encoding
        # For now, return normalized classical state as placeholder
        norm = np.linalg.norm(classical_state) + 1e-8
        state = classical_state / norm
        
        return state.astype(np.float32)
    
    def build_batch_states(
        self, 
        data: pd.DataFrame,
        normalize: Optional[bool] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Build states for a batch of rows efficiently.
        
        Args:
            data (pd.DataFrame): Data rows
            normalize (bool): Override default normalization
            use_cache (bool): Whether to use cache
            
        Returns:
            np.ndarray: Batch of states of shape (n_samples, feature_dim)
        """
        states = []
        for _, row in data.iterrows():
            state = self.build_classical_state(row, normalize, use_cache)
            states.append(state)
        
        return np.array(states, dtype=np.float32)
    
    def build_state_for_row_index(self, data: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Build state for a specific row index.
        
        Args:
            data (pd.DataFrame): Data DataFrame
            idx (int): Row index
            
        Returns:
            np.ndarray: State vector
        """
        if idx >= len(data) or idx < 0:
            idx = 0
        row = data.iloc[idx]
        return self.build_classical_state(row)
    
    def _safe_get(self, row: pd.Series, key: str, default: Any = None) -> Any:
        """
        Safely get a value from a Series.
        
        Args:
            row (pd.Series): Data row
            key (str): Column name
            default: Default value if key not found
            
        Returns:
            Any: Value or default
        """
        try:
            if key in row.index:
                value = row[key]
                if pd.isna(value):
                    return default
                return value
            return default
        except Exception:
            return default
    
    def _get_row_hash(self, row: pd.Series) -> Optional[str]:
        """
        Generate a hash for a row for caching.
        
        Args:
            row (pd.Series): Data row
            
        Returns:
            str: Row hash or None if cannot generate
        """
        try:
            # Use index if available
            if hasattr(row, 'name'):
                return f"row_{row.name}"
            return None
        except Exception:
            return None
    
    def _add_to_cache(self, key: str, value: np.ndarray) -> None:
        """
        Add a state to the cache.
        
        Args:
            key (str): Cache key
            value (np.ndarray): State value
        """
        if len(self._state_cache) >= self._max_cache_size:
            # Remove oldest entry
            self._state_cache.pop(next(iter(self._state_cache)))
        self._state_cache[key] = value.copy()
    
    def _one_hot_area(self, area: str) -> np.ndarray:
        """
        Convert area to one-hot encoding.
        
        Args:
            area (str): Area name
            
        Returns:
            np.ndarray: One-hot vector of length NUM_AREAS
        """
        one_hot = np.zeros(NUM_AREAS, dtype=np.float32)
        if pd.isna(area) or area is None:
            return one_hot
        
        area = str(area).strip()
        if area in AREA_TYPES:
            one_hot[AREA_TYPES.index(area)] = 1.0
        return one_hot
    
    def _one_hot_available(self, available: Any) -> np.ndarray:
        """
        Convert available networks to one-hot encoding.
        
        Args:
            available: Comma-separated network names or list
            
        Returns:
            np.ndarray: One-hot vector of length NUM_NETWORKS
        """
        one_hot = np.zeros(NUM_NETWORKS, dtype=np.float32)
        
        if available is None:
            return one_hot
        
        if pd.isna(available):
            return one_hot
        
        try:
            if isinstance(available, str):
                if available == '':
                    return one_hot
                available_list = [n.strip() for n in available.split(',')]
            elif isinstance(available, (list, tuple)):
                available_list = [str(n).strip() for n in available]
            else:
                available_list = [str(available).strip()]
            
            for i, network in enumerate(NETWORK_TYPES):
                if network in available_list:
                    one_hot[i] = 1.0
            
            # Ensure at least one network is available
            if np.sum(one_hot) == 0:
                one_hot[0] = 1.0  # Default to NR_5G
                
        except Exception as e:
            logger.debug(f"Error parsing available networks: {e}")
            one_hot[0] = 1.0  # Default to NR_5G
        
        return one_hot
    
    def _get_measurements(self, row: pd.Series) -> np.ndarray:
        """
        Extract network measurements from a row.
        
        Measurements: SNR_dB, SINR_dB, RSSI_dBm
        Repeated for each network type.
        
        Args:
            row (pd.Series): Data row
            
        Returns:
            np.ndarray: Measurements of length NUM_NETWORKS * 3
        """
        # Key measurements with fallback
        measurements = [
            self._safe_float(self._safe_get(row, 'SNR_dB', 0.0)),
            self._safe_float(self._safe_get(row, 'SINR_dB', 0.0)),
            self._safe_float(self._safe_get(row, 'RSSI_dBm', 0.0))
        ]
        
        # Get network type for variation
        network = self._safe_get(row, NETWORK_COLUMN, 'NR_5G')
        network_idx = NETWORK_TYPES.index(network) if network in NETWORK_TYPES else 0
        
        # Repeat for all networks with realistic variations
        full_measurements = []
        for i in range(NUM_NETWORKS):
            # Add variations based on network type and index
            variation = 1.0 + (i - network_idx) * 0.1  # -30% to +30%
            for m in measurements:
                full_measurements.append(m * variation)
        
        # Ensure correct length
        target_length = NUM_NETWORKS * 3
        if len(full_measurements) < target_length:
            full_measurements.extend([self.fallback_value] * (target_length - len(full_measurements)))
        elif len(full_measurements) > target_length:
            full_measurements = full_measurements[:target_length]
        
        return np.array(full_measurements, dtype=np.float32)
    
    def _get_ue_state(self, row: pd.Series) -> np.ndarray:
        """
        Extract UE (User Equipment) state features.
        
        Args:
            row (pd.Series): Data row
            
        Returns:
            np.ndarray: UE features (speed_ms only)
        """
        speed = self._safe_float(self._safe_get(row, 'speed_ms', 0.0))
        return np.array([speed], dtype=np.float32)
    
    def _get_ntn_state(self, row: pd.Series) -> np.ndarray:
        """
        Extract NTN-only features.
        
        Args:
            row (pd.Series): Data row
            
        Returns:
            np.ndarray: NTN features
        """
        ntn_features = [
            'altitude_m',
            'Doppler_Hz',
            'Propagation_Delay_ms',
            'Rain_Rate_mmhr',
            'Rain_Fade_dB'
        ]
        
        return np.array([
            self._safe_float(self._safe_get(row, feat, 0.0))
            for feat in ntn_features
        ], dtype=np.float32)
    
    def _safe_float(self, value: Any) -> float:
        """
        Safely convert a value to float.
        
        Args:
            value: Value to convert
            
        Returns:
            float: Converted value or fallback if conversion fails
        """
        try:
            if value is None:
                return self.fallback_value
            if pd.isna(value):
                return self.fallback_value
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                if value == '':
                    return self.fallback_value
                return float(value)
            return self.fallback_value
        except (ValueError, TypeError):
            return self.fallback_value
    
    def _normalize_state_vector(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize a state vector using fitted statistics.
        
        Args:
            state (np.ndarray): State vector
            
        Returns:
            np.ndarray: Normalized state
        """
        if not self._is_fitted:
            return state
        
        mean = self._state_stats['mean']
        std = self._state_stats['std']
        
        if mean is None or std is None:
            return state
        
        # Ensure same length
        if len(state) != len(mean):
            # Truncate or pad if needed
            min_len = min(len(state), len(mean))
            state_normalized = np.zeros_like(state, dtype=np.float32)
            
            # Normalize existing features
            state_normalized[:min_len] = (state[:min_len] - mean[:min_len]) / (std[:min_len] + 1e-8)
            
            # Pad if needed
            if len(state) > len(mean):
                state_normalized[len(mean):] = state[len(mean):]
            
            return state_normalized
        
        return (state - mean) / (std + 1e-8)
    
    def _apply_feature_selection(self, state: np.ndarray) -> np.ndarray:
        """
        Apply feature selection weights.
        
        Args:
            state (np.ndarray): State vector
            
        Returns:
            np.ndarray: Weighted state
        """
        if not self.feature_importance:
            return state
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        # Apply weights
        weighted_state = state.copy()
        for i, name in enumerate(feature_names):
            if i < len(weighted_state) and name in self.feature_importance:
                weighted_state[i] *= self.feature_importance[name]
        
        return weighted_state
    
    def get_state_dimension(self) -> int:
        """
        Get the dimension of the current state.
        
        Returns:
            int: State dimension
        """
        if self.state_type in ['classical', 'quantum']:
            return self._feature_dims['total_classical']
        elif self.state_type in ['lstm', 'gru']:
            return self.sequence_length * self._feature_dims['total_classical']
        else:
            raise ValueError(f"Unknown state type: {self.state_type}")
    
    def get_state_dimension_by_type(self, state_type: Optional[str] = None) -> int:
        """
        Get state dimension for a specific state type.
        
        Args:
            state_type (str): Type of state ('classical', 'lstm', 'gru', 'quantum')
            
        Returns:
            int: State dimension
        """
        stype = state_type or self.state_type
        if stype == 'classical':
            return self._feature_dims['total_classical']
        elif stype in ['lstm', 'gru']:
            return self.sequence_length * self._feature_dims['total_classical']
        elif stype == 'quantum':
            return self._feature_dims['total_classical']
        else:
            raise ValueError(f"Unknown state type: {stype}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features in the state.
        
        Returns:
            List[str]: Feature names
        """
        if self._feature_names_cache is not None:
            return self._feature_names_cache
        
        feature_names = []
        
        # Area features
        feature_names.extend([f'Area_{area}' for area in AREA_TYPES])
        
        # Available networks features
        feature_names.extend([f'Avail_{net}' for net in NETWORK_TYPES])
        
        # Measurement features
        for i in range(NUM_NETWORKS):
            feature_names.extend([
                f'Net{i}_SNR',
                f'Net{i}_SINR',
                f'Net{i}_RSSI'
            ])
        
        # UE features
        feature_names.extend(['speed_ms'])
        
        # NTN features (if included)
        if self.include_ntn_features:
            feature_names.extend([
                'altitude_m',
                'Doppler_Hz',
                'Propagation_Delay_ms',
                'Rain_Rate_mmhr',
                'Rain_Fade_dB'
            ])
        
        self._feature_names_cache = feature_names
        return feature_names
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions of different feature groups.
        
        Returns:
            Dict[str, int]: Feature dimensions
        """
        return self._feature_dims.copy()
    
    def build_fair_comparison_state(self, row: pd.Series) -> np.ndarray:
        """
        Build state for fair comparison between TN and NTN.
        
        Uses only features that exist in ALL network types.
        
        Args:
            row (pd.Series): Data row
            
        Returns:
            np.ndarray: Fair comparison state
        """
        features = []
        for feature in FAIR_COMPARISON_FEATURES:
            features.append(self._safe_float(self._safe_get(row, feature, 0.0)))
        
        return np.array(features, dtype=np.float32)
    
    def build_ntn_comparison_state(self, row: pd.Series) -> np.ndarray:
        """
        Build state for comparison between NTN networks only.
        
        Uses features that exist in all NTN networks.
        
        Args:
            row (pd.Series): Data row
            
        Returns:
            np.ndarray: NTN comparison state
        """
        features = []
        for feature in NTN_COMPARISON_FEATURES:
            features.append(self._safe_float(self._safe_get(row, feature, 0.0)))
        
        return np.array(features, dtype=np.float32)
    
    def normalize_value(self, value: float, feature_name: str) -> float:
        """
        Normalize a single value using fitted statistics.
        
        Args:
            value (float): Value to normalize
            feature_name (str): Name of the feature
            
        Returns:
            float: Normalized value
        """
        if not self._is_fitted:
            return value
        
        feature_names = self.get_feature_names()
        if feature_name in feature_names:
            idx = feature_names.index(feature_name)
            if idx < len(self._state_stats['mean']):
                mean = self._state_stats['mean'][idx]
                std = self._state_stats['std'][idx]
                return (value - mean) / (std + 1e-8)
        
        return value
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get information about the state configuration.
        
        Returns:
            Dict: State information
        """
        return {
            'state_type': self.state_type,
            'sequence_length': self.sequence_length,
            'include_ntn_features': self.include_ntn_features,
            'normalize_state': self.normalize_state,
            'use_feature_selection': self.use_feature_selection,
            'feature_dims': self._feature_dims,
            'total_dimension': self.get_state_dimension(),
            'is_fitted': self._is_fitted,
            'feature_names_count': len(self.get_feature_names()),
            'cache_size': len(self._state_cache),
            'max_cache_size': self._max_cache_size,
            'fitted_samples': self._state_stats['count']
        }
    
    def clear_cache(self) -> None:
        """Clear the state cache."""
        self._state_cache.clear()
        self._history_cache.clear()
        logger.info("✅ State cache cleared")
    
    def set_cache_size(self, max_size: int) -> None:
        """
        Set maximum cache size.
        
        Args:
            max_size (int): Maximum number of cached states
        """
        self._max_cache_size = max(100, max_size)
        # Trim cache if needed
        while len(self._state_cache) > self._max_cache_size:
            self._state_cache.pop(next(iter(self._state_cache)))
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """
        Get fitted state statistics.
        
        Returns:
            Dict: State statistics
        """
        if not self._is_fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'mean_shape': len(self._state_stats['mean']) if self._state_stats['mean'] is not None else 0,
            'std_shape': len(self._state_stats['std']) if self._state_stats['std'] is not None else 0,
            'samples': self._state_stats['count'],
            'mean_min': np.min(self._state_stats['mean']) if self._state_stats['mean'] is not None else 0,
            'mean_max': np.max(self._state_stats['mean']) if self._state_stats['mean'] is not None else 0,
            'std_min': np.min(self._state_stats['std']) if self._state_stats['std'] is not None else 0,
            'std_max': np.max(self._state_stats['std']) if self._state_stats['std'] is not None else 0
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING STATE BUILDER")
    print("="*80)
    
    from data_preprocessing.data_loader import DataLoader
    
    # Load data
    loader = DataLoader('data_raw/Hybrid_Network_TN_NTN_Final.csv')
    data = loader.load()
    
    print(f"\nData loaded: {len(data)} rows")
    
    # Test Classical State Builder
    print("\n📌 Testing Classical State Builder:")
    print("="*80)
    builder = StateBuilder(
        state_type='classical',
        normalize_state=True,
        include_ntn_features=False
    )
    
    # Fit statistics
    builder.fit(data.head(1000))
    
    sample_row = data.iloc[0]
    state = builder.build_classical_state(sample_row)
    print(f"   State shape: {state.shape}")
    print(f"   State dimension: {builder.get_state_dimension()}")
    print(f"   Feature names (first 10): {builder.get_feature_names()[:10]}")
    print(f"   State values (first 10): {state[:10]}")
    
    # Test LSTM State Builder
    print("\n📌 Testing LSTM State Builder:")
    print("="*80)
    lstm_builder = StateBuilder(
        state_type='lstm', 
        sequence_length=5,
        normalize_state=True
    )
    lstm_builder.fit(data.head(1000))
    
    history = [data.iloc[i] for i in range(5)]
    lstm_state = lstm_builder.build_lstm_state(history)
    print(f"   LSTM state shape: {lstm_state.shape}")
    print(f"   LSTM state dimension: {lstm_builder.get_state_dimension_by_type('lstm')}")
    
    # Test with NTN features
    print("\n📌 Testing State Builder with NTN features:")
    print("="*80)
    ntn_builder = StateBuilder(
        state_type='classical', 
        include_ntn_features=True,
        normalize_state=True
    )
    ntn_builder.fit(data.head(1000))
    
    ntn_state = ntn_builder.build_classical_state(sample_row)
    print(f"   State with NTN features shape: {ntn_state.shape}")
    print(f"   Feature names with NTN (total: {len(ntn_builder.get_feature_names())})")
    
    # Test batch building
    print("\n📌 Testing Batch State Building:")
    print("="*80)
    batch_states = builder.build_batch_states(data.head(10))
    print(f"   Batch states shape: {batch_states.shape}")
    
    # Test fair comparison
    print("\n📌 Testing Fair Comparison State:")
    print("="*80)
    fair_state = builder.build_fair_comparison_state(sample_row)
    print(f"   Fair comparison state shape: {fair_state.shape}")
    print(f"   Fair state values: {fair_state[:5]}")
    
    # Test NTN comparison
    print("\n📌 Testing NTN Comparison State:")
    print("="*80)
    ntn_compare_state = builder.build_ntn_comparison_state(sample_row)
    print(f"   NTN comparison state shape: {ntn_compare_state.shape}")
    print(f"   NTN compare values: {ntn_compare_state[:5]}")
    
    # Test normalize_value
    print("\n📌 Testing normalize_value:")
    print("="*80)
    normalized = builder.normalize_value(20.0, 'Area_Urban')
    print(f"   Normalized value: {normalized:.4f}")
    
    # Get state info
    print("\n📌 State Info:")
    print("="*80)
    info = builder.get_state_info()
    for key, value in info.items():
        if key != 'feature_dims':
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")
    
    # Get state statistics
    print("\n📌 State Statistics:")
    print("="*80)
    stats = builder.get_state_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ StateBuilder test completed successfully!")