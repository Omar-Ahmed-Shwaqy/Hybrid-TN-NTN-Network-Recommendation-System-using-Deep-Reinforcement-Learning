# =============================================================================
# FILE: Scr/environment/reward_calculator.py
# =============================================================================
# PURPOSE: Calculate QoE (Quality of Experience) reward for network selection.
#          Implements multi-objective reward function with handover penalty.
# =============================================================================
# This module handles:
# 1. Multi-objective reward (Throughput, Latency, Packet Loss, BER, SINR, SNR)
# 2. Handover penalty for network switching (OPTIMIZED)
# 3. Normalization of metrics (min-max scaling, robust scaling)
# 4. Reward clipping and shaping with adaptive bounds
# 5. Weighted combination of sub-objectives
# 6. QoE-based reward with user preferences
# 7. Dynamic reward shaping for better learning
# 8. QoS violation penalties (ENHANCED)
# 9. Area-specific reward weights (ENHANCED)
# =============================================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Import constants
from utils.constants import (
    REWARD_WEIGHTS, HANDOVER_PENALTY, REWARD_MIN, REWARD_MAX,
    REWARD_NORM_BOUNDS, NETWORK_TYPES, AREA_TYPES
)

logger = logging.getLogger(__name__)
# تقليل مستوى الـ logging
logger.setLevel(logging.WARNING)


class RewardCalculator:
    """
    QoE Reward Calculator for Hybrid Network Selection.
    
    OPTIMIZED REWARD DESIGN:
    1. Balanced weights for all metrics
    2. Higher handover penalty for stability
    3. Area-specific rewards for different environments
    4. Progressive QoS penalties
    5. Reward shaping for faster convergence
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        handover_penalty: float = 0.35,  # OPTIMIZED: Higher penalty
        normalize: bool = True,
        normalize_type: str = 'robust',
        use_dynamic_weights: bool = False,
        use_qos_penalty: bool = True,
        reward_shaping: bool = True,
        area_specific_weights: bool = True  # OPTIMIZED: Enabled by default
    ):
        """
        Initialize the reward calculator.
        
        Args:
            weights (Dict): Weights for each reward component
            handover_penalty (float): Penalty for switching networks
            normalize (bool): Whether to normalize metrics
            normalize_type (str): Type of normalization ('robust', 'minmax', 'zscore')
            use_dynamic_weights (bool): Whether to use dynamic weights
            use_qos_penalty (bool): Whether to apply QoS violation penalty
            reward_shaping (bool): Whether to apply reward shaping
            area_specific_weights (bool): Whether to use area-specific weights
        """
        # ===== OPTIMIZED WEIGHTS: Balanced for better learning =====
        self.weights = weights or {
            'throughput': 0.35,      # Reduced for balance
            'latency': 0.25,          # Same
            'packet_loss': 0.20,      # Increased for reliability
            'ber': 0.10,              # Same
            'snr': 0.05,              # Added for signal quality
            'sinr': 0.05              # Added for signal quality
        }
        
        self.handover_penalty = handover_penalty
        self.normalize = normalize
        self.normalize_type = normalize_type
        self.use_dynamic_weights = use_dynamic_weights
        self.use_qos_penalty = use_qos_penalty
        self.reward_shaping = reward_shaping
        self.area_specific_weights = area_specific_weights
        
        # Store normalization bounds (fitted from data)
        self.norm_bounds: Dict[str, Tuple[float, float]] = {}
        self.norm_stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False
        
        # ===== OPTIMIZED QoS thresholds =====
        self.qos_thresholds = {
            'throughput': 10.0,       # Min Mbps (increased)
            'latency': 30.0,          # Max ms (reduced for stricter QoS)
            'packet_loss': 2.0,       # Max % (reduced)
            'ber': 0.0005,            # Max BER (reduced)
            'snr': 5.0,               # Min SNR (increased)
            'sinr': 3.0               # Min SINR (increased)
        }
        
        # ===== OPTIMIZED: Area-specific weights =====
        self.area_weights = {
            'Urban': {
                'throughput': 0.40, 'latency': 0.30, 'packet_loss': 0.15, 
                'ber': 0.05, 'snr': 0.05, 'sinr': 0.05
            },
            'Indoor': {
                'throughput': 0.25, 'latency': 0.40, 'packet_loss': 0.20, 
                'ber': 0.05, 'snr': 0.05, 'sinr': 0.05
            },
            'Rural': {
                'throughput': 0.45, 'latency': 0.20, 'packet_loss': 0.15, 
                'ber': 0.05, 'snr': 0.10, 'sinr': 0.05
            },
            'Highway': {
                'throughput': 0.35, 'latency': 0.35, 'packet_loss': 0.15, 
                'ber': 0.05, 'snr': 0.05, 'sinr': 0.05
            },
            'Maritime': {
                'throughput': 0.40, 'latency': 0.20, 'packet_loss': 0.20, 
                'ber': 0.05, 'snr': 0.10, 'sinr': 0.05
            },
            'Desert': {
                'throughput': 0.45, 'latency': 0.20, 'packet_loss': 0.15, 
                'ber': 0.05, 'snr': 0.10, 'sinr': 0.05
            }
        }
        
        # ===== OPTIMIZED: QoS penalty weights =====
        self.qos_penalty_weights = {
            'throughput': 0.08,
            'latency': 0.08,
            'packet_loss': 0.06,
            'ber': 0.04,
            'snr': 0.02,
            'sinr': 0.02
        }
        
        logger.info(f"RewardCalculator initialized (OPTIMIZED)")
        logger.info(f"   Weights: {self.weights}")
        logger.info(f"   Handover penalty: {self.handover_penalty}")
        logger.info(f"   Area-specific weights: {self.area_specific_weights}")
        logger.info(f"   QoS penalty: {self.use_qos_penalty}")
        logger.info(f"   Reward shaping: {self.reward_shaping}")
    
    def fit_bounds(
        self, 
        df: pd.DataFrame, 
        quantile_low: float = 0.01, 
        quantile_high: float = 0.99
    ) -> None:
        """
        Fit normalization bounds from data.
        
        Uses quantiles to be robust to outliers.
        """
        # Mapping from metric name to column name in data
        metric_columns = {
            'throughput': 'Throughput_Mbps',
            'latency': 'Latency_ms',
            'packet_loss': 'Packet_Loss_pct',
            'ber': 'BER',
            'snr': 'SNR_dB',
            'sinr': 'SINR_dB',
            'rssi': 'RSSI_dBm'
        }
        
        for metric, column in metric_columns.items():
            if column in df.columns:
                data = df[column].dropna()
                if len(data) > 0:
                    if self.normalize_type == 'robust':
                        lower = data.quantile(quantile_low)
                        upper = data.quantile(quantile_high)
                        
                        if lower == upper:
                            lower = data.min()
                            upper = data.max()
                        
                        self.norm_bounds[metric] = (float(lower), float(upper))
                    
                    elif self.normalize_type == 'zscore':
                        self.norm_stats[metric] = {
                            'mean': float(data.mean()),
                            'std': float(data.std())
                        }
                    
                    elif self.normalize_type == 'minmax':
                        lower = data.min()
                        upper = data.max()
                        if lower == upper:
                            lower = 0
                            upper = 1
                        self.norm_bounds[metric] = (float(lower), float(upper))
        
        self.is_fitted = True
    
    def calculate_reward(
        self,
        row: pd.Series,
        selected_network: str,
        previous_network: Optional[str] = None,
        penalty: Optional[float] = None,
        area: Optional[str] = None,
        step: Optional[int] = None
    ) -> float:
        """
        Calculate the reward for a given network selection.
        
        OPTIMIZED: More balanced and stable reward.
        
        Args:
            row (pd.Series): Data row with network metrics
            selected_network (str): Selected network name
            previous_network (str): Previously selected network
            penalty (float): Handover penalty (uses default if None)
            area (str): Current area for area-specific weights
            step (int): Current step for progressive rewards
            
        Returns:
            float: Calculated reward (clipped between REWARD_MIN and REWARD_MAX)
        """
        # 1. Get weights (area-specific or default)
        weights = self._get_weights(area)
        
        # 2. Calculate QoE components
        throughput_score = self._calculate_throughput_score(row)
        latency_score = self._calculate_latency_score(row)
        packet_loss_score, ber_score = self._calculate_reliability_score(row)
        snr_score = self._calculate_snr_score(row)
        sinr_score = self._calculate_sinr_score(row)
        
        # 3. Combine components with weights
        reward = (
            weights.get('throughput', 0.35) * throughput_score +
            weights.get('latency', 0.25) * latency_score +
            weights.get('packet_loss', 0.20) * packet_loss_score +
            weights.get('ber', 0.10) * ber_score +
            weights.get('snr', 0.05) * snr_score +
            weights.get('sinr', 0.05) * sinr_score
        )
        
        # 4. Apply handover penalty (OPTIMIZED: Higher penalty)
        if previous_network is not None and selected_network != previous_network:
            penalty_value = penalty or self.handover_penalty
            reward -= penalty_value
        
        # 5. Apply QoS violation penalty (OPTIMIZED: Progressive)
        if self.use_qos_penalty:
            qos_penalty = self._calculate_qos_penalty_optimized(row, step)
            reward -= qos_penalty
        
        # 6. Apply reward shaping (OPTIMIZED: Better bonuses)
        if self.reward_shaping:
            shaping_bonus = self._calculate_shaping_bonus_optimized(row, selected_network)
            reward += shaping_bonus
        
        # 7. Clip reward to bounds
        reward = np.clip(reward, REWARD_MIN, REWARD_MAX)
        
        return float(reward)
    
    def _get_weights(self, area: Optional[str] = None) -> Dict[str, float]:
        """Get weights for reward calculation."""
        if self.area_specific_weights and area and area in self.area_weights:
            return self.area_weights[area]
        return self.weights
    
    def _normalize_value(self, value: float, metric: str, invert: bool = False) -> float:
        """Normalize a value using fitted bounds or statistics."""
        if not self.normalize:
            return value
        
        if self.normalize_type == 'robust':
            lower, upper = self.norm_bounds.get(metric, (0, 1))
            if upper > lower:
                normalized = (value - lower) / (upper - lower)
                if invert:
                    normalized = 1 - normalized
                return np.clip(normalized, 0, 1)
            return 0.5
        
        elif self.normalize_type == 'zscore':
            stats = self.norm_stats.get(metric, {'mean': 0, 'std': 1})
            if stats['std'] > 0:
                normalized = (value - stats['mean']) / stats['std']
                normalized = 1 / (1 + np.exp(-normalized))
                if invert:
                    normalized = 1 - normalized
                return np.clip(normalized, 0, 1)
            return 0.5
        
        elif self.normalize_type == 'minmax':
            lower, upper = self.norm_bounds.get(metric, (0, 1))
            if upper > lower:
                normalized = (value - lower) / (upper - lower)
                if invert:
                    normalized = 1 - normalized
                return np.clip(normalized, 0, 1)
            return 0.5
        
        return value
    
    def _calculate_throughput_score(self, row: pd.Series) -> float:
        """Calculate throughput score (higher is better)."""
        throughput = row.get('Throughput_Mbps', 0.0)
        if pd.isna(throughput):
            throughput = 0.0
        return self._normalize_value(float(throughput), 'throughput', invert=False)
    
    def _calculate_latency_score(self, row: pd.Series) -> float:
        """Calculate latency score (lower is better)."""
        latency = row.get('Latency_ms', 0.0)
        if pd.isna(latency):
            latency = 0.0
        return self._normalize_value(float(latency), 'latency', invert=True)
    
    def _calculate_reliability_score(self, row: pd.Series) -> Tuple[float, float]:
        """Calculate reliability scores (packet loss and BER)."""
        packet_loss = row.get('Packet_Loss_pct', 0.0)
        if pd.isna(packet_loss):
            packet_loss = 0.0
        pl_score = self._normalize_value(float(packet_loss), 'packet_loss', invert=True)
        
        ber = row.get('BER', 0.0)
        if pd.isna(ber):
            ber = 0.0
        ber_score = self._normalize_value(float(ber), 'ber', invert=True)
        
        return pl_score, ber_score
    
    def _calculate_snr_score(self, row: pd.Series) -> float:
        """Calculate SNR score (higher is better)."""
        snr = row.get('SNR_dB', 0.0)
        if pd.isna(snr):
            snr = 0.0
        return self._normalize_value(float(snr), 'snr', invert=False)
    
    def _calculate_sinr_score(self, row: pd.Series) -> float:
        """Calculate SINR score (higher is better)."""
        sinr = row.get('SINR_dB', 0.0)
        if pd.isna(sinr):
            sinr = 0.0
        return self._normalize_value(float(sinr), 'sinr', invert=False)
    
    def _calculate_qos_penalty_optimized(self, row: pd.Series, step: Optional[int] = None) -> float:
        """
        Calculate QoS violation penalty (OPTIMIZED: Progressive).
        
        Args:
            row (pd.Series): Data row
            step (int): Current step for progressive penalties
            
        Returns:
            float: Penalty value (0 if no violation)
        """
        penalty = 0.0
        
        # Progressive penalty (higher later in training)
        progress_factor = min(1.0, (step or 0) / 10000) if step else 1.0
        
        # Check throughput
        throughput = row.get('Throughput_Mbps', 0.0)
        if pd.isna(throughput):
            throughput = 0.0
        if float(throughput) < self.qos_thresholds['throughput']:
            penalty += self.qos_penalty_weights['throughput'] * (1 + progress_factor * 0.5)
        
        # Check latency
        latency = row.get('Latency_ms', 0.0)
        if pd.isna(latency):
            latency = 0.0
        if float(latency) > self.qos_thresholds['latency']:
            penalty += self.qos_penalty_weights['latency'] * (1 + progress_factor * 0.5)
        
        # Check packet loss
        packet_loss = row.get('Packet_Loss_pct', 0.0)
        if pd.isna(packet_loss):
            packet_loss = 0.0
        if float(packet_loss) > self.qos_thresholds['packet_loss']:
            penalty += self.qos_penalty_weights['packet_loss'] * (1 + progress_factor * 0.5)
        
        # Check BER
        ber = row.get('BER', 0.0)
        if pd.isna(ber):
            ber = 0.0
        if float(ber) > self.qos_thresholds['ber']:
            penalty += self.qos_penalty_weights['ber'] * (1 + progress_factor * 0.5)
        
        # Check SNR
        snr = row.get('SNR_dB', 0.0)
        if pd.isna(snr):
            snr = 0.0
        if float(snr) < self.qos_thresholds['snr']:
            penalty += self.qos_penalty_weights['snr'] * (1 + progress_factor * 0.5)
        
        # Check SINR
        sinr = row.get('SINR_dB', 0.0)
        if pd.isna(sinr):
            sinr = 0.0
        if float(sinr) < self.qos_thresholds['sinr']:
            penalty += self.qos_penalty_weights['sinr'] * (1 + progress_factor * 0.5)
        
        return np.clip(penalty, 0, 0.3)  # Max penalty 0.3
    
    def _calculate_shaping_bonus_optimized(self, row: pd.Series, selected_network: str) -> float:
        """
        Calculate reward shaping bonus (OPTIMIZED).
        
        Args:
            row (pd.Series): Data row
            selected_network (str): Selected network
            
        Returns:
            float: Shaping bonus
        """
        bonus = 0.0
        
        # 1. Network selection quality bonus
        throughput = row.get('Throughput_Mbps', 0.0)
        if pd.isna(throughput):
            throughput = 0.0
        
        if throughput > 50:
            bonus += 0.02
        if throughput > 100:
            bonus += 0.03
        
        # 2. Latency bonus (lower is better)
        latency = row.get('Latency_ms', 0.0)
        if pd.isna(latency):
            latency = 0.0
        if latency < 10:
            bonus += 0.03
        elif latency < 20:
            bonus += 0.02
        
        # 3. Packet loss bonus
        packet_loss = row.get('Packet_Loss_pct', 0.0)
        if pd.isna(packet_loss):
            packet_loss = 0.0
        if packet_loss < 1:
            bonus += 0.02
        
        # 4. SNR bonus
        snr = row.get('SNR_dB', 0.0)
        if pd.isna(snr):
            snr = 0.0
        if snr > 20:
            bonus += 0.02
        elif snr > 10:
            bonus += 0.01
        
        return np.clip(bonus, 0, 0.15)  # Max bonus 0.15
    
    def calculate_reward_for_action(
        self,
        row: pd.Series,
        action: int,
        previous_action: Optional[int] = None,
        area: Optional[str] = None,
        step: Optional[int] = None
    ) -> float:
        """
        Calculate reward for an action (network index).
        
        Args:
            row (pd.Series): Data row
            action (int): Selected network index (0-4)
            previous_action (int): Previous action index
            area (str): Current area
            step (int): Current step
            
        Returns:
            float: Calculated reward
        """
        selected_network = NETWORK_TYPES[action]
        previous_network = NETWORK_TYPES[previous_action] if previous_action is not None else None
        
        return self.calculate_reward(row, selected_network, previous_network, area=area, step=step)
    
    def set_handover_penalty(self, penalty: float) -> None:
        """Set the handover penalty."""
        self.handover_penalty = penalty
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set reward weights."""
        self.weights = weights
    
    def get_reward_components(
        self,
        row: pd.Series,
        selected_network: str,
        previous_network: Optional[str] = None,
        area: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get individual reward components for analysis.
        """
        weights = self._get_weights(area)
        
        throughput_score = self._calculate_throughput_score(row)
        latency_score = self._calculate_latency_score(row)
        pl_score, ber_score = self._calculate_reliability_score(row)
        snr_score = self._calculate_snr_score(row)
        sinr_score = self._calculate_sinr_score(row)
        
        components = {
            'throughput_raw': throughput_score,
            'latency_raw': latency_score,
            'packet_loss_raw': pl_score,
            'ber_raw': ber_score,
            'snr_raw': snr_score,
            'sinr_raw': sinr_score,
            'weighted_throughput': weights.get('throughput', 0.35) * throughput_score,
            'weighted_latency': weights.get('latency', 0.25) * latency_score,
            'weighted_packet_loss': weights.get('packet_loss', 0.20) * pl_score,
            'weighted_ber': weights.get('ber', 0.10) * ber_score,
            'weighted_snr': weights.get('snr', 0.05) * snr_score,
            'weighted_sinr': weights.get('sinr', 0.05) * sinr_score
        }
        
        is_handover = previous_network is not None and selected_network != previous_network
        if is_handover:
            components['handover_penalty'] = -self.handover_penalty
        else:
            components['handover_penalty'] = 0.0
        
        qos_penalty = self._calculate_qos_penalty_optimized(row)
        components['qos_penalty'] = -qos_penalty
        
        shaping_bonus = self._calculate_shaping_bonus_optimized(row, selected_network)
        components['shaping_bonus'] = shaping_bonus
        
        components['total'] = sum([
            components['weighted_throughput'],
            components['weighted_latency'],
            components['weighted_packet_loss'],
            components['weighted_ber'],
            components['weighted_snr'],
            components['weighted_sinr'],
            components['handover_penalty'],
            components['qos_penalty'],
            components['shaping_bonus']
        ])
        
        components['is_handover'] = is_handover
        components['weights'] = weights
        
        return components
    
    def get_reward_explanation(
        self,
        row: pd.Series,
        selected_network: str,
        previous_network: Optional[str] = None,
        area: Optional[str] = None
    ) -> str:
        """Get a human-readable explanation of the reward."""
        components = self.get_reward_components(row, selected_network, previous_network, area)
        weights = components.get('weights', self.weights)
        
        explanation = [
            "="*70,
            f"REWARD BREAKDOWN for {selected_network}",
            "="*70,
            f"Throughput:  {components['weighted_throughput']:.3f} "
            f"(score: {components['throughput_raw']:.3f} × {weights.get('throughput', 0.35):.2f})",
            f"Latency:     {components['weighted_latency']:.3f} "
            f"(score: {components['latency_raw']:.3f} × {weights.get('latency', 0.25):.2f})",
            f"Packet Loss: {components['weighted_packet_loss']:.3f} "
            f"(score: {components['packet_loss_raw']:.3f} × {weights.get('packet_loss', 0.20):.2f})",
            f"BER:         {components['weighted_ber']:.3f} "
            f"(score: {components['ber_raw']:.3f} × {weights.get('ber', 0.10):.2f})",
            f"SNR:         {components['weighted_snr']:.3f} "
            f"(score: {components['snr_raw']:.3f} × {weights.get('snr', 0.05):.2f})",
            f"SINR:        {components['weighted_sinr']:.3f} "
            f"(score: {components['sinr_raw']:.3f} × {weights.get('sinr', 0.05):.2f})"
        ]
        
        if components['handover_penalty'] < 0:
            explanation.append(f"Handover:    {components['handover_penalty']:.3f} (penalty)")
        
        if components['qos_penalty'] < 0:
            explanation.append(f"QoS Penalty: {components['qos_penalty']:.3f}")
        
        if components['shaping_bonus'] > 0:
            explanation.append(f"Shaping:     +{components['shaping_bonus']:.3f} (bonus)")
        
        explanation.append(f"\n{'─'*70}")
        explanation.append(f"TOTAL:       {components['total']:.3f}")
        
        if area and self.area_specific_weights:
            explanation.append(f"Area:        {area} (using area-specific weights)")
        
        explanation.append("="*70)
        
        return "\n".join(explanation)
    
    def get_qos_report(self, row: pd.Series) -> Dict[str, Union[bool, float, str]]:
        """Get QoS status report for a data row."""
        report = {}
        
        for metric, threshold in self.qos_thresholds.items():
            value = row.get(metric.replace('_', '_'), 0.0)
            if pd.isna(value):
                value = 0.0
            value = float(value)
            
            is_violation = False
            status = "OK"
            
            if metric in ['throughput', 'snr', 'sinr']:
                is_violation = value < threshold
                if is_violation:
                    status = f"Low: {value:.2f} < {threshold}"
            else:
                is_violation = value > threshold
                if is_violation:
                    status = f"High: {value:.2f} > {threshold}"
            
            report[metric] = {
                'value': value,
                'threshold': threshold,
                'violation': is_violation,
                'status': status
            }
        
        report['overall_status'] = 'Violation' if any(
            r['violation'] for r in report.values()
        ) else 'OK'
        
        return report


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING REWARD CALCULATOR")
    print("="*80)
    
    from data_preprocessing.data_loader import DataLoader
    
    loader = DataLoader('data_raw/Hybrid_Network_TN_NTN_Final.csv')
    data = loader.load()
    
    reward_calc = RewardCalculator(
        normalize=True,
        normalize_type='robust',
        use_qos_penalty=True,
        reward_shaping=True,
        area_specific_weights=True
    )
    
    reward_calc.fit_bounds(data.head(5000))
    
    sample_row = data.iloc[0]
    area = sample_row.get('Area', 'Urban')
    
    print(f"\nTesting reward for different networks (Area: {area}):")
    print("="*70)
    
    for network in ['NR_5G', 'WiFi', 'SAT (LEO)', 'HAPS', 'UAV']:
        reward = reward_calc.calculate_reward(sample_row, network, None, area=area)
        print(f"   {network}: {reward:.4f}")
    
    print("\nTesting reward with handover penalty:")
    print("="*70)
    reward_no_handover = reward_calc.calculate_reward(sample_row, 'NR_5G', None, area=area)
    reward_handover = reward_calc.calculate_reward(sample_row, 'NR_5G', 'WiFi', area=area)
    print(f"   No handover: {reward_no_handover:.4f}")
    print(f"   With handover: {reward_handover:.4f}")
    print(f"   Penalty applied: {reward_no_handover - reward_handover:.4f}")
    
    print("\nReward components for NR_5G:")
    print("="*70)
    components = reward_calc.get_reward_components(sample_row, 'NR_5G', 'WiFi', area=area)
    for key, value in components.items():
        if not isinstance(value, (dict, list)):
            print(f"   {key}: {value:.4f}")
    
    print("\nReward explanation:")
    print("="*70)
    print(reward_calc.get_reward_explanation(sample_row, 'NR_5G', 'WiFi', area=area))
    
    print("\nRewardCalculator test completed successfully!")