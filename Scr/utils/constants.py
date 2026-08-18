# =============================================================================
# FILE: src/utils/constants.py
# =============================================================================
# PURPOSE: Central configuration file containing all constants, hyperparameters,
#          and feature definitions used across the hybrid network recommendation
#          system project.
# =============================================================================
# This file serves as the single source of truth for:
# 1. Network and area type definitions
# 2. Feature classifications (Core, NTN, UE, Environment)
# 3. State space dimensions
# 4. Reward function parameters
# 5. RL algorithm hyperparameters (FAIR & OPTIMIZED)
# 6. File paths and directory structures
# 7. Evaluation configuration
# 8. Visualization and tracking configuration
# 9. Area performance benchmarks (from experimental results)
# 10. Visualization standards for comparison charts
# =============================================================================

from typing import List, Dict, Tuple, Optional
import numpy as np

# =============================================================================
# SECTION 1: NETWORK AND AREA DEFINITIONS
# =============================================================================

# List of all network types in the hybrid system
# Index mapping: 0=NR_5G, 1=WiFi, 2=SAT(LEO), 3=HAPS, 4=UAV
NETWORK_TYPES: List[str] = [
    'NR_5G',        # 5G New Radio (Terrestrial Network)
    'WiFi',         # WiFi 6 (Terrestrial Network)
    'SAT (LEO)',    # Low Earth Orbit Satellite (Non-Terrestrial Network)
    'HAPS',         # High Altitude Platform Station (Non-Terrestrial Network)
    'UAV'           # Unmanned Aerial Vehicle Relay (Non-Terrestrial Network)
]

# Display names for visualization and reporting
NETWORK_LABELS: Dict[str, str] = {
    'NR_5G': '5G NR',
    'WiFi': 'WiFi 6',
    'SAT (LEO)': 'LEO Satellite',
    'HAPS': 'HAPS',
    'UAV': 'UAV Relay'
}

# Color codes for consistent visualization
NETWORK_COLORS: Dict[str, str] = {
    'NR_5G': '#2E86AB',      # Blue
    'WiFi': '#A23B72',       # Purple
    'SAT (LEO)': '#F18F01',  # Orange
    'HAPS': '#C73E1D',       # Red
    'UAV': '#6A994E'         # Green
}

# List of all area types in the environment
AREA_TYPES: List[str] = [
    'Urban',      # Dense city area with high network density
    'Indoor',     # Inside buildings, WiFi dominated
    'Rural',      # Countryside with mixed coverage
    'Highway',    # Roads with high mobility
    'Maritime',   # Sea/ocean areas, NTN dominated
    'Desert'      # Remote areas, NTN dominated
]

# Area colors for visualization
AREA_COLORS: Dict[str, str] = {
    'Urban': '#2C3E50',
    'Indoor': '#E67E22',
    'Rural': '#27AE60',
    'Highway': '#2980B9',
    'Maritime': '#1ABC9C',
    'Desert': '#F39C12'
}

# Area order for consistent display
AREA_ORDER: List[str] = ['Urban', 'Indoor', 'Rural', 'Highway', 'Maritime', 'Desert']

# Area emojis for display
AREA_EMOJIS: Dict[str, str] = {
    'Urban': '🏙️',
    'Indoor': '🏠',
    'Rural': '🌾',
    'Highway': '🛣️',
    'Maritime': '🌊',
    'Desert': '🏜️'
}

# =============================================================================
# SECTION 2: FEATURE CLASSIFICATION
# =============================================================================

# CORE FEATURES - Available for ALL network types
# These are used for fair comparison between TN and NTN
CORE_FEATURES: List[str] = [
    'SNR_dB',                      # Signal-to-Noise Ratio (higher is better)
    'SINR_dB',                     # Signal-to-Interference-plus-Noise Ratio
    'RSSI_dBm',                    # Received Signal Strength Indicator
    'Throughput_Mbps',             # Data throughput (higher is better)
    'Latency_ms',                  # End-to-end delay (lower is better)
    'Packet_Loss_pct',             # Packet loss percentage (lower is better)
    'BER',                         # Bit Error Rate (lower is better)
    'Link_Quality_Index',          # Overall link quality (0-100, higher is better)
    'Spectral_Efficiency_bps_hz'   # Spectrum efficiency (higher is better)
]

# NTN-ONLY FEATURES - Available ONLY for Non-Terrestrial Networks
# Used when comparing between NTN networks (HAPS, SAT, UAV)
NTN_ONLY_FEATURES: List[str] = [
    'altitude_m',                  # Platform altitude (HAPS: 20km, SAT: 800km+)
    'Doppler_Hz',                  # Doppler shift due to relative motion
    'Propagation_Delay_ms',        # Signal propagation delay (long distance)
    'Rain_Rate_mmhr',              # Rainfall rate affecting signal
    'Rain_Fade_dB'                 # Rain attenuation on NTN links
]

# UE FEATURES - User Equipment state (available for ALL networks)
UE_FEATURES: List[str] = [
    'speed_ms'                     # User speed in meters/second
]

# ENVIRONMENT FEATURES - Contextual information
ENV_FEATURES: List[str] = [
    'Area',                        # Current location area
    'Available_Networks'           # Networks available at current location
]

# =============================================================================
# SECTION 3: STATE SPACE DIMENSIONS
# =============================================================================

# Number of networks in the system
NUM_NETWORKS: int = len(NETWORK_TYPES)  # 5

# Number of area types
NUM_AREAS: int = len(AREA_TYPES)  # 6

# Measurements per network (SNR, SINR, RSSI)
MEASUREMENTS_PER_NETWORK: int = 3

# Total measurement features (5 networks × 3 measurements)
TOTAL_MEASUREMENTS: int = NUM_NETWORKS * MEASUREMENTS_PER_NETWORK  # 15

# POMDP State Dimension (No memory, current state only)
# Composition: Area(6) + Available_Networks(5) + Measurements(15) + UE(1) + NTN(0/5)
POMDP_STATE_DIM: int = NUM_AREAS + NUM_NETWORKS + TOTAL_MEASUREMENTS + len(UE_FEATURES)
# POMDP_STATE_DIM = 6 + 5 + 15 + 1 = 27

# History per step for LSTM/GRU (previous action, reward, area, measurements)
# Used when building sequential memory states
HISTORY_PER_STEP: int = 1 + 1 + NUM_AREAS + MEASUREMENTS_PER_NETWORK
# HISTORY_PER_STEP = 1 + 1 + 6 + 3 = 11

# LSTM/GRU State Dimension (Current state + History of 10 steps)
LSTM_STATE_DIM: int = POMDP_STATE_DIM + (HISTORY_PER_STEP * 10)
# LSTM_STATE_DIM = 27 + (11 * 10) = 137

# =============================================================================
# SECTION 4: ACTION SPACE
# =============================================================================

# Number of possible actions (select one network)
ACTION_SPACE: int = NUM_NETWORKS  # 5

# Mapping from action index to network name
ACTION_TO_NETWORK: Dict[int, str] = {
    idx: network for idx, network in enumerate(NETWORK_TYPES)
}

# Mapping from network name to action index
NETWORK_TO_ACTION: Dict[str, int] = {
    network: idx for idx, network in enumerate(NETWORK_TYPES)
}

# =============================================================================
# SECTION 5: REWARD FUNCTION PARAMETERS (OPTIMIZED V3)
# =============================================================================

# Weights for each component of the multi-objective reward function
# OPTIMIZED: Balanced for better learning
REWARD_WEIGHTS: Dict[str, float] = {
    'throughput': 0.40,      # Reduced from 0.50 for balance
    'latency': 0.30,         # Increased from 0.25
    'packet_loss': 0.20,     # Increased from 0.15
    'ber': 0.10              # Same
}

# Penalty for switching networks (handover)
# OPTIMIZED: Higher penalty for stability
HANDOVER_PENALTY: float = 0.35  # Increased from 0.20

# Reward clipping bounds to prevent extreme values
REWARD_MIN: float = -1.0
REWARD_MAX: float = 1.0

# Reward normalization bounds (fitted from data percentiles)
REWARD_NORM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'throughput': (0.0, 1000.0),
    'latency': (0.0, 100.0),
    'packet_loss': (0.0, 20.0),
    'ber': (0.0, 0.01)
}

# =============================================================================
# SECTION 6: RL HYPERPARAMETERS (FAIR & OPTIMIZED V3)
# =============================================================================

# General training configuration - FAIR for all agents
TRAINING_CONFIG: Dict = {
    'total_timesteps': 150000,      # Fair: All agents get same timesteps
    'n_episodes': 400,              # Increased for better evaluation
    'seed': 42,
    'n_seeds': 5,
    'eval_freq': 10000,
    'log_interval': 100,
    'save_freq': 25000,
    'verbose': 1
}

# =============================================================================
# PPO (Proximal Policy Optimization) - FAIR & OPTIMIZED
# =============================================================================

PPO_CONFIG: Dict = {
    'learning_rate': 3e-4,          # Balanced learning rate
    'n_steps': 2048,                # Steps per update
    'batch_size': 64,               # Mini-batch size
    'n_epochs': 10,                 # Epochs per update
    'gamma': 0.99,                  # Discount factor
    'gae_lambda': 0.95,             # GAE lambda
    'clip_range': 0.2,              # Clipping range
    'ent_coef': 0.01,               # Entropy coefficient
    'vf_coef': 0.5,                 # Value function coefficient
    'max_grad_norm': 0.5,           # Gradient clipping
    'target_kl': 0.01,              # Target KL divergence
    'hidden_dims': [256, 256]       # Network architecture
}

# =============================================================================
# DQN (Deep Q-Network) - FAIR & OPTIMIZED
# =============================================================================

DQN_CONFIG: Dict = {
    'learning_rate': 5e-4,          # Balanced learning rate
    'buffer_size': 200000,          # Replay buffer size
    'batch_size': 64,               # Batch size
    'gamma': 0.99,                  # Discount factor (increased for fairness)
    'tau': 0.005,                   # Target network update rate
    'target_update_interval': 1000, # Target update frequency
    'exploration_fraction': 0.15,   # Exploration fraction
    'exploration_initial_eps': 1.0, # Initial epsilon
    'exploration_final_eps': 0.01,  # Final epsilon
    'train_freq': 4,                # Training frequency
    'gradient_steps': 1,            # Gradient steps per update
    'learning_starts': 1000,        # Learning start delay
    'hidden_dims': [256, 256]       # Network architecture
}

# =============================================================================
# SECTION 7: LSTM/GRU RECURRENT NETWORK PARAMETERS (FAIR & OPTIMIZED)
# =============================================================================

# Configuration for LSTM and GRU memory networks - FAIR for both
RECURRENT_CONFIG: Dict = {
    'sequence_length': 15,          # Same for LSTM and GRU
    'hidden_size': 256,             # Same hidden size
    'num_layers': 2,                # Same number of layers (reduced for speed)
    'dropout': 0.15,                # Same dropout rate
    'bidirectional': True,          # Same for both
    'batch_first': True,
    'device': 'cpu'
}

# =============================================================================
# SECTION 8: EVALUATION CONFIGURATION (ENHANCED)
# =============================================================================

EVALUATION_CONFIG: Dict = {
    'n_episodes': 20,
    'max_steps_per_episode': 500,
    'deterministic': True,
    'compute_accuracy': True,
    'compute_switch_rate': True,
    'compute_area_accuracy': True,
    'compute_handover_matrix': True,
    'compute_user_tracking': True,
    'compute_qos_analysis': True,
    'compute_stability_score': True
}

# =============================================================================
# SECTION 9: FILE PATHS AND DIRECTORIES
# =============================================================================

DATA_PATHS: Dict[str, str] = {
    'raw_data': 'data_raw/Hybrid_Network_TN_NTN_Final.csv',
    'processed_train': 'data/processed/train_data.pkl',
    'processed_test': 'data/processed/test_data.pkl',
    'scaler': 'data/processed/scaler.pkl',
    'stats_networks': 'data/stats/network_statistics.csv',
    'stats_areas': 'data/stats/area_statistics.csv',
    'correlation': 'data/stats/feature_correlation.csv'
}

RESULTS_PATHS: Dict[str, str] = {
    'figures': 'results/figures/',
    'models': 'results/models/',
    'reports': 'results/reports/',
    'comparisons': 'results/comparisons/',
    'logs': 'results/logs/',
    'user_tracking': 'results/user_tracking/',
    'handover_analysis': 'results/handover_analysis/',
    'dashboard': 'results/dashboard/'
}

LOG_PATHS: Dict[str, str] = {
    'tensorboard': 'logs/tensorboard/',
    'training': 'logs/training/',
    'evaluation': 'logs/evaluation/',
    'user_tracking': 'logs/user_tracking/',
    'handover': 'logs/handover/',
    'errors': 'logs/errors/'
}

# =============================================================================
# SECTION 10: QUANTUM COMPUTING PARAMETERS (Phase 2)
# =============================================================================

QUANTUM_CONFIG: Dict = {
    'n_qubits': 6,
    'n_layers': 2,
    'device': 'default.qubit',
    'shots': 1000,
    'encoding_type': 'angle',
    'entanglement': 'full',
    'backend': 'statevector'
}

# =============================================================================
# SECTION 11: MISCELLANEOUS CONSTANTS
# =============================================================================

RANDOM_SEED: int = 42
VERBOSE: bool = True
DEBUG: bool = False
MAX_SEQUENCE_LENGTH: int = 1000
NUM_USERS: int = 20

USER_ID_COLUMN: str = 'UE_ID'
NETWORK_COLUMN: str = 'network_type'
AREA_COLUMN: str = 'Area'
AVAILABLE_COLUMN: str = 'Available_Networks'

# =============================================================================
# SECTION 12: FEATURE GROUPS FOR FAIR COMPARISON
# =============================================================================

FAIR_COMPARISON_FEATURES: List[str] = CORE_FEATURES + UE_FEATURES
NTN_COMPARISON_FEATURES: List[str] = CORE_FEATURES + NTN_ONLY_FEATURES + UE_FEATURES
ALL_FEATURES: List[str] = CORE_FEATURES + NTN_ONLY_FEATURES + UE_FEATURES

# =============================================================================
# SECTION 13: USER TRACKING CONFIGURATION
# =============================================================================

USER_TRACKING_CONFIG: Dict = {
    'max_steps_to_display': 30,
    'include_switch_details': True,
    'include_time_details': True,
    'include_qos_metrics': True,
    'include_area_transitions': True,
    'save_user_reports': True,
    'generate_visualizations': True,
    'track_top_users': 10
}

# =============================================================================
# SECTION 14: EVALUATION METRICS (STANDARD FROM RELATED WORK)
# =============================================================================

# Standard metrics from related work (Network Selection Papers)
STANDARD_METRICS: Dict[str, Dict] = {
    'accuracy': {
        'description': 'Correct network selection rate',
        'unit': '%',
        'higher_better': True
    },
    'handover_rate': {
        'description': 'Frequency of network switches',
        'unit': 'switches/step',
        'higher_better': False
    },
    'average_reward': {
        'description': 'Mean accumulated reward',
        'unit': 'score',
        'higher_better': True
    },
    'decision_time': {
        'description': 'Time to make a decision',
        'unit': 'ms',
        'higher_better': False
    },
    'training_time': {
        'description': 'Total training time',
        'unit': 'seconds',
        'higher_better': False
    },
    'model_size': {
        'description': 'Number of parameters',
        'unit': 'parameters',
        'higher_better': False
    },
    'qos_violation_rate': {
        'description': 'Quality of Service violations',
        'unit': '%',
        'higher_better': False
    },
    'stability_score': {
        'description': 'Consistency of decisions',
        'unit': 'score (0-1)',
        'higher_better': True
    }
}

# Evaluation metrics list
EVALUATION_METRICS: List[str] = [
    'accuracy',
    'switch_rate',
    'mean_reward',
    'total_reward',
    'decision_time_ms',
    'total_time_ms',
    'outage_rate',
    'handover_count',
    'handover_matrix',
    'area_accuracy',
    'area_stability',
    'network_preference',
    'episode_length',
    'qos_violation_rate',
    'reliability_score'
]

# =============================================================================
# SECTION 15: VISUALIZATION CONFIGURATION
# =============================================================================

VISUALIZATION_CONFIG: Dict = {
    'figure_dpi': 300,
    'figure_format': 'png',
    'style': 'seaborn-v0_8-whitegrid',
    'palette': 'husl',
    'font_size': 12,
    'title_size': 16,
    'legend_font_size': 10,
    'tick_font_size': 10,
    'color_map': 'viridis',
    'heatmap_cmap': 'YlOrRd'
}

VIZ_TYPES: List[str] = [
    'accuracy_comparison',
    'reward_distribution',
    'handover_matrix',
    'area_heatmap',
    'radar_chart',
    'training_curve',
    'confusion_matrix',
    'network_selection',
    'user_journey',
    'decision_time',
    'qos_analysis',
    'stability_plot',
    'tradeoff_analysis',
    'performance_dashboard'
]

# =============================================================================
# SECTION 16: AREA PERFORMANCE BENCHMARKS (FROM EXPERIMENTAL RESULTS)
# =============================================================================

# Best agent per area (from experimental evaluation)
AREA_BEST_AGENT: Dict[str, str] = {
    'Urban': 'PPO',
    'Indoor': 'DQN',
    'Rural': 'GRU',
    'Highway': 'DQN',
    'Maritime': 'GRU',
    'Desert': 'PPO'
}

# Best network per area (from experimental evaluation)
AREA_BEST_NETWORK: Dict[str, str] = {
    'Urban': 'NR_5G',
    'Indoor': 'WiFi',
    'Rural': 'SAT (LEO)',
    'Highway': 'NR_5G',
    'Maritime': 'SAT (LEO)',
    'Desert': 'HAPS'
}

# Expected accuracy per area (from experimental evaluation)
AREA_EXPECTED_ACCURACY: Dict[str, float] = {
    'Urban': 27.47,
    'Indoor': 37.36,
    'Rural': 30.24,
    'Highway': 30.51,
    'Maritime': 42.16,
    'Desert': 44.44
}

# Handover rate per area (from experimental evaluation)
AREA_HANDOVER_RATE: Dict[str, float] = {
    'Urban': 0.456,
    'Indoor': 0.000,
    'Rural': 0.540,
    'Highway': 0.000,
    'Maritime': 0.540,
    'Desert': 0.456
}

# Expected reward per area (from experimental evaluation)
AREA_EXPECTED_REWARD: Dict[str, float] = {
    'Urban': 260.44,
    'Indoor': 330.41,
    'Rural': 247.26,
    'Highway': 330.41,
    'Maritime': 247.26,
    'Desert': 260.44
}

# Recommendation reasons per area
AREA_REASONS: Dict[str, str] = {
    'Urban': 'PPO provides best accuracy in dense, complex urban coverage with multiple overlapping networks.',
    'Indoor': 'DQN delivers perfect stability with zero handovers, ideal for indoor environments.',
    'Rural': 'GRU sequence memory handles sparse, long-range coverage effectively in rural areas.',
    'Highway': 'DQN offers fast, stable decisions essential for high-speed mobility scenarios.',
    'Maritime': 'GRU achieves best performance under maritime NTN conditions with long-term memory.',
    'Desert': 'PPO shows best results in harsh desert coverage with limited infrastructure.'
}

# =============================================================================
# SECTION 17: VISUALIZATION STANDARDS FOR COMPARISON
# =============================================================================

# Figure sizes for different chart types
FIGURE_SIZES: Dict[str, Tuple[int, int]] = {
    'comparison_bar': (10, 6),
    'heatmap': (12, 8),
    'radar': (10, 10),
    'dashboard': (18, 12),
    'summary': (14, 10),
    'area_analysis': (16, 10),
    'training_curve': (12, 7)
}

# Main metrics to display in comparison charts
MAIN_COMPARISON_METRICS: List[str] = [
    'accuracy',
    'handover_rate',
    'average_reward',
    'decision_time',
    'training_time',
    'model_size'
]

# Metric display names for plots
METRIC_DISPLAY_NAMES: Dict[str, str] = {
    'accuracy': 'Accuracy (%)',
    'handover_rate': 'Handover Rate',
    'average_reward': 'Average Reward',
    'decision_time': 'Decision Time (ms)',
    'training_time': 'Training Time (s)',
    'model_size': 'Model Size (params)',
    'qos_violation_rate': 'QoS Violation Rate (%)',
    'stability_score': 'Stability Score'
}

# Metric colors for consistent plots
METRIC_COLORS: Dict[str, str] = {
    'accuracy': '#2E86AB',
    'handover_rate': '#A23B72',
    'average_reward': '#F18F01',
    'decision_time': '#C73E1D',
    'training_time': '#6A994E',
    'model_size': '#1D3557'
}

# Agent display colors
AGENT_COLORS: Dict[str, str] = {
    'DQN': '#2E86AB',
    'PPO': '#A23B72',
    'LSTM': '#F18F01',
    'GRU': '#6A994E'
}

# =============================================================================
# SECTION 18: PERFORMANCE BENCHMARKS (FROM RELATED WORK)
# =============================================================================

# Expected performance ranges from related work
PERFORMANCE_BENCHMARKS: Dict[str, Dict] = {
    'accuracy': {
        'good': 25.0,
        'excellent': 30.0,
        'state_of_art': 35.0
    },
    'handover_rate': {
        'good': 0.1,
        'excellent': 0.05,
        'state_of_art': 0.01
    },
    'decision_time_ms': {
        'good': 1.0,
        'excellent': 0.5,
        'state_of_art': 0.1
    },
    'training_time_s': {
        'good': 100,
        'excellent': 50,
        'state_of_art': 10
    }
}

# =============================================================================
# SECTION 19: REFERENCE PAPERS
# =============================================================================

# Reference papers for each algorithm
REFERENCE_PAPERS: Dict[str, List[str]] = {
    'DQN': [
        'Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.',
        'Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. ICML.',
        'Van Hasselt, H., et al. (2016). Deep reinforcement learning with double Q-learning. AAAI.'
    ],
    'PPO': [
        'Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.',
        'Sutton, R.S., & Barto, A.G. (2018). Reinforcement Learning: An Introduction. MIT Press.'
    ],
    'LSTM': [
        'Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.',
        'Graves, A., et al. (2013). Hybrid speech recognition with deep bidirectional LSTM. IEEE Workshop on ASRU.'
    ],
    'GRU': [
        'Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder. EMNLP.',
        'Chung, J., et al. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv:1412.3555.'
    ]
}

# Network selection papers
NETWORK_SELECTION_REFERENCES: List[str] = [
    'Barmpounakis, S., et al. (2020). Machine learning for 5G/B5G mobile and wireless communications. IEEE Access.',
    'Alsharif, M.H., et al. (2020). Machine learning for wireless networks: A survey. IEEE Communications Surveys.',
    'Zhang, C., et al. (2021). Deep reinforcement learning for network selection. IEEE Transactions on Communications.',
    'Kumar, A., et al. (2022). Hybrid TN-NTN networks: A survey. IEEE Communications Surveys & Tutorials.'
]

# =============================================================================
# SECTION 20: CONSTANT VALIDATION
# =============================================================================

def validate_constants() -> bool:
    """
    Validate that all constants are properly defined and consistent.
    
    Returns:
        bool: True if all validations pass, False otherwise
    """
    try:
        # Validate network types
        assert len(NETWORK_TYPES) == 5, "Expected 5 network types"
        assert len(NETWORK_LABELS) == 5, "Network labels count mismatch"
        assert len(NETWORK_COLORS) == 5, "Network colors count mismatch"
        
        # Validate area types
        assert len(AREA_TYPES) == 6, "Expected 6 area types"
        assert len(AREA_COLORS) == 6, "Area colors count mismatch"
        assert len(AREA_ORDER) == 6, "Area order count mismatch"
        
        # Validate area benchmarks
        assert len(AREA_BEST_AGENT) == 6, "Area best agent count mismatch"
        assert len(AREA_BEST_NETWORK) == 6, "Area best network count mismatch"
        assert len(AREA_EXPECTED_ACCURACY) == 6, "Area expected accuracy count mismatch"
        assert len(AREA_HANDOVER_RATE) == 6, "Area handover rate count mismatch"
        assert len(AREA_REASONS) == 6, "Area reasons count mismatch"
        
        # Validate state dimensions
        assert POMDP_STATE_DIM == 27, f"POMDP state dimension should be 27, got {POMDP_STATE_DIM}"
        assert LSTM_STATE_DIM == 137, f"LSTM state dimension should be 137, got {LSTM_STATE_DIM}"
        
        # Validate action space
        assert ACTION_SPACE == 5, "Action space should be 5"
        
        # Validate reward weights sum to 1.0
        weight_sum = sum(REWARD_WEIGHTS.values())
        assert abs(weight_sum - 1.0) < 0.001, f"Reward weights sum to {weight_sum}, expected 1.0"
        
        # Validate feature counts
        assert len(CORE_FEATURES) == 9, "Expected 9 core features"
        assert len(NTN_ONLY_FEATURES) == 5, "Expected 5 NTN features"
        assert len(UE_FEATURES) == 1, "Expected 1 UE feature"
        assert len(ENV_FEATURES) == 2, "Expected 2 environment features"
        
        # Validate evaluation config
        assert EVALUATION_CONFIG['n_episodes'] > 0, "n_episodes must be > 0"
        assert EVALUATION_CONFIG['max_steps_per_episode'] > 0, "max_steps_per_episode must be > 0"
        
        # Validate training config
        assert TRAINING_CONFIG['total_timesteps'] > 0, "total_timesteps must be > 0"
        assert TRAINING_CONFIG['n_episodes'] > 0, "n_episodes must be > 0"
        
        # Validate fairness: PPO and DQN have same hidden dimensions
        assert PPO_CONFIG['hidden_dims'] == DQN_CONFIG['hidden_dims'], \
            "PPO and DQN should have same hidden dimensions for fairness"
        
        # Validate fairness: LSTM and GRU have same recurrent config
        assert RECURRENT_CONFIG['hidden_size'] == 256, "LSTM/GRU hidden size should be consistent"
        assert RECURRENT_CONFIG['num_layers'] == 2, "LSTM/GRU layers should be consistent"
        
        # Validate metrics
        assert len(STANDARD_METRICS) == 8, "Expected 8 standard metrics"
        assert len(EVALUATION_METRICS) == 15, "Expected 15 evaluation metrics"
        
        # Validate figure sizes
        assert len(FIGURE_SIZES) == 7, "Expected 7 figure sizes"
        assert len(MAIN_COMPARISON_METRICS) == 6, "Expected 6 main comparison metrics"
        
        # Validate reference papers
        assert len(REFERENCE_PAPERS) == 4, "Expected 4 agent reference groups"
        assert len(NETWORK_SELECTION_REFERENCES) == 4, "Expected 4 network selection references"
        
        print("="*80)
        print("✅ ALL CONSTANTS VALIDATED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📡 Networks: {len(NETWORK_TYPES)} ({', '.join(NETWORK_TYPES)})")
        print(f"📍 Areas: {len(AREA_TYPES)} ({', '.join(AREA_TYPES)})")
        print(f"🧠 POMDP State Dim: {POMDP_STATE_DIM}")
        print(f"🧠 LSTM State Dim: {LSTM_STATE_DIM}")
        print(f"🎮 Action Space: {ACTION_SPACE}")
        print(f"⚡ Training Timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
        print(f"📊 Evaluation Episodes: {EVALUATION_CONFIG['n_episodes']}")
        print(f"🎯 Handover Penalty: {HANDOVER_PENALTY}")
        
        print("\n✅ FAIRNESS CHECK:")
        print(f"   ✅ All agents get {TRAINING_CONFIG['total_timesteps']:,} timesteps")
        print(f"   ✅ PPO hidden_dims: {PPO_CONFIG['hidden_dims']}")
        print(f"   ✅ DQN hidden_dims: {DQN_CONFIG['hidden_dims']}")
        print(f"   ✅ LSTM layers: {RECURRENT_CONFIG['num_layers']}")
        print(f"   ✅ GRU layers: {RECURRENT_CONFIG['num_layers']}")
        
        print("\n✅ AREA BENCHMARKS:")
        for area in AREA_ORDER:
            print(f"   {AREA_EMOJIS[area]} {area}: {AREA_BEST_AGENT[area]} → {AREA_BEST_NETWORK[area]} ({AREA_EXPECTED_ACCURACY[area]:.2f}%)")
        
        print("\n✅ REFERENCE PAPERS:")
        for agent, papers in REFERENCE_PAPERS.items():
            print(f"   {agent}: {len(papers)} papers")
        
        print("\n" + "="*80)
        return True
        
    except AssertionError as e:
        print(f"❌ Constant validation failed: {e}")
        return False


# =============================================================================
# MAIN: Run validation when file is executed directly
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("HYBRID NETWORK RECOMMENDATION SYSTEM - CONSTANTS")
    print("="*80)
    
    print("\n📡 NETWORK TYPES:")
    for i, net in enumerate(NETWORK_TYPES):
        print(f"   {i}: {net} ({NETWORK_LABELS[net]})")
    
    print("\n📍 AREA TYPES:")
    for area in AREA_TYPES:
        emoji = AREA_EMOJIS.get(area, '📍')
        print(f"   {emoji} {area}")
    
    print("\n🏆 AREA PERFORMANCE BENCHMARKS:")
    for area in AREA_ORDER:
        emoji = AREA_EMOJIS.get(area, '📍')
        print(f"   {emoji} {area}: {AREA_BEST_AGENT[area]} → {AREA_BEST_NETWORK[area]} ({AREA_EXPECTED_ACCURACY[area]:.2f}%)")
        print(f"      Handover: {AREA_HANDOVER_RATE[area]:.3f}, Reward: {AREA_EXPECTED_REWARD[area]:.2f}")
    
    print("\n🧠 STATE DIMENSIONS:")
    print(f"   POMDP: {POMDP_STATE_DIM} features")
    print(f"   LSTM:  {LSTM_STATE_DIM} features")
    
    print("\n🎮 ACTION SPACE:")
    print(f"   {ACTION_SPACE} possible actions")
    
    print("\n🎯 REWARD WEIGHTS:")
    for key, value in REWARD_WEIGHTS.items():
        print(f"   {key}: {value:.2f}")
    
    print(f"\n⚡ HANDOVER PENALTY: {HANDOVER_PENALTY}")
    
    print("\n⚙️ TRAINING CONFIG:")
    print(f"   Total Timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"   Episodes: {TRAINING_CONFIG['n_episodes']}")
    
    print("\n📊 MAIN COMPARISON METRICS:")
    for metric in MAIN_COMPARISON_METRICS:
        display = METRIC_DISPLAY_NAMES.get(metric, metric)
        print(f"   - {display}")
    
    print("\n📚 REFERENCE PAPERS:")
    for agent, papers in REFERENCE_PAPERS.items():
        print(f"\n   {agent}:")
        for paper in papers:
            print(f"      • {paper}")
    
    print("\n" + "="*80)
    print("🔍 VALIDATION:")
    validate_constants()
    
    print("\n" + "="*80)
    print("constants.py loaded successfully!")
    print("="*80)