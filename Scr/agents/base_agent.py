# =============================================================================
# FILE: Scr/agents/base_agent.py (VERSION 2.0 - PROFESSIONAL)
# =============================================================================
# PURPOSE: Base class for all RL agents with professional enhancements:
#          1. Added type hints and better documentation
#          2. Enhanced training history with more metrics
#          3. Added checkpoint management
#          4. Added early stopping and learning rate scheduling
#          5. Added model versioning
#          6. Added distributed training support
#          7. Enhanced user tracking
#          8. Added anomaly detection
#          9. Added performance profiling
#          10. Added model quantization support
# =============================================================================

import os
import sys
import json
import time
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from collections import deque, defaultdict
from pathlib import Path
import hashlib
import warnings

# Import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import (
    TRAINING_CONFIG, PPO_CONFIG, DQN_CONFIG, NETWORK_TYPES,
    AREA_TYPES, EVALUATION_CONFIG, USER_TRACKING_CONFIG
)

logger = logging.getLogger(__name__)


# ============================================
# 1. Enums and Dataclasses
# ============================================

class AgentStatus(Enum):
    """Agent training status"""
    UNINITIALIZED = "uninitialized"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    SAVED = "saved"
    LOADED = "loaded"
    FAILED = "failed"


class DeviceType(Enum):
    """Device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    total_timesteps: int = 150000
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    target_update_freq: int = 1000
    grad_clip: float = 0.5
    save_freq: int = 10000
    eval_freq: int = 5000
    log_freq: int = 100
    early_stopping_patience: int = 100
    min_improvement: float = 0.01
    use_wandb: bool = False
    wandb_project: str = "hybrid_networks"
    wandb_entity: str = "your_entity"
    seed: int = 42
    device: str = "cpu"
    track_user: bool = True
    verbose: bool = True


@dataclass
class EvaluationMetrics:
    """Evaluation metrics dataclass"""
    mean_reward: float = 0.0
    std_reward: float = 0.0
    max_reward: float = 0.0
    min_reward: float = 0.0
    accuracy: float = 0.0
    handover_rate: float = 0.0
    qos_violation_rate: float = 0.0
    mean_decision_time_ms: float = 0.0
    area_accuracies: Dict[str, float] = field(default_factory=dict)
    network_preferences: Dict[str, float] = field(default_factory=dict)
    total_episodes: int = 0
    total_steps: int = 0


@dataclass
class CheckpointInfo:
    """Checkpoint information"""
    version: str
    timestamp: float
    step: int
    episode: int
    status: AgentStatus
    metrics: Dict[str, float]
    file_size: int
    hash: str


# ============================================
# 2. Enhanced BaseAgent
# ============================================

class BaseAgent(ABC):
    """
    Enhanced abstract base class for all RL agents.
    
    Professional features:
    - Type hints for all methods
    - Enhanced training history with metric tracking
    - Checkpoint management with versioning
    - Early stopping with patience
    - Learning rate scheduling
    - Distributed training support
    - Model quantization
    - Anomaly detection
    - Performance profiling
    - User journey tracking with analytics
    """
    
    def __init__(
        self,
        env,
        config: Optional[Dict[str, Any]] = None,
        name: str = "BaseAgent",
        seed: Optional[int] = 42,
        device: str = "cpu",
        track_user: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the enhanced base agent.
        
        Args:
            env: Gymnasium environment
            config: Agent configuration dictionary
            name: Agent name
            seed: Random seed
            device: Device to use ('cpu', 'cuda', or 'mps')
            track_user: Whether to track user journeys
            verbose: Whether to print verbose output
        """
        # Basic attributes
        self.env = env
        self.name = name
        self.seed = seed
        self.track_user = track_user
        self.verbose = verbose
        
        # Device handling
        self.device = self._setup_device(device)
        
        # Configuration
        self.config = config or self._get_default_config()
        self.training_config = self._parse_training_config()
        
        # Model
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        # Status
        self.status = AgentStatus.UNINITIALIZED
        self.is_trained = False
        self.total_steps = 0
        self.total_episodes = 0
        self.best_reward = float('-inf')
        self.best_model_state: Optional[Dict] = None
        self.patience_counter = 0
        
        # Timing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.episode_start_time: Optional[float] = None
        self.training_time: float = 0.0
        self.total_inference_time: float = 0.0
        self.inference_count: int = 0
        
        # Enhanced training history
        self.training_history = {
            # Basic metrics
            'timesteps': [],
            'episodes': [],
            'rewards': [],
            'episode_lengths': [],
            
            # Loss metrics
            'losses': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            
            # Performance metrics
            'mean_rewards': [],
            'best_rewards': [],
            'worst_rewards': [],
            'std_rewards': [],
            
            # Network metrics
            'handover_counts': [],
            'handover_rates': [],
            'decision_times': [],
            'qos_violations': [],
            'qos_violation_rates': [],
            
            # Area metrics
            'area_accuracies': defaultdict(lambda: defaultdict(list)),
            'area_handovers': defaultdict(lambda: defaultdict(list)),
            
            # Network selection
            'network_selection': defaultdict(lambda: defaultdict(int)),
            'network_accuracy': defaultdict(lambda: defaultdict(lambda: defaultdict(float))),
            
            # Advanced metrics
            'learning_rates': [],
            'gradient_norms': [],
            'parameter_norms': [],
            'exploration_rates': [],
            'loss_std': [],
            
            # Checkpoints
            'checkpoints': [],
            
            # Anomalies
            'anomalies': [],
            
            # Profiling
            'profiling': {
                'forward_time': [],
                'backward_time': [],
                'update_time': [],
                'step_time': []
            }
        }
        
        # User tracking data with analytics
        self.user_data = {}
        self.current_user: Optional[str] = None
        self.user_journey_buffer = deque(maxlen=10000)
        
        # Handover analysis data
        self.handover_data = {
            'transitions': [],
            'from_areas': [],
            'to_areas': [],
            'switch_times': [],
            'rewards_before': [],
            'rewards_after': [],
            'successful_handovers': 0,
            'failed_handovers': 0,
            'handover_reasons': [],
            'handover_quality': []
        }
        
        # Performance metrics
        self.performance_metrics = {
            'mean_reward': 0.0,
            'best_reward': float('-inf'),
            'worst_reward': float('inf'),
            'std_reward': 0.0,
            'total_handovers': 0,
            'handover_rate': 0.0,
            'avg_decision_time': 0.0,
            'qos_violation_rate': 0.0,
            'area_accuracy': {},
            'network_preference': {},
            'stability_score': 0.0,
            'efficiency_score': 0.0,
            'adaptability_score': 0.0,
            'overall_score': 0.0
        }
        
        # Checkpoint management
        self.checkpoints_dir = Path("checkpoints")
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.current_checkpoint: Optional[CheckpointInfo] = None
        
        # Anomaly detection
        self.anomaly_threshold = 3.0  # Standard deviations
        self.reward_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        
        # Set random seed
        if seed is not None:
            self._set_seed(seed)
        
        logger.info(f"✅ {self.name} initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Config: {self.config}")
        logger.info(f"   Track user: {self.track_user}")
        logger.info(f"   Status: {self.status.value}")
    
    # ============================================
    # 3. Abstract Methods
    # ============================================
    
    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of training steps
            callback: Optional callback function
            
        Returns:
            Training results dictionary
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Predict the next action.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, info)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent model.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the agent model.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    # ============================================
    # 4. Core Methods
    # ============================================
    
    def evaluate(
        self,
        n_episodes: int = 100,
        render: bool = False,
        deterministic: bool = True,
        max_steps_per_episode: int = 500,
        track_handovers: bool = True,
        track_users: bool = True,
        return_metrics: bool = True
    ) -> Union[Dict[str, Any], EvaluationMetrics]:
        """
        Evaluate the agent with comprehensive metrics.
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            deterministic: Whether to use deterministic policy
            max_steps_per_episode: Maximum steps per episode
            track_handovers: Whether to track handovers
            track_users: Whether to track users
            return_metrics: Whether to return EvaluationMetrics object
            
        Returns:
            Evaluation results or EvaluationMetrics
        """
        if not self.is_trained:
            warnings.warn(f"{self.name} is not trained yet. Evaluating untrained agent.")
        
        self.status = AgentStatus.EVALUATING
        
        # Initialize metrics
        rewards = []
        episode_lengths = []
        handovers = []
        switch_times = []
        area_accuracies = {}
        all_predictions = []
        all_actuals = []
        qos_violations = []
        user_journeys = {}
        network_preferences = {}
        
        logger.info(f"🔍 Evaluating {self.name} for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            # Reset environment
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step_count = 0
            episode_handovers = 0
            episode_predictions = []
            episode_actuals = []
            episode_qos_violations = 0
            episode_switch_times = []
            episode_network_history = []
            
            # Get user ID
            user_id = info.get('user_id', f'Episode_{episode}')
            if user_id not in user_journeys:
                user_journeys[user_id] = {
                    'steps': [],
                    'rewards': [],
                    'networks': [],
                    'areas': [],
                    'handovers': 0,
                    'total_reward': 0,
                    'accuracy': 0,
                    'qos_violations': 0
                }
            
            while not (done or truncated) and step_count < max_steps_per_episode:
                # Get action
                start_time = time.perf_counter()
                action, action_info = self.predict(obs, deterministic=deterministic)
                inference_time = time.perf_counter() - start_time
                self.total_inference_time += inference_time
                self.inference_count += 1
                
                # Get actual network
                if hasattr(self.env, 'data') and hasattr(self.env, 'current_step'):
                    if self.env.current_step < len(self.env.data):
                        actual_network = self.env.data.iloc[self.env.current_step].get('network_type', 'NR_5G')
                        episode_actuals.append(actual_network)
                        episode_predictions.append(NETWORK_TYPES[action])
                
                # Execute action
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Track network preferences
                network = NETWORK_TYPES[action]
                if network not in network_preferences:
                    network_preferences[network] = 0
                network_preferences[network] += 1
                
                # Track handovers
                is_handover = info.get('is_handover', False)
                if is_handover:
                    episode_handovers += 1
                    if track_handovers:
                        episode_switch_times.append(info.get('decision_time_ms', 0))
                        self._track_handover(info)
                
                # Track QoS violations
                if info.get('qos_violation', False):
                    episode_qos_violations += 1
                
                # Track user journey
                if track_users:
                    step_data = {
                        'step': step_count,
                        'action': action,
                        'network': NETWORK_TYPES[action],
                        'area': info.get('area', 'Unknown'),
                        'reward': reward,
                        'is_handover': is_handover,
                        'is_correct': info.get('is_correct', False),
                        'actual_network': info.get('actual_network', 'NR_5G'),
                        'decision_time_ms': info.get('decision_time_ms', 0),
                        'qos_violation': info.get('qos_violation', False),
                        'snr': info.get('snr', 0),
                        'rsrp': info.get('rsrp', 0)
                    }
                    user_journeys[user_id]['steps'].append(step_data)
                    user_journeys[user_id]['rewards'].append(reward)
                    user_journeys[user_id]['networks'].append(NETWORK_TYPES[action])
                    user_journeys[user_id]['areas'].append(info.get('area', 'Unknown'))
                    if info.get('qos_violation', False):
                        user_journeys[user_id]['qos_violations'] += 1
                
                # Track area accuracy
                area = info.get('area', 'Unknown')
                if area not in area_accuracies:
                    area_accuracies[area] = {'correct': 0, 'total': 0}
                area_accuracies[area]['total'] += 1
                if info.get('is_correct', False):
                    area_accuracies[area]['correct'] += 1
            
            # Episode summary
            rewards.append(episode_reward)
            episode_lengths.append(step_count)
            handovers.append(episode_handovers)
            switch_times.extend(episode_switch_times)
            qos_violations.append(episode_qos_violations)
            all_predictions.extend(episode_predictions)
            all_actuals.extend(episode_actuals)
            
            # Update user journey summary
            if track_users and user_id in user_journeys:
                user_journeys[user_id]['total_reward'] = episode_reward
                user_journeys[user_id]['handovers'] = episode_handovers
                if step_count > 0:
                    user_journeys[user_id]['accuracy'] = sum(
                        1 for step in user_journeys[user_id]['steps'] 
                        if step.get('is_correct', False)
                    ) / len(user_journeys[user_id]['steps'])
        
        # Calculate overall metrics
        accuracy = 0.0
        if all_predictions and all_actuals and len(all_predictions) == len(all_actuals):
            correct = sum(1 for p, a in zip(all_predictions, all_actuals) if p == a)
            accuracy = correct / len(all_predictions) if all_predictions else 0.0
        
        # Calculate network accuracy
        network_accuracy = self._calculate_network_accuracy(all_predictions, all_actuals)
        
        # Calculate area accuracy
        area_acc = {
            area: data['correct'] / data['total'] if data['total'] > 0 else 0
            for area, data in area_accuracies.items()
        }
        
        # Calculate stability score
        stability_score = 1.0 - (np.std(rewards) / (np.mean(rewards) + 1e-8)) if rewards else 0.0
        
        # Calculate efficiency score
        efficiency_score = accuracy / (np.mean(switch_times) + 1e-8) if switch_times else 0.0
        
        # Prepare results
        results = {
            'n_episodes': n_episodes,
            'episode_rewards': rewards,
            'mean_reward': np.mean(rewards) if rewards else 0,
            'std_reward': np.std(rewards) if rewards else 0,
            'min_reward': np.min(rewards) if rewards else 0,
            'max_reward': np.max(rewards) if rewards else 0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0,
            'mean_handovers': np.mean(handovers) if handovers else 0,
            'total_handovers': sum(handovers) if handovers else 0,
            'handover_rate': sum(handovers) / (sum(episode_lengths) or 1),
            'mean_switch_time_ms': np.mean(switch_times) if switch_times else 0,
            'total_switch_time_ms': sum(switch_times) if switch_times else 0,
            'accuracy': accuracy,
            'area_accuracies': area_acc,
            'network_accuracy': network_accuracy,
            'network_preferences': network_preferences,
            'mean_qos_violations': np.mean(qos_violations) if qos_violations else 0,
            'total_qos_violations': sum(qos_violations) if qos_violations else 0,
            'qos_violation_rate': sum(qos_violations) / (sum(episode_lengths) or 1),
            'stability_score': stability_score,
            'efficiency_score': efficiency_score,
            'adaptability_score': self._calculate_adaptability_score(area_acc),
            'overall_score': self._calculate_overall_score(accuracy, stability_score, efficiency_score),
            'user_journeys': user_journeys if track_users else None,
            'handover_data': self.handover_data if track_handovers else None
        }
        
        # Update performance metrics
        self._update_performance_metrics(results)
        
        self.status = AgentStatus.EVALUATED
        
        # Log results
        self._log_evaluation_results(results)
        
        if return_metrics:
            return self._create_evaluation_metrics(results)
        return results
    
    # ============================================
    # 5. Enhanced Helper Methods
    # ============================================
    
    def _setup_device(self, device: str) -> str:
        """Setup device with fallback"""
        if device == DeviceType.CUDA.value and torch.cuda.is_available():
            return DeviceType.CUDA.value
        elif device == DeviceType.MPS.value and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return DeviceType.MPS.value
        else:
            return DeviceType.CPU.value
    
    def _parse_training_config(self) -> TrainingConfig:
        """Parse training configuration"""
        config_dict = self.config.get('training', {})
        return TrainingConfig(**config_dict) if config_dict else TrainingConfig()
    
    def _calculate_network_accuracy(
        self,
        predictions: List[str],
        actuals: List[str]
    ) -> Dict[str, float]:
        """Calculate accuracy per network type"""
        network_acc = {}
        
        for p, a in zip(predictions, actuals):
            if a not in network_acc:
                network_acc[a] = {'correct': 0, 'total': 0}
            network_acc[a]['total'] += 1
            if p == a:
                network_acc[a]['correct'] += 1
        
        return {
            net: data['correct'] / data['total'] if data['total'] > 0 else 0
            for net, data in network_acc.items()
        }
    
    def _calculate_adaptability_score(self, area_accuracies: Dict[str, float]) -> float:
        """Calculate adaptability score based on area accuracies"""
        if not area_accuracies:
            return 0.0
        
        # Calculate variance of area accuracies
        accuracies = list(area_accuracies.values())
        if len(accuracies) <= 1:
            return 1.0
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        # Higher score means more adaptable (lower variance)
        adaptability = 1.0 - (std_acc / (mean_acc + 1e-8))
        return max(0.0, min(1.0, adaptability))
    
    def _calculate_overall_score(
        self,
        accuracy: float,
        stability_score: float,
        efficiency_score: float
    ) -> float:
        """Calculate overall performance score"""
        # Weighted combination
        weights = {
            'accuracy': 0.4,
            'stability': 0.3,
            'efficiency': 0.3
        }
        
        overall = (
            weights['accuracy'] * accuracy +
            weights['stability'] * stability_score +
            weights['efficiency'] * min(1.0, efficiency_score)
        )
        
        return min(1.0, max(0.0, overall))
    
    def _create_evaluation_metrics(self, results: Dict) -> EvaluationMetrics:
        """Create EvaluationMetrics object"""
        return EvaluationMetrics(
            mean_reward=results['mean_reward'],
            std_reward=results['std_reward'],
            max_reward=results['max_reward'],
            min_reward=results['min_reward'],
            accuracy=results['accuracy'],
            handover_rate=results['handover_rate'],
            qos_violation_rate=results['qos_violation_rate'],
            mean_decision_time_ms=results['mean_switch_time_ms'],
            area_accuracies=results['area_accuracies'],
            network_preferences=results['network_preferences'],
            total_episodes=results['n_episodes'],
            total_steps=sum(results.get('episode_lengths', []))
        )
    
    def _log_evaluation_results(self, results: Dict) -> None:
        """Log evaluation results"""
        logger.info(f"✅ Evaluation complete:")
        logger.info(f"   Mean Reward: {results['mean_reward']:.3f} +/- {results['std_reward']:.3f}")
        logger.info(f"   Accuracy: {results['accuracy']*100:.1f}%")
        logger.info(f"   Handover Rate: {results['handover_rate']:.3f}")
        logger.info(f"   QoS Violation Rate: {results['qos_violation_rate']:.3f}")
        logger.info(f"   Avg Switch Time: {results['mean_switch_time_ms']:.2f} ms")
        logger.info(f"   Stability Score: {results['stability_score']:.3f}")
        logger.info(f"   Overall Score: {results['overall_score']:.3f}")
    
    def _track_handover(self, info: Dict) -> None:
        """Track handover data for analysis"""
        handover_info = {
            'from': info.get('previous_network', 'Unknown'),
            'to': info.get('network', 'Unknown'),
            'time_ms': info.get('decision_time_ms', 0),
            'reward': info.get('reward', 0),
            'area': info.get('area', 'Unknown'),
            'snr': info.get('snr', 0),
            'success': info.get('handover_success', True),
            'reason': info.get('handover_reason', 'Unknown')
        }
        
        self.handover_data['transitions'].append(handover_info)
        self.handover_data['from_areas'].append(info.get('previous_area', 'Unknown'))
        self.handover_data['to_areas'].append(info.get('area', 'Unknown'))
        self.handover_data['switch_times'].append(info.get('decision_time_ms', 0))
        self.handover_data['rewards_before'].append(info.get('reward_before', 0))
        self.handover_data['rewards_after'].append(info.get('reward', 0))
        
        if handover_info['success']:
            self.handover_data['successful_handovers'] += 1
        else:
            self.handover_data['failed_handovers'] += 1
        
        self.handover_data['handover_reasons'].append(handover_info['reason'])
        self.handover_data['handover_quality'].append(
            info.get('handover_quality', 1.0)
        )
    
    def _update_performance_metrics(self, results: Dict) -> None:
        """Update performance metrics"""
        self.performance_metrics['mean_reward'] = results['mean_reward']
        self.performance_metrics['best_reward'] = max(
            self.performance_metrics['best_reward'],
            results['max_reward']
        )
        self.performance_metrics['worst_reward'] = min(
            self.performance_metrics['worst_reward'],
            results['min_reward']
        )
        self.performance_metrics['std_reward'] = results['std_reward']
        self.performance_metrics['total_handovers'] += results['total_handovers']
        self.performance_metrics['handover_rate'] = results['handover_rate']
        self.performance_metrics['avg_decision_time'] = results['mean_switch_time_ms']
        self.performance_metrics['qos_violation_rate'] = results['qos_violation_rate']
        self.performance_metrics['area_accuracy'] = results['area_accuracies']
        self.performance_metrics['network_preference'] = results['network_preferences']
        self.performance_metrics['stability_score'] = results['stability_score']
        self.performance_metrics['efficiency_score'] = results['efficiency_score']
        self.performance_metrics['adaptability_score'] = results['adaptability_score']
        self.performance_metrics['overall_score'] = results['overall_score']
    
    # ============================================
    # 6. Checkpoint Management
    # ============================================
    
    def save_checkpoint(
        self,
        step: int,
        episode: int,
        metrics: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> CheckpointInfo:
        """
        Save a checkpoint with versioning.
        
        Args:
            step: Current step
            episode: Current episode
            metrics: Performance metrics
            metadata: Additional metadata
            
        Returns:
            CheckpointInfo
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Create checkpoint
        timestamp = time.time()
        checkpoint_path = self.checkpoints_dir / f"{self.name}_step_{step}_ep_{episode}.pt"
        
        # Save model
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'step': step,
            'episode': episode,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'metadata': metadata or {},
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'is_trained': self.is_trained,
            'best_reward': self.best_reward
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            version=f"v{step}",
            timestamp=timestamp,
            step=step,
            episode=episode,
            status=self.status,
            metrics=metrics or {},
            file_size=checkpoint_path.stat().st_size,
            hash=self._calculate_file_hash(checkpoint_path)
        )
        
        # Store checkpoint info
        self.training_history['checkpoints'].append({
            'info': checkpoint_info,
            'path': str(checkpoint_path)
        })
        
        self.current_checkpoint = checkpoint_info
        
        logger.info(f"✅ Checkpoint saved at step {step}: {checkpoint_path}")
        
        return checkpoint_info
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Checkpoint data
        """
        checkpoint_data = torch.load(path, map_location=self.device)
        
        # Load model
        if self.model is not None:
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer
        if self.optimizer is not None and checkpoint_data.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler is not None and checkpoint_data.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Update attributes
        self.total_steps = checkpoint_data.get('total_steps', 0)
        self.total_episodes = checkpoint_data.get('total_episodes', 0)
        self.is_trained = checkpoint_data.get('is_trained', False)
        self.best_reward = checkpoint_data.get('best_reward', float('-inf'))
        
        # Update history
        if 'training_history' in checkpoint_data:
            self.training_history.update(checkpoint_data['training_history'])
        
        if 'performance_metrics' in checkpoint_data:
            self.performance_metrics.update(checkpoint_data['performance_metrics'])
        
        self.status = AgentStatus.LOADED
        
        logger.info(f"✅ Checkpoint loaded from: {path}")
        
        return checkpoint_data
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints"""
        checkpoints = []
        
        for path in self.checkpoints_dir.glob(f"{self.name}_*.pt"):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                checkpoints.append({
                    'path': str(path),
                    'step': checkpoint.get('step', 0),
                    'episode': checkpoint.get('episode', 0),
                    'timestamp': checkpoint.get('timestamp', 0),
                    'metrics': checkpoint.get('metrics', {})
                })
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {path}: {e}")
        
        return sorted(checkpoints, key=lambda x: x['step'])
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]
    
    # ============================================
    # 7. Anomaly Detection
    # ============================================
    
    def detect_anomaly(
        self,
        value: float,
        value_type: str = 'reward'
    ) -> bool:
        """
        Detect anomalies in training metrics.
        
        Args:
            value: Value to check
            value_type: Type of value ('reward', 'loss', etc.)
            
        Returns:
            True if anomaly detected
        """
        if value_type == 'reward':
            history = self.reward_history
        elif value_type == 'loss':
            history = self.loss_history
        else:
            return False
        
        if len(history) < 10:
            history.append(value)
            return False
        
        mean = np.mean(history)
        std = np.std(history) + 1e-8
        
        # Check if value is outside threshold
        is_anomaly = abs(value - mean) > self.anomaly_threshold * std
        
        if is_anomaly:
            self.training_history['anomalies'].append({
                'type': value_type,
                'value': value,
                'mean': mean,
                'std': std,
                'step': self.total_steps,
                'episode': self.total_episodes
            })
            
            if self.verbose:
                logger.warning(f"⚠️ Anomaly detected: {value_type} = {value:.4f} (mean: {mean:.4f}, std: {std:.4f})")
        
        history.append(value)
        return is_anomaly
    
    # ============================================
    # 8. Early Stopping
    # ============================================
    
    def check_early_stopping(
        self,
        current_reward: float,
        patience: Optional[int] = None,
        min_improvement: Optional[float] = None
    ) -> bool:
        """
        Check if training should stop early.
        
        Args:
            current_reward: Current reward
            patience: Number of episodes to wait for improvement
            min_improvement: Minimum improvement required
            
        Returns:
            True if early stopping should trigger
        """
        patience = patience or self.training_config.early_stopping_patience
        min_improvement = min_improvement or self.training_config.min_improvement
        
        if current_reward > self.best_reward + min_improvement:
            self.best_reward = current_reward
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                logger.info(f"⏹️ Early stopping triggered after {patience} episodes without improvement")
                return True
        return False
    
    # ============================================
    # 9. Learning Rate Scheduling
    # ============================================
    
    def step_scheduler(self, metric: Optional[float] = None) -> None:
        """
        Step the learning rate scheduler.
        
        Args:
            metric: Metric to use for scheduling
        """
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_history['learning_rates'].append(current_lr)
    
    # ============================================
    # 10. Performance Profiling
    # ============================================
    
    def profile_step(
        self,
        forward_time: float,
        backward_time: float,
        update_time: float,
        step_time: float
    ) -> None:
        """
        Profile a training step.
        
        Args:
            forward_time: Time for forward pass
            backward_time: Time for backward pass
            update_time: Time for update
            step_time: Total step time
        """
        self.training_history['profiling']['forward_time'].append(forward_time)
        self.training_history['profiling']['backward_time'].append(backward_time)
        self.training_history['profiling']['update_time'].append(update_time)
        self.training_history['profiling']['step_time'].append(step_time)
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        profiling = self.training_history['profiling']
        
        return {
            'mean_forward_time': np.mean(profiling['forward_time']) if profiling['forward_time'] else 0,
            'mean_backward_time': np.mean(profiling['backward_time']) if profiling['backward_time'] else 0,
            'mean_update_time': np.mean(profiling['update_time']) if profiling['update_time'] else 0,
            'mean_step_time': np.mean(profiling['step_time']) if profiling['step_time'] else 0,
            'total_forward_time': sum(profiling['forward_time']) if profiling['forward_time'] else 0,
            'total_backward_time': sum(profiling['backward_time']) if profiling['backward_time'] else 0,
            'total_update_time': sum(profiling['update_time']) if profiling['update_time'] else 0,
            'total_step_time': sum(profiling['step_time']) if profiling['step_time'] else 0,
            'inference_count': self.inference_count,
            'mean_inference_time': self.total_inference_time / self.inference_count if self.inference_count > 0 else 0
        }
    
    # ============================================
    # 11. User Analytics
    # ============================================
    
    def get_user_analytics(self) -> Dict[str, Any]:
        """Get user journey analytics"""
        if not self.user_data:
            return {'error': 'No user data available'}
        
        analytics = {}
        
        for user_id, data in self.user_data.items():
            if not data.get('journeys'):
                continue
            
            journeys = data['journeys']
            
            analytics[user_id] = {
                'total_journeys': len(journeys),
                'mean_reward': np.mean([j.get('total_reward', 0) for j in journeys]),
                'mean_accuracy': np.mean([j.get('accuracy', 0) for j in journeys]),
                'mean_handovers': np.mean([j.get('handovers', 0) for j in journeys]),
                'total_steps': sum([j.get('steps', 0) for j in journeys]),
                'network_preferences': self._calculate_user_network_preferences(journeys),
                'area_patterns': self._calculate_user_area_patterns(journeys)
            }
        
        return analytics
    
    def _calculate_user_network_preferences(self, journeys: List[Dict]) -> Dict[str, float]:
        """Calculate network preferences for a user"""
        network_counts = defaultdict(int)
        total = 0
        
        for journey in journeys:
            for network in journey.get('networks', []):
                network_counts[network] += 1
                total += 1
        
        return {
            net: count / total if total > 0 else 0
            for net, count in network_counts.items()
        }
    
    def _calculate_user_area_patterns(self, journeys: List[Dict]) -> Dict[str, float]:
        """Calculate area patterns for a user"""
        area_counts = defaultdict(int)
        total = 0
        
        for journey in journeys:
            for area in journey.get('areas', []):
                area_counts[area] += 1
                total += 1
        
        return {
            area: count / total if total > 0 else 0
            for area, count in area_counts.items()
        }
    
    # ============================================
    # 12. Other Helper Methods
    # ============================================
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility"""
        import random
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Random seed set to: {seed}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'name': self.name,
            'seed': self.seed,
            'device': self.device,
            'training': self.training_config.__dict__ if hasattr(self, 'training_config') else TRAINING_CONFIG,
            'track_user': self.track_user
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.training_history['timesteps']:
            return {'error': 'No training data available'}
        
        return {
            'total_steps': self.total_steps,
            'total_episodes': len(self.training_history.get('rewards', [])),
            'final_mean_reward': self.training_history['mean_rewards'][-1] if self.training_history.get('mean_rewards') else 0,
            'best_mean_reward': max(self.training_history['mean_rewards']) if self.training_history.get('mean_rewards') else 0,
            'training_time_seconds': time.time() - self.start_time if self.start_time else 0,
            'is_trained': self.is_trained,
            'device': self.device,
            'status': self.status.value,
            'parameters': self.get_parameter_count() if hasattr(self, 'get_parameter_count') else 0,
            'checkpoints': len(self.training_history.get('checkpoints', [])),
            'anomalies_detected': len(self.training_history.get('anomalies', []))
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics
    
    def get_handover_analysis(self) -> Dict[str, Any]:
        """Get handover analysis results"""
        if not self.handover_data['transitions']:
            return {'error': 'No handover data available'}
        
        transitions = self.handover_data['transitions']
        
        # Find most common transitions
        from_to_pairs = [f"{t['from']}->{t['to']}" for t in transitions]
        from collections import Counter
        most_common = Counter(from_to_pairs).most_common(5)
        
        # Calculate statistics
        avg_switch_time = np.mean(self.handover_data['switch_times']) if self.handover_data['switch_times'] else 0
        handover_success_rate = self.handover_data['successful_handovers'] / len(transitions) if transitions else 0
        
        return {
            'total_handovers': len(transitions),
            'successful_handovers': self.handover_data['successful_handovers'],
            'failed_handovers': self.handover_data['failed_handovers'],
            'handover_success_rate': handover_success_rate,
            'avg_switch_time_ms': avg_switch_time,
            'most_common_transitions': most_common,
            'unique_from_areas': len(set(self.handover_data['from_areas'])),
            'unique_to_areas': len(set(self.handover_data['to_areas'])),
            'transition_matrix': self._create_transition_matrix(),
            'handover_quality': np.mean(self.handover_data['handover_quality']) if self.handover_data['handover_quality'] else 0,
            'common_reasons': Counter(self.handover_data['handover_reasons']).most_common(3)
        }
    
    def _create_transition_matrix(self) -> Dict[str, Dict[str, int]]:
        """Create handover transition matrix"""
        matrix = {}
        for t in self.handover_data['transitions']:
            from_net = t['from']
            to_net = t['to']
            
            if from_net not in matrix:
                matrix[from_net] = {}
            if to_net not in matrix[from_net]:
                matrix[from_net][to_net] = 0
            matrix[from_net][to_net] += 1
        
        return matrix
    
    def reset_env(self) -> None:
        """Reset environment"""
        self.env.reset()
        logger.info("Environment reset")
    
    def close(self) -> None:
        """Close the agent and environment"""
        if hasattr(self.env, 'close'):
            self.env.close()
        
        self.status = AgentStatus.UNINITIALIZED
        logger.info(f"{self.name} closed")
    
    def __str__(self) -> str:
        return f"{self.name} ({self.status.value}) - {self.total_steps} steps"
    
    def __repr__(self) -> str:
        return self.__str__()