# =============================================================================
# FILE: Scr/training/trainer.py (VERSION 4.6 - FULLY FIXED)
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('numpy').setLevel(logging.ERROR)
logging.getLogger('pandas').setLevel(logging.ERROR)

import torch
torch._dynamo.config.suppress_errors = True

import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Import agents
from agents.base_agent import BaseAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.lstm_agent import LSTMAgent
from agents.gru_agent import GRUAgent

# Import environment
from environment.hybrid_network_env import HybridNetworkEnv
from data_preprocessing.data_loader import DataLoader
from data_preprocessing.data_splitter import DataSplitter
from data_preprocessing.feature_engineering import FeatureEngineer

# Import constants
from utils.constants import (
    TRAINING_CONFIG, EVALUATION_CONFIG, NETWORK_TYPES,
    USER_TRACKING_CONFIG, RECURRENT_CONFIG, 
    PPO_CONFIG, DQN_CONFIG,
    AREA_BEST_AGENT, AREA_BEST_NETWORK, AREA_EXPECTED_ACCURACY,
    AREA_TYPES
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def to_dict(obj):
    """Convert object to dict if needed"""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return {'accuracy': 0, 'mean_reward': 0, 'handover_rate': 0, 'mean_switch_time_ms': 0}


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    """
    Professional Trainer with progress display
    """
    
    def __init__(
        self, 
        output_dir: str = '../test_results/', 
        seed: int = 42, 
        verbose: bool = True, 
        use_gpu: bool = False,
        use_state_builder: bool = False,
        use_reward_calculator: bool = False
    ):
        self.output_dir = output_dir
        self.seed = seed
        self.verbose = verbose
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_state_builder = use_state_builder
        self.use_reward_calculator = use_reward_calculator
        
        self.results = {}
        self.agents = {}
        self.envs = {}
        
        self._create_directories()
        self._setup_logging()
        
        print(f"✅ Trainer initialized")
        print(f"   Output dir: {output_dir}")
        print(f"   GPU: {self.use_gpu}")
        print(f"   State Builder: {use_state_builder}")
        print(f"   Reward Calculator: {use_reward_calculator}")
    
    def _create_directories(self) -> None:
        dirs = [
            self.output_dir,
            f"{self.output_dir}/models/",
            f"{self.output_dir}/reports/",
            f"{self.output_dir}/figures/",
            f"{self.output_dir}/logs/",
            f"{self.output_dir}/comparisons/",
            f"{self.output_dir}/user_tracking/",
            f"{self.output_dir}/handover_analysis/"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _setup_logging(self) -> None:
        log_file = f"{self.output_dir}/logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        logger.addHandler(file_handler)
    
    def load_and_prepare_data(
        self, 
        data_path: str, 
        split_ratio: float = 0.8, 
        balance_data: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("\n📂 Loading data...")
        
        loader = DataLoader(data_path, balance_data=balance_data, random_state=self.seed)
        data = loader.load()
        
        splitter = DataSplitter(data, split_ratio=split_ratio)
        train_data, test_data = splitter.split_by_user(random_state=self.seed)
        
        engineer = FeatureEngineer(use_scaling=True, scaler_type='robust')
        train_data = engineer.fit_transform(train_data)
        test_data = engineer.transform(test_data)
        
        print(f"   Train: {len(train_data):,}, Test: {len(test_data):,}")
        return train_data, test_data
    
    def create_environments(
        self, 
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame, 
        state_type: str = 'classical', 
        sequence_length: int = 15
    ) -> Tuple[HybridNetworkEnv, HybridNetworkEnv]:
        train_env = HybridNetworkEnv(
            data=train_data,
            state_type=state_type,
            sequence_length=sequence_length,
            seed=self.seed,
            track_user=True,
            use_state_builder=self.use_state_builder,
            use_reward_calculator=self.use_reward_calculator
        )
        
        test_env = HybridNetworkEnv(
            data=test_data,
            state_type=state_type,
            sequence_length=sequence_length,
            seed=self.seed,
            track_user=True,
            use_state_builder=self.use_state_builder,
            use_reward_calculator=self.use_reward_calculator
        )
        
        return train_env, test_env
    
    def train_agent(
        self, 
        agent_type: str, 
        train_env: HybridNetworkEnv, 
        test_env: HybridNetworkEnv, 
        total_timesteps: int, 
        agent_config: Optional[Dict] = None
    ) -> Tuple[BaseAgent, Dict]:
        print(f"\n🏋️ Training {agent_type.upper()}...")
        print(f"   Total Steps: {total_timesteps:,}")
        print("-"*60)
        
        agent_classes = {
            'dqn': DQNAgent,
            'ppo': PPOAgent,
            'lstm': LSTMAgent,
            'gru': GRUAgent
        }
        
        agent_class = agent_classes.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        device = 'cuda' if self.use_gpu else 'cpu'
        
        # Create agent
        agent = agent_class(
            env=train_env,
            config=agent_config or {},
            name=f"{agent_type.upper()}_Agent",
            seed=self.seed,
            device=device
        )
        
        # ===== FIXED: PROGRESS CALLBACK =====
        def progress_callback(*args, **kwargs):
            """Update progress display - supports both DQN and PPO callback signatures"""
            try:
                step = None
                episode_reward = 0
                
                # Handle different callback signatures
                if len(args) >= 3:
                    # DQN: (step, episode_reward, episode_length)
                    step = args[0]
                    episode_reward = args[1]
                elif len(args) == 1:
                    # PPO: (agent_object)
                    agent_obj = args[0]
                    if hasattr(agent_obj, 'total_steps'):
                        step = agent_obj.total_steps
                        # Try to get last reward
                        if hasattr(agent_obj, 'training_history') and 'rewards' in agent_obj.training_history:
                            rewards = agent_obj.training_history['rewards']
                            if rewards:
                                episode_reward = rewards[-1]
                else:
                    return
                
                if step is not None and hasattr(agent, 'total_timesteps'):
                    progress = (step / agent.total_timesteps) * 100
                    if step % 100 == 0:
                        print(f"\r   Progress: {progress:5.1f}% | Step: {step:,}/{agent.total_timesteps:,} | Reward: {episode_reward:7.1f}", end='')
            except Exception:
                pass
        
        # Store total timesteps in agent for callback
        agent.total_timesteps = total_timesteps
        
        # Train the agent
        start_time = time.time()
        results = agent.train(
            total_timesteps=total_timesteps,
            callback=progress_callback
        )
        
        print()  # New line after progress bar
        
        training_time = time.time() - start_time
        
        results['training_time_seconds'] = training_time
        
        # Count parameters
        if hasattr(agent, 'get_parameter_count'):
            results['parameters'] = agent.get_parameter_count()
        elif hasattr(agent, 'policy_net'):
            results['parameters'] = sum(p.numel() for p in agent.policy_net.parameters())
        elif hasattr(agent, 'q_network'):
            results['parameters'] = sum(p.numel() for p in agent.q_network.parameters())
        else:
            results['parameters'] = 0
        
        self.agents[agent_type] = agent
        self.results[agent_type] = results
        
        # Evaluate the agent
        print(f"\n🔍 Evaluating {agent_type.upper()}...")
        eval_results = agent.evaluate(
            n_episodes=EVALUATION_CONFIG.get('n_episodes', 10),
            max_steps_per_episode=EVALUATION_CONFIG.get('max_steps_per_episode', 500),
            track_handovers=True,
            track_users=True
        )
        
        eval_dict = to_dict(eval_results)
        self.results[f"{agent_type}_eval"] = eval_dict
        
        accuracy = eval_dict.get('accuracy', 0)
        
        # Save model
        model_path = f"{self.output_dir}/models/{agent_type}_model.pt"
        agent.save(model_path)
        
        # Save training history
        history_path = f"{self.output_dir}/logs/{agent_type}_history.json"
        try:
            agent.save_training_history(history_path)
        except AttributeError:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(agent.training_history, f, indent=2, default=str)
        
        print(f"\n   ✅ {agent_type.upper()} complete in {training_time:.2f}s")
        print(f"   📊 Accuracy: {accuracy*100:.1f}%")
        print(f"   📦 Parameters: {results.get('parameters', 0):,}")
        
        return agent, results
    
    def train_all_agents(
        self, 
        data_path: str, 
        total_timesteps: int = 150000, 
        split_ratio: float = 0.8,
        sequence_length: int = 15, 
        use_kl_penalty: bool = True, 
        adaptive_lr: bool = True,
        balance_data: bool = True
    ) -> Dict[str, Dict]:
        print("\n" + "="*80)
        print("🚀 STARTING TRAINING PIPELINE")
        print("="*80)
        print(f"   Timesteps per Agent: {total_timesteps:,}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   State Builder: {self.use_state_builder}")
        print(f"   Reward Calculator: {self.use_reward_calculator}")
        print("="*80)
        
        # Load data
        train_data, test_data = self.load_and_prepare_data(
            data_path=data_path,
            split_ratio=split_ratio,
            balance_data=balance_data
        )
        
        # Agent configurations
        agent_configs = {
            'dqn': {'dqn': DQN_CONFIG.copy()},
            'ppo': {'ppo': PPO_CONFIG.copy()},
            'lstm': {'ppo': PPO_CONFIG.copy(), 'recurrent': RECURRENT_CONFIG.copy()},
            'gru': {'ppo': PPO_CONFIG.copy(), 'recurrent': RECURRENT_CONFIG.copy()}
        }
        
        for agent_type in ['lstm', 'gru']:
            agent_configs[agent_type]['recurrent']['sequence_length'] = sequence_length
        
        all_results = {}
        agent_types = ['dqn', 'ppo', 'lstm', 'gru']
        
        # Timing tracking
        total_start = time.time()
        
        # Train each agent
        for i, agent_type in enumerate(agent_types):
            print("\n" + "="*80)
            print(f"📈 AGENT {i+1}/{len(agent_types)}: {agent_type.upper()}")
            print("="*80)
            
            # Determine state type based on agent
            state_type = agent_type if agent_type in ['lstm', 'gru'] else 'classical'
            
            # Create environments
            train_env, test_env = self.create_environments(
                train_data=train_data,
                test_data=test_data,
                state_type=state_type,
                sequence_length=sequence_length
            )
            
            # Train agent
            agent, results = self.train_agent(
                agent_type=agent_type,
                train_env=train_env,
                test_env=test_env,
                total_timesteps=total_timesteps,
                agent_config=agent_configs[agent_type]
            )
            
            all_results[agent_type] = {
                'training': results,
                'evaluation': self.results.get(f"{agent_type}_eval", {})
            }
            
            # Close environments
            train_env.close()
            test_env.close()
            
            # Show progress
            elapsed = time.time() - total_start
            remaining_agents = len(agent_types) - (i + 1)
            avg_time_per_agent = elapsed / (i + 1)
            est_remaining = avg_time_per_agent * remaining_agents
            
            print(f"\n⏱️  Elapsed: {elapsed/60:.1f} min | Est. Remaining: {est_remaining/60:.1f} min")
        
        # Generate comprehensive comparison
        self._generate_comparison(all_results)
        
        # Generate recommendations report
        self._generate_recommendations(all_results)
        
        total_time = time.time() - total_start
        print("\n" + "="*80)
        print(f"🎉 ALL AGENTS TRAINED SUCCESSFULLY! (Total time: {total_time/60:.1f} min)")
        print("="*80)
        
        return all_results
    
    def _generate_comparison(self, all_results: Dict) -> None:
        print("\n📊 COMPARISON TABLE")
        print("-"*80)
        
        data = []
        for agent_type, results in all_results.items():
            eval_res = results.get('evaluation', {})
            train_res = results.get('training', {})
            
            data.append({
                'Agent': agent_type.upper(),
                'Accuracy (%)': eval_res.get('accuracy', 0) * 100,
                'Mean Reward': eval_res.get('mean_reward', 0),
                'Handover Rate': eval_res.get('handover_rate', 0),
                'Decision Time (ms)': eval_res.get('mean_switch_time_ms', 0),
                'Training Time (s)': train_res.get('training_time_seconds', 0),
                'Parameters': train_res.get('parameters', 0)
            })
        
        df = pd.DataFrame(data)
        
        print("\n┌─────────────┬──────────────┬──────────────┬──────────────┬──────────────────┬─────────────────┬──────────────┐")
        print("│ Agent       │ Accuracy %   │ Mean Reward  │ Handover     │ Decision Time    │ Training Time   │ Parameters   │")
        print("├─────────────┼──────────────┼──────────────┼──────────────┼──────────────────┼─────────────────┼──────────────┤")
        
        for _, row in df.iterrows():
            print(f"│ {row['Agent']:<11} │ {row['Accuracy (%)']:>12.1f} │ {row['Mean Reward']:>12.2f} │ {row['Handover Rate']:>12.3f} │ {row['Decision Time (ms)']:>16.2f} │ {row['Training Time (s)']:>15.2f} │ {row['Parameters']:>12,} │")
        
        print("└─────────────┴──────────────┴──────────────┴──────────────┴──────────────────┴─────────────────┴──────────────┘")
        
        # Best per metric
        print("\n🏆 BEST PER METRIC:")
        
        best_acc = df.loc[df['Accuracy (%)'].idxmax()]
        print(f"   🎯 Best Accuracy: {best_acc['Agent']} ({best_acc['Accuracy (%)']:.1f}%)")
        
        best_reward = df.loc[df['Mean Reward'].idxmax()]
        print(f"   💰 Best Reward: {best_reward['Agent']} ({best_reward['Mean Reward']:.2f})")
        
        best_handover = df.loc[df['Handover Rate'].idxmin()]
        print(f"   🔄 Best Handover: {best_handover['Agent']} ({best_handover['Handover Rate']:.3f})")
        
        best_time = df.loc[df['Training Time (s)'].idxmin()]
        print(f"   ⚡ Fastest Training: {best_time['Agent']} ({best_time['Training Time (s)']:.2f}s)")
        
        # Overall winner
        print("\n" + "="*80)
        scores = {}
        for _, row in df.iterrows():
            acc_score = row['Accuracy (%)'] / (df['Accuracy (%)'].max() + 0.001)
            reward_score = row['Mean Reward'] / (df['Mean Reward'].max() + 0.001)
            handover_score = 1 - (row['Handover Rate'] / (df['Handover Rate'].max() + 0.001))
            speed_score = 1 - (row['Training Time (s)'] / (df['Training Time (s)'].max() + 0.001))
            overall = (acc_score * 0.35 + reward_score * 0.25 + handover_score * 0.25 + speed_score * 0.15) * 100
            scores[row['Agent']] = overall
        
        winner = max(scores.items(), key=lambda x: x[1])
        print(f"🏆 OVERALL WINNER: {winner[0]}")
        print(f"   Overall Score: {winner[1]:.1f}%")
        print("="*80)
        
        # Save report
        report_path = f"{self.output_dir}/reports/comparison_report.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("FINAL COMPARISON REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n🏆 BEST PER METRIC\n")
            f.write("-"*80 + "\n")
            f.write(f"   Best Accuracy: {best_acc['Agent']} ({best_acc['Accuracy (%)']:.1f}%)\n")
            f.write(f"   Best Reward: {best_reward['Agent']} ({best_reward['Mean Reward']:.2f})\n")
            f.write(f"   Best Handover: {best_handover['Agent']} ({best_handover['Handover Rate']:.3f})\n")
            f.write(f"   Fastest Training: {best_time['Agent']} ({best_time['Training Time (s)']:.2f}s)\n")
            f.write("\n" + "="*80 + "\n")
            f.write(f"🏆 OVERALL WINNER: {winner[0]}\n")
            f.write(f"   Overall Score: {winner[1]:.1f}%\n")
            f.write("="*80 + "\n")
        
        print(f"\n✅ Report saved: {report_path}")
        
        csv_path = f"{self.output_dir}/reports/comparison_table.csv"
        df.to_csv(csv_path, index=False)
    
    def _generate_recommendations(self, all_results: Dict) -> None:
        print("\n" + "="*80)
        print("🎯 RECOMMENDATIONS BY AREA")
        print("="*80)
        
        recommendations = []
        
        for area in AREA_TYPES:
            agent = AREA_BEST_AGENT.get(area, 'N/A')
            network = AREA_BEST_NETWORK.get(area, 'N/A')
            accuracy = AREA_EXPECTED_ACCURACY.get(area, 0)
            
            emoji = {
                'Urban': '🏙️', 'Indoor': '🏠', 'Rural': '🌾',
                'Highway': '🛣️', 'Maritime': '🌊', 'Desert': '🏜️'
            }.get(area, '📍')
            
            recommendations.append({
                'area': f"{emoji} {area}",
                'agent': agent,
                'network': network,
                'accuracy': accuracy
            })
        
        print("\n┌─────────────┬─────────────────┬─────────────────┬─────────────────┐")
        print("│ Area        │ Best Agent      │ Best Network    │ Accuracy        │")
        print("├─────────────┼─────────────────┼─────────────────┼─────────────────┤")
        
        for rec in recommendations:
            print(f"│ {rec['area']:<11} │ {rec['agent']:<15} │ {rec['network']:<15} │ {rec['accuracy']:>5.1f}%     │")
        
        print("└─────────────┴─────────────────┴─────────────────┴─────────────────┘")
        
        rec_path = f"{self.output_dir}/reports/recommendations.txt"
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RECOMMENDATIONS BY AREA\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("┌─────────────┬─────────────────┬─────────────────┬─────────────────┐\n")
            f.write("│ Area        │ Best Agent      │ Best Network    │ Accuracy        │\n")
            f.write("├─────────────┼─────────────────┼─────────────────┼─────────────────┤\n")
            for rec in recommendations:
                f.write(f"│ {rec['area']:<11} │ {rec['agent']:<15} │ {rec['network']:<15} │ {rec['accuracy']:>5.1f}%     │\n")
            f.write("└─────────────┴─────────────────┴─────────────────┴─────────────────┘\n")
        
        print(f"\n✅ Recommendations saved: {rec_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    results_dir = os.getenv('RESULTS_DIR', 'test_results/')
    data_path = os.getenv('DATA_PATH', 'data_raw/Hybrid_Network_TN_NTN_Final.csv')
    total_timesteps = int(os.getenv('TOTAL_TIMESTEPS', '150000'))
    sequence_length = int(os.getenv('SEQUENCE_LENGTH', '15'))
    balance_data = os.getenv('BALANCE_DATA', 'True').strip().lower() in {'1', 'true', 'yes'}
    use_state_builder = os.getenv('USE_STATE_BUILDER', 'False').strip().lower() in {'1', 'true', 'yes'}
    use_reward_calculator = os.getenv('USE_REWARD_CALCULATOR', 'False').strip().lower() in {'1', 'true', 'yes'}

    trainer = Trainer(
        output_dir=results_dir, 
        verbose=True,
        use_gpu=os.getenv('USE_GPU', 'False').strip().lower() in {'1', 'true', 'yes'},
        use_state_builder=use_state_builder,
        use_reward_calculator=use_reward_calculator
    )
    
    trainer.train_all_agents(
        data_path=data_path,
        total_timesteps=total_timesteps,
        sequence_length=sequence_length,
        balance_data=balance_data,
    )


if __name__ == '__main__':
    main()