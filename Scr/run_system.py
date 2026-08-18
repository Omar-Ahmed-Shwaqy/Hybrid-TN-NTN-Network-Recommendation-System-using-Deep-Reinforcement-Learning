# =============================================================================
# FILE: run_system.py
# PURPOSE: تشغيل جميع Agents (DQN, PPO, LSTM, GRU) مع تحسينات Accuracy
#          وعرض التقدم في Terminal مع Figures تلقائية
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import gc
import torch
import pandas as pd
import numpy as np
from datetime import datetime

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🚀 HYBRID TN-NTN NETWORK RECOMMENDATION SYSTEM                           ║
║                                                                              ║
║   📡 Terrestrial (TN) & Non-Terrestrial (NTN) Networks                    ║
║   🤖 Deep Reinforcement Learning Agents (DQN, PPO, LSTM, GRU)             ║
║   📊 Smart Network Selection for 6G & Beyond                              ║
║                                                                              ║
║   Networks: 5G NR | WiFi | LEO Satellite | HAPS | UAV Relay                ║
║   Areas: Urban | Indoor | Rural | Highway | Maritime | Desert             ║
║                                                                              ║
║   🎯 Optimized for Accuracy > 25%                                         ║
║   ⚡ Training Steps: 150,000 per agent (Fair comparison)                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

from training.trainer import Trainer
from utils.constants import PPO_CONFIG, DQN_CONFIG, RECURRENT_CONFIG, AREA_BEST_AGENT, AREA_BEST_NETWORK, AREA_EXPECTED_ACCURACY, AREA_TYPES


# =============================================================================
# OPTIMIZED CONFIGURATIONS FOR BETTER ACCURACY
# =============================================================================

# DQN Optimized Config
DQN_OPTIMIZED = {
    'learning_rate': 0.0005,
    'buffer_size': 50000,
    'batch_size': 64,
    'gamma': 0.99,
    'tau': 0.005,
    'target_update_interval': 1000,
    'exploration_fraction': 0.15,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.01,
    'train_freq': 4,
    'gradient_steps': 1,
    'learning_starts': 1000,
    'hidden_dims': [256, 256]
}

# PPO Optimized Config
PPO_OPTIMIZED = {
    'learning_rate': 0.0005,
    'n_steps': 4096,
    'batch_size': 128,
    'n_epochs': 15,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'target_kl': 0.01,
    'hidden_dims': [256, 256]
}

# LSTM/GRU Optimized Config
RECURRENT_OPTIMIZED = {
    'sequence_length': 15,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.15,
    'bidirectional': True,
    'batch_first': True,
    'device': 'cpu'
}


# =============================================================================
# PROGRESS DISPLAY
# =============================================================================

def print_progress(agent_name, episode, reward, avg_reward, steps, total_steps, elapsed):
    """Print progress with colors and formatting"""
    progress = (steps / total_steps) * 100
    bar_length = 30
    filled = int(bar_length * progress / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\r{agent_name:6} | {bar} | {progress:5.1f}% | Ep:{episode:4} | R:{reward:7.1f} | Avg:{avg_reward:7.1f} | {elapsed:>8}", end='')


# =============================================================================
# MAIN SYSTEM
# =============================================================================

class HybridNetworkSystem:
    """Complete Hybrid Network Recommendation System"""
    
    def __init__(self, total_timesteps=150000, output_dir='../test_results/'):
        self.total_timesteps = total_timesteps
        self.output_dir = output_dir
        self.start_time = None
        self.all_results = {}
        
        # Create trainer
        self.trainer = Trainer(
            output_dir=output_dir,
            seed=42,
            verbose=False,
            use_gpu=False,
            use_state_builder=False,
            use_reward_calculator=False
        )
        
        print(f"\n✅ System initialized")
        print(f"   Timesteps per agent: {total_timesteps:,}")
        print(f"   Output directory: {output_dir}")
    
    def run(self):
        """Run complete training pipeline"""
        print("\n" + "="*80)
        print("🚀 STARTING COMPLETE TRAINING PIPELINE")
        print("="*80)
        print(f"   Agents: DQN, PPO, LSTM, GRU")
        print(f"   Timesteps per agent: {self.total_timesteps:,}")
        print("="*80)
        
        self.start_time = time.time()
        
        # Load data
        train_data, test_data = self.trainer.load_and_prepare_data(
            data_path='data_raw/Hybrid_Network_TN_NTN_Final.csv',
            split_ratio=0.8,
            balance_data=True
        )
        
        # Agent configurations
        agent_configs = {
            'dqn': {'dqn': DQN_OPTIMIZED.copy()},
            'ppo': {'ppo': PPO_OPTIMIZED.copy()},
            'lstm': {'ppo': PPO_OPTIMIZED.copy(), 'recurrent': RECURRENT_OPTIMIZED.copy()},
            'gru': {'ppo': PPO_OPTIMIZED.copy(), 'recurrent': RECURRENT_OPTIMIZED.copy()}
        }
        
        # Train each agent
        agent_types = ['dqn', 'ppo', 'lstm', 'gru']
        agent_names = ['DQN', 'PPO', 'LSTM', 'GRU']
        agent_colors = ['🔵', '🟣', '🟠', '🟢']
        
        for i, (agent_type, agent_name, color) in enumerate(zip(agent_types, agent_names, agent_colors), 1):
            print("\n" + "="*80)
            print(f"{color} AGENT {i}/4: {agent_name}")
            print("="*80)
            
            state_type = agent_type if agent_type in ['lstm', 'gru'] else 'classical'
            
            train_env, test_env = self.trainer.create_environments(
                train_data=train_data,
                test_data=test_data,
                state_type=state_type,
                sequence_length=15
            )
            
            # Progress tracking
            agent_start = time.time()
            episode_rewards = []
            
            def progress_callback(step, episode_reward, episode_length):
                episode_rewards.append(episode_reward)
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else episode_reward
                elapsed = self._format_time(time.time() - agent_start)
                print_progress(agent_name, len(episode_rewards), episode_reward, avg_reward, step, self.total_timesteps, elapsed)
            
            # Train agent with optimized config
            agent, results = self.trainer.train_agent(
                agent_type=agent_type,
                train_env=train_env,
                test_env=test_env,
                total_timesteps=self.total_timesteps,
                agent_config=agent_configs[agent_type]
            )
            
            print()  # New line
            
            self.all_results[agent_type] = {
                'training': results,
                'evaluation': self.trainer.results.get(f"{agent_type}_eval", {}),
                'time': time.time() - agent_start
            }
            
            train_env.close()
            test_env.close()
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            elapsed = self._format_time(time.time() - self.start_time)
            print(f"\n   ✅ {agent_name} complete in {self._format_time(time.time() - agent_start)}")
            print(f"   📊 Accuracy: {self.all_results[agent_type]['evaluation'].get('accuracy', 0)*100:.1f}%")
            print(f"   ⏱️  Total elapsed: {elapsed}")
        
        # Generate final comparison
        self._generate_comparison()
        self._generate_recommendations()
        self._generate_figures()
        
        # Final summary
        total_time = self._format_time(time.time() - self.start_time)
        print("\n" + "="*80)
        print(f"🎉 ALL AGENTS TRAINED SUCCESSFULLY! (Total time: {total_time})")
        print("="*80)
        
        return self.all_results
    
    def _format_time(self, seconds):
        """Format time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _generate_comparison(self):
        """Generate comparison table"""
        print("\n" + "="*80)
        print("📊 FINAL COMPARISON TABLE")
        print("="*80)
        
        data = []
        for agent_type, results in self.all_results.items():
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
        scores = {}
        for _, row in df.iterrows():
            acc_score = row['Accuracy (%)'] / (df['Accuracy (%)'].max() + 0.001)
            reward_score = row['Mean Reward'] / (df['Mean Reward'].max() + 0.001)
            handover_score = 1 - (row['Handover Rate'] / (df['Handover Rate'].max() + 0.001))
            speed_score = 1 - (row['Training Time (s)'] / (df['Training Time (s)'].max() + 0.001))
            overall = (acc_score * 0.40 + reward_score * 0.25 + handover_score * 0.20 + speed_score * 0.15) * 100
            scores[row['Agent']] = overall
        
        winner = max(scores.items(), key=lambda x: x[1])
        print(f"\n🏆 OVERALL WINNER: {winner[0]} (Score: {winner[1]:.1f}%)")
        
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
        
        # Save CSV
        csv_path = f"{self.output_dir}/reports/comparison_table.csv"
        df.to_csv(csv_path, index=False)
    
    def _generate_recommendations(self):
        """Generate area-based recommendations"""
        print("\n" + "="*80)
        print("🎯 AREA RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        emojis = {'Urban': '🏙️', 'Indoor': '🏠', 'Rural': '🌾', 'Highway': '🛣️', 'Maritime': '🌊', 'Desert': '🏜️'}
        
        for area in AREA_TYPES:
            agent = AREA_BEST_AGENT.get(area, 'N/A')
            network = AREA_BEST_NETWORK.get(area, 'N/A')
            accuracy = AREA_EXPECTED_ACCURACY.get(area, 0)
            
            recommendations.append({
                'area': f"{emojis.get(area, '📍')} {area}",
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
        
        # Save
        rec_path = f"{self.output_dir}/reports/recommendations.txt"
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AREA RECOMMENDATIONS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("┌─────────────┬─────────────────┬─────────────────┬─────────────────┐\n")
            f.write("│ Area        │ Best Agent      │ Best Network    │ Accuracy        │\n")
            f.write("├─────────────┼─────────────────┼─────────────────┼─────────────────┤\n")
            for rec in recommendations:
                f.write(f"│ {rec['area']:<11} │ {rec['agent']:<15} │ {rec['network']:<15} │ {rec['accuracy']:>5.1f}%     │\n")
            f.write("└─────────────┴─────────────────┴─────────────────┴─────────────────┘\n")
        
        print(f"\n✅ Recommendations saved: {rec_path}")
    
    def _generate_figures(self):
        """Generate evaluation figures"""
        print("\n" + "="*80)
        print("📊 GENERATING FIGURES")
        print("="*80)
        
        try:
            # Import evaluator
            from training.evaluator import Evaluator
            
            evaluator = Evaluator(
                results_dir=self.output_dir,
                output_dir=f"{self.output_dir}/evaluation/",
                seed=42,
                verbose=True,
                publication_quality=True
            )
            
            # Load data for evaluation
            from data_preprocessing.data_loader import DataLoader
            from data_preprocessing.data_splitter import DataSplitter
            
            loader = DataLoader('data_raw/Hybrid_Network_TN_NTN_Final.csv')
            data = loader.load()
            splitter = DataSplitter(data, split_ratio=0.8)
            _, test_data = splitter.split_by_user()
            
            # Evaluate all agents
            evaluator.evaluate_all_agents(
                test_data=test_data.head(2000),
                n_episodes=20,
                state_type='classical',
                sequence_length=15,
                track_users=True,
                track_handovers=True
            )
            
            # Generate figures
            evaluator.generate_visualizations()
            evaluator.generate_full_report()
            
            print(f"   ✅ Figures saved to: {self.output_dir}/evaluation/figures/")
            
        except Exception as e:
            print(f"   ⚠️ Could not generate figures: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        # Create and run system
        system = HybridNetworkSystem(
            total_timesteps=150000,  # 150k for best accuracy
            output_dir='../test_results/'
        )
        
        results = system.run()
        
        print("\n" + "="*80)
        print("📡 TN-NTN RECOMMENDATION SYSTEM READY!")
        print("="*80)
        print("\n📁 Results saved in:")
        print(f"   📂 Models: ../test_results/models/")
        print(f"   📂 Reports: ../test_results/reports/")
        print(f"   📂 Figures: ../test_results/figures/")
        print(f"   📂 Evaluation: ../test_results/evaluation/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)