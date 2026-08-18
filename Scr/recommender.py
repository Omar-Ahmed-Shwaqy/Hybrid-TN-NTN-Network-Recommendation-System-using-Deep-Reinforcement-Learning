# =============================================================================
# FILE: Scr/recommender.py
# PURPOSE: Lightweight, deployment-ready recommendations for TN-NTN system
# =============================================================================
# This module has NO PyTorch dependency - can run anywhere!
# =============================================================================

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class AgentType(Enum):
    """Available RL Agents"""
    DQN = "DQN"
    PPO = "PPO"
    GRU = "GRU"
    LSTM = "LSTM"


class NetworkType(Enum):
    """Available Networks"""
    NR_5G = "NR_5G"
    WIFI = "WiFi"
    SAT_LEO = "SAT (LEO)"
    HAPS = "HAPS"
    UAV = "UAV"


class AreaType(Enum):
    """Supported Areas"""
    URBAN = "Urban"
    INDOOR = "Indoor"
    RURAL = "Rural"
    HIGHWAY = "Highway"
    MARITIME = "Maritime"
    DESERT = "Desert"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class Recommendation:
    """Complete recommendation result"""
    agent: str                    # Best RL Agent
    network: str                  # Best Network
    expected_accuracy: float      # Expected accuracy (%)
    reason: str                   # Why this recommendation
    
    # Extended fields
    agent_full_name: Optional[str] = None
    network_full_name: Optional[str] = None
    handover_rate: Optional[float] = None
    expected_reward: Optional[float] = None
    
    def __post_init__(self):
        """Fill default full names if not provided"""
        if self.agent_full_name is None:
            agent_names = {
                'DQN': 'Deep Q-Network (Stable, Zero Handover)',
                'PPO': 'Proximal Policy Optimization (Best Overall)',
                'GRU': 'Gated Recurrent Unit (Long-term Memory)',
                'LSTM': 'Long Short-Term Memory (Not Recommended)'
            }
            object.__setattr__(self, 'agent_full_name', agent_names.get(self.agent, self.agent))
        
        if self.network_full_name is None:
            network_names = {
                'NR_5G': '5G Terrestrial - High Speed, Low Latency',
                'WiFi': 'Wi-Fi - Best for Indoor',
                'SAT (LEO)': 'LEO Satellite - Wide Coverage',
                'HAPS': 'High Altitude Platform - Medium Coverage',
                'UAV': 'Unmanned Aerial Vehicle - Flexible'
            }
            object.__setattr__(self, 'network_full_name', network_names.get(self.network, self.network))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (serializable)"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def print_pretty(self):
        """Print recommendation in a beautiful format"""
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎯 RECOMMENDATION                                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📍 Area                 : {self.area if hasattr(self, 'area') else 'N/A'}                         
║  🤖 Best Agent           : {self.agent}                                     
║  📝 Agent Description    : {self.agent_full_name}                          
║  📡 Best Network         : {self.network}                                  
║  📝 Network Description  : {self.network_full_name}                        
║  🎯 Expected Accuracy    : {self.expected_accuracy:.2f}%                   
║  🔄 Handover Rate        : {self.handover_rate or 'N/A'}                   
║  💰 Expected Reward      : {self.expected_reward or 'N/A'}                 
║  📝 Reason               : {self.reason}                                   
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)


@dataclass
class AreaInfo:
    """Detailed information about an area"""
    name: str
    description: str
    characteristics: List[str]
    best_agent: str
    best_network: str
    accuracy: float


# =============================================================================
# SMART RECOMMENDER - MAIN CLASS
# =============================================================================

class SmartRecommender:
    """
    Lightweight recommender for TN-NTN hybrid networks.
    
    Returns the best evaluated RL-agent/network pair for each supported area.
    No PyTorch dependency - can be deployed anywhere.
    
    Usage:
        recommender = SmartRecommender()
        
        # Get recommendation for an area
        rec = recommender.recommend('Maritime')
        print(rec['agent'])  # 'GRU'
        print(rec['network'])  # 'SAT (LEO)'
        
        # Get all recommendations
        all_recs = recommender.get_all_recommendations()
        
        # Compare two areas
        recommender.compare_areas('Urban', 'Rural')
    """
    
    # =====================================================================
    # KNOWLEDGE BASE - From experimental results
    # =====================================================================
    
    # Core policy: Best agent and network per area
    _POLICY: Dict[str, Recommendation] = {
        'Urban': Recommendation(
            agent='PPO',
            network='NR_5G',
            expected_accuracy=27.47,
            reason='Best accuracy in dense, complex urban coverage with multiple networks.',
            handover_rate=0.456,
            expected_reward=260.44
        ),
        'Indoor': Recommendation(
            agent='DQN',
            network='WiFi',
            expected_accuracy=37.36,
            reason='Perfect stability with zero handovers. Ideal for indoor environments.',
            handover_rate=0.000,
            expected_reward=330.41
        ),
        'Rural': Recommendation(
            agent='GRU',
            network='SAT (LEO)',
            expected_accuracy=30.24,
            reason='Sequence memory handles sparse, long-range coverage in rural areas.',
            handover_rate=0.540,
            expected_reward=247.26
        ),
        'Highway': Recommendation(
            agent='DQN',
            network='NR_5G',
            expected_accuracy=30.51,
            reason='Fast, stable decisions essential for high-speed mobility.',
            handover_rate=0.000,
            expected_reward=330.41
        ),
        'Maritime': Recommendation(
            agent='GRU',
            network='SAT (LEO)',
            expected_accuracy=42.16,
            reason='Best result under maritime NTN conditions with long-term memory.',
            handover_rate=0.540,
            expected_reward=247.26
        ),
        'Desert': Recommendation(
            agent='PPO',
            network='HAPS',
            expected_accuracy=44.44,
            reason='Best result in harsh desert coverage with limited infrastructure.',
            handover_rate=0.456,
            expected_reward=260.44
        ),
    }
    
    # Area emojis for display
    _AREA_EMOJIS = {
        'Urban': '🏙️',
        'Indoor': '🏠',
        'Rural': '🌾',
        'Highway': '🛣️',
        'Maritime': '🌊',
        'Desert': '🏜️'
    }
    
    # Agent descriptions
    _AGENT_DESCRIPTIONS = {
        'DQN': 'Deep Q-Network - Best for stability',
        'PPO': 'Proximal Policy Optimization - Best overall',
        'GRU': 'Gated Recurrent Unit - Best for memory',
        'LSTM': 'Long Short-Term Memory - Not recommended'
    }
    
    # Network descriptions
    _NETWORK_DESCRIPTIONS = {
        'NR_5G': '5G Terrestrial - Fast, low latency',
        'WiFi': 'Wi-Fi - Best indoors',
        'SAT (LEO)': 'LEO Satellite - Wide coverage',
        'HAPS': 'High Altitude Platform - Medium coverage',
        'UAV': 'UAV - Flexible, limited coverage'
    }
    
    # Area characteristics
    _AREA_INFO = {
        'Urban': AreaInfo(
            name='Urban',
            description='Dense city with multiple networks and high interference',
            characteristics=['High network density', 'Signal interference', 'Many users'],
            best_agent='PPO',
            best_network='NR_5G',
            accuracy=27.47
        ),
        'Indoor': AreaInfo(
            name='Indoor',
            description='Indoor environments like buildings, malls, offices',
            characteristics=['WiFi preferred', 'Weak 5G signal', 'Limited mobility'],
            best_agent='DQN',
            best_network='WiFi',
            accuracy=37.36
        ),
        'Rural': AreaInfo(
            name='Rural',
            description='Countryside with sparse coverage and long distances',
            characteristics=['Limited networks', 'Medium coverage', 'Long distances'],
            best_agent='GRU',
            best_network='SAT (LEO)',
            accuracy=30.24
        ),
        'Highway': AreaInfo(
            name='Highway',
            description='High-speed roads with frequent network changes',
            characteristics=['High speed', 'Frequent handovers', 'Stability required'],
            best_agent='DQN',
            best_network='NR_5G',
            accuracy=30.51
        ),
        'Maritime': AreaInfo(
            name='Maritime',
            description='Sea/ocean environments with NTN-only coverage',
            characteristics=['NTN only', 'Difficult conditions', 'Intermittent coverage'],
            best_agent='GRU',
            best_network='SAT (LEO)',
            accuracy=42.16
        ),
        'Desert': AreaInfo(
            name='Desert',
            description='Harsh desert with limited infrastructure',
            characteristics=['NTN only', 'Harsh conditions', 'Limited coverage'],
            best_agent='PPO',
            best_network='HAPS',
            accuracy=44.44
        ),
    }
    
    # =====================================================================
    # PUBLIC METHODS
    # =====================================================================
    
    def recommend(self, area: str) -> Dict[str, Any]:
        """
        Get recommendation for a specific area.
        
        Args:
            area: Area name (case-insensitive)
            
        Returns:
            Dictionary with recommendation details
            
        Raises:
            ValueError: If area is not recognized
            
        Example:
            >>> recommender = SmartRecommender()
            >>> rec = recommender.recommend('Maritime')
            >>> print(rec['agent'])
            'GRU'
        """
        normalised = str(area).strip().casefold()
        
        for name, recommendation in self._POLICY.items():
            if name.casefold() == normalised:
                result = asdict(recommendation)
                result['area'] = name
                result['area_emoji'] = self._AREA_EMOJIS.get(name, '📍')
                return result
        
        valid = ', '.join(self._POLICY.keys())
        raise ValueError(f"Unknown area '{area}'. Choose from: {valid}.")
    
    def recommend_agent(self, area: str) -> str:
        """
        Get only the best agent for an area.
        
        Args:
            area: Area name
            
        Returns:
            Best agent name (e.g., 'PPO')
        """
        rec = self.recommend(area)
        return rec['agent']
    
    def recommend_network(self, area: str) -> str:
        """
        Get only the best network for an area.
        
        Args:
            area: Area name
            
        Returns:
            Best network name (e.g., 'NR_5G')
        """
        rec = self.recommend(area)
        return rec['network']
    
    def get_area_info(self, area: str) -> Dict[str, Any]:
        """
        Get detailed information about an area.
        
        Args:
            area: Area name
            
        Returns:
            Area information dictionary
        """
        if area in self._AREA_INFO:
            info = self._AREA_INFO[area]
            return {
                'name': info.name,
                'description': info.description,
                'characteristics': info.characteristics,
                'best_agent': info.best_agent,
                'best_network': info.best_network,
                'accuracy': info.accuracy,
                'emoji': self._AREA_EMOJIS.get(area, '📍')
            }
        raise ValueError(f"Unknown area: {area}")
    
    def get_all_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get recommendations for all areas.
        
        Returns:
            Dictionary with all area recommendations
        """
        result = {}
        for area in self._POLICY.keys():
            result[area] = self.recommend(area)
        return result
    
    def get_all_areas(self) -> List[str]:
        """Get list of all supported areas"""
        return list(self._POLICY.keys())
    
    def get_all_agents(self) -> List[str]:
        """Get list of all agents"""
        return list(set(rec.agent for rec in self._POLICY.values()))
    
    def get_all_networks(self) -> List[str]:
        """Get list of all networks"""
        return list(set(rec.network for rec in self._POLICY.values()))
    
    def compare_areas(self, area1: str, area2: str) -> Dict[str, Any]:
        """
        Compare recommendations for two areas.
        
        Args:
            area1: First area
            area2: Second area
            
        Returns:
            Comparison dictionary
            
        Example:
            >>> recommender.compare_areas('Urban', 'Rural')
            {
                'areas': ['Urban', 'Rural'],
                'agents': ['PPO', 'GRU'],
                'networks': ['NR_5G', 'SAT (LEO)'],
                'accuracies': [27.47, 30.24]
            }
        """
        rec1 = self.recommend(area1)
        rec2 = self.recommend(area2)
        
        return {
            'areas': [area1, area2],
            'agents': [rec1['agent'], rec2['agent']],
            'networks': [rec1['network'], rec2['network']],
            'accuracies': [rec1['expected_accuracy'], rec2['expected_accuracy']],
            'reasons': [rec1['reason'], rec2['reason']]
        }
    
    def print_recommendation(self, area: str):
        """Print a beautiful recommendation for an area"""
        rec = self.recommend(area)
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  {self._AREA_EMOJIS.get(area, '📍')} RECOMMENDATION FOR: {area.upper()}                            
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🤖 Best Agent           : {rec['agent']} ({self._AGENT_DESCRIPTIONS.get(rec['agent'], '')})    
║  📡 Best Network         : {rec['network']} ({self._NETWORK_DESCRIPTIONS.get(rec['network'], '')})
║  🎯 Expected Accuracy    : {rec['expected_accuracy']:.2f}%                 
║  🔄 Handover Rate        : {rec.get('handover_rate', 'N/A')}                
║  💰 Expected Reward      : {rec.get('expected_reward', 'N/A')}              
║                                                                              ║
║  📝 Reason               : {rec['reason']}                                  
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
    
    def print_all_recommendations(self):
        """Print all recommendations in a table"""
        print("\n" + "="*100)
        print("📊 ALL RECOMMENDATIONS BY AREA")
        print("="*100)
        
        print("\n┌─────────────┬─────────────────┬─────────────────┬─────────────────┬──────────────────────┐")
        print("│ Area        │ Best Agent      │ Best Network    │ Accuracy        │ Handover Rate        │")
        print("├─────────────┼─────────────────┼─────────────────┼─────────────────┼──────────────────────┤")
        
        for area in self._POLICY.keys():
            rec = self.recommend(area)
            emoji = self._AREA_EMOJIS.get(area, '📍')
            print(f"│ {emoji} {area:<9} │ {rec['agent']:<15} │ {rec['network']:<15} │ {rec['expected_accuracy']:>5.1f}%     │ {rec.get('handover_rate', 0):>6.3f}               │")
        
        print("└─────────────┴─────────────────┴─────────────────┴─────────────────┴──────────────────────┘")
    
    def print_summary(self):
        """Print a summary of the recommendation system"""
        print("\n" + "="*80)
        print("📊 TN-NTN RECOMMENDATION SYSTEM - SUMMARY")
        print("="*80)
        
        print(f"\n📌 Supported Areas: {len(self._POLICY)}")
        for area in self._POLICY.keys():
            emoji = self._AREA_EMOJIS.get(area, '📍')
            print(f"   {emoji} {area}")
        
        agents = self.get_all_agents()
        print(f"\n🤖 Available Agents: {', '.join(agents)}")
        
        networks = self.get_all_networks()
        print(f"📡 Available Networks: {', '.join(networks)}")
        
        # Best overall accuracy
        best_area = max(self._POLICY.items(), key=lambda x: x[1].expected_accuracy)
        print(f"\n🏆 Best Performance: {best_area[0]} ({best_area[1].expected_accuracy:.2f}%)")
        print(f"   Agent: {best_area[1].agent}, Network: {best_area[1].network}")
    
    def to_json(self) -> str:
        """Export all recommendations to JSON"""
        import json
        data = {}
        for area, rec in self._POLICY.items():
            data[area] = asdict(rec)
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def save_to_file(self, filename: str = 'recommendations.json'):
        """Save all recommendations to a JSON file"""
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({area: asdict(rec) for area, rec in self._POLICY.items()}, 
                     f, indent=2, ensure_ascii=False)
        print(f"✅ Recommendations saved to {filename}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def quick_recommend(area: str) -> Dict[str, Any]:
    """
    Quick function to get a recommendation.
    
    Usage:
        rec = quick_recommend('Maritime')
        print(rec['agent'])  # GRU
    """
    recommender = SmartRecommender()
    return recommender.recommend(area)


def quick_recommend_agent(area: str) -> str:
    """Quick function to get just the best agent"""
    recommender = SmartRecommender()
    return recommender.recommend_agent(area)


def quick_recommend_network(area: str) -> str:
    """Quick function to get just the best network"""
    recommender = SmartRecommender()
    return recommender.recommend_network(area)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING SMART RECOMMENDER")
    print("="*80)
    
    # Create recommender
    recommender = SmartRecommender()
    
    # 1. Get recommendation for an area
    print("\n📌 1. Recommendation for Maritime:")
    rec = recommender.recommend('Maritime')
    print(f"   Agent: {rec['agent']}")
    print(f"   Network: {rec['network']}")
    print(f"   Accuracy: {rec['expected_accuracy']:.2f}%")
    print(f"   Reason: {rec['reason']}")
    
    # 2. Print a pretty recommendation
    print("\n📌 2. Pretty Recommendation:")
    recommender.print_recommendation('Desert')
    
    # 3. Print all recommendations
    recommender.print_all_recommendations()
    
    # 4. Compare two areas
    print("\n📌 3. Comparison:")
    comparison = recommender.compare_areas('Urban', 'Rural')
    print(f"   {comparison['areas'][0]}: {comparison['agents'][0]} -> {comparison['networks'][0]} ({comparison['accuracies'][0]:.2f}%)")
    print(f"   {comparison['areas'][1]}: {comparison['agents'][1]} -> {comparison['networks'][1]} ({comparison['accuracies'][1]:.2f}%)")
    
    # 5. Summary
    recommender.print_summary()
    
    # 6. Quick functions
    print("\n📌 4. Quick Functions:")
    print(f"   Best agent for Indoor: {quick_recommend_agent('Indoor')}")
    print(f"   Best network for Highway: {quick_recommend_network('Highway')}")
    
    print("\n✅ SmartRecommender test completed!")