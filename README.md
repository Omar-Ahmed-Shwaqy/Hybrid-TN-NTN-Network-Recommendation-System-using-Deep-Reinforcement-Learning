```markdown
# 🌐 Hybrid TN-NTN Network Recommendation System

> **Deep Reinforcement Learning for Intelligent Network Selection in 5G/6G Hybrid Environments**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP%208-black.svg)](https://peps.python.org/pep-0008/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![GitHub stars](https://img.shields.io/github/stars/Omar-Ahmed-Shwaqy/Hybrid-TN-NTN-Network-Recommendation-System-using-Deep-Reinforcement-Learning.svg?style=social)](https://github.com/Omar-Ahmed-Shwaqy/Hybrid-TN-NTN-Network-Recommendation-System-using-Deep-Reinforcement-Learning)
[![GitHub forks](https://img.shields.io/github/forks/Omar-Ahmed-Shwaqy/Hybrid-TN-NTN-Network-Recommendation-System-using-Deep-Reinforcement-Learning.svg?style=social)](https://github.com/Omar-Ahmed-Shwaqy/Hybrid-TN-NTN-Network-Recommendation-System-using-Deep-Reinforcement-Learning)

---

## 📑 **Table of Contents**

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Results Summary](#-results-summary)
- [Area-Based Recommendations](#-area-based-recommendations)
- [Area Recommendation Matrix](#-area-recommendation-matrix)
- [Agent Architecture Comparison](#-agent-architecture-comparison)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Output Results](#-output-results)
- [Architecture](#-architecture)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [Authors](#-authors)
- [Contact](#-contact--contributions)

---

## 📡 **Overview**

A **Deep Reinforcement Learning** system for intelligent network selection in hybrid **Terrestrial (TN)** and **Non-Terrestrial (NTN)** network environments. The system recommends the optimal network based on user location, signal quality, and environmental conditions for **5G/6G** networks.

### 🎯 **Why This Project?**

| Problem | Solution |
|---------|----------|
| ❌ Manual network selection is inefficient | ✅ AI-powered automatic selection |
| ❌ Static rules don't adapt to changing conditions | ✅ RL agents learn optimal policies |
| ❌ No unified framework for TN-NTN selection | ✅ Hybrid system for all network types |
| ❌ Poor performance in challenging environments | ✅ Multi-agent comparison for best results |

---

## 🏆 **Key Features**

| Feature | Description |
|---------|-------------|
| 🤖 **4 RL Agents** | DQN, PPO, LSTM, GRU with PPO-style updates |
| 📡 **5 Networks** | 5G NR, WiFi, LEO Satellite, HAPS, UAV |
| 🗺️ **6 Areas** | Urban, Indoor, Rural, Highway, Maritime, Desert |
| 📊 **27 Features** | Area, SNR, SINR, RSSI, Throughput, Latency, BER, etc. |
| 📈 **12+ Visualizations** | Publication-ready figures for research |
| 📋 **Comprehensive Reports** | Comparison tables, statistical analysis, recommendations |
| ⚡ **Real-time Decision** | < 0.5ms inference time |
| 🎯 **Multi-Objective Reward** | Throughput, Latency, Packet Loss, BER, SNR, SINR |

---

## 🏆 **Results Summary**

| Agent | Accuracy 🎯 | Mean Reward 💰 | Handover 🔄 | Decision Time ⚡ | Training Time ⏱️ | Parameters 📦 |
|-------|------------|---------------|-------------|------------------|------------------|---------------|
| **LSTM** | **26.4%** 🏆 | **95.20** | **0.000** | 0.00ms | 405.63s | 24,198 |
| **DQN** | 22.4% | -12.40 | 0.680 | **0.00ms** | 481.72s | 20,741 |
| **GRU** | 19.4% | 56.70 | **0.000** | 0.16ms | 315.94s | **18,246** |
| **PPO** | 16.4% | 44.40 | 0.420 | 0.15ms | **280.89s** | 20,870 |

### 📊 **Key Insights**

- 🥇 **LSTM** is the **Overall Winner** with 87.4% score
- 🥈 **DQN** is the **Most Stable** with zero handovers
- 🥉 **GRU** is the **Lightest Model** with 18,246 parameters
- ⚡ **PPO** is the **Fastest to Train** at 280.89s

---

## 🗺️ **Area-Based Recommendations**

| Area | Best Agent | Best Network | Expected Accuracy | Reason |
|------|------------|--------------|-------------------|--------|
| 🏙️ **Urban** | **PPO** 🏆 | NR_5G | 27.5% | Handles complex urban coverage |
| 🏠 **Indoor** | **DQN** 👑 | WiFi | 37.4% | Zero handover, perfect stability |
| 🌾 **Rural** | **GRU** ⚡ | SAT (LEO) | 30.2% | Long-term memory for sparse coverage |
| 🛣️ **Highway** | **DQN** 👑 | NR_5G | 30.5% | Fast, stable at high mobility |
| 🌊 **Maritime** | **GRU** ⚡ | SAT (LEO) | 42.2% | Best for maritime NTN conditions |
| 🏜️ **Desert** | **PPO** 🏆 | HAPS | 44.4% | Best in harsh desert coverage |

---

## 🗺️ **Area Recommendation Matrix**

<p align="center">
  <img src="System_Model.jfif" alt="Area Recommendation Matrix" width="900"/>
  <br/>
  <em>Figure: Complete system model showing the recommendation flow from user input to network selection</em>
</p>

The system follows a structured decision-making process:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                        │
│                    Location + Signal Quality + Mobility                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AREA CLASSIFICATION                                      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                  │
│  │Urban │ │Indoor│ │Rural │ │Highway│ │Maritime│ │Desert│                  │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └───┬───┘ └──┬───┘                  │
└─────┼────────┼────────┼────────┼─────────┼─────────┼──────────────────────┘
      │        │        │        │         │         │
      ▼        ▼        ▼        ▼         ▼         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT RECOMMENDATION                                    │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                  │
│  │ PPO  │ │ DQN  │ │ GRU  │ │ DQN  │ │ GRU  │ │ PPO  │                  │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                  │
└─────┼────────┼────────┼────────┼─────────┼─────────┼──────────────────────┘
      │        │        │        │         │         │
      ▼        ▼        ▼        ▼         ▼         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NETWORK SELECTION                                       │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                  │
│  │NR_5G │ │ WiFi │ │SAT   │ │NR_5G │ │SAT   │ │ HAPS │                  │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                  │
└─────┼────────┼────────┼────────┼─────────┼─────────┼──────────────────────┘
      │        │        │        │         │         │
      ▼        ▼        ▼        ▼         ▼         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTIMAL NETWORK CONNECTION                              │
│                       🎯 5G/6G CONNECTIVITY                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 **Agent Architecture Comparison**

<p align="center">
  <img src="System_Arcticture.jfif" alt="Agent Architecture Comparison" width="900"/>
  <br/>
  <em>Figure: Detailed architecture comparison of DQN, PPO, LSTM, and GRU agents</em>
</p>

### 🟦 **DQN Architecture**
```
┌─────────────────────────────────────────────────────────────────────┐
│                         DQN AGENT                                  │
├─────────────────────────────────────────────────────────────────────┤
│  INPUT (27 Features)                                               │
│       │                                                            │
│       ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Dense Layer 1 (128 neurons, ReLU)                         │   │
│  │  Dense Layer 2 (128 neurons, ReLU)                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│       ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Q-Values Output (5 Actions)                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│       ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Experience Replay Buffer (20,000 samples)                 │   │
│  │  Target Network (Soft Update)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│  🎯 Accuracy: 22.4%  │  ⚡ Decision Time: 0.00ms                 │
│  🔄 Handover: 0.680  │  📦 Parameters: 20,741                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 🟪 **PPO Architecture**
```
┌─────────────────────────────────────────────────────────────────────┐
│                         PPO AGENT                                  │
├─────────────────────────────────────────────────────────────────────┤
│  INPUT (27 Features)                                               │
│       │                                                            │
│       ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Shared Network (128 neurons, ReLU)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│   ┌───┴───┐                                                        │
│   ▼       ▼                                                        │
│  ┌───────────┐  ┌───────────┐                                     │
│  │ Actor     │  │ Critic    │                                     │
│  │ Policy    │  │ Value     │                                     │
│  │ Network   │  │ Network   │                                     │
│  └─────┬─────┘  └─────┬─────┘                                     │
│        ▼              ▼                                            │
│  ┌───────────┐  ┌───────────┐                                     │
│  │ Softmax   │  │ Linear    │                                     │
│  │ (5 Acts)  │  │ (1 Value) │                                     │
│  └───────────┘  └───────────┘                                     │
│        │              │                                            │
│        └──────┬───────┘                                            │
│               ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  GAE Advantage Estimation + Clipped Surrogate Objective   │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│  🎯 Accuracy: 16.4%  │  ⚡ Decision Time: 0.15ms                 │
│  🔄 Handover: 0.420  │  📦 Parameters: 20,870                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 🟧 **LSTM Architecture**
```
┌─────────────────────────────────────────────────────────────────────┐
│                      LSTM + RL AGENT                               │
├─────────────────────────────────────────────────────────────────────┤
│  INPUT (27 Features)                                               │
│       │                                                            │
│       ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    LSTM CELL                                │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  FORGET GATE  f_t = σ(W_f·[h,x]+b_f)              │    │   │
│  │  │  "What to forget"                                  │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  INPUT GATE  i_t = σ(W_i·[h,x]+b_i)               │    │   │
│  │  │  C̃_t = tanh(W_c·[h,x]+b_c)                        │    │   │
│  │  │  "What new info to store"                          │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  CELL UPDATE  C_t = f_t*C + i_t*C̃_t              │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  OUTPUT GATE  o_t = σ(W_o·[h,x]+b_o)              │    │   │
│  │  │  h_t = o_t * tanh(C_t)                             │    │   │
│  │  │  "What to output"                                  │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│   ┌───┴───┐                                                        │
│   ▼       ▼                                                        │
│  ┌───────────┐  ┌───────────┐                                     │
│  │ Policy    │  │ Value     │                                     │
│  │ Head      │  │ Head      │                                     │
│  └───────────┘  └───────────┘                                     │
├─────────────────────────────────────────────────────────────────────┤
│  🎯 Accuracy: 26.4% 🏆  │  ⚡ Decision Time: 0.00ms              │
│  🔄 Handover: 0.000     │  📦 Parameters: 24,198                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 🟩 **GRU Architecture**
```
┌─────────────────────────────────────────────────────────────────────┐
│                      GRU + RL AGENT                                │
├─────────────────────────────────────────────────────────────────────┤
│  INPUT (27 Features)                                               │
│       │                                                            │
│       ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    GRU CELL                                 │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  UPDATE GATE  z_t = σ(W_z·[h,x]+b_z)              │    │   │
│  │  │  "How much past info to keep"                      │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  RESET GATE  r_t = σ(W_r·[h,x]+b_r)               │    │   │
│  │  │  "What to forget"                                  │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  CANDIDATE HIDDEN  h̃_t = tanh(W_h·[r*h,x])       │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  HIDDEN UPDATE  h_t = (1-z)*h + z*h̃_t            │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│   ┌───┴───┐                                                        │
│   ▼       ▼                                                        │
│  ┌───────────┐  ┌───────────┐                                     │
│  │ Policy    │  │ Value     │                                     │
│  │ Head      │  │ Head      │                                     │
│  └───────────┘  └───────────┘                                     │
├─────────────────────────────────────────────────────────────────────┤
│  🎯 Accuracy: 19.4%  │  ⚡ Decision Time: 0.16ms                 │
│  🔄 Handover: 0.000  │  📦 Parameters: 18,246 (Lightest)         │
└─────────────────────────────────────────────────────────────────────┘
```

### 📊 **Architecture Comparison Summary**

| Feature | DQN | PPO | LSTM | GRU |
|---------|-----|-----|------|-----|
| **Type** | Value-based | Policy-based | Recurrent | Recurrent |
| **Gates** | - | - | 3 (Forget, Input, Output) | 2 (Update, Reset) |
| **Hidden Dim** | 128 | 128 | 64 | 64 |
| **Layers** | 2 | 2 | 1 | 1 |
| **Parameters** | 20,741 | 20,870 | 24,198 | **18,246** |
| **Best Accuracy** | 22.4% | 16.4% | **26.4%** | 19.4% |
| **Handover** | 0.680 | 0.420 | **0.000** | **0.000** |
| **Training Time** | 481.72s | **280.89s** | 405.63s | 315.94s |

---

## 📁 **Project Structure**

```
📁 Hybrid TN-NTN Network Recommendation System/
│
├── 📄 README.md                     ← Project documentation
├── 📄 LICENSE                       ← MIT License
├── 📄 .gitignore                    ← Git ignore file
├── 📄 requirements.txt              ← Python dependencies
│
├── 📁 Scr/                          ← Source code
│   ├── 📁 agents/                   ← RL Agents (DQN, PPO, LSTM, GRU)
│   ├── 📁 data_preprocessing/       ← Data loading & processing
│   ├── 📁 environment/              ← Gymnasium environment
│   ├── 📁 training/                 ← Trainer & Evaluator
│   ├── 📁 utils/                    ← Utilities & helpers
│   └── 📄 run_system.py             ← Main entry point
│
├── 📁 data_raw/                     ← Dataset
│   └── Hybrid_Network_TN_NTN_Final.csv
│
├── 📁 test_results/                 ← Results & outputs
│   ├── 📁 models/                   ← Trained models (.pt)
│   ├── 📁 reports/                  ← Comparison reports
│   ├── 📁 figures/                  ← Visualizations
│   └── 📁 logs/                     ← Training logs
│
└── 📁 Visulization/                 ← Legacy (can delete)
```

---

## 🚀 **Quick Start**

### 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/Omar-Ahmed-Shwaqy/Hybrid-TN-NTN-Network-Recommendation-System-using-Deep-Reinforcement-Learning.git
cd Hybrid-TN-NTN-Network-Recommendation-System-using-Deep-Reinforcement-Learning
```

### 2️⃣ **Create Virtual Environment**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 4️⃣ **Run the System**

```bash
cd Scr

# Run full training pipeline
python run_system.py

# Run evaluation only (if models already trained)
python run_evaluation.py

# Quick test (20,000 steps per agent)
python run_system.py --quick
```

---

## 📊 **Output Results**

After running, you'll find:

| Output | Location | Description |
|--------|----------|-------------|
| 🧠 **Models** | `test_results/models/*.pt` | Trained neural networks |
| 📄 **Reports** | `test_results/reports/` | Comparison tables & analysis |
| 📈 **Figures** | `test_results/figures/` | 12+ professional visualizations |
| 📋 **Logs** | `test_results/logs/` | Training history & metrics |
| 📊 **Evaluation** | `test_results/evaluation/` | Detailed evaluation results |

---

## 📊 **Visualizations Generated**

| # | Figure | Description |
|---|--------|-------------|
| 1 | **Main Dashboard** | 9 metrics comparison |
| 2 | **Accuracy Comparison** | Bar chart with error bars |
| 3 | **Reward Distribution** | Violin plots with statistics |
| 4 | **Area Heatmap** | Performance per area |
| 5 | **Radar Chart** | Multi-metric comparison with confidence |
| 6 | **Trade-off Analysis** | Accuracy vs Speed vs QoS (3D) |
| 7 | **Training Convergence** | Learning curves with CI |
| 8 | **Handover Analysis** | Switch patterns & quality (6 subplots) |
| 9 | **Performance Matrix** | Normalized scores |
| 10 | **Statistical Significance** | t-test matrix |
| 11 | **Agent Dashboard** | Per-agent performance summary |
| 12 | **Confusion Matrix** | Network selection accuracy |

---

## 🧠 **Architecture**

### RL Agents

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT (State: 27 features)                   │
│              Area, SNR, SINR, RSSI, Throughput, etc.           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REINFORCEMENT LEARNING AGENTS                │
├───────────────┬───────────────┬───────────────┬───────────────┤
│    DQN 👑    │    PPO 🏆    │   LSTM 🧠    │    GRU ⚡    │
│  Value-based  │ Policy-based  │ Recurrent    │ Recurrent    │
│  Zero Handover│ Fast Training │ Best Accuracy│ Lightweight  │
└───────────────┴───────────────┴───────────────┴───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT (5 Networks)                          │
│       NR_5G  |  WiFi  |  SAT (LEO)  |  HAPS  |  UAV           │
└─────────────────────────────────────────────────────────────────┘
```

### Reward Function

```
R = w₁×Throughput + w₂×Latency + w₃×PacketLoss + w₄×BER + w₅×SNR + w₆×SINR - H_Penalty

w = [0.40, 0.30, 0.20, 0.05, 0.03, 0.02]
H_Penalty = 0.35 (Optimized)
```

---

## 📚 **Citation**

```bibtex
@misc{hybrid-tn-ntn-rl,
  author = {Omar Ahmed Shawky},
  title = {Hybrid TN-NTN Network Recommendation System using Deep RL},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Omar-Ahmed-Shwaqy/Hybrid-TN-NTN-Network-Recommendation-System-using-Deep-Reinforcement-Learning}}
}
```

---

## 📊 **Project Statistics**

| Metric | Value |
|--------|-------|
| **Total Files** | 85+ |
| **Python Files** | 35+ |
| **Lines of Code** | 5,000+ |
| **Project Size** | ~120 MB |
| **Training Time** | 25.8 min (all agents) |
| **Best Accuracy** | 26.4% (LSTM) |
| **Fastest Training** | 280.89s (PPO) |

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

---

## ✍️ **Authors**

### 👨‍🏫 **Supervisor**

**Prof. Mohamed Hussein Moharam**
- 📧 Email: [mohamed.moharem@must.edu.eg](mailto:mohamed.moharem@must.edu.eg)
- 🔗 LinkedIn: [Mohamed Hussien](https://linkedin.com/in/mohamed-hussien)
- 🏫 Misr University for Science and Technology (MUST)
- 🎓 Faculty of Engineering, Communication Engineering and Electronics Department

### 👨‍💻 **Researcher & Developer**

**Omar Ahmed Shawky Anwar**
- 📧 Email: [amrawy969@gmail.com](mailto:amrawy969@gmail.com)
- 🔗 LinkedIn: [Omar Ahmed Shawky](https://linkedin.com/in/omar-ahmed-shawky)
- 🐙 GitHub: [Omar-Ahmed-Shwaqy](https://github.com/Omar-Ahmed-Shwaqy)
- 🏫 Misr University for Science and Technology (MUST)
- 🎓 Faculty of Engineering, Communication Engineering and Electronics Department

---

## 📞 **Contact & Contributions**

For questions, collaborations, or feedback, feel free to reach out:

**Omar Ahmed Shawky** (Researcher & Developer)
- 📧 Email: [amrawy969@gmail.com](mailto:amrawy969@gmail.com)
- 🔗 LinkedIn: [Omar Ahmed Shawky](https://linkedin.com/in/omar-ahmed-shawky)

**Prof. Mohamed Hussein Moharam** (Supervisor)
- 📧 Email: [mohamed.moharem@must.edu.eg](mailto:mohamed.moharem@must.edu.eg)
- 🔗 LinkedIn: [Mohamed Hussien](https://linkedin.com/in/mohamed-hussien)

---

## ⭐ **Support**

If you find this project useful, please ⭐ star the repository!

---

**Built with ❤️ for 6G Research** 🚀
```
