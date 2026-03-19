# 🐍 Snake AI Using Deep Q-Learning (DQN)

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-supported-76B900.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-red.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.49-purple.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="Images/score13.gif" alt="Gif - AI DQL">
</p>

## 📝 Project Description

This project is a **continuation** of my previous Snake AI series :

- 🎮 The Snake game itself : [snake_game](https://github.com/Thibault-GAREL/snake_game)
- 🧬 First AI version using NEAT (NeuroEvolution) : [AI_snake_genetic_version](https://github.com/Thibault-GAREL/AI_snake_genetic_version)

This time, the agent learns to play Snake using **Deep Q-Learning (DQN)** with PyTorch and CUDA support. Unlike NEAT which evolves a population of networks over generations, DQN is a **reinforcement learning** approach : a single agent interacts with the environment, stores its experiences in a replay buffer, and learns by minimizing the Bellman error. 🤖🎯

The project also includes a full **Explainable AI (XAI)** suite to analyze what the network has actually learned, going beyond just performance metrics.

---

## 🚀 Features

🧠 **Double DQN** with target network — reduces Q-value overestimation

⚡ **CUDA support** — automatic GPU detection and training

🗂️ **Experience Replay** — 100k transition buffer for stable learning

📉 **ε-greedy exploration** with exponential decay

💾 **Auto-save** — best model and periodic checkpoints

📊 **Full XAI suite** — 4 independent analysis scripts

---

## ⚙️ How it works

🕹️ The AI controls a snake on a 10×14 grid. At each step, it receives a **state vector of 16 features** (distances to walls/body and distances to food in 8 directions) and outputs **Q-values for 4 actions** (UP, RIGHT, DOWN, LEFT).

🧠 The network is a fully-connected MLP (16 → 256 → 128 → 64 → 4) trained with the Double DQN algorithm. A separate target network is updated every 500 steps to stabilize training.

📈 A reward shaping signal guides the agent toward food even before it reaches it, making early training much more efficient.

---

## 🗺️ Network Architecture

```
Input (16)  →  Linear(256) → BatchNorm → ReLU
            →  Linear(128) → ReLU
            →  Linear(64)  → ReLU
            →  Linear(4)   →  Q-values
```

<details>
<summary>📋 State vector — 16 input features</summary>

### Distances to walls / body (8 inputs)

| # | Feature |
|---|---------|
| 0 | `distance_bord_N` — Distance to obstacle North |
| 1 | `distance_bord_NE` — Distance to obstacle North-East |
| 2 | `distance_bord_E` — Distance to obstacle East |
| 3 | `distance_bord_SE` — Distance to obstacle South-East |
| 4 | `distance_bord_S` — Distance to obstacle South |
| 5 | `distance_bord_SW` — Distance to obstacle South-West |
| 6 | `distance_bord_W` — Distance to obstacle West |
| 7 | `distance_bord_NW` — Distance to obstacle North-West |

### Distances to food (8 inputs)

| # | Feature |
|---|---------|
| 8  | `distance_food_N` — Distance to food North |
| 9  | `distance_food_NE` — Distance to food North-East |
| 10 | `distance_food_E` — Distance to food East |
| 11 | `distance_food_SE` — Distance to food South-East |
| 12 | `distance_food_S` — Distance to food South |
| 13 | `distance_food_SW` — Distance to food South-West |
| 14 | `distance_food_W` — Distance to food West |
| 15 | `distance_food_NW` — Distance to food North-West |

### Output — 4 actions

| # | Action |
|---|--------|
| 0 | `UP` |
| 1 | `RIGHT` |
| 2 | `DOWN` |
| 3 | `LEFT` |

</details>

---

## 🔬 Explainable AI (XAI) Suite

One of the key aspects of this project is understanding **what the network actually learned**, not just how well it performs. Four dedicated scripts analyze the model from different angles :

| Script | Analysis | Output |
|--------|----------|--------|
| `xai_qvalues.py` | Q-value heatmaps, confidence map, temporal evolution | `xai_qvalues/` |
| `xai_features.py` | Permutation importance, weight variance, feature-action correlation | `xai_features/` |
| `xai_activations.py` | Dead neurons, specialization, t-SNE / UMAP projection | `xai_activations/` |
| `xai_shap.py` | SHAP DeepExplainer — beeswarm, waterfall, force plots, summary heatmap | `xai_shap/` |

<details>
<summary>📸 See the XAI analyses — results & interpretation</summary>

---

### 🔷 Q-values analysis (`xai_qvalues.py`)

#### Q-value heatmaps
![xai_heatmaps](xai_qvalues/xai_heatmaps.png)

Each subplot shows the Q-value estimated by the network for one action, at every cell of the grid (food fixed at column 5, row 3 — marked with ★). The dominant color is warm (positive Q-values), which means the agent generally sees all positions as potentially rewarding. The **border cells** show clearly colder (darker/blue) values for dangerous actions — for example the RIGHT heatmap shows a dark blue column on the right wall, and the UP heatmap darkens near the top row. This confirms the network has learned to discount actions that lead toward walls.

#### Confidence map & learned policy
![xai_confidence](xai_qvalues/xai_confidence.png)

The left map shows `Q_max − Q_2nd_max` : almost the entire grid is dark (near zero gap), meaning the agent is **systematically uncertain** — it rarely has a strongly dominant action. Only two cells stand out with a high gap : the top-left corner (forced UP) and the bottom-right corner (forced LEFT), which are border situations where the choice is obvious. The right map shows the learned policy with one arrow per cell : the agent tends to steer **LEFT and UP** in most of the grid, which is consistent with a strategy of circling back toward the food placed in the upper-center.

#### Temporal Q-value evolution
![xai_temporal](xai_qvalues/xai_temporal.png)

Three episodes are shown. **Episode 1** (score 0, 500 steps) shows a very regular oscillating pattern where Q-values for all 4 actions alternate symmetrically — the agent is stuck in a loop, going back and forth without eating. **Episode 2** (score 0, 500 steps) is the most interesting : all 4 Q-values are nearly flat and perfectly superimposed around 5.5, with almost no variance — the agent has converged to a near-constant policy that avoids death but never finds the food. **Episode 3** (score 5, 57 steps) shows a much more dynamic pattern : Q-values fluctuate strongly, food events (green dotted lines) coincide with Q-value spikes, and the red death line appears at the end after a last sharp divergence between actions.

---

### 🔷 Feature importance (`xai_features.py`)

#### Permutation importance
![xai_permutation](xai_features/xai_permutation.png)

When any feature is shuffled, the score drops by approximately **6.5 to 8.5 points** from a baseline of 8.55 — which means **every single feature is critical**. No feature can be removed without a near-total collapse in performance. The ranking shows `Dist. food W` as the most important (drop = 8.45), followed closely by all other food distances, then wall distances. The radar chart confirms this : the polygon is **large and nearly regular**, meaning no single feature dominates — the agent relies on the full 16-dimensional state equally. This is a strong sign that the network has not found shortcuts and genuinely uses all available information.

#### Weight variance (W₁ analysis)
![xai_variance](xai_features/xai_variance.png)

The L2 norm of each input column in the first weight matrix shows **wall distances systematically dominate** (Mur NE = 8.07, Mur NW = 7.91, Mur SE = 7.90) while food diagonal distances are lowest (Food SE = 4.18, Food NW = 4.62). The scatter plot (bottom right) places all wall features in the **high L2 / high std quadrant** (strong importance) while food diagonal features fall in the bottom-left (lower structural weight). This suggests the first layer focuses more on obstacle avoidance than food tracking — directional food alignment is handled later in the network.

#### Feature-action correlation
![xai_correlation](xai_features/xai_correlation.png)

The Pearson correlation heatmap reveals clean and interpretable patterns : `Dist. food N` correlates strongly with UP (r = +0.30), `Dist. food E` with RIGHT (r = +0.30), `Dist. mur SE` with DOWN (r = +0.20), and `Dist. mur W` with LEFT (r = +0.20). Wall distances show negative correlations with the action that would lead toward them, confirming the agent avoids obstacles. The food distance features act as triggers : when food is visible in a direction, the agent is more likely to move that way.

#### Sensory profile per action
![xai_mean_per_action](xai_features/xai_mean_per_action.png)

For each action, all 16 features are near zero in the **food rows** (top half), except for the directionally relevant one. For example, when choosing UP, `Dist. food N` has a non-zero mean while all other food features are 0. The **wall features** (bottom half) all show consistent non-zero values across all actions, reflecting that wall proximity information is always present regardless of the decision. Each action has a slightly distinct wall distance profile, confirming the agent integrates spatial context when deciding.

---

### 🔷 Internal activations (`xai_activations.py`)

#### Distribution & dead neurons
![xai_distribution](xai_activations/xai_distribution.png)

The histograms (left column) show a heavily **right-skewed exponential distribution** for all three layers — most activations are near zero, with a long sparse tail of high values. This is typical of ReLU networks with sparse representations. The dead neuron analysis (middle column) reveals a severe situation : **Layer 1 has 45.7% dead neurons** (117/256), **Layer 2 has 75%** (96/128), and **Layer 3 has 73.4%** (47/64). This progressive increase in dead neuron rate suggests significant under-utilization of capacity — the network has converged to a sparse solution using only ~25-55% of its neurons. The temporal heatmaps (right column) confirm this : only the top rows (highest variance neurons) show intermittent activity, while most neurons (dark blue) remain completely silent throughout the 200 steps.

#### Neuron specialization
![xai_specialization](xai_activations/xai_specialization.png)

The specialization score distributions (left column) show most neurons have scores near zero (not specialized), but a small fraction exhibits scores up to 0.58 (Layer 1), 1.66 (Layer 2), and 1.34 (Layer 3). The heatmaps (center column) reveal that **"Food alignée H"** consistently activates the most specialized neurons across all layers — this situation has very few observations (n=12), making neurons that respond to it stand out strongly. The **"Neutre"** and **"Danger"** situations dominate in terms of volume (n=488–640) but produce more distributed activations. The top-5 neuron profiles (right column) show Layer 2 neuron #10 as the most specialized (score=1.66), nearly exclusively active during horizontal food alignment — a dedicated "food east/west detector" neuron.

#### t-SNE projection
![xai_tsne](xai_activations/xai_tsne.png)

The t-SNE projections across all 3 layers show a **diffuse cloud with no clear cluster separation** when colored by situation or action. This contrasts with what a well-disentangled representation would show (distinct colored clusters). However, the score-colored view reveals a subtle gradient : higher-score states (orange/red) tend to appear slightly more concentrated toward the center of the cloud, while zero-score states (dark blue) are scattered at the periphery. The lack of clean clustering suggests the network has learned a **continuous, overlapping representation** rather than discrete situation-specific modes.

#### UMAP projection
![xai_umap](xai_activations/xai_umap.png)

The UMAP projections show a striking contrast with t-SNE : the activations collapse into a **very dense central cluster** with a small number of isolated outlier points. This extreme density suggests the network's internal representations are highly similar across most game states — it processes the majority of situations in a near-identical way, only producing distinct activations for a few exceptional states (the outlier points). The isolated points in Layer 3 (64n) correspond to rare situations such as food alignment or corner positions, confirming that the final layer slightly differentiates edge cases from the common baseline.

---

### 🔷 SHAP analysis (`xai_shap.py`)

#### Beeswarm plot
![xai_shap_beeswarm](xai_shap/xai_shap_beeswarm.png)

The beeswarm plots show that **wall distance features dominate SHAP impact** across all 4 actions, with food features contributing very little individually. The spread of SHAP values is much wider for wall features (reaching ±10) than for food features (mostly within ±1). High-value wall features (red dots) tend to have strong negative SHAP values — a wall that is very close sharply reduces the Q-value of the action pointing toward it. The food features show almost no color variation (all cold blue), meaning their values are rarely non-zero, which is consistent with the sparse directional food encoding used in the state vector.

#### Waterfall plots
![xai_shap_waterfall](xai_shap/xai_shap_waterfall.png)

In all 8 situations, the agent systematically chooses **LEFT** as the dominant action, starting from a base value E[f(x)] ≈ 4.5–6.5 and accumulating positive contributions from wall features to reach a final Q-value around 5.5–7.5. The **"Food alignée V"** situation shows the smallest final Q-value (~5.3), with the most balanced positive/negative contributions — the agent is least certain in this case. The **Danger situations** (N, E, S, W) all show a large positive contribution from the wall feature of the opposite direction (`Mur W` contributing +1.35 in Danger N, +0.80 in Danger W), confirming the agent is pushed away from the danger by the wall distance signal on the safe side.

#### SHAP summary heatmap
![xai_shap_heatmap](xai_shap/xai_shap_heatmap.png)

The global importance ranking (bottom left) confirms wall distances are the most impactful features : `Mur W` (0.503), `Mur NW` (0.448), `Mur E` (0.438) lead the ranking, while food diagonal features (Food SW, Food NE, Food SE) are nearly irrelevant (< 0.01). The signed SHAP heatmap (top right) shows that wall features have **strong positive SHAP values for LEFT** (`Mur NW` = +0.252, `Mur W` = +0.172, `Mur SW` = +0.206), meaning the network is biased toward turning left when walls are on the right side. The feature × situation heatmap (bottom right) reveals `Dist. food W` becomes extremely important specifically during **"Food alignée H"** (horizontal alignment), which is the one situation where the food is directly to the west — confirming the network correctly focuses on the relevant directional feature when food is visible.

</details>

---

## 📂 Repository structure

```bash
├── snake.py                # Snake game (from snake_game repo)
├── dql.py                  # DQN agent, network, replay buffer
├── main.py                 # Training loop + SnakeEnv wrapper
│
├── xai_qvalues.py          # XAI — Q-value analysis
├── xai_features.py         # XAI — Feature importance
├── xai_activations.py      # XAI — Internal activations
├── xai_shap.py             # XAI — SHAP explanations
│
├── model_best.pth          # Best model checkpoint (auto-saved)
├── model_final.pth         # Final model checkpoint
│
├── xai_qvalues/            # Output plots — Q-values
├── xai_features/           # Output plots — Feature importance
├── xai_activations/        # Output plots — Activations
├── xai_shap/               # Output plots + HTML — SHAP
│
├── LICENSE
└── README.md
```

---

## 💻 Run it on Your PC

Clone the repository and install dependencies :

```bash
git clone https://github.com/Thibault-GAREL/AI_snake_DQL.git
cd AI_snake_DQL

python -m venv .venv
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pygame numpy matplotlib scipy scikit-learn
pip install shap          # for xai_shap.py
pip install umap-learn    # optional, for xai_activations.py --umap
```

### Train the agent

```bash
python main.py                        # silent training (fast)
python main.py --show-every 100       # display every 100 episodes
python main.py --load                 # resume from model_best.pth
python main.py --episodes 10000       # custom episode count
```

### Evaluate a trained model

```bash
python main.py --eval                 # greedy evaluation, visual
```

### Run XAI analyses

```bash
python xai_qvalues.py                 # all Q-value plots
python xai_features.py                # all feature importance plots
python xai_activations.py --tsne      # t-SNE projection
python xai_shap.py --beeswarm         # SHAP beeswarm plot
python xai_shap.py                    # all SHAP plots
```

---

## ⚙️ Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GAMMA` | 0.95 | Discount factor |
| `LEARNING_RATE` | 1e-3 | Adam optimizer |
| `BATCH_SIZE` | 128 | Mini-batch size |
| `REPLAY_CAPACITY` | 100 000 | Replay buffer size |
| `EPS_START / END` | 1.0 → 0.01 | ε-greedy exploration range |
| `EPS_DECAY` | 0.9995 | Multiplicative decay per episode |
| `TARGET_UPDATE_FREQ` | 500 steps | Hard update of target network |

---

## 📖 Inspiration / Sources

Built entirely from scratch 😆 !

Code created by me 😎, Thibault GAREL — [Github](https://github.com/Thibault-GAREL)
