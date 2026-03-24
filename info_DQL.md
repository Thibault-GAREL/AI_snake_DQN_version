# 📊 Information DQL - Snake AI

## 🔍 Résumé XAI — Vue d'ensemble

L'analyse XAI de l'agent Snake (Double DQN, 16 entrées, 3 couches cachées 256→128→64) révèle une politique apprise fonctionnelle mais sous-optimale. Les analyses SHAP et de permutation d'importance confirment que les **distances aux murs dominent massivement** les décisions (top 5 : Mur W, NW, E, SW, N avec des importances SHAP > 0.4), tandis que les distances à la nourriture restent quasi-inexploitées sauf quand elles sont alignées. Le réseau souffre d'un **fort taux de neurones morts** (45 % en couche 1, 75 % en couche 2, 73 % en couche 3), signe d'une sur-paramétrisation, et les projections t-SNE/UMAP montrent une **représentation interne peu discriminante** entre situations de jeu. La politique apprise présente un **biais systématique vers l'action LEFT**, visible sur les heatmaps de Q-values et dans tous les waterfalls SHAP quelle que soit la situation. L'évolution temporelle des Q-values illustre clairement deux modes d'échec (boucle oscillante, boucle plate) et un mode réussite (Q-values variables, score 5), confirmant que l'agent a appris à survivre via l'évitement des murs mais n'a pas développé de stratégie active de recherche de nourriture.

---

## 🎯 Configuration d'Entraînement

### Temps d'entraînement

- **Nombre d'épisodes** : 5 000 épisodes
- **Étapes max par épisode** : 500 steps
- **Total d'interactions** : Jusqu'à 2 500 000 steps
- **Sauvegarde périodique** : Tous les 200 épisodes
- **Durée estimée** : Non chronométrée dans le code (dépend du matériel)
  - _Note_ : CPU : plusieurs heures | GPU (CUDA) : ~30-60 minutes

---

## 🧠 Architecture du Réseau Neuronal

### Nombre de neurones par couche

| Couche       | Type                          | Neurones | Description                                      |
| ------------ | ----------------------------- | -------- | ------------------------------------------------ |
| Input        | -                             | 16       | État : 8 distances murs + 8 distances nourriture |
| **Couche 1** | **Linear + BatchNorm + ReLU** | **256**  | Première couche cachée                           |
| **Couche 2** | **Linear + ReLU**             | **128**  | Deuxième couche cachée                           |
| **Couche 3** | **Linear + ReLU**             | **64**   | Troisième couche cachée                          |
| Output       | Linear                        | 4        | Q-values (1 par action)                          |

### Détails de l'architecture

```
Input (16-dim)
  ↓
Linear(16 → 256) + BatchNorm + ReLU
  ↓
Linear(256 → 128) + ReLU
  ↓
Linear(128 → 64) + ReLU
  ↓
Linear(64 → 4) [Q-values]
```

**Total de neurones actifs** : 256 + 128 + 64 = **448 neurones**

### Utilisation effective

- Couche 1 : **~54.3% de neurones actifs** (45.7% morts)
- Couche 2 : **~25% de neurones actifs** (75% morts)
- Couche 3 : **~26.6% de neurones actifs** (73.4% morts)

---

## 💾 Mémoire et Batch

### Replay Buffer (Expérience Replay)

| Paramètre           | Valeur  | Description                                 |
| ------------------- | ------- | ------------------------------------------- |
| **REPLAY_CAPACITY** | 100 000 | Taille max du buffer circulaire             |
| **MIN_REPLAY_SIZE** | 1 000   | Transitions minimum avant d'apprendre       |
| **BATCH_SIZE**      | 128     | Taille des mini-batches pour l'entraînement |

### Calcul mémoire estimé

- Chaque transition : ~40 bytes (état 16 float32 + action int + reward float + next_state 16 float32 + done bool)
- **Mémoire totale buffer** : 100 000 × 40 bytes ≈ **4 MB**

### Optimizer & Loss

- **Optimizer** : Adam (lr=1e-3, weight_decay=1e-5)
- **Loss Function** : SmoothL1Loss (Huber Loss) — robuste aux valeurs aberrantes

---

## 📈 Performance et Scores

### Score Moyen (Baseline)

- **Score moyen de référence** : **8.55** (meilleur agent)
- Chaque score représente le nombre de nourriture mangée
- Épisode max : 500 steps (limite temps)

### Scores Observés

| Situation                   | Score | Notes                                               |
| --------------------------- | ----- | --------------------------------------------------- |
| **Lors de l'apprentissage** | 0-13  | Variable selon convergence                          |
| **Episode 1 (XAI)**         | 0     | Agent bloqué dans une boucle                        |
| **Episode 2 (XAI)**         | 0     | Agent évite muerte mais ne trouve pas la nourriture |
| **Episode 3 (XAI)**         | 5     | Agent actif, trouve la nourriture                   |
| **GIF démonstration**       | 13    | **Score maximum enregistré**                        |
| **Baseline finale**         | 8.55  | Score moyen du meilleur modèle                      |

### Chute de performance (Permutation Features)

- Chaque feature perturbée : chute de **6.5 à 8.5 points**
- Features les plus critiques :
  1. `Dist. food W` (Ouest) — drop = 8.45
  2. Toutes distances nourriture (équilibrées)
  3. Distances murs (moins critiques)

---

## ⚙️ Hyperparamètres Clés

### Double DQN

- **Epsilon initial** : 1.0 (exploration max)
- **Epsilon final** : 0.01 (exploitation min)
- **Epsilon decay** : 0.9995 multiplicatif par épisode
- **Gamma (discount factor)** : 0.95
- **Target Network Update Freq** : Tous les 500 steps

### Récompenses (Reward Shaping)

| Action                         | Récompense |
| ------------------------------ | ---------- |
| Manger la nourriture           | +10.0      |
| Collision (mort)               | -10.0      |
| Se rapprocher de la nourriture | +0.5       |
| S'éloigner de la nourriture    | -0.5       |
| Chaque step                    | -0.01      |

---

## 📊 État Vectoriel d'Entrée (16 features)

### Distances aux obstacles (8 inputs)

Distances normalisées à la diagonale de la grille (13×14)

| #   | Direction | Feature            |
| --- | --------- | ------------------ |
| 0   | Nord      | `distance_bord_N`  |
| 1   | NE        | `distance_bord_NE` |
| 2   | Est       | `distance_bord_E`  |
| 3   | SE        | `distance_bord_SE` |
| 4   | Sud       | `distance_bord_S`  |
| 5   | SW        | `distance_bord_SW` |
| 6   | Ouest     | `distance_bord_W`  |
| 7   | NW        | `distance_bord_NW` |

### Distances à la nourriture (8 inputs)

Actives **seulement** si la nourriture est alignée dans cette direction (sinon = 0)

| #   | Direction | Feature            |
| --- | --------- | ------------------ |
| 8   | Nord      | `distance_food_N`  |
| 9   | NE        | `distance_food_NE` |
| 10  | Est       | `distance_food_E`  |
| 11  | SE        | `distance_food_SE` |
| 12  | Sud       | `distance_food_S`  |
| 13  | SW        | `distance_food_SW` |
| 14  | Ouest     | `distance_food_W`  |
| 15  | NW        | `distance_food_NW` |

---

## 🎓 Apprentissage et Convergence

### Dynamique par époque

| Phase           | Épisodes  | Comportement                              |
| --------------- | --------- | ----------------------------------------- |
| **Exploration** | 0-500     | Agent oscille au hasard, ε=1.0            |
| **Transition**  | 500-2000  | Premiers apprentissages, ε decay linéaire |
| **Convergence** | 2000-5000 | Stabilisation progressive, ε≈0.05-0.01    |

### Caractéristiques observées en XAI

- **Représentation interne** : Très dense, peu de distinction par situation
- **Dead neurons croissant** : 45% → 75% → 73% (L1 → L2 → L3)
- **Specialization** : Faible, sauf pour situations rares (Food alignée)
- **Policy dominante** : Biais fort vers l'action LEFT

---

## 🖥️ Matériel & Device

### Configuration

- **Device** : Automatique (CUDA si disponible, sinon CPU)
- **PyTorch Version** : 2.x
- **Support NVIDIA** : CUDA 11.8+ (optionnel)
- **RAM estimée** : ~500 MB pour l'agent seul

---

## 📝 Modèles Sauvegardés

```
model_best.pth       ← Meilleur score trouvé
model_final.pth      ← État après 5000 épisodes
model_best1.pth      ← Checkpoint alternatif
model_ep{N}.pth      ← Checkpoint périodiques (200, 400, 600, ..., 5000)
```

Chaque modèle stocke :

- État du réseau online
- État du réseau cible
- État de l'optimiseur
- Epsilon courant
- Nombre de steps
- Numéro d'épisode

---

## 🔬 Résumé XAI

### Importance des features (SHAP)

**Top 5 features les plus impactantes** :

1. `Dist. Mur Ouest (W)` — 0.503
2. `Dist. Mur NW` — 0.448
3. `Dist. Mur Est (E)` — 0.438
4. `Dist. Mur SW` — 0.431
5. `Dist. Mur N` — 0.427

**Least important** : Distances nourriture diagonales (< 0.01)

### Stratégie apprise

- Biais dominant : Tourner **LEFT**
- Évitement murs : Guidé par features de distance murs
- Recherche nourriture : Réactive seulement quand alignée

---

## 💡 Insights

### Efficacité neuronale

Seuls **25-55% des neurones** sont réellement utilisés → réseau sur-paramétré, opportunité d'optimisation

### Bottleneck

Le réseau n'a appris qu'une **policy locale** (éviter les murs, tourner left), pas de stratégie globale de recherche

### Features critiques

**Tous les 16 inputs sont nécessaires** — pas de redondance majeure détectée (permutation importance : drop = 6.5-8.5 pour chaque feature)
