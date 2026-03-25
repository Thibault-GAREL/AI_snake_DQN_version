"""
dql.py — Agent Deep Q-Learning pour Snake
==========================================
Architecture : DQN avec Double DQN + Experience Replay + Target Network
Input  : vecteur d'état à 16 features (distances murs + nourriture, 8 directions)
Output : Q-values pour 4 actions (UP=0, RIGHT=1, DOWN=2, LEFT=3)
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Hyperparamètres
# ─────────────────────────────────────────────
STATE_DIM    = 16       # 8 distances bords/corps + 8 distances nourriture
ACTION_DIM   = 4        # UP, RIGHT, DOWN, LEFT

# Réseau
HIDDEN_1     = 256
HIDDEN_2     = 128
HIDDEN_3     = 64

# Entraînement
LEARNING_RATE      = 1e-3       # Adam lr — bon compromis stabilité/vitesse pour DQN
GAMMA              = 0.95       # Discount factor — valeur légèrement < 1 car horizon court
BATCH_SIZE         = 128        # Taille des mini-batches
REPLAY_CAPACITY    = 100_000    # Taille du replay buffer
MIN_REPLAY_SIZE    = 1_000      # Nombre de transitions avant de commencer à apprendre
TARGET_UPDATE_FREQ = 500        # Fréquence (en steps) de mise à jour du réseau cible

# Exploration ε-greedy
EPS_START  = 1.0        # Exploration maximale au début
EPS_END    = 0.01       # Exploration minimale
EPS_DECAY  = 0.9995     # Décroissance multiplicative par épisode


# ─────────────────────────────────────────────
#  Sélection automatique du device (CUDA / CPU)
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    """Retourne le meilleur device disponible."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DQL] CUDA disponible — GPU : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[DQL] CUDA non disponible — entraînement sur CPU")
    return device


# ─────────────────────────────────────────────
#  Réseau de neurones
# ─────────────────────────────────────────────
class DQNetwork(nn.Module):
    """
    Réseau fully-connected 3 couches cachées.
    Batch Normalization sur la 1ère couche pour stabiliser l'entraînement
    sur des inputs de magnitudes très différentes (distances en pixels).
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super(DQNetwork, self).__init__()

        self.net = nn.Sequential(
            # Couche d'entrée
            nn.Linear(state_dim, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.ReLU(),

            # Couche cachée 1
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),

            # Couche cachée 2
            nn.Linear(HIDDEN_2, HIDDEN_3),
            nn.ReLU(),

            # Couche de sortie — Q-values brutes (pas de softmax : on maximise)
            nn.Linear(HIDDEN_3, action_dim),
        )

        # Initialisation des poids (He/Kaiming pour ReLU)
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  Experience Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    """
    Buffer circulaire stockant les transitions (s, a, r, s', done).
    Utilise numpy pour un stockage compact et un sampling rapide.
    """

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        """Ajoute une transition au buffer."""
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int = BATCH_SIZE):
        """Retourne un batch aléatoire sous forme de tenseurs."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────
#  Agent DQN (Double DQN)
# ─────────────────────────────────────────────
class DQNAgent:
    """
    Agent implémentant Double DQN :
      - online_net  : réseau principal, mis à jour à chaque step
      - target_net  : réseau cible, mis à jour périodiquement (copie hard)
    Double DQN réduit la surestimation des Q-values en découplant
    la sélection d'action (online) et l'évaluation (target).
    """

    def __init__(
        self,
        state_dim:   int   = STATE_DIM,
        action_dim:  int   = ACTION_DIM,
        device:      torch.device | None = None,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = device if device is not None else get_device()

        # Réseaux
        self.online_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # Le réseau cible est toujours en mode éval

        # Optimiseur & loss
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5,    # L2 regularisation légère
        )
        # Huber loss : moins sensible aux outliers que MSE
        self.criterion = nn.SmoothL1Loss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

        # Compteurs
        self.epsilon    = EPS_START
        self.steps_done = 0     # Total de steps (pour update target)
        self.episode    = 0     # Numéro d'épisode

    # ── Sélection d'action ──────────────────────
    def select_action(self, state) -> int:
        """
        Politique ε-greedy :
          - avec proba ε → action aléatoire (exploration)
          - sinon         → argmax Q(s, ·) (exploitation)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)   # shape [1, STATE_DIM]

        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_tensor)   # [1, ACTION_DIM]
        self.online_net.train()

        return q_values.argmax(dim=1).item()

    # ── Entraînement sur un mini-batch ──────────
    def learn(self) -> float | None:
        """
        Effectue une étape de gradient sur un mini-batch.
        Retourne la loss (float) ou None si le buffer est trop petit.
        """
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Conversion en tenseurs GPU
        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device)
        rewards_t     = torch.tensor(rewards,     device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device)

        # ── Q-values courants : Q_online(s, a) ──
        q_current = self.online_net(states_t)                     # [B, 4]
        q_current = q_current.gather(1, actions_t.unsqueeze(1))   # [B, 1]

        # ── Double DQN : cible ──────────────────
        # 1) online_net sélectionne la meilleure action dans s'
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)  # [B, 1]
            # 2) target_net évalue la Q-value de cette action
            q_next = self.target_net(next_states_t).gather(1, next_actions)            # [B, 1]
            q_target = rewards_t.unsqueeze(1) + GAMMA * q_next * (1.0 - dones_t.unsqueeze(1))

        # ── Mise à jour ──────────────────────────
        loss = self.criterion(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — évite les explosions de gradient
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Mise à jour du réseau cible ──────────
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ── Décroissance epsilon ─────────────────────
    def decay_epsilon(self):
        """À appeler une fois par épisode."""
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        self.episode += 1

    # ── Sauvegarde / chargement ──────────────────
    def save(self, path: str = "models/model.pth"):
        """Sauvegarde le modèle entraîné."""
        import os; os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_state_dict":  self.online_net.state_dict(),
            "target_state_dict":  self.target_net.state_dict(),
            "optimizer_state":    self.optimizer.state_dict(),
            "epsilon":            self.epsilon,
            "steps_done":         self.steps_done,
            "episode":            self.episode,
        }, path)
        print(f"[DQL] Modèle sauvegardé → {path}")

    def load(self, path: str = "models/model.pth"):
        """Charge un modèle sauvegardé."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon    = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
        self.episode    = checkpoint["episode"]
        print(f"[DQL] Modèle chargé ← {path}  (épisode {self.episode})")
