import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

import os

# import snake

input_dim = 16 #state dim : 8 bords, 8 foods

nb_loop_train = 60 * 6000 # 6000 = 1 min

nb_loop_train += 1

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Réseau tout simple : état -> couches cachées -> valeurs Q
        self.fc1 = nn.Linear(state_dim,32)
        self.fc2 = nn.Linear(32, action_dim)


    def forward(self, x):
        # Passage avant (calcul des Q-values)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    # def __init__(self, state_dim, action_dim):
    #     super(QNetwork, self).__init__()
    #     # Réseau tout simple : état -> couches cachées -> valeurs Q
    #     self.fc1 = nn.Linear(state_dim, 64)
    #     self.fc2 = nn.Linear(64, 32)
    #     self.fc3 = nn.Linear(32, 16)
    #     self.fc4 = nn.Linear(16, action_dim)
    #
    # def forward(self, x):
    #     # Passage avant (calcul des Q-values)
    #     x = torch.relu(self.fc1(x))
    #     x = torch.relu(self.fc2(x))
    #     x = torch.relu(self.fc3(x))
    #     return self.fc4(x)


# -------- Replay buffer (mémoire d’expériences) --------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done): # C'est une liste de tuple avec dans le tuple tout les imputs
        # On stocke un tuple d’expérience
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # On tire un batch aléatoire
        return random.sample(self.buffer, batch_size)

# -------- Agent DQN --------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-2)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10000, min_lr=1e-5)

        # Copie initiale du réseau vers le target network
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.gamma = 0.99       # facteur de discount
        self.epsilon = 1.0      # exploration initiale
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        self.replay_buffer = ReplayBuffer(10000)

    def select_action(self, state, action_dim):
        # Choix epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # on rajoute une dimension pour pas que ça coince
            q_values = self.q_net(state) #ici, on call le réseau : on fait tout bien + forward()
            return q_values.argmax().item() #on prend la plus grand valeur (on prend donc l'action avec le meilleur reward)

    def train_step(self, batch_size):
        # Vérifie qu’on a assez d’expériences
        if len(self.replay_buffer.buffer) < batch_size:
            return

        # Échantillonnage
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Conversion en tenseurs PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-values courantes
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q-targets avec le réseau cible
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)

        # Perte
        loss = (q_values - target).pow(2).mean()

        # Rétropropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step(loss)

        # Mise à jour epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        # On recopie les poids du réseau principal vers le target
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, filepath):
        """Sauvegarde le modèle principal et le target network"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load_model(self, filepath):
        """Charge un modèle sauvegardé"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Modèle chargé depuis {filepath}")
        else:
            print(f"Fichier {filepath} non trouvé")

print("CUDA disponible :", torch.cuda.is_available())
print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun")