"""
main.py — Boucle d'entraînement DQL pour Snake
================================================
Lance N épisodes, collecte les transitions, entraîne l'agent,
affiche les statistiques et sauvegarde le meilleur modèle.

Usage :
    python main.py            # entraînement complet
    python main.py --load     # reprend depuis model.pth
    python main.py --eval     # mode évaluation (pas d'entraînement)
"""

import sys
import argparse
import random
import math
from collections import deque

import numpy as np
import pygame

# ── Imports locaux ──────────────────────────────
from dql import DQNAgent, get_device, EPS_END

# ── Import de ton jeu (adapté pour le DQL) ──────
#    On importe uniquement les éléments nécessaires.
#    Les globals pygame (display, clock…) sont gérés ici.
import snake as game


# ═══════════════════════════════════════════════
#  Hyperparamètres d'entraînement
# ═══════════════════════════════════════════════
NUM_EPISODES      = 5_000      # Nombre total d'épisodes
MAX_STEPS         = 500        # Limite de steps par épisode (= stop_iteration de snake.py)
SAVE_EVERY        = 200        # Sauvegarde le modèle tous les N épisodes
PRINT_EVERY       = 50         # Log console tous les N épisodes
SHOW_EVERY        = 500        # Affichage visuel tous les N épisodes (0 = jamais)

# Récompenses
REWARD_FOOD       =  10.0      # Manger la nourriture
REWARD_DEATH      = -10.0      # Collision (mur ou corps)
REWARD_CLOSER     =   0.5      # Se rapprocher de la nourriture
REWARD_FARTHER    =  -0.5      # S'éloigner de la nourriture
REWARD_STEP       =  -0.01     # Légère pénalité par step (encourage l'efficacité)


# ═══════════════════════════════════════════════
#  Classe d'environnement (wrapper autour de snake.py)
# ═══════════════════════════════════════════════
class SnakeEnv:
    """
    Encapsule la logique de snake.py dans une interface gym-like :
        reset()  → state
        step(action) → next_state, reward, done, info
    """

    def __init__(self):
        self.width      = game.width
        self.height     = game.height
        self.rect_w     = game.rect_width
        self.rect_h     = game.rect_height

    def reset(self):
        """Réinitialise l'environnement et retourne l'état initial."""
        self.my_snake    = game.Manager_snake()
        first_segment    = game.Snake(5 * self.rect_w, 5 * self.rect_h)
        self.my_snake.add_snake(first_segment)
        self.food        = game.generated_food(self.my_snake)
        self.score       = 0
        self.iteration   = 0
        self._prev_dist  = self._manhattan_to_food()
        return self._get_state()

    def step(self, action: int):
        """
        Applique l'action, retourne (next_state, reward, done, info).
        Actions : 0=UP  1=RIGHT  2=DOWN  3=LEFT
        """
        self.iteration += 1

        # ── Mise à jour de la direction ──────────
        # Interdit le demi-tour (cohérent avec snake.py)
        direction_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        opposites     = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        new_dir       = direction_map[action]
        if new_dir != opposites.get(self.my_snake.direction, ""):
            self.my_snake.direction = new_dir

        # ── Mouvement ───────────────────────────
        ate_food_before = (
            self.my_snake.list_snake[0].x == self.food.x and
            self.my_snake.list_snake[0].y == self.food.y
        )

        # Simuler l'agrandissement avant le move si la tête est sur la nourriture
        # (dans ton snake.py l'agrandissement se fait avant move dans la boucle)
        if ate_food_before:
            extra = game.Snake(self.my_snake.list_snake[-1].x, self.my_snake.list_snake[-1].y)
            self.my_snake.add_snake(extra)
            self.food  = game.generated_food(self.my_snake)
            self.score += 1

        alive = self.my_snake.move()

        # ── Calcul de la récompense ──────────────
        reward = REWARD_STEP

        if not alive:
            reward  = REWARD_DEATH
            done    = True
        else:
            done = self.iteration >= MAX_STEPS

            # Nourriture mangée
            if ate_food_before:
                reward += REWARD_FOOD
            else:
                # Récompense de shaping : rapprochement/éloignement
                curr_dist = self._manhattan_to_food()
                if curr_dist < self._prev_dist:
                    reward += REWARD_CLOSER
                else:
                    reward += REWARD_FARTHER
                self._prev_dist = curr_dist

        next_state = self._get_state()
        info       = {"score": self.score, "iteration": self.iteration}
        return next_state, reward, done, info

    def render(self):
        """Dessine l'état courant (uniquement si pygame est initialisé)."""
        if not game.show:
            return
        game.display.fill(game.BLACK)
        pygame.draw.rect(
            game.display, game.RED,
            (self.food.x, self.food.y, self.rect_w, self.rect_h)
        )
        self.my_snake.draw_snake()
        game.print_display(f"Score : {self.score}", game.WHITE, {'topleft': (10, 10)})
        pygame.display.update()
        game.clock.tick(game.vitesse)

    # ── Calcul de l'état ─────────────────────────
    def _get_state(self) -> list:
        """
        Construit le vecteur d'état normalisé de 16 features.
        Correspond exactement aux features utilisées dans ta boucle snake.py.
        Les distances sont normalisées par la diagonale de la grille
        pour que toutes les entrées soient dans [0, 1].
        """
        diag = math.sqrt(self.width**2 + self.height**2)

        raw = [
            # 8 distances obstacles (bords + corps du serpent) dans 8 directions
            game.distance_bord_north(self.my_snake),
            game.distance_bord_north_est(self.my_snake),
            game.distance_bord_est(self.my_snake),
            game.distance_bord_south_est(self.my_snake),
            game.distance_bord_south(self.my_snake),
            game.distance_bord_south_west(self.my_snake),
            game.distance_bord_west(self.my_snake),
            game.distance_bord_north_west(self.my_snake),

            # 8 distances nourriture dans 8 directions (0 si pas alignée)
            game.distance_food_north(self.my_snake, self.food),
            game.distance_food_north_est(self.my_snake, self.food),
            game.distance_food_est(self.my_snake, self.food),
            game.distance_food_south_est(self.my_snake, self.food),
            game.distance_food_south(self.my_snake, self.food),
            game.distance_food_south_west(self.my_snake, self.food),
            game.distance_food_west(self.my_snake, self.food),
            game.distance_food_north_west(self.my_snake, self.food),
        ]

        # Normalisation [0, 1]
        return [v / diag for v in raw]

    def _manhattan_to_food(self) -> float:
        """Distance de Manhattan entre la tête et la nourriture."""
        head = self.my_snake.list_snake[0]
        return abs(head.x - self.food.x) + abs(head.y - self.food.y)


# ═══════════════════════════════════════════════
#  Boucle d'entraînement principale
# ═══════════════════════════════════════════════
def train(agent: DQNAgent, env: SnakeEnv, num_episodes: int, show_every: int):
    """
    Boucle DQL principale.
    Retourne l'historique des scores.
    """
    scores_history  = []
    losses_history  = []
    best_score      = -float("inf")
    recent_scores   = deque(maxlen=100)   # Moyenne glissante sur 100 épisodes

    print(f"\n{'═'*55}")
    print(f"  Démarrage entraînement — {num_episodes} épisodes")
    print(f"  Device : {agent.device}")
    print(f"  Epsilon initial : {agent.epsilon:.3f}")
    print(f"{'═'*55}\n")

    for episode in range(1, num_episodes + 1):

        # ── Activation de l'affichage ponctuellement ──
        render_this_ep = show_every > 0 and (episode % show_every == 0)
        game.show = render_this_ep and pygame.get_init()

        state = env.reset()
        episode_reward = 0.0
        episode_losses = []

        done = False
        while not done:
            # Gestion des événements pygame (fermeture fenêtre)
            if game.show:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save("model_interrupt.pth")
                        pygame.quit()
                        sys.exit()

            # Sélection de l'action
            action = agent.select_action(state)

            # Step dans l'environnement
            next_state, reward, done, info = env.step(action)

            # Stockage dans le replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Apprentissage
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            # Rendu visuel si activé
            if game.show:
                env.render()

            episode_reward += reward
            state = next_state

        # ── Fin d'épisode ────────────────────────────
        score = info["score"]
        agent.decay_epsilon()
        scores_history.append(score)
        recent_scores.append(score)
        avg_score = np.mean(recent_scores)
        avg_loss  = np.mean(episode_losses) if episode_losses else 0.0
        losses_history.append(avg_loss)

        # Sauvegarde du meilleur modèle
        if score > best_score:
            best_score = score
            agent.save("model_best.pth")

        # Sauvegarde périodique
        if episode % SAVE_EVERY == 0:
            agent.save(f"model_ep{episode}.pth")

        # Log console
        if episode % PRINT_EVERY == 0:
            print(
                f"Ep {episode:>5}/{num_episodes} | "
                f"Score {score:>3} | "
                f"Moy-100 {avg_score:>5.2f} | "
                f"Meilleur {best_score:>3} | "
                f"ε {agent.epsilon:.4f} | "
                f"Loss {avg_loss:.5f} | "
                f"Buffer {len(agent.replay_buffer):>6}"
            )

    print(f"\n[TRAIN] Terminé. Meilleur score : {best_score}")
    return scores_history, losses_history


# ═══════════════════════════════════════════════
#  Mode évaluation (pas d'entraînement)
# ═══════════════════════════════════════════════
def evaluate(agent: DQNAgent, env: SnakeEnv, num_episodes: int = 10):
    """Lance N épisodes en mode greedy pur (ε=0) et affiche les résultats."""
    agent.epsilon = 0.0
    game.show     = True

    scores = []
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done  = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.select_action(state)
            state, _, done, info = env.step(action)
            env.render()

        print(f"[EVAL] Épisode {ep} — Score : {info['score']}")
        scores.append(info['score'])

    print(f"\n[EVAL] Score moyen : {np.mean(scores):.2f}  |  Max : {max(scores)}")


# ═══════════════════════════════════════════════
#  Point d'entrée
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Entraînement DQL pour Snake")
    parser.add_argument("--load",     action="store_true", help="Charger model.pth avant l'entraînement")
    parser.add_argument("--eval",     action="store_true", help="Mode évaluation uniquement")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Nombre d'épisodes")
    parser.add_argument("--show-every", type=int, default=SHOW_EVERY,
                        help="Afficher visuellement tous les N épisodes (0 = jamais)")
    args = parser.parse_args()

    # ── Initialisation de pygame ──────────────────
    # On force l'affichage en mode éval, on le coupe en entraînement rapide
    if args.eval or args.show_every > 0:
        game.show = True
        if not pygame.get_init():
            pygame.init()
            game.display  = pygame.display.set_mode((game.width, int(game.height)))
            pygame.display.set_caption("Snake — DQL")
            game.clock    = pygame.time.Clock()
            game.fonttype = pygame.font.SysFont(None, 30)
    else:
        game.show = False
        game.display = None
        if not pygame.get_init():
            pygame.init()   # Nécessaire pour certaines opérations internes

    # ── Création de l'agent et de l'environnement ──
    device = get_device()
    agent  = DQNAgent(device=device)
    env    = SnakeEnv()

    if args.load or args.eval:
        try:
            agent.load("model_best1.pth")
        except FileNotFoundError:
            print("[WARN] model_best.pth introuvable — démarrage à zéro.")

    # ── Lancement ─────────────────────────────────
    if args.eval:
        evaluate(agent, env, num_episodes=20)
    else:
        train(agent, env, num_episodes=args.episodes, show_every=args.show_every)
        agent.save("model_final.pth")

    if pygame.get_init():
        pygame.quit()


if __name__ == "__main__":
    main()


# python main.py                    # entraînement silencieux (rapide)
# python main.py --show-every 100  # affiche tous les 100 épisodes
# python main.py --load            # reprend depuis model_best.pth
# python main.py --eval            # évaluation visuelle pure