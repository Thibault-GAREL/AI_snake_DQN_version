"""
main.py — Boucle d'entraînement DQL pour Snake
================================================
Lance N épisodes, collecte les transitions, entraîne l'agent,
affiche les statistiques et sauvegarde le meilleur modèle.

Usage :
    python main.py                     # entraînement complet
    python main.py --load              # reprend depuis model_best.pth
    python main.py --eval              # mode évaluation (pas d'entraînement)
    python main.py --episodes 10000    # nombre d'épisodes
    python main.py --show-every 1000   # affichage tous les N épisodes

Changements v2 (2026-03-30) :
  - _get_state() : 16 → 28 features standardisées (voir input.md)
  - Stagnation limit : fin d'épisode si > STAGNATION_LIMIT steps sans manger
  - TrainingLogger : CSV par épisode + JSON résumé + PNG courbe d'apprentissage
  - Timing : durée totale d'entraînement enregistrée dans le résumé JSON
  - Structure de sortie : models/ et results/ avec convention de nommage
"""

import sys
import os
import argparse
import random
import math
import json
import time
import csv
from collections import deque
from datetime import date

import numpy as np
import pygame
import matplotlib
matplotlib.use("Agg")   # backend sans fenêtre pour la sauvegarde PNG
import matplotlib.pyplot as plt

# ── Imports locaux ──────────────────────────────
from dql import (
    DQNAgent, get_device, EPS_END,
    STATE_DIM, LEARNING_RATE, GAMMA, BATCH_SIZE,
    REPLAY_CAPACITY, TARGET_UPDATE_FREQ, EPS_START, EPS_DECAY,
)
import snake as game


# ═══════════════════════════════════════════════
#  Hyperparamètres d'entraînement
# ═══════════════════════════════════════════════
NUM_EPISODES      = 10_000     # 10k épisodes (5k insuffisant avec 28 features)
MAX_STEPS         = 500        # Limite de steps par épisode
STAGNATION_LIMIT  = 200        # Fin d'épisode si > N steps sans manger
SAVE_EVERY        = 500        # Sauvegarde checkpoint tous les N épisodes
PRINT_EVERY       = 100        # Log console tous les N épisodes
SHOW_EVERY        = 1_000      # Affichage visuel tous les N épisodes (0 = jamais)

# Récompenses
REWARD_FOOD       =  10.0      # Manger la nourriture
REWARD_DEATH      = -10.0      # Collision (mur ou corps) ou stagnation
REWARD_CLOSER     =   0.5      # Se rapprocher de la nourriture
REWARD_FARTHER    =  -0.5      # S'éloigner de la nourriture
REWARD_STEP       =  -0.01     # Légère pénalité par step (encourage l'efficacité)

# Nommage du run
MODEL_NAME        = "dqn-28feat"


# ═══════════════════════════════════════════════
#  Logger d'entraînement
# ═══════════════════════════════════════════════
class TrainingLogger:
    """
    Enregistre les métriques d'entraînement dans :
      - metrics.csv  : une ligne par épisode
      - summary.json : hyperparamètres + résultats finals + durée
      - training_curve.png : courbe de score lissée
    """

    def __init__(self, results_dir: str, hyperparams: dict):
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir  = results_dir
        self.hyperparams  = hyperparams
        self.start_time   = time.time()

        # CSV per-episode
        csv_path = os.path.join(results_dir, "metrics.csv")
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["episode", "score", "avg_100", "best_score", "epsilon",
             "avg_loss", "buffer_size", "elapsed_s"]
        )

        self._scores = []

    def log_episode(
        self,
        episode: int,
        score: int,
        avg_100: float,
        best_score: int,
        epsilon: float,
        avg_loss: float,
        buffer_size: int,
    ):
        elapsed = round(time.time() - self.start_time, 1)
        self._csv_writer.writerow(
            [episode, score, round(avg_100, 4), best_score,
             round(epsilon, 6), round(avg_loss, 6), buffer_size, elapsed]
        )
        self._csv_file.flush()
        self._scores.append(score)

    def finalize(self) -> float:
        """Ferme le CSV, sauvegarde le JSON et la courbe PNG. Retourne la durée en secondes."""
        self._csv_file.close()

        duration_s = time.time() - self.start_time

        # ── Résumé JSON ──────────────────────────
        summary = {
            **self.hyperparams,
            "training_duration_s":   round(duration_s, 1),
            "training_duration_min": round(duration_s / 60, 2),
            "total_episodes":        len(self._scores),
            "final_mean_100":        round(float(np.mean(self._scores[-100:])), 2) if self._scores else 0,
            "final_mean_all":        round(float(np.mean(self._scores)), 2) if self._scores else 0,
            "final_best_score":      int(max(self._scores)) if self._scores else 0,
        }
        json_path = os.path.join(self.results_dir, "summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[LOG] Résumé sauvegardé → {json_path}")

        # ── Courbe PNG ───────────────────────────
        self._save_training_curve()

        return duration_s

    def _save_training_curve(self):
        scores = np.array(self._scores, dtype=float)
        episodes = np.arange(1, len(scores) + 1)

        window = min(100, len(scores))
        smoothed = np.convolve(scores, np.ones(window) / window, mode="valid")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(episodes, scores, alpha=0.25, color="steelblue", linewidth=0.8, label="Score brut")
        ax.plot(
            episodes[window - 1:], smoothed,
            color="steelblue", linewidth=2.0, label=f"Moyenne glissante ({window} ep)"
        )
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Score")
        ax.set_title(f"Courbe d'apprentissage — {MODEL_NAME}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        png_path = os.path.join(self.results_dir, "training_curve.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[LOG] Courbe sauvegardée → {png_path}")


# ═══════════════════════════════════════════════
#  Classe d'environnement (wrapper autour de snake.py)
# ═══════════════════════════════════════════════
class SnakeEnv:
    """
    Encapsule la logique de snake.py dans une interface gym-like :
        reset()  → state (28 features)
        step(action) → next_state, reward, done, info
    """

    def __init__(self):
        self.width      = game.width
        self.height     = game.height
        self.rect_w     = game.rect_width
        self.rect_h     = game.rect_height

        # Nombre total de cellules (grille 16×8 = 128)
        self._max_cells = int(self.width // self.rect_w) * int(self.height // self.rect_h)

    def reset(self):
        """Réinitialise l'environnement et retourne l'état initial."""
        self.my_snake        = game.Manager_snake()
        first_segment        = game.Snake(5 * self.rect_w, 5 * self.rect_h)
        self.my_snake.add_snake(first_segment)
        self.food            = game.generated_food(self.my_snake)
        self.score           = 0
        self.iteration       = 0
        self.steps_since_food = 0
        self._prev_dist      = self._manhattan_to_food()
        return self._get_state()

    def step(self, action: int):
        """
        Applique l'action, retourne (next_state, reward, done, info).
        Actions : 0=UP  1=RIGHT  2=DOWN  3=LEFT
        """
        self.iteration += 1

        # ── Mise à jour de la direction ──────────
        direction_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        opposites     = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        new_dir       = direction_map[action]
        if new_dir != opposites.get(self.my_snake.direction, ""):
            self.my_snake.direction = new_dir

        # ── Détection nourriture avant move ─────
        ate_food_before = (
            self.my_snake.list_snake[0].x == self.food.x and
            self.my_snake.list_snake[0].y == self.food.y
        )

        if ate_food_before:
            extra = game.Snake(self.my_snake.list_snake[-1].x, self.my_snake.list_snake[-1].y)
            self.my_snake.add_snake(extra)
            self.food              = game.generated_food(self.my_snake)
            self.score             += 1
            self.steps_since_food  = 0
        else:
            self.steps_since_food += 1

        alive = self.my_snake.move()

        # ── Calcul de la récompense ──────────────
        reward = REWARD_STEP

        if not alive:
            reward = REWARD_DEATH
            done   = True
        elif self.steps_since_food >= STAGNATION_LIMIT:
            # Stagnation : trop longtemps sans manger → fin d'épisode
            reward = REWARD_DEATH
            done   = True
        else:
            done = self.iteration >= MAX_STEPS

            if ate_food_before:
                reward += REWARD_FOOD
            else:
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

    # ── Calcul de l'état — 28 features ──────────
    def _get_state(self) -> list:
        """
        Construit le vecteur d'état standardisé à 28 features (input.md).

        Groupe 1  [0-7]  : distances obstacle 8 directions, normalisées / diag
        Groupe 2  [8-15] : distances nourriture 8 directions (sparse), normalisées / diag
        Groupe 3  [16-17]: food_delta_x, food_delta_y — continus, toujours non-nuls
        Groupe 4  [18-21]: danger immédiat N/E/S/W — binaire {0, 1}
        Groupe 5  [22-25]: direction courante one-hot — binaire {0, 1}
        Groupe 6  [26-27]: length_norm, urgency — [0, 1]
        """
        diag = math.sqrt(self.width**2 + self.height**2)
        head = self.my_snake.list_snake[0]

        # ── Groupe 1 : distances obstacles ───────
        danger_distances = [
            game.distance_bord_north(self.my_snake)      / diag,
            game.distance_bord_north_est(self.my_snake)  / diag,
            game.distance_bord_est(self.my_snake)        / diag,
            game.distance_bord_south_est(self.my_snake)  / diag,
            game.distance_bord_south(self.my_snake)      / diag,
            game.distance_bord_south_west(self.my_snake) / diag,
            game.distance_bord_west(self.my_snake)       / diag,
            game.distance_bord_north_west(self.my_snake) / diag,
        ]

        # ── Groupe 2 : distances nourriture (sparse) ──
        food_distances = [
            game.distance_food_north(self.my_snake, self.food)      / diag,
            game.distance_food_north_est(self.my_snake, self.food)  / diag,
            game.distance_food_est(self.my_snake, self.food)        / diag,
            game.distance_food_south_est(self.my_snake, self.food)  / diag,
            game.distance_food_south(self.my_snake, self.food)      / diag,
            game.distance_food_south_west(self.my_snake, self.food) / diag,
            game.distance_food_west(self.my_snake, self.food)       / diag,
            game.distance_food_north_west(self.my_snake, self.food) / diag,
        ]

        # ── Groupe 3 : food delta continu ────────
        # Toujours non-nul — résout l'angle mort des features sparse (~80% à 0)
        food_delta_x = (self.food.x - head.x) / self.width
        food_delta_y = (self.food.y - head.y) / self.height

        # ── Groupe 4 : danger immédiat N/E/S/W ───
        # Cohérent avec snake.py : collision = mur OU corps (tail inclus)
        body_pos = {(s.x, s.y) for s in self.my_snake.list_snake}

        danger_N = 1.0 if (
            head.y - self.rect_h < 0 or
            (head.x, head.y - self.rect_h) in body_pos
        ) else 0.0

        danger_E = 1.0 if (
            head.x + self.rect_w >= self.width or
            (head.x + self.rect_w, head.y) in body_pos
        ) else 0.0

        danger_S = 1.0 if (
            head.y + self.rect_h >= self.height or
            (head.x, head.y + self.rect_h) in body_pos
        ) else 0.0

        danger_W = 1.0 if (
            head.x - self.rect_w < 0 or
            (head.x - self.rect_w, head.y) in body_pos
        ) else 0.0

        # ── Groupe 5 : direction one-hot ─────────
        # Absolu (N/E/S/W) — stable quelle que soit l'orientation du serpent
        direction = self.my_snake.direction
        dir_UP    = 1.0 if direction == "UP"    else 0.0
        dir_RIGHT = 1.0 if direction == "RIGHT" else 0.0
        dir_DOWN  = 1.0 if direction == "DOWN"  else 0.0
        dir_LEFT  = 1.0 if direction == "LEFT"  else 0.0

        # ── Groupe 6 : contexte temporel ─────────
        length_norm = (self.my_snake.lenght - 1) / (self._max_cells - 1)
        urgency     = min(self.steps_since_food / MAX_STEPS, 1.0)

        return (
            danger_distances             # [0-7]
            + food_distances             # [8-15]
            + [food_delta_x, food_delta_y]  # [16-17]
            + [danger_N, danger_E, danger_S, danger_W]  # [18-21]
            + [dir_UP, dir_RIGHT, dir_DOWN, dir_LEFT]   # [22-25]
            + [length_norm, urgency]     # [26-27]
        )

    def _manhattan_to_food(self) -> float:
        """Distance de Manhattan entre la tête et la nourriture."""
        head = self.my_snake.list_snake[0]
        return abs(head.x - self.food.x) + abs(head.y - self.food.y)


# ═══════════════════════════════════════════════
#  Boucle d'entraînement principale
# ═══════════════════════════════════════════════
def train(
    agent: DQNAgent,
    env: SnakeEnv,
    num_episodes: int,
    show_every: int,
    models_dir: str,
    logger: TrainingLogger,
):
    """Boucle DQL principale. Retourne l'historique des scores."""
    scores_history  = []
    losses_history  = []
    best_score      = -float("inf")
    recent_scores   = deque(maxlen=100)

    print(f"\n{'═'*60}")
    print(f"  Démarrage entraînement — {num_episodes} épisodes")
    print(f"  Device     : {agent.device}")
    print(f"  State dim  : {agent.state_dim} features")
    print(f"  Modèles    : {models_dir}/")
    print(f"{'═'*60}\n")

    for episode in range(1, num_episodes + 1):

        render_this_ep = show_every > 0 and (episode % show_every == 0)
        game.show = render_this_ep and pygame.get_init()

        state = env.reset()
        episode_reward = 0.0
        episode_losses = []

        done = False
        while not done:
            if game.show:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save(os.path.join(models_dir, "model_interrupt.pth"))
                        pygame.quit()
                        sys.exit()

            action     = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            if game.show:
                env.render()

            episode_reward += reward
            state = next_state

        # ── Fin d'épisode ────────────────────────
        score     = info["score"]
        agent.decay_epsilon()
        scores_history.append(score)
        recent_scores.append(score)
        avg_score = float(np.mean(recent_scores))
        avg_loss  = float(np.mean(episode_losses)) if episode_losses else 0.0
        losses_history.append(avg_loss)

        if score > best_score:
            best_score = score
            agent.save(os.path.join(models_dir, "model_best.pth"))

        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(models_dir, f"model_ep{episode}.pth"))

        # Logger CSV
        logger.log_episode(
            episode, score, avg_score, best_score,
            agent.epsilon, avg_loss, len(agent.replay_buffer),
        )

        if episode % PRINT_EVERY == 0:
            print(
                f"Ep {episode:>6}/{num_episodes} | "
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
def evaluate(agent: DQNAgent, env: SnakeEnv, num_episodes: int = 20):
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
#  Sélection automatique du meilleur modèle
# ═══════════════════════════════════════════════
def find_best_model(base_dir: str = ".") -> str | None:
    """
    Cherche le meilleur model_best.pth en comparant les summary.json.

    Critères (par ordre de priorité) :
      1. final_best_score le plus élevé
      2. final_mean_100 le plus élevé (tie-breaker)
      3. Date de modification du fichier .pth (fallback si pas de summary.json)

    Retourne le chemin absolu vers le model_best.pth retenu, ou None.
    """
    models_root  = os.path.join(base_dir, "models")
    results_root = os.path.join(base_dir, "results")

    candidates = []  # (final_best_score, final_mean_100, mtime, path)

    # ── Runs avec summary.json ────────────────
    if os.path.isdir(results_root):
        for run_name in os.listdir(results_root):
            summary_path = os.path.join(results_root, run_name, "summary.json")
            model_path   = os.path.join(models_root,  run_name, "model_best.pth")
            if not os.path.isfile(summary_path) or not os.path.isfile(model_path):
                continue
            try:
                with open(summary_path, encoding="utf-8") as f:
                    s = json.load(f)
                best_score = s.get("final_best_score", -1)
                mean_100   = s.get("final_mean_100",   -1.0)
                mtime      = os.path.getmtime(model_path)
                candidates.append((best_score, mean_100, mtime, model_path))
            except (json.JSONDecodeError, OSError):
                continue

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        chosen = candidates[0]
        print(f"[AUTO] Meilleur modèle sélectionné : {chosen[3]}")
        print(f"       best_score={chosen[0]}  mean_100={chosen[1]:.2f}")
        return chosen[3]

    # ── Fallback : model_best.pth le plus récent ──
    best_path, best_mtime = None, -1.0
    if os.path.isdir(models_root):
        for run_name in os.listdir(models_root):
            model_path = os.path.join(models_root, run_name, "model_best.pth")
            if os.path.isfile(model_path):
                mtime = os.path.getmtime(model_path)
                if mtime > best_mtime:
                    best_mtime, best_path = mtime, model_path

    if best_path:
        print(f"[AUTO] Aucun summary.json trouvé — modèle le plus récent : {best_path}")
        return best_path

    return None


# ═══════════════════════════════════════════════
#  Point d'entrée
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Entraînement DQL pour Snake — 28 features")
    parser.add_argument("--load",       action="store_true",      help="Charger model_best.pth avant l'entraînement")
    parser.add_argument("--eval",       action="store_true",      help="Mode évaluation uniquement")
    parser.add_argument("--episodes",   type=int, default=NUM_EPISODES, help="Nombre d'épisodes")
    parser.add_argument("--show-every", type=int, default=SHOW_EVERY,  help="Afficher tous les N épisodes (0=jamais)")
    parser.add_argument("--run",        type=int, default=1,      help="Numéro de run (pour les dossiers)")
    args = parser.parse_args()

    # ── Dossiers du run ───────────────────────────
    today      = date.today().isoformat()
    run_suffix = f"{MODEL_NAME}_run-{args.run:02d}_date-{today}"
    models_dir  = os.path.join("models",  run_suffix)
    results_dir = os.path.join("results", run_suffix)
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── Initialisation pygame ─────────────────────
    if args.eval or args.show_every > 0:
        game.show = True
        if not pygame.get_init():
            pygame.init()
            game.display  = pygame.display.set_mode((game.width, int(game.height)))
            pygame.display.set_caption("Snake — DQL 28feat")
            game.clock    = pygame.time.Clock()
            game.fonttype = pygame.font.SysFont(None, 30)
    else:
        game.show = False
        game.display = None
        if not pygame.get_init():
            pygame.init()

    # ── Création de l'agent ───────────────────────
    device = get_device()
    agent  = DQNAgent(state_dim=STATE_DIM, device=device)
    env    = SnakeEnv()

    if args.load or args.eval:
        model_path = find_best_model()
        if model_path:
            agent.load(model_path)
        else:
            print("[WARN] Aucun model_best.pth trouvé dans models/ — démarrage à zéro.")

    # ── Lancement ─────────────────────────────────
    if args.eval:
        evaluate(agent, env, num_episodes=20)
    else:
        # Hyperparamètres pour le logger
        hyperparams = {
            "model_name":        MODEL_NAME,
            "run":               args.run,
            "state_dim":         STATE_DIM,
            "architecture":      f"{STATE_DIM}→{256}→{256}→{128}→4 (LayerNorm)",
            "learning_rate":     LEARNING_RATE,
            "gamma":             GAMMA,
            "batch_size":        BATCH_SIZE,
            "replay_capacity":   REPLAY_CAPACITY,
            "target_update_freq": TARGET_UPDATE_FREQ,
            "eps_start":         EPS_START,
            "eps_end":           EPS_END,
            "eps_decay":         EPS_DECAY,
            "num_episodes":      args.episodes,
            "max_steps":         MAX_STEPS,
            "stagnation_limit":  STAGNATION_LIMIT,
            "reward_food":       REWARD_FOOD,
            "reward_death":      REWARD_DEATH,
            "reward_closer":     REWARD_CLOSER,
            "reward_farther":    REWARD_FARTHER,
            "reward_step":       REWARD_STEP,
        }

        logger = TrainingLogger(results_dir, hyperparams)

        scores, losses = train(
            agent, env,
            num_episodes=args.episodes,
            show_every=args.show_every,
            models_dir=models_dir,
            logger=logger,
        )

        agent.save(os.path.join(models_dir, "model_final.pth"))

        duration = logger.finalize()
        print(f"\n[DONE] Durée totale : {duration/60:.1f} min")
        print(f"[DONE] Modèles  → {models_dir}/")
        print(f"[DONE] Résultats → {results_dir}/")

    if pygame.get_init():
        pygame.quit()


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────
# Commandes utiles :
#   python main.py                         # entraînement silencieux
#   python main.py --show-every 1000       # affiche tous les 1000 épisodes
#   python main.py --load                  # reprend depuis model_best.pth
#   python main.py --eval                  # évaluation visuelle
#   python main.py --run 2                 # run n°2 (dossiers séparés)
# ─────────────────────────────────────────────────────────────
