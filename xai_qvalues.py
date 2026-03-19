"""
xai_qvalues.py — Analyse XAI : Q-values du DQN Snake
======================================================
3 visualisations :
  1. Heatmaps des Q-values des 4 actions sur la grille (position fixe nourriture)
  2. Q-value gap (confiance de l'agent) sur la grille
  3. Évolution temporelle des Q-values pendant un épisode complet

Usage :
    python xai_qvalues.py                  # toutes les visualisations
    python xai_qvalues.py --heatmap        # uniquement les heatmaps
    python xai_qvalues.py --gap            # uniquement le gap de confiance
    python xai_qvalues.py --temporal       # uniquement l'évolution temporelle
    python xai_qvalues.py --model best     # utiliser model_best.pth (défaut)
    python xai_qvalues.py --model final    # utiliser model_final.pth
"""

import argparse
import math
import random
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

import torch

# ── Imports locaux ──────────────────────────────
from dql import DQNAgent, get_device, STATE_DIM, ACTION_DIM
from main import SnakeEnv
import snake as game

# ── Pygame en mode headless ─────────────────────
import pygame
pygame.init()
game.show    = False
game.display = None

# ── Dossier de sortie dédié ──────────────────────
OUT_DIR = "xai_qvalues"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    """Retourne le chemin complet dans le dossier de sortie."""
    return os.path.join(OUT_DIR, filename)


# ═══════════════════════════════════════════════
#  Constantes & palettes
# ═══════════════════════════════════════════════
ACTION_NAMES   = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS  = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

# Colormap personnalisée : bleu froid (Q faible) → rouge chaud (Q élevé)
CMAP_QVAL = LinearSegmentedColormap.from_list(
    "qval", ["#0D1B2A", "#1B4F72", "#2E86C1", "#F39C12", "#E74C3C", "#FFFFFF"]
)
CMAP_GAP = LinearSegmentedColormap.from_list(
    "gap", ["#1A1A2E", "#16213E", "#0F3460", "#533483", "#E94560"]
)

GRID_W = int(game.width  // game.rect_width)   # 10 colonnes
GRID_H = int(game.height // game.rect_height)  # 14 lignes
DIAG   = math.sqrt(game.width**2 + game.height**2)


# ═══════════════════════════════════════════════
#  Utilitaires
# ═══════════════════════════════════════════════
def load_agent(model_name: str = "best") -> DQNAgent:
    """Charge l'agent depuis le modèle sauvegardé."""
    device = get_device()
    agent  = DQNAgent(device=device)
    path   = f"model_{model_name}.pth"
    try:
        agent.load(path)
    except FileNotFoundError:
        print(f"[WARN] {path} introuvable — poids aléatoires (agent non entraîné).")
    agent.online_net.eval()
    return agent


def get_qvalues(agent: DQNAgent, state: list) -> np.ndarray:
    """Retourne les Q-values numpy pour un état donné."""
    state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q = agent.online_net(state_t)
    return q.cpu().numpy().squeeze()   # shape [4]


def build_state_at(col: int, row: int, food_col: int, food_row: int) -> list:
    """
    Construit un état normalisé en plaçant la tête du serpent
    en (col, row) et la nourriture en (food_col, food_row).
    Le serpent est de longueur 1 (pas de corps) pour éviter
    les collisions parasites lors du scan de grille.
    """
    # Créer un serpent temporaire d'un seul segment
    tmp_snake = game.Manager_snake()
    tmp_snake.add_snake(game.Snake(col * game.rect_width, row * game.rect_height))

    tmp_food  = game.food(food_col * game.rect_width, food_row * game.rect_height)

    raw = [
        game.distance_bord_north(tmp_snake),
        game.distance_bord_north_est(tmp_snake),
        game.distance_bord_est(tmp_snake),
        game.distance_bord_south_est(tmp_snake),
        game.distance_bord_south(tmp_snake),
        game.distance_bord_south_west(tmp_snake),
        game.distance_bord_west(tmp_snake),
        game.distance_bord_north_west(tmp_snake),
        game.distance_food_north(tmp_snake, tmp_food),
        game.distance_food_north_est(tmp_snake, tmp_food),
        game.distance_food_est(tmp_snake, tmp_food),
        game.distance_food_south_est(tmp_snake, tmp_food),
        game.distance_food_south(tmp_snake, tmp_food),
        game.distance_food_south_west(tmp_snake, tmp_food),
        game.distance_food_west(tmp_snake, tmp_food),
        game.distance_food_north_west(tmp_snake, tmp_food),
    ]
    return [v / DIAG for v in raw]


def scan_grid(agent: DQNAgent, food_col: int, food_row: int):
    """
    Parcourt toutes les cellules de la grille et calcule
    les Q-values pour chaque position de tête.

    Retourne :
        qmap  : np.ndarray [GRID_H, GRID_W, 4]  — Q-values par action
        best  : np.ndarray [GRID_H, GRID_W]      — action choisie (argmax)
        gap   : np.ndarray [GRID_H, GRID_W]      — Q1 - Q2 (confiance)
    """
    qmap = np.zeros((GRID_H, GRID_W, ACTION_DIM), dtype=np.float32)

    for row in range(GRID_H):
        for col in range(GRID_W):
            state  = build_state_at(col, row, food_col, food_row)
            qvals  = get_qvalues(agent, state)
            qmap[row, col] = qvals

    sorted_q = np.sort(qmap, axis=2)          # tri croissant sur les actions
    best     = np.argmax(qmap, axis=2)
    gap      = sorted_q[:, :, -1] - sorted_q[:, :, -2]   # max - 2e max

    return qmap, best, gap


# ═══════════════════════════════════════════════
#  Visualisation 1 — Heatmaps des Q-values
# ═══════════════════════════════════════════════
def plot_qvalue_heatmaps(agent: DQNAgent, food_col: int = 5, food_row: int = 3):
    """
    4 heatmaps (une par action) + 1 carte de la politique (action choisie).
    La position de la nourriture est marquée d'une étoile.
    """
    qmap, best, gap = scan_grid(agent, food_col, food_row)

    fig = plt.figure(figsize=(20, 9), facecolor="#0D1117")
    fig.suptitle(
        "Q-values par action — position de la nourriture fixée",
        fontsize=18, fontweight="bold", color="white", y=1.01
    )

    # Layout : 4 heatmaps + 1 politique
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

    vmin = qmap.min()
    vmax = qmap.max()

    for i, action_name in enumerate(ACTION_NAMES):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#0D1117")

        im = ax.imshow(
            qmap[:, :, i],
            cmap=CMAP_QVAL, vmin=vmin, vmax=vmax,
            interpolation="nearest", aspect="auto"
        )

        # Étoile = nourriture
        ax.scatter(food_col, food_row, marker="*", s=300,
                   color="#FFD700", zorder=5, label="Nourriture")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title(f"{action_name}", color=ACTION_COLORS[i],
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Colonne", color="#AAAAAA", fontsize=8)
        ax.set_ylabel("Ligne", color="#AAAAAA", fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(out("xai_heatmaps.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[XAI] Sauvegarde -> {out('xai_heatmaps.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Visualisation 2 — Q-value gap (confiance)
# ═══════════════════════════════════════════════
def plot_confidence_map(agent: DQNAgent, food_col: int = 5, food_row: int = 3):
    """
    Carte de confiance : Q_max - Q_2ndmax.
    Zones sombres = agent hésitant, zones claires = agent sûr de lui.
    Superposé : flèche indiquant l'action choisie en chaque cellule.
    """
    qmap, best, gap = scan_grid(agent, food_col, food_row)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0D1117")
    fig.suptitle(
        "Confiance de l'agent (Q-gap) & Politique apprise",
        fontsize=16, fontweight="bold", color="white"
    )

    # ── Gauche : heatmap du gap ──────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#0D1117")
    im = ax1.imshow(gap, cmap=CMAP_GAP, interpolation="nearest", aspect="auto")
    ax1.scatter(food_col, food_row, marker="*", s=400, color="#FFD700", zorder=5)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Q_max − Q_2nd", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax1.set_title("Confiance (gap Q-values)", color="white", fontsize=13, pad=10)
    ax1.set_xlabel("Colonne", color="#AAAAAA", fontsize=9)
    ax1.set_ylabel("Ligne",   color="#AAAAAA", fontsize=9)
    ax1.tick_params(colors="#888888", labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")

    # ── Droite : politique (action choisie par cellule) ──
    ax2 = axes[1]
    ax2.set_facecolor("#0D1117")

    # Fond coloré par action choisie
    policy_rgb = np.zeros((GRID_H, GRID_W, 3))
    color_table = {
        0: np.array([0.31, 0.76, 0.97]),   # UP    bleu clair
        1: np.array([0.51, 0.78, 0.52]),   # RIGHT vert
        2: np.array([1.00, 0.72, 0.30]),   # DOWN  orange
        3: np.array([0.94, 0.38, 0.57]),   # LEFT  rose
    }
    for r in range(GRID_H):
        for c in range(GRID_W):
            policy_rgb[r, c] = color_table[best[r, c]]

    # Pondérer l'intensité par la confiance (normalisée)
    gap_norm = (gap - gap.min()) / (gap.max() - gap.min() + 1e-8)
    alpha    = 0.35 + 0.65 * gap_norm
    for ch in range(3):
        policy_rgb[:, :, ch] *= alpha

    ax2.imshow(policy_rgb, interpolation="nearest", aspect="auto")

    # Flèches indiquant l'action
    arrows = {0: (0, -0.35), 1: (0.35, 0), 2: (0, 0.35), 3: (-0.35, 0)}
    for r in range(GRID_H):
        for c in range(GRID_W):
            dx, dy = arrows[best[r, c]]
            ax2.annotate(
                "", xy=(c + dx, r + dy), xytext=(c, r),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
            )

    ax2.scatter(food_col, food_row, marker="*", s=400, color="#FFD700", zorder=5)

    # Légende
    legend_patches = [
        mpatches.Patch(color=tuple(color_table[i]), label=ACTION_NAMES[i])
        for i in range(4)
    ]
    ax2.legend(handles=legend_patches, loc="upper right",
               fontsize=8, facecolor="#1A1A2E", edgecolor="#444",
               labelcolor="white")

    ax2.set_title("Politique apprise (action choisie)", color="white", fontsize=13, pad=10)
    ax2.set_xlabel("Colonne", color="#AAAAAA", fontsize=9)
    ax2.set_ylabel("Ligne",   color="#AAAAAA", fontsize=9)
    ax2.tick_params(colors="#888888", labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(out("xai_confidence.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[XAI] Sauvegarde -> {out('xai_confidence.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Visualisation 3 — Évolution temporelle
# ═══════════════════════════════════════════════
def plot_temporal_qvalues(agent: DQNAgent, num_episodes: int = 3):
    """
    Lance N épisodes complets en mode greedy et enregistre
    les Q-values à chaque step.
    Affiche :
      - L'évolution des 4 Q-values (une courbe par action) dans le temps
      - Les moments où l'agent mange (ligne verte) ou meurt (ligne rouge)
      - La Q-value max (certitude de l'agent)
    """
    agent.epsilon = 0.0   # Greedy pur pour l'analyse
    env = SnakeEnv()

    all_episodes = []

    for ep in range(num_episodes):
        state   = env.reset()
        done    = False
        ep_data = {"qvals": [], "events": [], "scores": []}

        step = 0
        prev_score = 0

        while not done:
            qvals  = get_qvalues(agent, state)
            action = int(np.argmax(qvals))

            ep_data["qvals"].append(qvals.copy())

            state, reward, done, info = env.step(action)

            if info["score"] > prev_score:
                ep_data["events"].append((step, "food"))
                prev_score = info["score"]
            if done and info["iteration"] < 500:
                ep_data["events"].append((step, "death"))

            ep_data["scores"].append(info["score"])
            step += 1

        all_episodes.append(ep_data)
        print(f"[XAI] Épisode {ep+1} terminé — Score : {info['score']} ({step} steps)")

    # ── Tracé ────────────────────────────────────
    fig, axes = plt.subplots(
        num_episodes, 1,
        figsize=(18, 5 * num_episodes),
        facecolor="#0D1117",
        squeeze=False
    )
    fig.suptitle(
        "Évolution temporelle des Q-values pendant l'épisode",
        fontsize=16, fontweight="bold", color="white", y=1.01
    )

    for ep_idx, ep_data in enumerate(all_episodes):
        ax  = axes[ep_idx, 0]
        ax.set_facecolor("#0D1B2A")

        qvals_arr = np.array(ep_data["qvals"])   # [T, 4]
        T         = len(qvals_arr)
        steps     = np.arange(T)

        # Remplissage sous les courbes
        for i in range(ACTION_DIM):
            ax.fill_between(steps, qvals_arr[:, i], alpha=0.08,
                            color=ACTION_COLORS[i])
            ax.plot(steps, qvals_arr[:, i],
                    label=ACTION_NAMES[i], color=ACTION_COLORS[i],
                    linewidth=1.4, alpha=0.9)

        # Q-value max (enveloppe supérieure)
        ax.plot(steps, qvals_arr.max(axis=1),
                color="white", linewidth=0.8, linestyle="--",
                alpha=0.5, label="Q_max")

        # Événements
        for step_ev, ev_type in ep_data["events"]:
            if ev_type == "food":
                ax.axvline(x=step_ev, color="#2ECC71", linewidth=1.5,
                           linestyle=":", alpha=0.8)
                ax.text(step_ev + 0.5, ax.get_ylim()[1] * 0.92,
                        "🍎", fontsize=10, color="#2ECC71", alpha=0.9)
            elif ev_type == "death":
                ax.axvline(x=step_ev, color="#E74C3C", linewidth=2.0,
                           linestyle="-", alpha=0.9)
                ax.text(step_ev + 0.5, ax.get_ylim()[1] * 0.92,
                        "💀", fontsize=10, color="#E74C3C")

        score_final = ep_data["scores"][-1] if ep_data["scores"] else 0
        ax.set_title(
            f"Épisode {ep_idx + 1}  —  Score final : {score_final}  |  Steps : {T}",
            color="white", fontsize=12, pad=8
        )
        ax.set_xlabel("Step", color="#AAAAAA", fontsize=9)
        ax.set_ylabel("Q-value", color="#AAAAAA", fontsize=9)
        ax.tick_params(colors="#888888", labelsize=8)
        ax.legend(loc="upper left", fontsize=8, facecolor="#0D1117",
                  edgecolor="#444444", labelcolor="white", framealpha=0.8,
                  ncol=5)
        ax.grid(axis="y", color="#1E3A5F", linewidth=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E3A5F")

    plt.tight_layout()
    plt.savefig(out("xai_temporal.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[XAI] Sauvegarde -> {out('xai_temporal.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Point d'entrée
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="XAI — Analyse Q-values DQN Snake")
    parser.add_argument("--heatmap",  action="store_true", help="Heatmaps Q-values par action")
    parser.add_argument("--gap",      action="store_true", help="Carte de confiance + politique")
    parser.add_argument("--temporal", action="store_true", help="Évolution temporelle Q-values")
    parser.add_argument("--model",    type=str, default="best",
                        help="Modèle à charger : 'best' | 'final' | 'epXXXX' (défaut : best)")
    parser.add_argument("--food-col", type=int, default=5, help="Colonne de la nourriture (défaut : 5)")
    parser.add_argument("--food-row", type=int, default=3, help="Ligne de la nourriture (défaut : 3)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Nombre d'épisodes pour l'analyse temporelle (défaut : 3)")
    args = parser.parse_args()

    # Si aucun flag → tout afficher
    run_all = not (args.heatmap or args.gap or args.temporal)

    agent = load_agent(args.model)

    if run_all or args.heatmap:
        print("\n[XAI] Génération des heatmaps Q-values...")
        plot_qvalue_heatmaps(agent, food_col=args.food_col, food_row=args.food_row)

    if run_all or args.gap:
        print("\n[XAI] Génération de la carte de confiance...")
        plot_confidence_map(agent, food_col=args.food_col, food_row=args.food_row)

    if run_all or args.temporal:
        print(f"\n[XAI] Analyse temporelle sur {args.episodes} épisode(s)...")
        plot_temporal_qvalues(agent, num_episodes=args.episodes)

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# python xai_qvalues.py                          # tout générer
# python xai_qvalues.py --heatmap                # uniquement les heatmaps
# python xai_qvalues.py --gap                    # uniquement confiance + politique
# python xai_qvalues.py --temporal --episodes 5  # 5 épisodes temporels
# python xai_qvalues.py --food-col 7 --food-row 9 # changer la position de la nourriture
# python xai_qvalues.py --model final            # utiliser model_final.pth