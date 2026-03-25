"""
xai_features.py — Analyse XAI : Feature Importance du DQN Snake
=================================================================
3 analyses :
  1. Permutation Importance  : brouiller chaque feature → mesurer la chute de score
  2. Variance des activations : poids de la 1ère couche → features ignorées vs utilisées
  3. Corrélation features/actions : quelle feature déclenche quelle action ?

Usage :
    python xai_features.py                    # toutes les analyses
    python xai_features.py --permutation      # uniquement permutation importance
    python xai_features.py --variance         # uniquement variance des activations
    python xai_features.py --correlation      # uniquement corrélation features/actions
    python xai_features.py --model best       # modèle à charger (défaut : best)
    python xai_features.py --episodes 30      # épisodes pour permutation (défaut : 20)
"""

import argparse
import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

import torch
import torch.nn as nn

import pygame

from dql import DQNAgent, get_device, ACTION_DIM
from main import SnakeEnv
import snake as game

# ── Pygame headless ─────────────────────────────
pygame.init()
game.show    = False
game.display = None

# ── Dossier de sortie dédié ──────────────────────
OUT_DIR = "xai_features"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    """Retourne le chemin complet dans le dossier de sortie."""
    return os.path.join(OUT_DIR, filename)


# ═══════════════════════════════════════════════
#  Noms des 16 features
# ═══════════════════════════════════════════════
FEATURE_NAMES = [
    "Dist. mur  N",
    "Dist. mur  NE",
    "Dist. mur  E",
    "Dist. mur  SE",
    "Dist. mur  S",
    "Dist. mur  SW",
    "Dist. mur  W",
    "Dist. mur  NW",
    "Dist. food N",
    "Dist. food NE",
    "Dist. food E",
    "Dist. food SE",
    "Dist. food S",
    "Dist. food SW",
    "Dist. food W",
    "Dist. food NW",
]

ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

# ── Palettes sombres cohérentes ──────────────────
BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

CMAP_IMPORTANCE = LinearSegmentedColormap.from_list(
    "imp", ["#0D1B2A", "#1A3A5C", "#1F618D", "#2E86C1", "#F39C12", "#E74C3C"]
)
CMAP_CORR = LinearSegmentedColormap.from_list(
    "corr", ["#C0392B", "#922B21", "#1A1A2E", "#1A5276", "#2E86C1"]
)
CMAP_VAR = LinearSegmentedColormap.from_list(
    "var", ["#0D1B2A", "#154360", "#1F618D", "#AED6F1", "#EBF5FB"]
)


# ═══════════════════════════════════════════════
#  Utilitaires
# ═══════════════════════════════════════════════
def load_agent(model_name: str = "best") -> DQNAgent:
    device = get_device()
    agent  = DQNAgent(device=device)
    try:
        agent.load(f"models/model_{model_name}.pth")
    except FileNotFoundError:
        print(f"[WARN] models/model_{model_name}.pth introuvable — poids aléatoires.")
    agent.online_net.eval()
    agent.epsilon = 0.0
    return agent


def run_episode(agent: DQNAgent, env: SnakeEnv,
                noise_feature: int = -1,
                noise_std: float = 0.0,
                shuffle_feature: int = -1) -> tuple[int, list, list]:
    """
    Joue un épisode complet.
    - noise_feature  : index de la feature à bruiter (-1 = aucun)
    - noise_std      : écart-type du bruit gaussien
    - shuffle_feature: index de la feature à permuter (-1 = aucun)
    Retourne (score, liste_states, liste_actions).
    """
    state = env.reset()
    done  = False
    states_log  = []
    actions_log = []

    # Pour la permutation : collecter d'abord tous les états ?
    # Non : on perturbe en ligne pour rester réaliste.

    while not done:
        s = list(state)

        if shuffle_feature >= 0:
            s[shuffle_feature] = np.random.uniform(0, 1)   # valeur aléatoire uniforme

        if noise_feature >= 0 and noise_std > 0:
            s[noise_feature] = float(np.clip(
                s[noise_feature] + np.random.normal(0, noise_std), 0, 1
            ))

        state_t = torch.tensor(s, dtype=torch.float32,
                               device=agent.device).unsqueeze(0)
        with torch.no_grad():
            q = agent.online_net(state_t)
        action = int(q.argmax(dim=1).item())

        states_log.append(s)
        actions_log.append(action)
        state, _, done, info = env.step(action)

    return info["score"], states_log, actions_log


def apply_style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)


# ═══════════════════════════════════════════════
#  Analyse 1 — Permutation Importance
# ═══════════════════════════════════════════════
def compute_permutation_importance(agent: DQNAgent, env: SnakeEnv,
                                   n_episodes: int = 20) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Pour chaque feature :
      1. Jouer n_episodes épisodes avec cette feature randomisée
      2. Comparer au score baseline (toutes features intactes)
    Retourne (drop_mean, baseline_mean, drop_std).
    """
    n_features = len(FEATURE_NAMES)

    # ── Baseline ──────────────────────────────────
    print(f"  [PI] Calcul du baseline ({n_episodes} épisodes)...")
    baseline_scores = [run_episode(agent, env)[0] for _ in range(n_episodes)]
    baseline_mean   = float(np.mean(baseline_scores))
    print(f"  [PI] Score baseline moyen : {baseline_mean:.2f}")

    # ── Permutation par feature ──────────────────
    drops     = np.zeros(n_features)
    drops_std = np.zeros(n_features)

    for fi in range(n_features):
        shuffled = [run_episode(agent, env, shuffle_feature=fi)[0]
                    for _ in range(n_episodes)]
        mean_sh   = float(np.mean(shuffled))
        drop      = baseline_mean - mean_sh
        drops[fi]     = max(drop, 0.0)   # importance ≥ 0
        drops_std[fi] = float(np.std(shuffled))
        print(f"  [PI] Feature {fi:>2} ({FEATURE_NAMES[fi]:<18}) : "
              f"score moyen={mean_sh:.2f}  drop={drop:+.2f}")

    return drops, baseline_mean, drops_std


def plot_permutation_importance(drops: np.ndarray, baseline: float,
                                drops_std: np.ndarray):
    n = len(FEATURE_NAMES)
    order = np.argsort(drops)[::-1]   # tri décroissant

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG,
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(
        f"Permutation Importance — baseline score : {baseline:.2f}",
        fontsize=16, fontweight="bold", color="white"
    )

    # ── Gauche : barplot horizontal ──────────────
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)

    norm   = Normalize(vmin=drops.min(), vmax=drops.max())
    colors = [CMAP_IMPORTANCE(norm(drops[order[i]])) for i in range(n)]

    bars = ax.barh(
        range(n), drops[order],
        xerr=drops_std[order],
        color=colors, edgecolor="#1A1A2E",
        error_kw=dict(ecolor="#AAAAAA", lw=1.2, capsize=3),
        height=0.72
    )

    # Valeurs numériques au bout des barres
    for i, (bar, drop) in enumerate(zip(bars, drops[order])):
        ax.text(drop + 0.02, i, f"{drop:.2f}",
                va="center", ha="left", color=TEXT_COL, fontsize=8)

    # Séparation murs / nourriture
    sep = sum(1 for i in order if i < 8)
    ax.axhline(y=n - sep - 0.5, color="#F39C12", linewidth=1.2,
               linestyle="--", alpha=0.7)
    ax.text(drops.max() * 0.98, n - sep - 0.3,
            "── murs  /  nourriture ──",
            color="#F39C12", fontsize=8, ha="right", alpha=0.8)

    ax.set_yticks(range(n))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order],
                       color=TEXT_COL, fontsize=9)
    apply_style(ax, "Chute de score par feature brouillée",
                xlabel="Drop de score moyen (baseline − bruité)")

    # ── Droite : radar chart des 8 top features ──
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.set_aspect("equal")

    top8       = order[:8]
    drops_top8 = drops[top8]
    vals       = drops_top8 / (drops_top8.max() + 1e-8)   # normalisation [0,1]
    labs       = [FEATURE_NAMES[i] for i in top8]
    N          = len(top8)
    angles     = [2 * math.pi * k / N for k in range(N)] + [0]
    vals_r     = list(vals) + [vals[0]]

    # ── Toile d'araignée avec labels de niveau ───────
    for level in [0.25, 0.5, 0.75, 1.0]:
        ring_xs = [level * math.cos(a) for a in angles]
        ring_ys = [level * math.sin(a) for a in angles]
        ax2.plot(ring_xs, ring_ys, color=GRID_COL, linewidth=0.7, alpha=0.6)
        # Label de niveau positionné à 3h (angle=0)
        drop_val = level * drops_top8.max()
        ax2.text(level + 0.04, 0.02,
                 f"{int(level*100)}%\n({drop_val:.1f})",
                 color="#7A9CC0", fontsize=6, va="center", alpha=0.9,
                 multialignment="center")

    for a in angles[:-1]:
        ax2.plot([0, math.cos(a)], [0, math.sin(a)],
                 color=GRID_COL, linewidth=0.7, alpha=0.6)

    # ── Aire colorée ──────────────────────────────────
    xs = [v * math.cos(a) for v, a in zip(vals_r, angles)]
    ys = [v * math.sin(a) for v, a in zip(vals_r, angles)]
    ax2.fill(xs, ys, color="#2E86C1", alpha=0.30)
    ax2.plot(xs, ys, color="#4FC3F7", linewidth=2.2)
    ax2.scatter(xs[:-1], ys[:-1], color="#FFD700", s=70, zorder=5)

    # ── Labels des features avec rang + drop brut ────
    for rank, (i, a, lab) in enumerate(zip(range(N), angles[:-1], labs)):
        raw_drop  = drops_top8[i]
        # Couleur selon catégorie (mur vs food)
        feat_idx  = top8[i]
        lab_color = ACTION_COLORS[0] if feat_idx < 8 else ACTION_COLORS[2]
        full_lab  = f"#{rank+1} {lab}\ndrop = {raw_drop:.2f}"
        ax2.text(1.38 * math.cos(a), 1.38 * math.sin(a), full_lab,
                 ha="center", va="center", color=lab_color, fontsize=7,
                 fontweight="bold", multialignment="center")

    # ── Bloc d'explication intégré ────────────────────
    explanation = (
        "Comment lire ce graphe ?\n\n"
        "• Chaque axe = une des 8 features\n"
        "  les plus importantes du réseau.\n\n"
        "• Le rayon d'un point = la chute\n"
        "  de score causée par cette feature\n"
        "  quand on la brouille (randomise).\n\n"
        "• 100% = feature la plus critique\n"
        "  (drop le plus grand).\n\n"
        "• Un polygone grand et régulier\n"
        "  = plusieurs features cruciales.\n\n"
        f"• Labels en {ACTION_COLORS[0]} = distances murs\n"
        f"• Labels en {ACTION_COLORS[2]} = distances food"
    )
    ax2.text(0, -1.72, explanation,
             ha="center", va="top", color="#99AABB",
             fontsize=6.8, style="italic", multialignment="left",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0D1B2A",
                       edgecolor=GRID_COL, alpha=0.85))

    ax2.set_xlim(-1.75, 1.75)
    ax2.set_ylim(-3.20, 1.75)
    ax2.axis("off")
    ax2.set_title(
        "Top 8 features — Radar d'importance\n"
        "(rayon ∝ chute de score quand la feature est brouillée)",
        color="white", fontsize=11, fontweight="bold", pad=12
    )

    plt.tight_layout()
    plt.savefig(out("xai_permutation.png"), dpi=150, bbox_inches="tight",
                facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_permutation.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Analyse 2 — Variance des activations (poids W1)
# ═══════════════════════════════════════════════
def compute_weight_variance(agent: DQNAgent) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrait les poids de la 1ère couche linéaire (shape [256, 16]).
    Pour chaque feature d'entrée (colonne) :
      - L2 norm des poids : importance structurelle
      - Std des poids     : dispersion
    Retourne (l2_norms, weight_std, weight_matrix).
    """
    # Récupération du 1er Linear dans le Sequential
    first_linear = None
    for layer in agent.online_net.net:
        if isinstance(layer, nn.Linear):
            first_linear = layer
            break

    W = first_linear.weight.detach().cpu().numpy()   # [256, 16]
    l2_norms = np.linalg.norm(W, axis=0)             # [16] — norme par feature
    stds     = W.std(axis=0)                         # [16]
    return l2_norms, stds, W


def plot_weight_variance(l2_norms: np.ndarray, stds: np.ndarray, W: np.ndarray):
    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    fig.suptitle(
        "Analyse des poids de la 1ère couche — Features utilisées vs ignorées",
        fontsize=16, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # ── Haut gauche : L2 norm par feature (barplot) ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL_BG)
    order = np.argsort(l2_norms)[::-1]

    norm_c = Normalize(vmin=l2_norms.min(), vmax=l2_norms.max())
    colors = [CMAP_VAR(norm_c(l2_norms[i])) for i in order]

    ax1.barh(range(16), l2_norms[order], color=colors,
             edgecolor="#0D1117", height=0.7)
    ax1.set_yticks(range(16))
    ax1.set_yticklabels([FEATURE_NAMES[i] for i in order],
                        color=TEXT_COL, fontsize=8)
    for i, v in enumerate(l2_norms[order]):
        ax1.text(v + 0.002, i, f"{v:.3f}", va="center",
                 color=TEXT_COL, fontsize=7)
    apply_style(ax1, "Norme L2 des poids par feature d'entrée",
                xlabel="‖W[:,i]‖₂ — importance structurelle")

    # ── Haut droite : std par feature ───────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL_BG)
    order2 = np.argsort(stds)[::-1]
    colors2 = [CMAP_IMPORTANCE(norm_c(stds[i])) for i in order2]

    ax2.barh(range(16), stds[order2], color=colors2,
             edgecolor="#0D1117", height=0.7)
    ax2.set_yticks(range(16))
    ax2.set_yticklabels([FEATURE_NAMES[i] for i in order2],
                        color=TEXT_COL, fontsize=8)
    for i, v in enumerate(stds[order2]):
        ax2.text(v + 0.0005, i, f"{v:.4f}", va="center",
                 color=TEXT_COL, fontsize=7)
    apply_style(ax2, "Écart-type des poids par feature d'entrée",
                xlabel="std(W[:,i]) — dispersion des connexions")

    # ── Bas gauche : heatmap des poids W1 ───────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL_BG)

    # On montre les 64 premiers neurones de la couche cachée pour lisibilité
    W_show = W[:64, :]
    vabs   = np.abs(W_show).max()
    im = ax3.imshow(W_show.T, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                    aspect="auto", interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Valeur du poids", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    ax3.set_yticks(range(16))
    ax3.set_yticklabels(FEATURE_NAMES, color=TEXT_COL, fontsize=7)
    ax3.set_xlabel("Neurone (couche cachée 1, 64 premiers)", color=TEXT_COL, fontsize=8)
    ax3.set_title("Matrice des poids W₁ (features × neurones)",
                  color="white", fontsize=11, fontweight="bold", pad=8)
    ax3.tick_params(colors="#8899AA", labelsize=7)
    for spine in ax3.spines.values():
        spine.set_edgecolor(GRID_COL)

    # ── Bas droite : scatter L2 vs std ──────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL_BG)

    # Couleur : murs (bleu) vs nourriture (orange)
    colors_sc = [ACTION_COLORS[0] if i < 8 else ACTION_COLORS[2]
                 for i in range(16)]
    sc = ax4.scatter(l2_norms, stds, c=colors_sc, s=90,
                     edgecolors="#222244", linewidths=0.8, zorder=3)

    for i, (x, y) in enumerate(zip(l2_norms, stds)):
        ax4.annotate(FEATURE_NAMES[i], (x, y),
                     textcoords="offset points", xytext=(5, 3),
                     color=TEXT_COL, fontsize=6.5, alpha=0.85)

    # Quadrant : features « ignorées » vs « importantes »
    ax4.axvline(x=np.median(l2_norms), color="#F39C12",
                linestyle="--", linewidth=1, alpha=0.6)
    ax4.axhline(y=np.median(stds), color="#F39C12",
                linestyle="--", linewidth=1, alpha=0.6)
    ax4.text(np.median(l2_norms) * 0.3, stds.max() * 0.93,
             "faible\nimportance", color="#F39C12", fontsize=7.5, alpha=0.7)
    ax4.text(np.median(l2_norms) * 1.08, stds.max() * 0.93,
             "forte\nimportance", color="#F39C12", fontsize=7.5, alpha=0.7)

    legend_p = [
        mpatches.Patch(color=ACTION_COLORS[0], label="Distances murs (0–7)"),
        mpatches.Patch(color=ACTION_COLORS[2], label="Distances nourriture (8–15)"),
    ]
    ax4.legend(handles=legend_p, fontsize=8, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white")
    apply_style(ax4, "Nuage L2-norm vs Std (quadrant d'importance)",
                xlabel="Norme L2", ylabel="Écart-type")

    plt.savefig(out("xai_variance.png"), dpi=150, bbox_inches="tight",
                facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_variance.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Analyse 3 — Corrélation features / actions
# ═══════════════════════════════════════════════
def compute_feature_action_correlation(agent: DQNAgent, env: SnakeEnv,
                                       n_episodes: int = 20
                                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Joue n_episodes épisodes, collecte tous les (état, action_choisie).
    Calcule :
      - corr_matrix [16, 4] : corrélation de Pearson entre feature_i et (action==j)
      - mean_per_action [4, 16] : valeur moyenne de chaque feature quand action=j
      - std_per_action  [4, 16] : écart-type
    """
    all_states  = []
    all_actions = []

    print(f"  [Corr] Collecte de données sur {n_episodes} épisodes...")
    for ep in range(n_episodes):
        score, states, actions = run_episode(agent, env)
        all_states.extend(states)
        all_actions.extend(actions)

    states_arr  = np.array(all_states,  dtype=np.float32)   # [T, 16]
    actions_arr = np.array(all_actions, dtype=np.int32)      # [T]
    print(f"  [Corr] {len(actions_arr)} transitions collectées.")

    n_feat = len(FEATURE_NAMES)
    corr_matrix = np.zeros((n_feat, ACTION_DIM))

    for fi in range(n_feat):
        for ai in range(ACTION_DIM):
            binary = (actions_arr == ai).astype(float)
            r, _   = pearsonr(states_arr[:, fi], binary)
            corr_matrix[fi, ai] = r if not np.isnan(r) else 0.0

    # Moyenne et std par action
    mean_per_action = np.zeros((ACTION_DIM, n_feat))
    std_per_action  = np.zeros((ACTION_DIM, n_feat))
    for ai in range(ACTION_DIM):
        mask = actions_arr == ai
        if mask.sum() > 0:
            mean_per_action[ai] = states_arr[mask].mean(axis=0)
            std_per_action[ai]  = states_arr[mask].std(axis=0)

    return corr_matrix, mean_per_action, std_per_action


def plot_feature_action_correlation(corr_matrix: np.ndarray,
                                    mean_per_action: np.ndarray,
                                    std_per_action: np.ndarray):
    fig = plt.figure(figsize=(20, 12), facecolor=BG)
    fig.suptitle(
        "Corrélation Features → Actions  —  Ce qui déclenche chaque décision",
        fontsize=16, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.5)

    # ── Heatmap centrale [16 × 4] ────────────────
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_heat.set_facecolor(PANEL_BG)

    vabs = np.abs(corr_matrix).max()
    im   = ax_heat.imshow(corr_matrix, cmap=CMAP_CORR,
                          vmin=-vabs, vmax=vabs,
                          aspect="auto", interpolation="nearest")

    # Annotations des valeurs
    for fi in range(len(FEATURE_NAMES)):
        for ai in range(ACTION_DIM):
            v = corr_matrix[fi, ai]
            c = "white" if abs(v) > 0.15 else "#888888"
            ax_heat.text(ai, fi, f"{v:+.2f}", ha="center", va="center",
                         color=c, fontsize=8, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Corrélation de Pearson", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    ax_heat.set_xticks(range(ACTION_DIM))
    ax_heat.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_heat.set_yticks(range(len(FEATURE_NAMES)))
    ax_heat.set_yticklabels(FEATURE_NAMES, color=TEXT_COL, fontsize=8)
    ax_heat.set_title("Corrélation\nfeature × action", color="white",
                      fontsize=12, fontweight="bold", pad=10)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(GRID_COL)

    # Séparateur murs / nourriture
    ax_heat.axhline(y=7.5, color="#F39C12", linewidth=1.5,
                    linestyle="--", alpha=0.7)

    # ── 4 barplots (un par action) ───────────────
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for ai, (row, col) in enumerate(positions):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL_BG)

        vals   = corr_matrix[:, ai]
        order  = np.argsort(np.abs(vals))[::-1]
        ypos   = range(len(FEATURE_NAMES))

        bar_colors = [
            ACTION_COLORS[ai] if v >= 0 else "#E74C3C"
            for v in vals[order]
        ]
        bars = ax.barh(list(ypos), vals[order],
                       color=bar_colors, edgecolor="#0D1117",
                       alpha=0.85, height=0.7)

        # Ligne zéro
        ax.axvline(x=0, color="#AAAAAA", linewidth=1.0, alpha=0.5)

        ax.set_yticks(list(ypos))
        ax.set_yticklabels([FEATURE_NAMES[i] for i in order],
                           color=TEXT_COL, fontsize=7.5)
        ax.set_xlim(-vabs * 1.2, vabs * 1.2)

        apply_style(
            ax,
            f"Action : {ACTION_NAMES[ai]}",
            xlabel="Corrélation de Pearson"
        )
        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=11,
                     fontweight="bold", pad=8)

        # Top feature annotée
        top_feat = order[0]
        ax.text(vals[order][0] * 1.05, 0,
                f"  top: {FEATURE_NAMES[top_feat]}",
                va="center", color="#FFD700", fontsize=7)

    plt.savefig(out("xai_correlation.png"), dpi=150, bbox_inches="tight",
                facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_correlation.png')}")
    plt.show()

    # ── Bonus : valeur moyenne des features par action ──
    _plot_mean_per_action(mean_per_action, std_per_action)


def _plot_mean_per_action(mean_per_action: np.ndarray, std_per_action: np.ndarray):
    """
    Pour chaque action : barplot horizontal des valeurs moyennes de features.
    Chaque subplot affiche ses propres labels en ordonnée pour une lecture autonome.
    """
    fig, axes = plt.subplots(1, ACTION_DIM, figsize=(26, 9), facecolor=BG)
    fig.suptitle(
        "Profil sensoriel par action  —  Valeur moyenne de chaque feature quand l'agent choisit cette action\n"
        "Lecture : une barre longue = cette feature est élevée lors de ce choix  |  "
        "barres d'erreur = écart-type  |  trait orange = séparation murs / nourriture",
        fontsize=12, fontweight="bold", color="white", y=1.02
    )

    ypos = np.arange(len(FEATURE_NAMES))

    # Catégories : couleur de fond des labels pour distinguer murs vs food
    label_colors_y = [ACTION_COLORS[0]] * 8 + [ACTION_COLORS[2]] * 8

    for ai, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG)
        means = mean_per_action[ai]
        stds  = std_per_action[ai]

        # ── Fond alterné pour chaque ligne (lisibilité) ──
        for row in range(len(FEATURE_NAMES)):
            bg_c = "#0F2233" if row % 2 == 0 else PANEL_BG
            ax.axhspan(row - 0.5, row + 0.5, color=bg_c, alpha=0.5, zorder=0)

        # ── Barres ──────────────────────────────────────
        bars = ax.barh(ypos, means, xerr=stds,
                       color=ACTION_COLORS[ai], alpha=0.82,
                       edgecolor="#0D1117", height=0.65, zorder=2,
                       error_kw=dict(ecolor="#AAAAAA", lw=1, capsize=3))

        # Valeur numérique au bout de chaque barre
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 0.01, i, f"{m:.2f}",
                    va="center", ha="left", color=TEXT_COL,
                    fontsize=6.5, alpha=0.85)

        # ── Labels ordonnée : tous les subplots ─────────
        ax.set_yticks(ypos)
        ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
        # Colorier chaque label selon sa catégorie (mur vs food)
        for tick, col in zip(ax.get_yticklabels(), label_colors_y):
            tick.set_color(col)

        # ── Séparateur murs / nourriture ─────────────────
        ax.axhline(y=7.5, color="#F39C12", linewidth=1.5,
                   linestyle="--", alpha=0.75, zorder=3)

        # Annotations catégories sur le 1er subplot seulement (pour ne pas surcharger)
        if ai == 0:
            ax.text(-0.06, 3.5, "MURS", color=ACTION_COLORS[0],
                    fontsize=7, fontweight="bold", rotation=90,
                    va="center", ha="center",
                    transform=ax.get_yaxis_transform())
            ax.text(-0.06, 11.5, "FOOD", color=ACTION_COLORS[2],
                    fontsize=7, fontweight="bold", rotation=90,
                    va="center", ha="center",
                    transform=ax.get_yaxis_transform())

        # ── Légende catégories (1er subplot) ─────────────
        if ai == 0:
            legend_p = [
                mpatches.Patch(color=ACTION_COLORS[0], label="Distances murs (feat. 0–7)"),
                mpatches.Patch(color=ACTION_COLORS[2], label="Distances nourriture (feat. 8–15)"),
            ]
            ax.legend(handles=legend_p, loc="lower right", fontsize=7,
                      facecolor="#0D1117", edgecolor="#444", labelcolor="white")

        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=12,
                     fontweight="bold", pad=10)
        ax.set_xlabel("Valeur normalisée [0 – 1]", color=TEXT_COL, fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.tick_params(axis="x", colors="#8899AA", labelsize=8)
        ax.tick_params(axis="y", colors="#8899AA", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5, zorder=1)

    plt.tight_layout()
    plt.savefig(out("xai_mean_per_action.png"), dpi=150, bbox_inches="tight",
                facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_mean_per_action.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Point d'entrée
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="XAI — Feature Importance DQN Snake"
    )
    parser.add_argument("--permutation", action="store_true",
                        help="Permutation Importance")
    parser.add_argument("--variance",    action="store_true",
                        help="Variance des activations / poids W1")
    parser.add_argument("--correlation", action="store_true",
                        help="Corrélation features × actions")
    parser.add_argument("--model",    type=str, default="best",
                        help="Modèle : 'best' | 'final' | 'epXXXX' (défaut : best)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Épisodes pour permutation / corrélation (défaut : 20)")
    args = parser.parse_args()

    run_all = not (args.permutation or args.variance or args.correlation)

    agent = load_agent(args.model)
    env   = SnakeEnv()

    if run_all or args.variance:
        print("\n[XAI] ── Variance des poids (couche 1) ──────────────")
        l2, stds, W = compute_weight_variance(agent)
        plot_weight_variance(l2, stds, W)

    if run_all or args.permutation:
        print("\n[XAI] ── Permutation Importance ─────────────────────")
        drops, baseline, drops_std = compute_permutation_importance(
            agent, env, n_episodes=args.episodes
        )
        plot_permutation_importance(drops, baseline, drops_std)

    if run_all or args.correlation:
        print("\n[XAI] ── Corrélation features × actions ─────────────")
        corr, means, stds_a = compute_feature_action_correlation(
            agent, env, n_episodes=args.episodes
        )
        plot_feature_action_correlation(corr, means, stds_a)

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# python xai_features.py                        # tout générer (20 épisodes)
# python xai_features.py --variance             # rapide, pas d'épisodes requis
# python xai_features.py --permutation --episodes 50   # plus précis
# python xai_features.py --correlation --episodes 30
# python xai_features.py --model final          # avec model_final.pth