"""
xai_activations.py — Analyse XAI : Activations internes du DQN Snake
=====================================================================
3 analyses :
  1. Distribution des activations par couche
       → histogrammes, taux de neurones morts (ReLU → toujours 0)
  2. Neurones spécialisés
       → quels neurones s'activent uniquement dans des situations précises
         (danger imminent, nourriture alignée, serpent long…)
  3. t-SNE / UMAP des activations
       → projection 2D des états vus → clusters de situations similaires

Architecture du réseau (depuis dql.py) :
    net[0]  Linear(16 → 256)
    net[1]  BatchNorm1d(256)
    net[2]  ReLU            ← couche 1 (post-BN)
    net[3]  Linear(256 → 128)
    net[4]  ReLU            ← couche 2
    net[5]  Linear(128 → 64)
    net[6]  ReLU            ← couche 3
    net[7]  Linear(64 → 4)  ← sortie Q-values

Usage :
    python xai_activations.py                   # toutes les analyses
    python xai_activations.py --distribution    # histogrammes + neurones morts
    python xai_activations.py --specialization  # neurones spécialisés
    python xai_activations.py --tsne            # t-SNE des activations
    python xai_activations.py --umap            # UMAP des activations (+ rapide)
    python xai_activations.py --model best      # modèle (défaut : best)
    python xai_activations.py --episodes 15     # épisodes de collecte (défaut : 10)
"""

import argparse
import os
import math
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

import torch
import torch.nn as nn

import pygame

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dql import DQNAgent, get_device, ACTION_DIM, HIDDEN_1, HIDDEN_2, HIDDEN_3
from main import SnakeEnv
import snake as game

# ── Pygame headless ─────────────────────────────
pygame.init()
game.show    = False
game.display = None

# ── Dossier de sortie ────────────────────────────
OUT_DIR = "xai_activations"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ═══════════════════════════════════════════════
#  Constantes
# ═══════════════════════════════════════════════
# Couches ReLU dans net.Sequential (indices)
LAYER_HOOKS = {
    "Couche 1\n(post-BN, 256n)":  2,   # ReLU après Linear+BN
    "Couche 2\n(128n)":            4,   # ReLU après Linear
    "Couche 3\n(64n)":             6,   # ReLU après Linear
}
LAYER_SIZES = {
    "Couche 1\n(post-BN, 256n)": HIDDEN_1,
    "Couche 2\n(128n)":          HIDDEN_2,
    "Couche 3\n(64n)":           HIDDEN_3,
}
LAYER_KEYS  = list(LAYER_HOOKS.keys())

ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

# Situations de jeu (labels pour la spécialisation)
SITUATION_NAMES = [
    "Danger N",
    "Danger E",
    "Danger S",
    "Danger W",
    "Food alignée\nH",
    "Food alignée\nV",
    "Serpent long\n(≥5)",
    "Neutre",
]

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

CMAP_DEAD = LinearSegmentedColormap.from_list(
    "dead", ["#2ECC71", "#F39C12", "#E74C3C"]
)
CMAP_SPEC = LinearSegmentedColormap.from_list(
    "spec", ["#0D1B2A", "#154360", "#1F618D", "#D4AC0D", "#E74C3C"]
)
CMAP_TSNE = LinearSegmentedColormap.from_list(
    "tsne", ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]
)


# ═══════════════════════════════════════════════
#  Chargement du modèle
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


# ═══════════════════════════════════════════════
#  Collecte des activations via forward hooks
# ═══════════════════════════════════════════════
class ActivationCollector:
    """
    Pose des forward hooks sur les couches ReLU du réseau.
    Chaque appel forward() accumule les activations dans self.data.
    """
    def __init__(self, agent: DQNAgent):
        self.agent   = agent
        self.data    = {k: [] for k in LAYER_KEYS}
        self._hooks  = []
        self._register()

    def _register(self):
        for layer_name, idx in LAYER_HOOKS.items():
            def make_hook(name):
                def hook(module, inp, out):
                    self.data[name].append(out.detach().cpu().numpy())
                return hook
            h = self.agent.online_net.net[idx].register_forward_hook(
                make_hook(layer_name)
            )
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()

    def clear(self):
        self.data = {k: [] for k in LAYER_KEYS}

    def get_arrays(self) -> dict:
        """Retourne {layer_name: np.ndarray [T, N_neurons]}."""
        return {k: np.vstack(v) for k, v in self.data.items() if len(v) > 0}


def collect_episodes(agent: DQNAgent, env: SnakeEnv,
                     collector: ActivationCollector,
                     n_episodes: int = 10
                     ) -> tuple[list, list, list, list]:
    """
    Joue n_episodes épisodes, collecte activations + métadonnées.
    Retourne :
        states_log    : liste de vecteurs d'état [16]
        actions_log   : liste d'entiers (action choisie)
        situations_log: liste d'entiers (index de situation)
        scores_log    : liste de scores (score courant au step t)
    """
    states_log     = []
    actions_log    = []
    situations_log = []
    scores_log     = []
    collector.clear()

    for ep in range(n_episodes):
        state = env.reset()
        done  = False
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32,
                               device=agent.device).unsqueeze(0)
            with torch.no_grad():
                q = agent.online_net(s_t)
            action = int(q.argmax(dim=1).item())

            states_log.append(list(state))
            actions_log.append(action)
            situations_log.append(_classify_situation(state, env))
            scores_log.append(env.score)

            state, _, done, info = env.step(action)

        print(f"  [Collect] Épisode {ep+1}/{n_episodes} — score {info['score']} "
              f"({len(states_log)} steps total)")

    return states_log, actions_log, situations_log, scores_log


def _classify_situation(state: list, env: SnakeEnv) -> int:
    """
    Classe l'état courant en une des 8 situations prédéfinies.
    Les features (normalisées) sont dans l'ordre :
    [0..7] = distances murs/corps N NE E SE S SW W NW
    [8..15]= distances food N NE E SE S SW W NW
    DIAG   = diagonale de la grille (valeur de normalisation)
    """
    diag       = math.sqrt(game.width**2 + game.height**2)
    DANGER_THR = 50 / diag     # danger si obstacle à ≤ 50px (1 case)

    d_n  = state[0]   # danger nord
    d_e  = state[2]   # danger est
    d_s  = state[4]   # danger sud
    d_w  = state[6]   # danger ouest

    food_h = state[10] + state[14]   # food est + food ouest  (alignée horizontalement)
    food_v = state[8]  + state[12]   # food nord + food sud   (alignée verticalement)

    snake_len = env.my_snake.lenght

    if d_n <= DANGER_THR and d_n > 0:   return 0  # Danger N
    if d_e <= DANGER_THR and d_e > 0:   return 1  # Danger E
    if d_s <= DANGER_THR and d_s > 0:   return 2  # Danger S
    if d_w <= DANGER_THR and d_w > 0:   return 3  # Danger W
    if food_h > 0:                       return 4  # Food alignée H
    if food_v > 0:                       return 5  # Food alignée V
    if snake_len >= 5:                   return 6  # Serpent long
    return 7                                        # Neutre


def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    if title:
        ax.set_title(title, color="white", fontsize=11,
                     fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


# ═══════════════════════════════════════════════
#  Analyse 1 — Distribution des activations
# ═══════════════════════════════════════════════
def plot_distribution(act_arrays: dict):
    """
    Pour chaque couche :
      - Histogramme global des valeurs d'activation
      - Taux de neurones morts (activation = 0 sur > 80% des steps)
      - Heatmap temporelle (neurones × time, 200 premiers steps)
    """
    n_layers = len(LAYER_KEYS)

    fig = plt.figure(figsize=(22, 6 * n_layers), facecolor=BG)
    fig.suptitle(
        "Distribution des activations par couche\n"
        "Neurones morts = ReLU → toujours 0  |  "
        "Heatmap = activité temporelle (bleu=0, chaud=actif)",
        fontsize=14, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(n_layers, 3, figure=fig,
                           wspace=0.38, hspace=0.55,
                           width_ratios=[1.4, 1, 2])

    for row, layer_name in enumerate(LAYER_KEYS):
        acts = act_arrays[layer_name]   # [T, N]
        T, N = acts.shape

        # ── Taux de mort par neurone ──────────────────
        dead_thr  = 0.80   # mort si actif < 20% du temps
        frac_zero = (acts == 0).mean(axis=0)       # [N]
        is_dead   = frac_zero > dead_thr
        n_dead    = is_dead.sum()
        pct_dead  = 100 * n_dead / N

        # ── Col 0 : histogramme des activations ──────
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_facecolor(PANEL_BG)

        vals_flat = acts.flatten()
        nonzero   = vals_flat[vals_flat > 1e-6]
        ax0.hist(vals_flat, bins=80, color="#1F618D", alpha=0.6,
                 label="Toutes", edgecolor="none")
        if len(nonzero):
            ax0.hist(nonzero, bins=80, color="#F39C12", alpha=0.75,
                     label="Non-nulles", edgecolor="none")

        ax0.axvline(x=0, color="#E74C3C", linewidth=1.5,
                    linestyle="--", label="x = 0")
        ax0.set_yscale("log")
        apply_style(ax0,
                    title=f"{layer_name.strip()} — Distribution",
                    xlabel="Valeur d'activation", ylabel="Fréquence (log)")
        ax0.legend(fontsize=7, facecolor="#0D1117",
                   edgecolor="#444", labelcolor="white")

        # Annotation : stats
        stats_txt = (
            f"min  = {acts.min():.3f}\n"
            f"max  = {acts.max():.3f}\n"
            f"mean = {acts.mean():.3f}\n"
            f"std  = {acts.std():.3f}\n"
            f"morts = {n_dead}/{N} ({pct_dead:.1f}%)"
        )
        ax0.text(0.97, 0.97, stats_txt,
                 transform=ax0.transAxes, va="top", ha="right",
                 color=TEXT_COL, fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#0D1B2A",
                           edgecolor=GRID_COL, alpha=0.9))

        # ── Col 1 : barplot des neurones morts ───────
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.set_facecolor(PANEL_BG)

        # Tri par taux de zéro croissant → mort en haut
        order     = np.argsort(frac_zero)[::-1]
        top_n     = min(40, N)
        bar_colors = [CMAP_DEAD(frac_zero[order[i]]) for i in range(top_n)]

        ax1.barh(range(top_n), frac_zero[order[:top_n]],
                 color=bar_colors, edgecolor="none", height=0.85)
        ax1.axvline(x=dead_thr, color="#E74C3C", linewidth=1.2,
                    linestyle="--", alpha=0.9, label=f"Seuil mort ({int(dead_thr*100)}%)")
        ax1.set_yticks([])
        ax1.set_xlim(0, 1.05)

        apply_style(ax1,
                    title=f"Neurones morts (top {top_n})\n"
                           f"{n_dead}/{N} morts ({pct_dead:.1f}%)",
                    xlabel="Fraction de steps à 0")
        ax1.legend(fontsize=7, facecolor="#0D1117",
                   edgecolor="#444", labelcolor="white")

        # Colorbar mort
        sm = ScalarMappable(cmap=CMAP_DEAD,
                            norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("Frac. 0", color=TEXT_COL, fontsize=7)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=6)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
        cbar.set_ticks([0, 0.5, dead_thr, 1.0])
        cbar.set_ticklabels(["actif", "50%", "mort", "100%"])

        # ── Col 2 : heatmap temporelle ───────────────
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.set_facecolor(PANEL_BG)

        # Sous-échantillonnage : 200 steps × 80 neurones triés par activité
        t_show  = min(200, T)
        n_show  = min(80, N)
        # Tri des neurones par variance décroissante (les plus intéressants)
        var_order = np.argsort(acts.var(axis=0))[::-1][:n_show]
        heat_data = acts[:t_show, var_order].T   # [n_show, t_show]

        vmax_h = np.percentile(heat_data, 95)
        im = ax2.imshow(
            heat_data,
            cmap=CMAP_SPEC, vmin=0, vmax=max(vmax_h, 1e-6),
            aspect="auto", interpolation="nearest"
        )
        ax2.set_xlabel("Step (temps)", color=TEXT_COL, fontsize=8)
        ax2.set_ylabel(f"Neurone (top {n_show} par variance)",
                       color=TEXT_COL, fontsize=8)
        ax2.set_title(
            f"Activité temporelle — {n_show} neurones × {t_show} steps\n"
            "(triés par variance décroissante — foncé=inactif, chaud=actif fort)",
            color="white", fontsize=10, fontweight="bold", pad=8
        )
        ax2.tick_params(colors="#8899AA", labelsize=7)
        for spine in ax2.spines.values():
            spine.set_edgecolor(GRID_COL)

        cbar2 = plt.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
        cbar2.set_label("Activation", color=TEXT_COL, fontsize=7)
        cbar2.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=6)
        plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    plt.savefig(out("xai_distribution.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_distribution.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Analyse 2 — Neurones spécialisés
# ═══════════════════════════════════════════════
def compute_specialization(act_arrays: dict,
                            situations: list) -> dict:
    """
    Pour chaque couche et chaque neurone, calcule son activation moyenne
    dans chacune des 8 situations.
    Retourne un dict {layer: np.ndarray [N_neurons, 8_situations]}.
    """
    situations_arr = np.array(situations)
    result = {}

    for layer_name, acts in act_arrays.items():
        N    = acts.shape[1]
        S    = len(SITUATION_NAMES)
        means = np.zeros((N, S))
        for si in range(S):
            mask = situations_arr == si
            if mask.sum() > 0:
                means[:, si] = acts[mask].mean(axis=0)
        result[layer_name] = means

    return result


def plot_specialization(spec_data: dict, situations: list,
                        act_arrays: dict):
    """
    3 visualisations de la spécialisation :
      A) Score de spécialisation par neurone (max − mean sur les situations)
      B) Heatmap moyenne d'activation [situation × top-N neurones]
      C) Top-5 neurones les plus spécialisés par couche avec leur profil
    """
    situations_arr = np.array(situations)
    sit_counts = [(situations_arr == si).sum() for si in range(len(SITUATION_NAMES))]

    fig = plt.figure(figsize=(24, 7 * len(LAYER_KEYS)), facecolor=BG)
    fig.suptitle(
        "Neurones spécialisés — Quels neurones s'activent dans quelles situations ?\n"
        "Score de spécialisation = max_situation(act_moy) − mean_situations(act_moy)  "
        "|  Score élevé = neurone très sélectif",
        fontsize=13, fontweight="bold", color="white", y=1.005
    )

    gs = gridspec.GridSpec(len(LAYER_KEYS), 3, figure=fig,
                           wspace=0.42, hspace=0.60,
                           width_ratios=[1.2, 2, 1.8])

    for row, layer_name in enumerate(LAYER_KEYS):
        means  = spec_data[layer_name]   # [N, S]
        acts   = act_arrays[layer_name]  # [T, N]
        N      = means.shape[0]

        # Score de spécialisation : max − mean
        spec_score = means.max(axis=1) - means.mean(axis=1)   # [N]
        top_idx    = np.argsort(spec_score)[::-1]             # trié décroissant

        # ── Col 0 : distribution des scores de spécialisation ──
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_facecolor(PANEL_BG)

        n_spec_bins = 50
        ax0.hist(spec_score, bins=n_spec_bins,
                 color="#1F618D", alpha=0.7, edgecolor="none")
        ax0.axvline(x=np.percentile(spec_score, 90),
                    color="#E74C3C", linewidth=1.5, linestyle="--",
                    label="90e percentile (très spécialisé)")
        ax0.axvline(x=np.percentile(spec_score, 50),
                    color="#F39C12", linewidth=1.0, linestyle=":",
                    label="médiane")

        apply_style(ax0,
                    title=f"{layer_name.strip()} — Score de spécialisation",
                    xlabel="max − mean des activations par situation",
                    ylabel="Nombre de neurones")
        ax0.legend(fontsize=7, facecolor="#0D1117",
                   edgecolor="#444", labelcolor="white")

        ax0.text(0.97, 0.97,
                 f"Top neurone : n°{top_idx[0]}\n"
                 f"Score max   : {spec_score[top_idx[0]]:.3f}\n"
                 f"Situation   : {SITUATION_NAMES[means[top_idx[0]].argmax()].replace(chr(10),' ')}",
                 transform=ax0.transAxes, va="top", ha="right",
                 color="#FFD700", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#0D1B2A",
                           edgecolor="#F39C12", alpha=0.9))

        # ── Col 1 : heatmap [situation × top-40 neurones] ──
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.set_facecolor(PANEL_BG)

        top40      = top_idx[:40]
        heat       = means[top40, :].T   # [S, 40]
        vmax_h     = np.percentile(means, 95)
        im = ax1.imshow(heat, cmap=CMAP_SPEC,
                        vmin=0, vmax=max(vmax_h, 1e-6),
                        aspect="auto", interpolation="nearest")

        ax1.set_yticks(range(len(SITUATION_NAMES)))
        ax1.set_yticklabels(
            [f"{n.replace(chr(10),' ')}  (n={sit_counts[i]})"
             for i, n in enumerate(SITUATION_NAMES)],
            color=TEXT_COL, fontsize=8
        )
        ax1.set_xlabel("Neurone (top 40 les plus spécialisés)", color=TEXT_COL, fontsize=8)
        ax1.set_title(
            "Activation moyenne par situation × neurone\n"
            "(neurones triés par score de spécialisation décroissant)",
            color="white", fontsize=10, fontweight="bold", pad=8
        )
        ax1.tick_params(colors="#8899AA", labelsize=8)
        for spine in ax1.spines.values():
            spine.set_edgecolor(GRID_COL)

        cbar1 = plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.03)
        cbar1.set_label("Activation moy.", color=TEXT_COL, fontsize=7)
        cbar1.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=6)
        plt.setp(cbar1.ax.yaxis.get_ticklabels(), color=TEXT_COL)

        # ── Col 2 : profil des top-5 neurones ────────
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.set_facecolor(PANEL_BG)

        top5      = top_idx[:5]
        sit_pos   = np.arange(len(SITUATION_NAMES))
        bar_w     = 0.15
        palette   = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8"]

        for rank, nidx in enumerate(top5):
            offset = (rank - 2) * bar_w
            ax2.bar(sit_pos + offset, means[nidx],
                    width=bar_w * 0.9,
                    color=palette[rank], alpha=0.85,
                    label=f"Neurone #{nidx} (score={spec_score[nidx]:.2f})",
                    edgecolor="#0D1117")

        ax2.set_xticks(sit_pos)
        ax2.set_xticklabels(
            [s.replace("\n", " ") for s in SITUATION_NAMES],
            rotation=35, ha="right", color=TEXT_COL, fontsize=7.5
        )
        ax2.legend(fontsize=7, facecolor="#0D1117",
                   edgecolor="#444", labelcolor="white",
                   loc="upper right")
        apply_style(ax2,
                    title="Profil des 5 neurones les plus spécialisés",
                    ylabel="Activation moyenne")
        ax2.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.5)

    plt.savefig(out("xai_specialization.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_specialization.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Analyse 3 — t-SNE / UMAP des activations
# ═══════════════════════════════════════════════
def _run_tsne(data: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    import sklearn
    from packaging import version

    # n_iter renommé max_iter dans scikit-learn >= 1.4
    iter_kwarg = (
        "max_iter"
        if version.parse(sklearn.__version__) >= version.parse("1.4")
        else "n_iter"
    )

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, data.shape[0] - 1),
        learning_rate="auto",
        init="pca",
        random_state=42,
        **{iter_kwarg: 1000},
    )
    return tsne.fit_transform(data)


def _run_umap(data: np.ndarray) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        )
        return reducer.fit_transform(data)
    except ImportError:
        print("  [WARN] umap-learn non installé. "
              "Installez-le avec : pip install umap-learn --break-system-packages\n"
              "  Fallback sur t-SNE.")
        return _run_tsne(data)


def plot_projection(act_arrays: dict,
                    situations: list,
                    actions: list,
                    scores: list,
                    method: str = "tsne"):
    """
    Projette les activations de chaque couche en 2D.
    3 colorisations :
      - par situation de jeu
      - par action choisie
      - par score courant
    """
    situations_arr = np.array(situations)
    actions_arr    = np.array(actions)
    scores_arr     = np.array(scores, dtype=float)

    method_label = "t-SNE" if method == "tsne" else "UMAP"

    # Sous-échantillonnage si trop de points (t-SNE lent)
    MAX_POINTS = 3000
    T_total    = len(situations)
    if T_total > MAX_POINTS:
        idx = np.random.choice(T_total, MAX_POINTS, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(T_total)

    fig = plt.figure(figsize=(24, 8 * len(LAYER_KEYS)), facecolor=BG)
    fig.suptitle(
        f"Projection {method_label} des activations internes — "
        "Chaque point = un état de jeu\n"
        "Gauche : colorié par situation  |  "
        "Centre : colorié par action choisie  |  "
        "Droite : colorié par score",
        fontsize=13, fontweight="bold", color="white", y=1.005
    )

    gs = gridspec.GridSpec(len(LAYER_KEYS), 3, figure=fig,
                           wspace=0.30, hspace=0.55)

    sit_colors_map = {
        0: "#E74C3C",   # Danger N
        1: "#F39C12",   # Danger E
        2: "#F1C40F",   # Danger S
        3: "#2ECC71",   # Danger W
        4: "#3498DB",   # Food H
        5: "#9B59B6",   # Food V
        6: "#1ABC9C",   # Serpent long
        7: "#95A5A6",   # Neutre
    }

    for row, layer_name in enumerate(LAYER_KEYS):
        acts_full = act_arrays[layer_name]   # [T, N]
        acts_sub  = acts_full[idx]

        print(f"  [{method_label}] Couche {row+1}/3 — "
              f"{acts_sub.shape[0]} points × {acts_sub.shape[1]} dims…")

        if method == "tsne":
            proj = _run_tsne(acts_sub)
        else:
            proj = _run_umap(acts_sub)

        x, y         = proj[:, 0], proj[:, 1]
        sits_sub     = situations_arr[idx]
        actions_sub  = actions_arr[idx]
        scores_sub   = scores_arr[idx]

        ALPHA = 0.55
        SIZE  = 8

        # ── Gauche : par situation ────────────────────
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_facecolor(PANEL_BG)

        for si, sname in enumerate(SITUATION_NAMES):
            mask = sits_sub == si
            if mask.sum() == 0:
                continue
            ax0.scatter(x[mask], y[mask],
                        c=sit_colors_map[si], s=SIZE, alpha=ALPHA,
                        label=f"{sname.replace(chr(10),' ')} ({mask.sum()})",
                        edgecolors="none")

        ax0.legend(fontsize=6.5, facecolor="#0D1117",
                   edgecolor="#444", labelcolor="white",
                   markerscale=2, loc="best",
                   framealpha=0.85)
        apply_style(ax0,
                    title=f"{layer_name.strip()} — {method_label}\nColoré par situation",
                    xlabel=f"{method_label}-1", ylabel=f"{method_label}-2")

        # ── Centre : par action choisie ───────────────
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.set_facecolor(PANEL_BG)

        for ai, aname in enumerate(ACTION_NAMES):
            mask = actions_sub == ai
            if mask.sum() == 0:
                continue
            ax1.scatter(x[mask], y[mask],
                        c=ACTION_COLORS[ai], s=SIZE, alpha=ALPHA,
                        label=f"{aname} ({mask.sum()})",
                        edgecolors="none")

        ax1.legend(fontsize=7, facecolor="#0D1117",
                   edgecolor="#444", labelcolor="white",
                   markerscale=2, loc="best",
                   framealpha=0.85)
        apply_style(ax1,
                    title=f"{layer_name.strip()} — {method_label}\nColoré par action choisie",
                    xlabel=f"{method_label}-1", ylabel=f"{method_label}-2")

        # ── Droite : par score ────────────────────────
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.set_facecolor(PANEL_BG)

        CMAP_SCORE = LinearSegmentedColormap.from_list(
            "score", ["#0D1B2A", "#1F618D", "#2ECC71", "#F39C12", "#E74C3C"]
        )
        sc = ax2.scatter(x, y, c=scores_sub,
                         cmap=CMAP_SCORE, s=SIZE, alpha=ALPHA,
                         edgecolors="none",
                         vmin=scores_sub.min(), vmax=max(scores_sub.max(), 1))

        cbar = plt.colorbar(sc, ax=ax2, fraction=0.04, pad=0.03)
        cbar.set_label("Score courant", color=TEXT_COL, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

        apply_style(ax2,
                    title=f"{layer_name.strip()} — {method_label}\nColoré par score",
                    xlabel=f"{method_label}-1", ylabel=f"{method_label}-2")

    fname = f"xai_{method}.png"
    plt.savefig(out(fname), dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out(fname)}")
    plt.show()


# ═══════════════════════════════════════════════
#  Point d'entrée
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="XAI — Analyse des activations internes DQN Snake"
    )
    parser.add_argument("--distribution",   action="store_true",
                        help="Histogrammes + neurones morts + heatmap temporelle")
    parser.add_argument("--specialization", action="store_true",
                        help="Neurones spécialisés par situation de jeu")
    parser.add_argument("--tsne",           action="store_true",
                        help="Projection t-SNE des activations")
    parser.add_argument("--umap",           action="store_true",
                        help="Projection UMAP des activations (plus rapide que t-SNE)")
    parser.add_argument("--model",    type=str, default="best",
                        help="Modèle : 'best' | 'final' | 'epXXXX' (défaut : best)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Épisodes de collecte (défaut : 10)")
    args = parser.parse_args()

    run_all = not (args.distribution or args.specialization
                   or args.tsne or args.umap)

    # ── Chargement ────────────────────────────────
    agent     = load_agent(args.model)
    env       = SnakeEnv()
    collector = ActivationCollector(agent)

    # ── Collecte des activations ──────────────────
    print(f"\n[XAI] Collecte sur {args.episodes} épisode(s)…")
    states, actions, situations, scores = collect_episodes(
        agent, env, collector, n_episodes=args.episodes
    )
    act_arrays = collector.get_arrays()
    collector.remove()

    total_steps = len(actions)
    print(f"[XAI] {total_steps} steps collectés.\n")

    # ── Analyses ──────────────────────────────────
    if run_all or args.distribution:
        print("[XAI] ── Distribution des activations ──────────────")
        plot_distribution(act_arrays)

    if run_all or args.specialization:
        print("[XAI] ── Neurones spécialisés ───────────────────────")
        spec = compute_specialization(act_arrays, situations)
        plot_specialization(spec, situations, act_arrays)

    if run_all or args.tsne:
        print("[XAI] ── t-SNE des activations ──────────────────────")
        try:
            from sklearn.manifold import TSNE
            plot_projection(act_arrays, situations, actions, scores,
                            method="tsne")
        except ImportError:
            print("  [WARN] scikit-learn non installé.")
            print("  Installez : pip install scikit-learn --break-system-packages")

    if run_all or args.umap:
        print("[XAI] ── UMAP des activations ───────────────────────")
        plot_projection(act_arrays, situations, actions, scores,
                        method="umap")

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# python xai_activations.py                          # tout (10 épisodes)
# python xai_activations.py --distribution           # rapide, pas de sklearn
# python xai_activations.py --specialization
# python xai_activations.py --tsne --episodes 20     # plus de données = meilleure projection
# python xai_activations.py --umap --episodes 20     # plus rapide que t-SNE
# python xai_activations.py --model final