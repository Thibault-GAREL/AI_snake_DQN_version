"""
xai_shap.py — Analyse XAI : SHAP (SHapley Additive exPlanations) pour DQN Snake
==================================================================================
Utilise shap.DeepExplainer (compatible PyTorch) pour attribuer à chacune des
16 features sa contribution à chaque décision Q-value.

4 visualisations :
  1. Beeswarm plot  — vue globale : impact de chaque feature sur l'ensemble
                      des décisions (sur N états collectés)
  2. Waterfall plot — vue locale  : décomposition d'une décision précise
                      (état le plus représentatif de chaque situation)
  3. Force plot     — vue locale interactive (HTML) : visualisation additive
                      des contributions feature par feature
  4. Summary heatmap — matrice SHAP [feature × action] moyennée sur tous
                       les états : quelle feature pèse sur quelle action ?

Installation requise :
    pip install shap --break-system-packages

Usage :
    python xai_shap.py                    # toutes les visualisations
    python xai_shap.py --beeswarm         # beeswarm global
    python xai_shap.py --waterfall        # waterfall par situation
    python xai_shap.py --force            # force plots (HTML)
    python xai_shap.py --heatmap          # summary heatmap
    python xai_shap.py --model best       # modèle (défaut : best)
    python xai_shap.py --episodes 15      # épisodes de collecte (défaut : 12)
    python xai_shap.py --background 200   # taille du background SHAP (défaut : 150)
"""

import argparse
import os
import math
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

import torch

import pygame

warnings.filterwarnings("ignore")

from dql import DQNAgent, get_device, ACTION_DIM, STATE_DIM
from main import SnakeEnv
import snake as game

# ── Pygame headless ─────────────────────────────
pygame.init()
game.show    = False
game.display = None

# ── Dossier de sortie ────────────────────────────
OUT_DIR = "xai_shap"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ═══════════════════════════════════════════════
#  Constantes
# ═══════════════════════════════════════════════
FEATURE_NAMES = [
    # Danger distances [0:8]
    "Mur N",       "Mur NE",      "Mur E",       "Mur SE",
    "Mur S",       "Mur SW",      "Mur W",       "Mur NW",
    # Food distances [8:16]
    "Food N",      "Food NE",     "Food E",      "Food SE",
    "Food S",      "Food SW",     "Food W",      "Food NW",
    # Food delta [16:18]
    "Food dX",     "Food dY",
    # Danger binaire [18:22]
    "Dng bin N",   "Dng bin E",   "Dng bin S",   "Dng bin W",
    # Direction one-hot [22:26]
    "Dir UP",      "Dir RIGHT",   "Dir DOWN",    "Dir LEFT",
    # Contexte [26:28]
    "Longueur",    "Urgence",
]
ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

SITUATION_NAMES = [
    "Danger N", "Danger E", "Danger S", "Danger W",
    "Food alignée H", "Food alignée V", "Serpent long", "Neutre",
]
SITUATION_COLORS = [
    "#E74C3C", "#F39C12", "#F1C40F", "#2ECC71",
    "#3498DB", "#9B59B6", "#1ABC9C", "#95A5A6",
]

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

# Colormap divergente rouge–blanc–bleu pour SHAP (négatif=rouge, positif=bleu)
CMAP_SHAP = LinearSegmentedColormap.from_list(
    "shap_div", ["#C0392B", "#E8A090", "#F5F5F5", "#90C8E8", "#1A5276"]
)
CMAP_ABS = LinearSegmentedColormap.from_list(
    "shap_abs", ["#0D1B2A", "#154360", "#1F618D", "#F39C12", "#E74C3C"]
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


def _classify_situation(state: list, env: SnakeEnv) -> int:
    """Même classifieur que dans xai_activations.py."""
    diag       = math.sqrt(game.width**2 + game.height**2)
    DANGER_THR = 50 / diag
    d_n, d_e, d_s, d_w = state[0], state[2], state[4], state[6]
    food_h  = state[10] + state[14]
    food_v  = state[8]  + state[12]
    slen    = env.my_snake.lenght
    if d_n <= DANGER_THR and d_n > 0: return 0
    if d_e <= DANGER_THR and d_e > 0: return 1
    if d_s <= DANGER_THR and d_s > 0: return 2
    if d_w <= DANGER_THR and d_w > 0: return 3
    if food_h > 0:                     return 4
    if food_v > 0:                     return 5
    if slen >= 5:                      return 6
    return 7


def collect_states(agent: DQNAgent, env: SnakeEnv,
                   n_episodes: int = 12
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Joue n_episodes épisodes en greedy, retourne :
        states_arr    : [T, 16]  float32
        actions_arr   : [T]      int32
        situations_arr: [T]      int32
    """
    all_states     = []
    all_actions    = []
    all_situations = []

    for ep in range(n_episodes):
        state = env.reset()
        done  = False
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32,
                               device=agent.device).unsqueeze(0)
            with torch.no_grad():
                q = agent.online_net(s_t)
            action = int(q.argmax(dim=1).item())

            all_states.append(list(state))
            all_actions.append(action)
            all_situations.append(_classify_situation(state, env))

            state, _, done, info = env.step(action)

        print(f"  [Collect] Épisode {ep+1}/{n_episodes} — score {info['score']} "
              f"| total steps : {len(all_states)}")

    return (
        np.array(all_states,     dtype=np.float32),
        np.array(all_actions,    dtype=np.int32),
        np.array(all_situations, dtype=np.int32),
    )


def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    if title:
        ax.set_title(title, color="white", fontsize=11,
                     fontweight="bold", pad=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


# ═══════════════════════════════════════════════
#  Calcul des valeurs SHAP
# ═══════════════════════════════════════════════
def compute_shap_values(agent: DQNAgent,
                        states: np.ndarray,
                        background_size: int = 150
                        ) -> tuple:
    """
    Calcule les valeurs SHAP via shap.DeepExplainer.

    DeepExplainer utilise une version efficace d'Integrated Gradients
    adaptée aux réseaux PyTorch. Il compare chaque état à un ensemble de
    'background samples' (référence neutre) pour estimer la contribution
    marginale de chaque feature (valeur de Shapley).

    Retourne :
        shap_values : list[np.ndarray]  — un array [T,16] par action
        expected_val: np.ndarray        — valeur de base E[f(x)] par action
        background_t: torch.Tensor      — background utilisé
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap non installé.\n"
            "Installez-le avec : pip install shap --break-system-packages"
        )

    # ── Background : sous-ensemble représentatif ──
    # On utilise un k-means sur les états pour avoir un fond diversifié
    bg_size = min(background_size, len(states))
    bg_idx  = np.random.choice(len(states), bg_size, replace=False)
    bg_np   = states[bg_idx]
    background_t = torch.tensor(bg_np, dtype=torch.float32,
                                 device=agent.device)

    print(f"  [SHAP] Background : {bg_size} états de référence")
    print(f"  [SHAP] Calcul sur {len(states)} états…")

    # ── DeepExplainer ─────────────────────────────
    # Le modèle doit être en mode eval avec BatchNorm désactivée sur single sample
    # On wrappe pour gérer le batch dimension
    agent.online_net.eval()

    explainer   = shap.DeepExplainer(agent.online_net, background_t)
    states_t    = torch.tensor(states, dtype=torch.float32,
                               device=agent.device)

    # shap_values : liste de 4 arrays [T, 28], un par output (action)
    # check_additivity=False nécessaire car LayerNorm n'est pas supporté par DeepExplainer
    shap_values = explainer.shap_values(states_t, check_additivity=False)
    expected    = explainer.expected_value

    # ── Normalisation de la sortie SHAP ──────────────────────────────────────
    # SHAP 0.49+ retourne un seul array [T, STATE_DIM, ACTION_DIM] au lieu
    # d'une liste. On normalise vers : liste de ACTION_DIM arrays [T, STATE_DIM].

    if isinstance(expected, torch.Tensor):
        expected = expected.cpu().numpy()

    T  = len(states)       # nombre d'états
    F  = STATE_DIM         # 16 features
    A  = ACTION_DIM        # 4 actions

    # Conversion en numpy
    if isinstance(shap_values, (list, tuple)):
        shap_np = [
            sv.cpu().numpy() if isinstance(sv, torch.Tensor) else np.array(sv)
            for sv in shap_values
        ]
    else:
        shap_np = (shap_values.cpu().numpy()
                   if isinstance(shap_values, torch.Tensor)
                   else np.array(shap_values))

    # ── Cas 1 : liste de A arrays ─────────────────
    if isinstance(shap_np, list):
        result = []
        for sv in shap_np:
            if sv.shape == (T, F):
                result.append(sv)
            elif sv.shape == (F, T):
                result.append(sv.T)
            else:
                result.append(sv)
        shap_values = result

    # ── Cas 2 : un seul array numpy ──────────────
    else:
        arr = shap_np
        print(f"  [SHAP] Raw shape : {arr.shape}  — normalisation en cours…")

        if arr.ndim == 3:
            if arr.shape == (T, F, A):      # [T, F, A]  → le plus courant en 0.49
                shap_values = [arr[:, :, ai] for ai in range(A)]
            elif arr.shape == (A, T, F):    # [A, T, F]
                shap_values = [arr[ai] for ai in range(A)]
            elif arr.shape == (A, F, T):    # [A, F, T]
                shap_values = [arr[ai].T for ai in range(A)]
            elif arr.shape == (T, A, F):    # [T, A, F]
                shap_values = [arr[:, ai, :] for ai in range(A)]
            else:
                # Fallback générique : identifier l'axe de taille A
                a_ax = arr.shape.index(A) if A in arr.shape else 2
                shap_values = [
                    np.take(arr, ai, axis=a_ax) for ai in range(A)
                ]
                # S'assurer que chaque sv est [T, F]
                shap_values = [
                    sv if sv.shape == (T, F) else sv.T
                    for sv in shap_values
                ]

        elif arr.ndim == 2:
            if arr.shape == (T, F):
                shap_values = [arr] * A
            elif arr.shape == (F, T):
                shap_values = [arr.T] * A
            elif arr.shape == (T, A):
                # SHAP résumé par action — on ne peut pas reconstruire les features
                raise ValueError(
                    f"Shape SHAP (T, A) = {arr.shape} : "
                    "DeepExplainer a retourné un résumé par action, "
                    "pas les contributions par feature. "
                    "Essayez de réduire background_size."
                )
            else:
                raise ValueError(f"Shape SHAP 2D inattendue : {arr.shape}")
        else:
            raise ValueError(f"Format SHAP non reconnu : ndim={arr.ndim}, shape={arr.shape}")

    print(f"  [SHAP] ✓ Valeurs SHAP normalisées — shape par action : {shap_values[0].shape}")
    return shap_values, expected, background_t


# ═══════════════════════════════════════════════
#  Visualisation 1 — Beeswarm plot
# ═══════════════════════════════════════════════
def plot_beeswarm(shap_values: list, states: np.ndarray):
    """
    Beeswarm plot pour chaque action (4 subplots).
    Chaque point = un état.  Axe X = valeur SHAP (impact).
    Couleur = valeur de la feature (bleu froid = faible, rouge chaud = élevé).

    Lecture :
      - Features en haut = plus d'impact global
      - Point à droite   = la feature pousse vers cette action
      - Point à gauche   = la feature freine cette action
      - Couleur rouge    = feature à haute valeur quand elle impacte
    """
    fig, axes = plt.subplots(1, 4, figsize=(26, 9), facecolor=BG)
    fig.suptitle(
        "SHAP Beeswarm — Impact de chaque feature sur chaque action\n"
        "Chaque point = un état de jeu  |  "
        "Axe X : valeur SHAP (+ = pousse vers l'action, − = freine)  |  "
        "Couleur : valeur normalisée de la feature (froid=faible, chaud=élevé)",
        fontsize=12, fontweight="bold", color="white", y=1.02
    )

    # Feature order : trié par |SHAP| moyen toutes actions confondues
    mean_abs_all = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    feat_order   = np.argsort(mean_abs_all)   # croissant → top en haut du barh

    CMAP_FEAT = matplotlib.colormaps.get_cmap("coolwarm")

    for ai, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG)
        sv    = shap_values[ai]          # [T, 16]
        T     = sv.shape[0]

        y_positions = []
        for rank, fi in enumerate(feat_order):
            shap_fi = sv[:, fi]          # [T]  — valeurs SHAP de la feature fi
            feat_fi = states[:, fi]      # [T]  — valeurs réelles de la feature

            # Jitter vertical pour éviter l'empilement (beeswarm simplifié)
            jitter = np.random.uniform(-0.35, 0.35, size=T)
            y_vals = rank + jitter

            # Normalisation couleur de la feature [0, 1]
            f_min, f_max = feat_fi.min(), feat_fi.max()
            if f_max > f_min:
                feat_norm = (feat_fi - f_min) / (f_max - f_min)
            else:
                feat_norm = np.zeros_like(feat_fi)

            sc = ax.scatter(
                shap_fi, y_vals,
                c=feat_norm, cmap=CMAP_FEAT,
                s=6, alpha=0.55, edgecolors="none",
                vmin=0, vmax=1
            )
            y_positions.append(rank)

        # Ligne zéro
        ax.axvline(x=0, color="#AAAAAA", linewidth=1.0,
                   linestyle="--", alpha=0.6)

        # Axe Y : noms des features
        ax.set_yticks(range(STATE_DIM))
        ax.set_yticklabels(
            [FEATURE_NAMES[fi] for fi in feat_order],
            color=TEXT_COL, fontsize=8
        )

        # Séparateur murs / nourriture sur l'axe Y
        # Compter combien de features "mur" (index < 8) sont dans l'ordre
        n_mur = sum(1 for fi in feat_order if fi < 8)
        ax.axhline(y=n_mur - 0.5, color="#F39C12", linewidth=1.2,
                   linestyle=":", alpha=0.7)

        apply_style(
            ax,
            title=f"Action : {ACTION_NAMES[ai]}",
            xlabel="Valeur SHAP  (impact sur Q-value)"
        )
        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=12,
                     fontweight="bold", pad=9)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)

    # Colorbar globale (valeur de feature)
    sm = ScalarMappable(cmap=CMAP_FEAT, norm=Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.008, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Valeur de la feature (normalisée)",
                   color=TEXT_COL, fontsize=9, rotation=270, labelpad=14)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Faible", "Moyen", "Élevé"])

    plt.subplots_adjust(right=0.90, wspace=0.45)
    plt.savefig(out("xai_shap_beeswarm.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_shap_beeswarm.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Visualisation 2 — Waterfall plot
# ═══════════════════════════════════════════════
def plot_waterfall(shap_values: list, states: np.ndarray,
                   situations: np.ndarray, expected: np.ndarray):
    """
    Pour chaque situation de jeu (8 situations), sélectionne l'état
    le plus représentatif (plus proche du centroïde SHAP) et trace
    un waterfall plot pour l'action la plus probable.

    Waterfall : part de E[f(x)] (valeur de base), ajoute chaque contribution
    SHAP l'une après l'autre → arrive à f(x) (Q-value prédite).
    Bleu = contribution positive, rouge = contribution négative.
    """
    n_sit   = len(SITUATION_NAMES)
    n_cols  = 4
    n_rows  = math.ceil(n_sit / n_cols)

    fig = plt.figure(figsize=(24, 6 * n_rows), facecolor=BG)
    fig.suptitle(
        "SHAP Waterfall — Décomposition d'une décision représentative par situation\n"
        "Chaque barre = contribution d'une feature à la Q-value finale  |  "
        "Départ = E[f(x)] (valeur de base)  →  Arrivée = Q-value prédite",
        fontsize=12, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           wspace=0.45, hspace=0.65)

    for si in range(n_sit):
        row = si // n_cols
        col = si  % n_cols
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL_BG)

        mask = situations == si
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        # ── État représentatif ────────────────────────
        # Sélectionne l'état dont la somme des |SHAP| pour l'action choisie
        # est la plus proche de la médiane (ni cas extrême, ni cas trivial)
        indices = np.where(mask)[0]

        # Action la plus fréquente dans cette situation
        from collections import Counter
        action_counts = Counter()
        for idx in indices:
            sv_all = np.array([shap_values[ai][idx] for ai in range(ACTION_DIM)])
            best_a = int(sv_all.sum(axis=1).argmax())
            action_counts[best_a] += 1
        dominant_action = action_counts.most_common(1)[0][0]

        sv_action   = shap_values[dominant_action]   # [T, 16]
        total_shap  = np.abs(sv_action[indices]).sum(axis=1)
        median_val  = np.median(total_shap)
        rep_idx     = indices[np.argmin(np.abs(total_shap - median_val))]

        shap_rep    = sv_action[rep_idx]        # [16]
        state_rep   = states[rep_idx]           # [16]
        base_val    = float(expected[dominant_action]) if hasattr(expected, '__len__') else float(expected)

        # ── Tri par |SHAP| décroissant ───────────────
        order      = np.argsort(np.abs(shap_rep))[::-1]
        feat_names = [FEATURE_NAMES[i] for i in order]
        feat_vals  = state_rep[order]
        shap_ord   = shap_rep[order]

        # ── Calcul des positions cumulatives ─────────
        cumulative  = np.zeros(len(order) + 1)
        cumulative[0] = base_val
        for k, s in enumerate(shap_ord):
            cumulative[k + 1] = cumulative[k] + s
        final_val = cumulative[-1]

        # ── Tracé des barres waterfall ────────────────
        bar_bottoms = cumulative[:-1].copy()
        bar_heights = shap_ord.copy()

        # Pour les barres négatives, bottom = cumulative après (visuellement correct)
        for k in range(len(shap_ord)):
            if shap_ord[k] < 0:
                bar_bottoms[k] = cumulative[k + 1]
                bar_heights[k] = -shap_ord[k]

        colors_wf = ["#2E86C1" if s >= 0 else "#C0392B" for s in shap_ord]

        bars = ax.barh(
            range(len(order)), bar_heights,
            left=bar_bottoms,
            color=colors_wf, edgecolor="#0D1117",
            height=0.68, alpha=0.88
        )

        # Valeur SHAP annotée sur chaque barre
        for k, (b, h, s) in enumerate(zip(bar_bottoms, bar_heights, shap_ord)):
            x_txt = b + h + (0.01 if s >= 0 else -0.01)
            ha    = "left" if s >= 0 else "right"
            ax.text(x_txt, k, f"{s:+.3f}",
                    va="center", ha=ha, fontsize=6.5,
                    color="#AADDFF" if s >= 0 else "#FFAAAA")

        # ── Lignes de référence ───────────────────────
        ax.axvline(x=base_val,  color="#F39C12", linewidth=1.2,
                   linestyle="--", alpha=0.8, label=f"E[f(x)]={base_val:.2f}")
        ax.axvline(x=final_val, color="#2ECC71", linewidth=1.4,
                   linestyle="-", alpha=0.8, label=f"f(x)={final_val:.2f}")

        # Axe Y : feature + valeur réelle
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(
            [f"{FEATURE_NAMES[order[k]]}  [{feat_vals[k]:.2f}]"
             for k in range(len(order))],
            fontsize=7, color=TEXT_COL
        )

        apply_style(
            ax,
            title=f"{SITUATION_NAMES[si]}\n→ action : {ACTION_NAMES[dominant_action]}",
            xlabel="Q-value (contributions cumulées)"
        )
        ax.set_title(
            f"{SITUATION_NAMES[si]}  →  {ACTION_NAMES[dominant_action]}",
            color=SITUATION_COLORS[si], fontsize=10, fontweight="bold", pad=7
        )
        ax.legend(fontsize=6.5, facecolor="#0D1117",
                  edgecolor="#444", labelcolor="white", loc="lower right")
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.4)

        # Légende couleurs
        pos_p = mpatches.Patch(color="#2E86C1", label="Contribution +")
        neg_p = mpatches.Patch(color="#C0392B", label="Contribution −")
        ax.legend(handles=[pos_p, neg_p],
                  fontsize=6.5, facecolor="#0D1117",
                  edgecolor="#444", labelcolor="white",
                  loc="lower right")

    plt.savefig(out("xai_shap_waterfall.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_shap_waterfall.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Visualisation 3 — Force plot (HTML)
# ═══════════════════════════════════════════════
def plot_force(shap_values: list, states: np.ndarray,
               situations: np.ndarray, expected: np.ndarray):
    """
    Génère un force plot HTML interactif pour chaque situation.
    Le force plot montre comment chaque feature « pousse » ou « freine »
    la Q-value par rapport à la valeur de base E[f(x)].
    Forces rouges = poussent vers le haut, bleues = tirent vers le bas.

    Sauvegarde :
      - xai_shap/xai_force_sit{i}.html   (un par situation)
      - xai_shap/xai_force_global.html   (tous les états empilés)
    """
    try:
        import shap
    except ImportError:
        print("[SKIP] shap non installé — force plot ignoré.")
        return

    shap.initjs()

    for si in range(len(SITUATION_NAMES)):
        mask    = situations == si
        if mask.sum() == 0:
            continue

        indices = np.where(mask)[0][:50]   # 50 états max par situation

        # Action dominante
        from collections import Counter
        action_counts = Counter()
        for idx in indices:
            sv_all = np.array([shap_values[ai][idx] for ai in range(ACTION_DIM)])
            action_counts[int(sv_all.sum(axis=1).argmax())] += 1
        dominant_action = action_counts.most_common(1)[0][0]

        sv_sit   = shap_values[dominant_action][indices]    # [n, 16]
        st_sit   = states[indices]                          # [n, 16]
        base_val = (expected[dominant_action]
                    if hasattr(expected, '__len__')
                    else float(expected))

        html_path = out(f"xai_force_sit{si}_{SITUATION_NAMES[si].replace(' ','_')}.html")

        try:
            fp = shap.force_plot(
                base_value=float(base_val),
                shap_values=sv_sit,
                features=st_sit,
                feature_names=FEATURE_NAMES,
                show=False,
                matplotlib=False,
            )
            shap.save_html(html_path, fp)
            print(f"[XAI] Sauvegarde -> {html_path}")
        except Exception as e:
            print(f"[WARN] Force plot situation {si} : {e}")

    # ── Force plot global (tous les états) ────────
    # On prend l'action la plus choisie globalement
    from collections import Counter
    all_best = []
    for i in range(len(situations)):
        sv_all = np.array([shap_values[ai][i] for ai in range(ACTION_DIM)])
        all_best.append(int(sv_all.sum(axis=1).argmax()))

    global_dominant = Counter(all_best).most_common(1)[0][0]
    sv_global       = shap_values[global_dominant]   # [T, 16]
    base_global     = (expected[global_dominant]
                       if hasattr(expected, '__len__')
                       else float(expected))

    # Limiter à 500 points pour le HTML
    MAX_HTML = 500
    idx_html = np.linspace(0, len(situations) - 1, min(MAX_HTML, len(situations)),
                           dtype=int)

    html_global = out("xai_force_global.html")
    try:
        fp_global = shap.force_plot(
            base_value=float(base_global),
            shap_values=sv_global[idx_html],
            features=states[idx_html],
            feature_names=FEATURE_NAMES,
            show=False,
            matplotlib=False,
        )
        shap.save_html(html_global, fp_global)
        print(f"[XAI] Sauvegarde -> {html_global}")
    except Exception as e:
        print(f"[WARN] Force plot global : {e}")


# ═══════════════════════════════════════════════
#  Visualisation 4 — Summary heatmap
# ═══════════════════════════════════════════════
def plot_summary_heatmap(shap_values: list, states: np.ndarray,
                          situations: np.ndarray):
    """
    4 sous-figures :

    A) Heatmap |SHAP| moyen [feature × action]
       → quelle feature pèse sur quelle action globalement

    B) Heatmap SHAP signé [feature × action] (divergent)
       → signe de l'influence : est-ce que la feature augmente ou diminue la Q-value ?

    C) Barplot du |SHAP| moyen par feature (toutes actions confondues)
       → ranking global d'importance

    D) Heatmap |SHAP| moyen [feature × situation]
       → dans quelle situation chaque feature est-elle la plus influente ?
    """
    # ── Matrices de base ──────────────────────────
    # [16, 4] — valeur SHAP moyenne par feature × action
    mean_abs_matrix  = np.zeros((STATE_DIM, ACTION_DIM))
    mean_sign_matrix = np.zeros((STATE_DIM, ACTION_DIM))

    for ai in range(ACTION_DIM):
        mean_abs_matrix[:, ai]  = np.abs(shap_values[ai]).mean(axis=0)
        mean_sign_matrix[:, ai] = shap_values[ai].mean(axis=0)

    # [16] — importance globale (moyenne des |SHAP| sur toutes les actions)
    global_importance = mean_abs_matrix.mean(axis=1)
    feat_order        = np.argsort(global_importance)   # croissant

    # [16, 8] — |SHAP| moyen par feature × situation
    mean_sit_matrix = np.zeros((STATE_DIM, len(SITUATION_NAMES)))
    for si in range(len(SITUATION_NAMES)):
        mask = situations == si
        if mask.sum() == 0:
            continue
        for ai in range(ACTION_DIM):
            mean_sit_matrix[:, si] += np.abs(shap_values[ai][mask]).mean(axis=0)
        mean_sit_matrix[:, si] /= ACTION_DIM

    # ── Figure ────────────────────────────────────
    fig = plt.figure(figsize=(24, 14), facecolor=BG)
    fig.suptitle(
        "SHAP Summary — Vue globale de l'importance des features\n"
        "Calculé sur l'ensemble des états collectés",
        fontsize=14, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.55)

    # ── A) Heatmap |SHAP| feature × action ───────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(PANEL_BG)

    data_a = mean_abs_matrix[feat_order, :]
    vmax_a = np.percentile(data_a, 97)
    im_a   = ax_a.imshow(data_a, cmap=CMAP_ABS,
                          vmin=0, vmax=max(vmax_a, 1e-6),
                          aspect="auto", interpolation="nearest")

    for fi_r, fi in enumerate(feat_order):
        for ai in range(ACTION_DIM):
            v = mean_abs_matrix[fi, ai]
            c = "white" if v > vmax_a * 0.5 else TEXT_COL
            ax_a.text(ai, fi_r, f"{v:.3f}", ha="center", va="center",
                      color=c, fontsize=8)

    ax_a.set_xticks(range(ACTION_DIM))
    ax_a.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_a.set_yticks(range(STATE_DIM))
    ax_a.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    ax_a.axhline(y=sum(1 for i in feat_order if i < 8) - 0.5,
                 color="#F39C12", linewidth=1.2, linestyle="--", alpha=0.7)

    cbar_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("|SHAP| moyen", color=TEXT_COL, fontsize=8)
    cbar_a.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_a.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_a, title="|SHAP| moyen par feature × action\n"
                             "(importance absolue — plus clair = plus impactant)")

    # ── B) Heatmap SHAP signé ─────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(PANEL_BG)

    data_b = mean_sign_matrix[feat_order, :]
    vabs_b = np.abs(data_b).max()
    norm_b = TwoSlopeNorm(vcenter=0, vmin=-vabs_b, vmax=vabs_b)
    im_b   = ax_b.imshow(data_b, cmap=CMAP_SHAP,
                          norm=norm_b, aspect="auto",
                          interpolation="nearest")

    for fi_r, fi in enumerate(feat_order):
        for ai in range(ACTION_DIM):
            v  = mean_sign_matrix[fi, ai]
            col = "white" if abs(v) > vabs_b * 0.4 else TEXT_COL
            ax_b.text(ai, fi_r, f"{v:+.3f}", ha="center", va="center",
                      color=col, fontsize=8)

    ax_b.set_xticks(range(ACTION_DIM))
    ax_b.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_b.set_yticks(range(STATE_DIM))
    ax_b.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    ax_b.axhline(y=sum(1 for i in feat_order if i < 8) - 0.5,
                 color="#F39C12", linewidth=1.2, linestyle="--", alpha=0.7)

    cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("SHAP signé moyen", color=TEXT_COL, fontsize=8)
    cbar_b.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_b.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_b, title="SHAP signé moyen par feature × action\n"
                             "(bleu = impact +, rouge = impact −)")

    # ── C) Barplot importance globale ─────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(PANEL_BG)

    gi_sorted = global_importance[feat_order]
    norm_c    = Normalize(vmin=gi_sorted.min(), vmax=gi_sorted.max())
    colors_c  = [CMAP_ABS(norm_c(v)) for v in gi_sorted]

    bars = ax_c.barh(range(STATE_DIM), gi_sorted,
                     color=colors_c, edgecolor="#0D1117", height=0.72)

    # Fond alterné
    for k in range(STATE_DIM):
        bg = "#0F2233" if k % 2 == 0 else PANEL_BG
        ax_c.axhspan(k - 0.5, k + 0.5, color=bg, alpha=0.4, zorder=0)

    for k, v in enumerate(gi_sorted):
        ax_c.text(v + 0.0003, k, f"{v:.4f}",
                  va="center", color=TEXT_COL, fontsize=7.5)

    ax_c.set_yticks(range(STATE_DIM))
    ax_c.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8.5)
    ax_c.axhline(y=sum(1 for i in feat_order if i < 8) - 0.5,
                 color="#F39C12", linewidth=1.2, linestyle="--", alpha=0.7)

    # Annotations catégories
    n_mur  = sum(1 for i in feat_order if i < 8)
    n_food = STATE_DIM - n_mur
    ax_c.text(gi_sorted.max() * 0.98, n_mur / 2 - 0.5,
              "MURS", color=ACTION_COLORS[0], fontsize=8,
              fontweight="bold", va="center", ha="right", alpha=0.7)
    ax_c.text(gi_sorted.max() * 0.98, n_mur + n_food / 2 - 0.5,
              "FOOD", color=ACTION_COLORS[2], fontsize=8,
              fontweight="bold", va="center", ha="right", alpha=0.7)

    apply_style(ax_c,
                title="Importance SHAP globale (toutes actions)\n"
                      "Rang ↑ = feature la plus influente sur les décisions",
                xlabel="|SHAP| moyen (toutes actions)")
    ax_c.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)

    # ── D) Heatmap feature × situation ───────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(PANEL_BG)

    data_d = mean_sit_matrix[feat_order, :]
    vmax_d = np.percentile(data_d, 97)
    im_d   = ax_d.imshow(data_d, cmap=CMAP_ABS,
                          vmin=0, vmax=max(vmax_d, 1e-6),
                          aspect="auto", interpolation="nearest")

    ax_d.set_xticks(range(len(SITUATION_NAMES)))
    ax_d.set_xticklabels(
        [s.replace(" ", "\n") for s in SITUATION_NAMES],
        color=TEXT_COL, fontsize=7.5
    )
    ax_d.set_yticks(range(STATE_DIM))
    ax_d.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    ax_d.axhline(y=sum(1 for i in feat_order if i < 8) - 0.5,
                 color="#F39C12", linewidth=1.2, linestyle="--", alpha=0.7)

    # Colorier les colonnes par couleur de situation
    for si, col in enumerate(SITUATION_COLORS):
        ax_d.axvline(x=si - 0.5, color=col, linewidth=0.6, alpha=0.4)

    cbar_d = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label("|SHAP| moyen", color=TEXT_COL, fontsize=8)
    cbar_d.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_d.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_d,
                title="|SHAP| moyen par feature × situation de jeu\n"
                      "(quelle feature devient cruciale dans quelle situation ?)")

    plt.savefig(out("xai_shap_heatmap.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde -> {out('xai_shap_heatmap.png')}")
    plt.show()


# ═══════════════════════════════════════════════
#  Point d'entrée
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="XAI — SHAP pour DQN Snake"
    )
    parser.add_argument("--beeswarm",  action="store_true",
                        help="Beeswarm plot global")
    parser.add_argument("--waterfall", action="store_true",
                        help="Waterfall plot par situation")
    parser.add_argument("--force",     action="store_true",
                        help="Force plots HTML interactifs")
    parser.add_argument("--heatmap",   action="store_true",
                        help="Summary heatmap feature × action / situation")
    parser.add_argument("--model",      type=str, default="best",
                        help="Modèle : 'best'|'final'|'epXXXX' (défaut : best)")
    parser.add_argument("--episodes",   type=int, default=12,
                        help="Épisodes de collecte (défaut : 12)")
    parser.add_argument("--background", type=int, default=150,
                        help="Taille du background SHAP (défaut : 150)")
    args = parser.parse_args()

    run_all = not (args.beeswarm or args.waterfall
                   or args.force or args.heatmap)

    # ── Vérification de shap ──────────────────────
    try:
        import shap
        print(f"[XAI] shap version : {shap.__version__}")
    except ImportError:
        print("[ERREUR] shap non installé.")
        print("         pip install shap --break-system-packages")
        return

    # ── Chargement & collecte ─────────────────────
    agent = load_agent(args.model)
    env   = SnakeEnv()

    print(f"\n[XAI] Collecte sur {args.episodes} épisode(s)…")
    states, actions, situations = collect_states(
        agent, env, n_episodes=args.episodes
    )
    print(f"[XAI] {len(states)} états collectés.\n")

    # ── Calcul SHAP ───────────────────────────────
    print("[XAI] Calcul des valeurs SHAP (DeepExplainer)…")
    shap_values, expected, _ = compute_shap_values(
        agent, states, background_size=args.background
    )

    # ── Visualisations ────────────────────────────
    if run_all or args.beeswarm:
        print("\n[XAI] ── Beeswarm plot ────────────────────────")
        plot_beeswarm(shap_values, states)

    if run_all or args.waterfall:
        print("\n[XAI] ── Waterfall plot ───────────────────────")
        plot_waterfall(shap_values, states, situations, expected)

    if run_all or args.heatmap:
        print("\n[XAI] ── Summary heatmap ──────────────────────")
        plot_summary_heatmap(shap_values, states, situations)

    if run_all or args.force:
        print("\n[XAI] ── Force plots (HTML) ───────────────────")
        plot_force(shap_values, states, situations, expected)

    print(f"\n[XAI] Analyse SHAP terminée. Fichiers dans : {OUT_DIR}/")


if __name__ == "__main__":
    main()


# python xai_shap.py                             # tout (12 épisodes)
# python xai_shap.py --beeswarm                  # le plus informatif
# python xai_shap.py --waterfall
# python xai_shap.py --heatmap
# python xai_shap.py --force                     # génère les HTML
# python xai_shap.py --episodes 25 --background 300  # plus précis
# python xai_shap.py --model final
