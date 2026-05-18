"""
Cluster 2D EMC classes using the common-lines similarity scores.

Two methods are available and can be combined:

  hierarchical  – scipy agglomerative clustering; shows a dendrogram so you
                  can choose a cut threshold visually.  Classes are then
                  assigned by cutting at --threshold (default: show only).

  spectral      – sklearn SpectralClustering on the affinity matrix; requires
                  --n-clusters k.

Usage
-----
  # Just visualise the score matrix and dendrogram:
  python -m emc3.cluster_classes common_lines.h5

  # Cut the dendrogram at distance 0.4 and save:
  python -m emc3.cluster_classes common_lines.h5 --threshold 0.4

  # Spectral clustering into 3 groups:
  python -m emc3.cluster_classes common_lines.h5 --n-clusters 3

  # Both methods at once:
  python -m emc3.cluster_classes common_lines.h5 --threshold 0.4 --n-clusters 3

    emc3/cluster_classes.py — importable module + CLI:

  ┌───────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
  │           Function            │                                  Role                                   │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ load_scores                   │ Read scores, class_ids from common_lines.h5                             │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ hierarchical(scores, linkage) │ scipy agglomerative on 1 − clip(scores, 0, 1); returns linkage matrix Z │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ cut_dendrogram(Z, threshold)  │ Cut Z at a distance threshold → integer labels                          │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ spectral(scores, n_clusters)  │ sklearn SpectralClustering on the clipped affinity matrix               │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ plot_dendrogram               │ Dendrogram with optional red cut line                                   │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ plot_score_matrix             │ Heatmap reordered by cluster, with boundary lines                       │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ save_clusters                 │ Write labels + per-cluster class lists to HDF5                          │
  └───────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘

  The output class_clusters.h5 has class_ids, cluster_labels, and one dataset per cluster (cluster_0, cluster_1, …) listing the member class IDs.
"""

import numpy as np
import h5py
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_scores(common_lines_file):
    """Return (scores, class_ids) from common_lines.h5."""
    with h5py.File(common_lines_file, 'r') as f:
        scores    = f['scores'][()]
        class_ids = f['class_ids'][()]
    return scores, class_ids


def save_clusters(class_ids, labels, output_file):
    """Write cluster labels to HDF5.  Overwrites if file exists."""
    output_file = Path(output_file)
    with h5py.File(output_file, 'w') as f:
        f['class_ids']      = class_ids
        f['cluster_labels'] = labels
        for k in np.unique(labels):
            f[f'cluster_{k}'] = class_ids[labels == k]
    print(f'Saved cluster labels → {output_file}')


# ---------------------------------------------------------------------------
# Distance / affinity helpers
# ---------------------------------------------------------------------------

def to_affinity(scores):
    """Clip scores to [0,1] for use as a precomputed affinity matrix."""
    A = np.clip(scores, 0.0, 1.0).astype(float)
    np.fill_diagonal(A, 1.0)
    return A


def to_distance(scores):
    """Convert clipped scores to a distance matrix in [0,1]."""
    D = 1.0 - np.clip(scores, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def hierarchical(scores, linkage='average'):
    """
    Run scipy agglomerative clustering.

    Parameters
    ----------
    scores  : (N, N) symmetric score matrix
    linkage : 'average' | 'complete' | 'single'

    Returns
    -------
    Z : linkage matrix (scipy format)
    """
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance  import squareform

    D         = to_distance(scores)
    condensed = squareform(D, checks=False)
    return sp_linkage(condensed, method=linkage)


def cut_dendrogram(Z, threshold):
    """Return cluster labels by cutting the linkage at *threshold* distance."""
    from scipy.cluster.hierarchy import fcluster
    labels = fcluster(Z, t=threshold, criterion='distance') - 1  # 0-indexed
    return labels.astype(int)


def spectral(scores, n_clusters, random_state=42):
    """
    Run sklearn SpectralClustering on the clipped affinity matrix.

    Parameters
    ----------
    scores     : (N, N) score matrix
    n_clusters : number of clusters
    """
    from sklearn.cluster import SpectralClustering
    A      = to_affinity(scores)
    labels = SpectralClustering(
        n_clusters    = n_clusters,
        affinity      = 'precomputed',
        assign_labels = 'kmeans',
        random_state  = random_state,
        n_init        = 20,
    ).fit_predict(A)
    return labels.astype(int)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dendrogram(Z, class_ids, ax, threshold=None):
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(
        Z,
        labels       = [str(c) for c in class_ids],
        ax           = ax,
        leaf_rotation= 90,
        color_threshold = threshold if threshold is not None else 0,
    )
    if threshold is not None:
        ax.axhline(threshold, color='r', linestyle='--', linewidth=1,
                   label=f'cut = {threshold:.3f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('class ID')
    ax.set_ylabel('distance  (1 – clipped score)')
    ax.set_title('Hierarchical clustering')


def plot_score_matrix(scores, class_ids, labels, ax):
    """Score matrix heatmap, rows/cols reordered by cluster then class_id."""
    order       = np.lexsort((class_ids, labels))
    S           = scores[np.ix_(order, order)]
    tick_labels = [f'{class_ids[i]}\n(c{labels[i]})' for i in order]

    vmin = np.percentile(S, 1)
    vmax = np.percentile(S, 99)

    im = ax.imshow(S, vmin=vmin, vmax=vmax, cmap='RdBu_r',
                   aspect='auto', interpolation='nearest')
    n  = len(class_ids)
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=90,
                                                 fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_title('Score matrix (reordered by cluster)')

    # draw cluster boundaries
    boundaries = np.where(np.diff(labels[order]))[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color='k', linewidth=0.8)
        ax.axvline(b, color='k', linewidth=0.8)

    plt_ref = ax.get_figure()
    plt_ref.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def print_clusters(class_ids, labels):
    print('\nCluster assignments:')
    for k in np.unique(labels):
        members = class_ids[labels == k]
        print(f'  cluster {k:2d}: classes {list(members)}')


# ---------------------------------------------------------------------------
# Triplet geometric consistency (spherical law of cosines)
# ---------------------------------------------------------------------------

def _sph_triangle_valid_vec(A, B, C, tol=1e-6):
    """
    Vectorised: return bool array — True where (A,B,C) are valid vertex angles
    of a spherical triangle.  All inputs are radians, shape (n_triplets,).

    Uses the dual spherical law of cosines: given vertex angles A, B, C, the
    three implied side lengths cos(a), cos(b), cos(c) must all be in (-1, 1).
    This is necessary and sufficient (unlike the weaker sum-and-inequality test).
    """
    in_range = ((A > tol) & (A < np.pi - tol) &
                (B > tol) & (B < np.pi - tol) &
                (C > tol) & (C < np.pi - tol))
    sA, sB, sC = np.sin(A), np.sin(B), np.sin(C)
    cA, cB, cC = np.cos(A), np.cos(B), np.cos(C)
    safe = in_range & (sB * sC > tol) & (sA * sC > tol) & (sA * sB > tol)
    # replace denominators with 1.0 where unsafe to avoid divide-by-zero warnings
    dBC = np.where(safe, sB * sC, 1.0)
    dAC = np.where(safe, sA * sC, 1.0)
    dAB = np.where(safe, sA * sB, 1.0)
    cos_a = np.where(safe, (cA + cB * cC) / dBC, 2.0)
    cos_b = np.where(safe, (cB + cA * cC) / dAC, 2.0)
    cos_c = np.where(safe, (cC + cA * cB) / dAB, 2.0)
    return (in_range &
            (cos_a > -1 + tol) & (cos_a < 1 - tol) &
            (cos_b > -1 + tol) & (cos_b < 1 - tol) &
            (cos_c > -1 + tol) & (cos_c < 1 - tol))


def _load_top_phi_candidates(common_lines_file, sel, Nrot, n_peaks):
    """
    For every pair (i, j) with i < j in the subset *sel* (local indices into
    class_ids), load the C matrix slice and return the top-*n_peaks* angle
    candidates.

    Returns
    -------
    cands : dict  (i, j) -> ndarray shape (n_peaks, 2)
            Each row is (phi_in_class_i, phi_in_class_j) in radians.
            Only keys with i < j are stored; access (j, i) by swapping columns.
    """
    N_all = len(sel)   # number of classes in the full file — read below

    def _pair_idx(a, b, N):
        return a * N - a * (a + 1) // 2 + b - a - 1

    cands = {}
    scale = 2.0 * np.pi / Nrot

    with h5py.File(common_lines_file, 'r') as f:
        N_file = int(f['N_classes'][()])
        C_ds   = f['C']
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                gi, gj = int(sel[i]), int(sel[j])
                # global pair index (gi < gj guaranteed since i < j and sel is sorted)
                if gi > gj:
                    gi, gj = gj, gi
                    swapped = True
                else:
                    swapped = False
                k   = _pair_idx(gi, gj, N_file)
                C_s = C_ds[k]                            # (Nrot, Nrot)
                # top-n_peaks flat indices
                if n_peaks >= C_s.size:
                    flat = np.arange(C_s.size)
                else:
                    flat = np.argpartition(C_s.ravel(), -n_peaks)[-n_peaks:]
                rows, cols = np.unravel_index(flat, C_s.shape)
                phi_i = rows * scale
                phi_j = cols * scale
                if swapped:
                    phi_i, phi_j = phi_j, phi_i
                cands[(i, j)] = np.stack([phi_i, phi_j], axis=1)

    return cands


def triplet_consistency(common_lines_file, class_subset=None, n_peaks=1):
    """
    Check geometric consistency of common-line angles via spherical triangles.

    For three classes (i, j, k) from the same 3D structure the angle each
    class sees between its two common lines must be a vertex angle of a valid
    spherical triangle.

    Symmetry note
    -------------
    A molecule with K-fold point-group symmetry produces K equivalent peaks
    in the Pearson correlation surface C[φᵢ, φⱼ] per class pair.  ``best_phi``
    stores only the global maximum.  For a triplet, picking inconsistent
    symmetry-related peaks causes the test to fail even for valid triplets.
    Set ``n_peaks`` to the symmetry order K (e.g. 6 for C6, 12 for D6) to
    load the top-K candidates from the stored C matrix and try all K³
    combinations per triplet.

    Parameters
    ----------
    common_lines_file : str or Path
    class_subset      : array-like of config class IDs to test, or None (all)
    n_peaks           : int — number of candidate peaks per pair to consider.
                        1  → fast, uses only best_phi (good for C1 molecules).
                        K  → correct for molecules with symmetry order K.

    Returns
    -------
    pair_scores  : (N, N) float — fraction of consistent triplets per pair
    class_scores : (N,)  float — mean pair score per class
    class_ids    : (N,)  int   — config class IDs for the subset
    """
    with h5py.File(common_lines_file, 'r') as f:
        all_ids  = f['class_ids'][()]
        best_phi = f['best_phi'][()]          # (N_all, N_all, 2)
        Nrot     = int(f['Nrot'][()])

    if class_subset is None:
        sel = np.arange(len(all_ids))
    else:
        sel = np.array([np.where(all_ids == c)[0][0]
                        for c in np.asarray(class_subset)])

    N = len(sel)
    if N < 3:
        raise ValueError('Need at least 3 classes to check triplet consistency')

    # --- build candidate angle sets per pair ------------------------------
    if n_peaks == 1:
        # fast path: use only best_phi
        phi = best_phi[np.ix_(sel, sel)][:, :, 0] * (2.0 * np.pi / Nrot)
        # wrap in same dict structure for uniform code below
        cands = {}
        for i in range(N):
            for j in range(i + 1, N):
                cands[(i, j)] = np.array([[phi[i, j], phi[j, i]]])
    else:
        print(f'  Loading C matrix for {N*(N-1)//2} pairs '
              f'(n_peaks={n_peaks}) …')
        cands = _load_top_phi_candidates(common_lines_file, sel, Nrot, n_peaks)

    # --- enumerate all unordered triplets (i < j < k) --------------------
    from itertools import combinations, product as iproduct

    # precompute all 8 sign-flip combinations once
    _signs = np.array(list(iproduct((1, -1), repeat=3)), dtype=np.float64)  # (8, 3)

    trips  = list(combinations(range(N), 3))
    consistent = np.zeros(len(trips), dtype=bool)

    for t_idx, (i, j, k) in enumerate(trips):
        cij = cands[(i, j)]   # (n_peaks, 2): col0=phi_in_i, col1=phi_in_j
        cik = cands[(i, k)]
        cjk = cands[(j, k)]

        found = False
        for (phi_ij, phi_ji), (phi_ik, phi_ki), (phi_jk, phi_kj) in \
                iproduct(cij, cik, cjk):

            alpha_i = (phi_ij - phi_ik) % np.pi
            alpha_j = (phi_ji - phi_jk) % np.pi
            alpha_k = (phi_ki - phi_kj) % np.pi

            # try all 8 sign choices (α ↔ π−α) at once — vectorised
            A = np.where(_signs[:, 0] > 0, alpha_i, np.pi - alpha_i)
            B = np.where(_signs[:, 1] > 0, alpha_j, np.pi - alpha_j)
            C = np.where(_signs[:, 2] > 0, alpha_k, np.pi - alpha_k)
            if _sph_triangle_valid_vec(A, B, C).any():
                found = True
                break
        consistent[t_idx] = found

    # --- aggregate to pair scores ----------------------------------------
    trips_arr = np.array(trips, dtype=np.int32)
    i_t, j_t, k_t = trips_arr[:, 0], trips_arr[:, 1], trips_arr[:, 2]

    pair_consistent = np.zeros((N, N), dtype=np.float64)
    pair_counts     = np.zeros((N, N), dtype=np.int64)
    c_f = consistent.astype(np.float64)

    for a, b in [(i_t, j_t), (j_t, i_t),
                 (i_t, k_t), (k_t, i_t),
                 (j_t, k_t), (k_t, j_t)]:
        np.add.at(pair_consistent, (a, b), c_f)
        np.add.at(pair_counts,     (a, b), 1)

    with np.errstate(invalid='ignore'):
        pair_scores = np.where(pair_counts > 0,
                               pair_consistent / pair_counts, 0.0)

    class_scores = pair_scores.mean(axis=1)
    return pair_scores, class_scores, all_ids[sel]


def print_consistency(class_scores, class_ids):
    print('\nTriplet-consistency scores (fraction of valid spherical triangles):')
    order = np.argsort(class_scores)[::-1]
    for idx in order:
        print(f'  class {class_ids[idx]:3d}  {class_scores[idx]:.3f}')


def plot_consistency(pair_scores, class_scores, class_ids, axes):
    """
    axes[0] : pair-consistency heatmap
    axes[1] : per-class bar chart
    """
    ax0, ax1 = axes
    n = len(class_ids)
    order = np.argsort(class_scores)[::-1]

    S  = pair_scores[np.ix_(order, order)]
    tl = [str(class_ids[i]) for i in order]

    im = ax0.imshow(S, vmin=0, vmax=1, cmap='YlGn',
                    aspect='auto', interpolation='nearest')
    ax0.set_xticks(range(n)); ax0.set_xticklabels(tl, rotation=90, fontsize=7)
    ax0.set_yticks(range(n)); ax0.set_yticklabels(tl, fontsize=7)
    ax0.set_title('Triplet consistency per pair\n(fraction of valid spherical triangles)')
    ax0.get_figure().colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1.barh(range(n), class_scores[order], color='steelblue')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels([str(class_ids[i]) for i in order], fontsize=7)
    ax1.set_xlabel('mean triplet-consistency score')
    ax1.set_title('Per-class consistency')
    ax1.axvline(0.5, color='r', linestyle='--', linewidth=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('common_lines_file',
                    help='path to common_lines.h5')
    ap.add_argument('--linkage',    default='average',
                    choices=['average', 'complete', 'single'],
                    help='hierarchical linkage method (default: average)')
    ap.add_argument('--threshold',  type=float, default=None,
                    help='cut the dendrogram at this distance to assign labels')
    ap.add_argument('--n-clusters', type=int,   default=None,
                    help='run spectral clustering into this many groups')
    ap.add_argument('--out',        type=str,   default=None,
                    help='output HDF5 for cluster labels '
                         '(default: class_clusters.h5 next to input)')
    ap.add_argument('--classes',    type=int,   nargs='+', default=None,
                    help='subset of class IDs to check triplet consistency '
                         '(default: use all classes, or the computed cluster '
                         'labels when --threshold / --n-clusters is given)')
    ap.add_argument('--n-peaks',   type=int,   default=1,
                    help='number of candidate common-line peaks to consider '
                         'per pair (default: 1 = fast, uses best_phi only). '
                         'Set to the molecule symmetry order K for symmetric '
                         'molecules, e.g. 6 for C6, 12 for D6.')
    args = ap.parse_args()

    fnam      = Path(args.common_lines_file)
    scores, class_ids = load_scores(fnam)
    N         = len(class_ids)

    print(f'Loaded {N} classes from {fnam}')
    print(f'Score range: [{scores[~np.eye(N, dtype=bool)].min():.4f}, '
          f'{scores[~np.eye(N, dtype=bool)].max():.4f}]')

    Z = hierarchical(scores, linkage=args.linkage)

    # ---- decide on labels -----------------------------------------------
    hier_labels     = None
    spectral_labels = None

    if args.threshold is not None:
        hier_labels = cut_dendrogram(Z, args.threshold)
        print(f'\nHierarchical (threshold={args.threshold}):')
        print_clusters(class_ids, hier_labels)

    if args.n_clusters is not None:
        spectral_labels = spectral(scores, args.n_clusters)
        print(f'\nSpectral (k={args.n_clusters}):')
        print_clusters(class_ids, spectral_labels)

    # use whichever labels are available (prefer spectral if both given)
    labels_for_plot = spectral_labels if spectral_labels is not None \
                      else hier_labels if hier_labels is not None \
                      else np.zeros(N, dtype=int)

    # save if any labels were computed
    if hier_labels is not None or spectral_labels is not None:
        out = args.out or str(fnam.parent / 'class_clusters.h5')
        save_clusters(class_ids, labels_for_plot, out)

    # ---- triplet consistency ---------------------------------------------
    # determine which classes to check
    if args.classes is not None:
        consistency_subset = np.array(args.classes)
    elif hier_labels is not None or spectral_labels is not None:
        # run separately per cluster
        consistency_subset = None   # handled per-cluster below
    else:
        consistency_subset = class_ids  # all

    per_cluster_consistency = {}
    if consistency_subset is not None:
        print(f'\nChecking triplet consistency for '
              f'{len(consistency_subset)} classes …')
        pair_scores, class_scores, cs_ids = triplet_consistency(
            fnam, class_subset=consistency_subset, n_peaks=args.n_peaks)
        print_consistency(class_scores, cs_ids)
        per_cluster_consistency['subset'] = (pair_scores, class_scores, cs_ids)
    elif labels_for_plot is not None:
        for k in np.unique(labels_for_plot):
            members = class_ids[labels_for_plot == k]
            if len(members) < 3:
                continue
            print(f'\nChecking triplet consistency for cluster {k} '
                  f'({len(members)} classes) …')
            ps, cs, ids = triplet_consistency(fnam, class_subset=members,
                                              n_peaks=args.n_peaks)
            print_consistency(cs, ids)
            per_cluster_consistency[k] = (ps, cs, ids)

    # ---- plot ------------------------------------------------------------
    has_labels      = args.threshold is not None or args.n_clusters is not None
    has_consistency = bool(per_cluster_consistency)

    n_top   = 1 + has_labels                        # dendrogram + score matrix
    n_extra = 2 * len(per_cluster_consistency)       # pair + bar per cluster
    fig, axes = plt.subplots(
        1, n_top + n_extra,
        figsize=(7 * (n_top + n_extra), max(4, N // 4))
    )
    axes = np.atleast_1d(axes)

    plot_dendrogram(Z, class_ids, axes[0], threshold=args.threshold)

    col = 1
    if has_labels:
        plot_score_matrix(scores, class_ids, labels_for_plot, axes[col])
        col += 1

    for key, (ps, cs, ids) in per_cluster_consistency.items():
        plot_consistency(ps, cs, ids, axes[col:col + 2])
        title = f'cluster {key}' if key != 'subset' else 'selected subset'
        axes[col].set_title(f'Triplet consistency — {title}\n'
                            + axes[col].get_title().split('\n', 1)[-1])
        col += 2

    fig.tight_layout()
    plt.show()
