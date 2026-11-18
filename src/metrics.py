"""Evaluation metrics for PathSnap candidate classification and route reconstruction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def compute_candidate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute candidate-level classification metrics (binary: is_reference_arc)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    return {
        "cand_accuracy": float(accuracy_score(y_true, y_pred)),
        "cand_precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "cand_recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "cand_f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "cand_macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "cand_macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "cand_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cand_log_loss": float(log_loss(y_true, y_prob_clipped, labels=[0, 1])),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n_candidates": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
    }


def compute_observation_top1_and_miss(
    cand_df: pd.DataFrame,
    scores: np.ndarray,
    obs_meta_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Compute observation-level Top-1 accuracy (including candidate misses in denominator).

    Parameters
    ----------
    cand_df : DataFrame with columns ['trip_id', 'obs_id', 'arc_id', 'is_target']
    scores : 1D array of predicted probability/score per candidate row
    obs_meta_df : DataFrame with one row per observation:
        ['trip_id', 'obs_id', 'ref_arc_id', 'is_scored', 'candidate_has_ref']

    Returns
    -------
    metrics : dict with:
        - obs_top1_overall: correct / total_scored_obs (candidate misses count as errors!)
        - obs_top1_given_covered: correct / covered_scored_obs
        - candidate_miss_rate: missed_scored_obs / total_scored_obs
        - n_total_obs, n_scored_obs, n_covered_scored_obs, n_missed_scored_obs
    pred_obs_df : DataFrame of one top-1 predicted arc per observation
    """
    work = cand_df[["trip_id", "obs_id", "arc_id"]].copy()
    work["score"] = np.asarray(scores, dtype=np.float64)
    # Deterministic tie-breaking: highest score, then original row order (sorted by distance)
    work["row_ord"] = np.arange(len(work))
    top_idx = (
        work.sort_values(["trip_id", "obs_id", "score", "row_ord"], ascending=[True, True, False, True])
        .drop_duplicates(subset=["trip_id", "obs_id"], keep="first")
    )
    pred_obs = obs_meta_df.merge(
        top_idx[["trip_id", "obs_id", "arc_id", "score"]].rename(columns={"arc_id": "pred_arc_id", "score": "pred_score"}),
        on=["trip_id", "obs_id"],
        how="left",
    )
    pred_obs["pred_arc_id"] = pred_obs["pred_arc_id"].fillna(-1).astype(np.int64)
    pred_obs["pred_score"] = pred_obs["pred_score"].fillna(0.0).astype(np.float64)

    scored = pred_obs[pred_obs["is_scored"]].copy()
    n_scored = len(scored)
    n_covered = int(scored["candidate_has_ref"].sum())
    n_missed = int(n_scored - n_covered)
    n_correct = int((scored["pred_arc_id"] == scored["ref_arc_id"]).sum())

    metrics = {
        "n_total_obs": int(len(pred_obs)),
        "n_scored_obs": int(n_scored),
        "n_covered_scored_obs": int(n_covered),
        "n_missed_scored_obs": int(n_missed),
        "candidate_miss_rate": float(n_missed / max(n_scored, 1)),
        "obs_top1_overall": float(n_correct / max(n_scored, 1)),
        "obs_top1_given_covered": float(n_correct / max(n_covered, 1)),
    }
    return metrics, pred_obs


def compute_route_continuity_for_trip(
    pred_arcs: np.ndarray,
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    adjacency: Optional[Dict[int, set]] = None,
) -> Dict[str, float]:
    """Compute route continuity for one trip's predicted observation-level arc sequence.

    Consecutive valid predicted arcs (a_prev, a_curr) are:
      - same_arc: a_prev == a_curr (continuing on the same directed arc)
      - direct_connected: to_node(a_prev) == from_node(a_curr) (strict 1-hop directed continuation)
      - two_hop_connected: a_curr reachable from to_node(a_prev) via 1 intermediate arc (covers short GPS skips)
    Primary Route Continuity = fraction of consecutive valid transitions that are same_arc OR direct_connected.
    """
    n_arcs = len(from_nodes)
    valid = [int(a) for a in pred_arcs if 0 <= int(a) < n_arcs]
    if len(valid) <= 1:
        return {
            "route_continuity": 1.0,
            "transition_continuity_changes_only": 1.0,
            "n_transitions": 0,
            "n_continuous": 0,
            "n_arc_changes": 0,
            "n_connected_changes": 0,
        }

    n_trans = len(valid) - 1
    n_cont = 0
    n_changes = 0
    n_conn_changes = 0
    for a1, a2 in zip(valid[:-1], valid[1:]):
        if a1 == a2:
            n_cont += 1
        else:
            n_changes += 1
            if to_nodes[a1] == from_nodes[a2]:
                n_cont += 1
                n_conn_changes += 1

    return {
        "route_continuity": float(n_cont / n_trans),
        "transition_continuity_changes_only": float(n_conn_changes / max(n_changes, 1)),
        "n_transitions": int(n_trans),
        "n_continuous": int(n_cont),
        "n_arc_changes": int(n_changes),
        "n_connected_changes": int(n_conn_changes),
    }


def compress_arc_sequence(arc_seq: np.ndarray) -> List[int]:
    """Collapse consecutive duplicate arc IDs (and drop invalid -1 entries) into a traversed arc list."""
    out: List[int] = []
    for a in arc_seq:
        ai = int(a)
        if ai < 0:
            continue
        if not out or out[-1] != ai:
            out.append(ai)
    return out


def compute_sequence_metrics_for_trip(
    pred_obs_arcs: np.ndarray,
    ref_route_arcs: np.ndarray,
    arc_lengths: np.ndarray,
) -> Dict[str, float]:
    """Compute sequence-level route reconstruction metrics comparing compressed predicted route vs .route.

    Metrics:
      - arc_set_f1: F1 score over unique arc IDs in compressed predicted route vs reference .route
      - length_weighted_f1 (Route Match F1 / RMF-style): F1 weighted by arc length in meters
      - lcs_f1: Longest Common Subsequence F1 preserving chronological arc traversal order
    """
    pred_seq = compress_arc_sequence(pred_obs_arcs)
    ref_seq = [int(a) for a in ref_route_arcs if 0 <= int(a) < len(arc_lengths)]

    pred_set = set(pred_seq)
    ref_set = set(ref_seq)
    inter = pred_set.intersection(ref_set)

    prec = len(inter) / max(len(pred_set), 1)
    rec = len(inter) / max(len(ref_set), 1)
    arc_set_f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    # Length-weighted precision/recall/F1
    len_inter = float(sum(arc_lengths[a] for a in inter))
    len_pred = float(sum(arc_lengths[a] for a in pred_set))
    len_ref = float(sum(arc_lengths[a] for a in ref_set))
    lw_prec = len_inter / max(len_pred, 1e-9)
    lw_rec = len_inter / max(len_ref, 1e-9)
    lw_f1 = 2 * lw_prec * lw_rec / max(lw_prec + lw_rec, 1e-12)

    # Order-aware LCS F1 on compressed sequences (typically ~50-200 arcs, very fast DP)
    m, n = len(pred_seq), len(ref_seq)
    if m == 0 or n == 0:
        lcs_f1 = 0.0
    else:
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            pi = pred_seq[i - 1]
            for j in range(1, n + 1):
                if pi == ref_seq[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
            prev, curr = curr, prev
        lcs_len = prev[n]
        lcs_prec = lcs_len / m
        lcs_rec = lcs_len / n
        lcs_f1 = 2 * lcs_prec * lcs_rec / max(lcs_prec + lcs_rec, 1e-12)

    return {
        "seq_arc_precision": float(prec),
        "seq_arc_recall": float(rec),
        "seq_arc_f1": float(arc_set_f1),
        "seq_length_weighted_f1": float(lw_f1),
        "seq_lcs_f1": float(lcs_f1),
        "pred_compressed_len": int(m),
        "ref_route_len": int(n),
    }
