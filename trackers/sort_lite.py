"""
trackers/sort_lite.py
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.config import IOU_THRESHOLD, MAX_MISSED_FRAMES, BBox, Track

_DIM_X = 7
_DIM_Z = 4

_Q_DIAG = np.array([1., 1., 1., 1e-2, 1e-2, 1e-2, 1e-4], dtype=np.float64)
_R_DIAG = np.array([1., 1., 10., 10.], dtype=np.float64)
_P0_VEL_SCALE = 10.0

_HIGH_IOU = 0.35
_LOW_IOU  = 0.15

_MAHA_GATE = 9.4877

_MIN_HIT_STREAK = 2


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a : (N, 4)  [x1,y1,x2,y2]
    b : (M, 4)
    → (N, M) float32 IoU matrix
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _bboxes_to_arr(bboxes: List[BBox]) -> np.ndarray:
    return np.array([[b.x1, b.y1, b.x2, b.y2] for b in bboxes], dtype=np.float32)


class _KalmanBox:
    """
    Constant-velocity KF on [cx, cy, s, r] measurements (s=area, r=aspect).
    Uses float64 internally for numerical stability.
    """

    _F = np.eye(_DIM_X, dtype=np.float64)
    for _i in range(4):
        _F[_i, _i + 3] = 1.0

    _H = np.zeros((_DIM_Z, _DIM_X), dtype=np.float64)
    np.fill_diagonal(_H, 1.0)

    _Q = np.diag(_Q_DIAG)
    _R = np.diag(_R_DIAG)

    def __init__(self, bbox: BBox) -> None:
        cx, cy, s, r = self._meas(bbox)
        self.x = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(_DIM_X, dtype=np.float64)
        self.P[4:, 4:] *= _P0_VEL_SCALE
        # Innovation covariance (updated each predict for Mahalanobis gate)
        self.S: np.ndarray = self._H @ self.P @ self._H.T + self._R


    def predict(self) -> BBox:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        self.S = self._H @ self.P @ self._H.T + self._R
        return self._to_bbox()

    def update(self, bbox: BBox) -> BBox:
        z = np.array(self._meas(bbox), dtype=np.float64)
        y = z - self._H @ self.x
        K = self.P @ self._H.T @ np.linalg.inv(self.S)
        self.x += K @ y
        I_KH  = np.eye(_DIM_X) - K @ self._H
        self.P = I_KH @ self.P
        return self._to_bbox()

    def mahalanobis(self, bbox: BBox) -> float:
        """Mahalanobis distance between *bbox* measurement and current state."""
        z = np.array(self._meas(bbox), dtype=np.float64)
        y = z - self._H @ self.x
        try:
            S_inv = np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(y @ S_inv @ y)

    def velocity(self) -> Tuple[float, float]:
        """Return (vx, vy) in pixel/frame derived from state."""
        return float(self.x[4]), float(self.x[5])


    @staticmethod
    def _meas(b: BBox) -> Tuple[float, float, float, float]:
        w  = b.x2 - b.x1
        h  = b.y2 - b.y1
        cx = b.x1 + w * 0.5
        cy = b.y1 + h * 0.5
        s  = float(w * h)
        r  = w / (h + 1e-5)
        return cx, cy, s, r

    def _to_bbox(self) -> BBox:
        cx, cy, s, r = self.x[:4]
        s = max(s, 1.0)
        w = float(np.sqrt(abs(s * r)))
        h = float(s / (w + 1e-5))
        return BBox(
            int(cx - w * 0.5), int(cy - h * 0.5),
            int(cx + w * 0.5), int(cy + h * 0.5),
        )


class _KalmanTrack:
    __slots__ = ("track", "kf", "hit_streak", "age")

    def __init__(self, track_id: int, bbox: BBox, color: Tuple[int, int, int]) -> None:
        self.track      = Track(track_id, bbox, color=color)
        self.kf         = _KalmanBox(bbox)
        self.hit_streak: int = 0
        self.age:        int = 0

    @property
    def is_confirmed(self) -> bool:
        return self.hit_streak >= _MIN_HIT_STREAK

    def predict(self) -> BBox:
        pred = self.kf.predict()
        self.track.bbox = pred
        self.age += 1
        return pred

    def update(self, bbox: BBox) -> None:
        self.track.bbox   = self.kf.update(bbox)
        self.track.missed = 0
        self.hit_streak  += 1
        # Propagate velocity to Track for downstream zone logic
        vx, vy = self.kf.velocity()
        self.track.vx = vx
        self.track.vy = vy

    def mark_missed(self) -> None:
        self.track.missed += 1
        self.hit_streak    = 0

    def adaptive_iou_thresh(self, base: float = _HIGH_IOU) -> float:
        """
        Lower the required IoU for fast-moving or recently-spawned tracks
        to reduce ID switches on partial occlusion.
        """
        speed = (self.track.vx ** 2 + self.track.vy ** 2) ** 0.5
        # Fast vehicles cover more ground → accept slightly lower overlap
        speed_penalty = min(0.12, speed * 0.004)
        # Young tracks are less reliable → require slightly higher overlap
        age_bonus = 0.02 if self.age < 5 else 0.0
        return max(0.10, base - speed_penalty + age_bonus)


class SortLiteTracker:

    def __init__(self) -> None:
        self._next_id: int = 1
        self._tracks: Dict[int, _KalmanTrack] = {}


    def update(self, detections: List[BBox]) -> List[Track]:
        # 1. Predict all existing tracks forward one step
        for kt in self._tracks.values():
            kt.predict()

        if not detections:
            for kt in self._tracks.values():
                kt.mark_missed()
            return self._prune()

        det_arr = _bboxes_to_arr(detections)

        # 2. Two-stage ByteTrack matching
        unmatched_tracks = list(self._tracks.keys())
        unmatched_dets   = list(range(len(detections)))

        unmatched_tracks, unmatched_dets = self._match_stage(
            det_arr, unmatched_tracks, unmatched_dets,
            iou_threshold=_HIGH_IOU, use_adaptive=True)

        if unmatched_tracks and unmatched_dets:
            unmatched_tracks, unmatched_dets = self._match_stage(
                det_arr, unmatched_tracks, unmatched_dets,
                iou_threshold=_LOW_IOU, use_adaptive=False)

        # 3. Increment missed for truly unmatched tracks
        for tid in unmatched_tracks:
            self._tracks[tid].mark_missed()

        # 4. Spawn new tracks for unmatched detections
        for c in unmatched_dets:
            self._spawn(detections[c])

        return self._prune()

    @property
    def tracks(self) -> Dict[int, Track]:
        return {tid: kt.track for tid, kt in self._tracks.items()}


    def _match_stage(
        self,
        det_arr: np.ndarray,
        track_ids: List[int],
        det_indices: List[int],
        iou_threshold: float,
        use_adaptive: bool,
    ) -> Tuple[List[int], List[int]]:
        if not track_ids or not det_indices:
            return track_ids, det_indices

        trk_arr = _bboxes_to_arr([self._tracks[tid].track.bbox for tid in track_ids])
        sub_det = det_arr[det_indices]

        iou  = _iou_matrix(trk_arr, sub_det)  # (n_tracks, k_dets)
        for ri, tid in enumerate(track_ids):
            kt = self._tracks[tid]
            for ci, di in enumerate(det_indices):
                if iou[ri, ci] == 0.0:
                    continue
                maha = kt.kf.mahalanobis(
                    BBox(*[int(v) for v in det_arr[di]])
                )
                if maha > _MAHA_GATE:
                    iou[ri, ci] = 0.0

        cost = 1.0 - iou
        row_ind, col_ind = linear_sum_assignment(cost)

        unmatched_t = set(range(len(track_ids)))
        unmatched_d = set(range(len(det_indices)))

        for r, c in zip(row_ind, col_ind):
            thresh = (
                self._tracks[track_ids[r]].adaptive_iou_thresh(iou_threshold)
                if use_adaptive else iou_threshold
            )
            if iou[r, c] >= thresh:
                tid  = track_ids[r]
                didx = det_indices[c]
                self._tracks[tid].update(BBox(*[int(v) for v in det_arr[didx]]))
                unmatched_t.discard(r)
                unmatched_d.discard(c)

        remaining_tracks = [track_ids[i] for i in sorted(unmatched_t)]
        remaining_dets   = [det_indices[i] for i in sorted(unmatched_d)]
        return remaining_tracks, remaining_dets

    def _spawn(self, bbox: BBox) -> None:
        color = self._color(self._next_id)
        self._tracks[self._next_id] = _KalmanTrack(self._next_id, bbox, color)
        self._next_id += 1

    def _prune(self) -> List[Track]:
        self._tracks = {
            tid: kt for tid, kt in self._tracks.items()
            if kt.track.missed <= MAX_MISSED_FRAMES
        }
        return [kt.track for kt in self._tracks.values() if kt.is_confirmed]

    @staticmethod
    def _color(seed: int) -> Tuple[int, int, int]:
        rng = np.random.default_rng(seed * 137 + 42)
        return tuple(int(x) for x in rng.integers(80, 220, 3))