"""Counting logic, independent of the detector.

Everything here works on plain bounding boxes, so it can be tested without
YOLOv8 or a video. pipeline.py feeds it the tracked detections of each frame.
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    class_name: str  # "car" or "truck"
    track_id: int


@dataclass
class TrackState:
    """What the counter knows about one tracked vehicle in the current frame."""
    track_id: int
    box: tuple[int, int, int, int]
    centroid: tuple[int, int]
    stable_class: str
    in_roi: bool
    status: str  # "", "ENTER" or "EXIT"
    trail: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class CounterConfig:
    """Geometry and filter settings. Fractions are relative to frame size."""
    allowed_classes: tuple[str, ...] = ("car", "truck")
    roi: tuple[float, float, float, float] = (0.05, 0.55, 0.95, 0.98)  # left, top, right, bottom
    line_y: float = 0.88
    band: float = 0.08  # half-height of the counting band around the line
    min_area: int = 6000  # px², filters distant/partial detections
    max_area: int = 500000
    truck_merge_iou: float = 0.10
    class_vote_window: int = 15
    trail_length: int = 12


def iou_xyxy(a, b) -> float:
    """Intersection over union of two boxes given as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def merge_truck_trailer(detections: list[Detection], iou_thresh: float = 0.10) -> list[Detection]:
    """Merge overlapping truck boxes (tractor + trailer) so one truck counts once.

    For every pair of truck detections with IoU >= iou_thresh, only the larger
    box is kept. Cars are never merged.
    """
    keep = [True] * len(detections)
    for i, a in enumerate(detections):
        if not keep[i] or a.class_name != "truck":
            continue
        for j in range(i + 1, len(detections)):
            b = detections[j]
            if not keep[j] or b.class_name != "truck":
                continue
            if iou_xyxy(a.box, b.box) >= iou_thresh:
                if _area(a.box) >= _area(b.box):
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [d for d, k in zip(detections, keep) if k]


class VehicleCounter:
    """Counts vehicles crossing a horizontal line, by direction and class.

    Moving from below the line to above it counts as ENTER, from above to
    below as EXIT. A vehicle is only counted when its centroid is inside the
    ROI, its box area is within limits, and it is within the band around the
    line. Each track ID counts at most once per direction.
    """

    def __init__(self, frame_width: int, frame_height: int, config: CounterConfig | None = None):
        self.cfg = config or CounterConfig()
        w, h = frame_width, frame_height
        left, top, right, bottom = self.cfg.roi
        self.roi = (int(left * w), int(top * h), int(right * w), int(bottom * h))
        self.line_y = int(self.cfg.line_y * h)
        self.band = int(self.cfg.band * h)

        self.class_history = defaultdict(lambda: deque(maxlen=self.cfg.class_vote_window))
        self.trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.last_side: dict[int, str] = {}
        self.entered_ids: set[int] = set()
        self.exited_ids: set[int] = set()
        self.enter = Counter({c: 0 for c in self.cfg.allowed_classes})
        self.exit = Counter({c: 0 for c in self.cfg.allowed_classes})

    @property
    def total_enter(self) -> int:
        return sum(self.enter.values())

    @property
    def total_exit(self) -> int:
        return sum(self.exit.values())

    def stable_class(self, track_id: int) -> str:
        """Majority vote over the last few frames, so a flickering label doesn't change the count."""
        history = self.class_history[track_id]
        if not history:
            return self.cfg.allowed_classes[0]
        # most_common is stable for ties (first seen wins), unlike max(set(...))
        return Counter(history).most_common(1)[0][0]

    def in_roi(self, cx: int, cy: int) -> bool:
        left, top, right, bottom = self.roi
        return left <= cx <= right and top <= cy <= bottom

    def update(self, detections: list[Detection]) -> list[TrackState]:
        """Process one frame's tracked detections. Returns per-track state for drawing."""
        detections = [d for d in detections if d.class_name in self.cfg.allowed_classes]
        detections = merge_truck_trailer(detections, self.cfg.truck_merge_iou)

        states = []
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            tid = det.track_id

            self.class_history[tid].append(det.class_name)
            cls = self.stable_class(tid)
            self.trails[tid].append((cx, cy))

            inside = self.in_roi(cx, cy)
            good_area = self.cfg.min_area <= (x2 - x1) * (y2 - y1) <= self.cfg.max_area
            if inside and good_area:
                self._check_crossing(tid, cls, cy)

            status = "ENTER" if tid in self.entered_ids else "EXIT" if tid in self.exited_ids else ""
            states.append(TrackState(
                track_id=tid, box=(x1, y1, x2, y2), centroid=(cx, cy), stable_class=cls,
                in_roi=inside, status=status, trail=self.trails[tid][-self.cfg.trail_length:],
            ))
        return states

    def _check_crossing(self, tid: int, cls: str, cy: int) -> None:
        side = "above" if cy < self.line_y else "below"
        previous = self.last_side.get(tid)
        self.last_side[tid] = side
        if previous is None or previous == side:
            return
        if abs(cy - self.line_y) > self.band:
            return  # a jump across the line far from it is treated as tracking noise
        if previous == "below" and tid not in self.entered_ids:
            self.entered_ids.add(tid)
            self.enter[cls] += 1
        elif previous == "above" and tid not in self.exited_ids:
            self.exited_ids.add(tid)
            self.exit[cls] += 1

    def summary(self) -> dict:
        return {
            "entered": {"total": self.total_enter, **dict(self.enter)},
            "exited": {"total": self.total_exit, **dict(self.exit)},
        }
