"""Video pipeline: read frames, detect and track vehicles, count, draw, write video."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from counting import CounterConfig, Detection, TrackState, VehicleCounter

# BGR colours
CAR_COLOR = (255, 0, 0)
TRUCK_COLOR = (0, 255, 255)
TRAIL_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 0)


class Tracker(Protocol):
    def __call__(self, frame: np.ndarray) -> list[Detection]: ...


class YoloByteTracker:
    """YOLOv8 detection + ByteTrack tracking via Ultralytics."""

    def __init__(self, model_path: str = "yolov8l.pt", conf: float = 0.5,
                 classes: tuple[str, ...] = ("car", "truck"), device: str | None = None):
        from ultralytics import YOLO  # imported here so the rest works without it

        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.names = self.model.names
        self.class_ids = [i for i, n in self.names.items() if n in classes]

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=self.class_ids,
            tracker="bytetrack.yaml",  # without this, Ultralytics defaults to BoT-SORT
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes if results else None
        if boxes is None or boxes.id is None:
            return []
        return [
            Detection(box=tuple(b), class_name=self.names[int(c)], track_id=int(t))
            for b, c, t in zip(
                boxes.xyxy.cpu().numpy().tolist(),
                boxes.cls.cpu().numpy(),
                boxes.id.cpu().numpy(),
            )
        ]


def put_text_box(frame, text, org, font_scale=0.7, text_color=(255, 255, 255),
                 box_color=(0, 0, 0), alpha=0.6, thickness=2, padding=5):
    """Text on a semi-transparent background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x1, y1 = max(x - padding, 0), max(y - th - padding, 0)
    x2, y2 = min(x + tw + padding, frame.shape[1]), min(y + padding, frame.shape[0])
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, counter: VehicleCounter, states: list[TrackState]) -> np.ndarray:
    w = frame.shape[1]
    roi_left, roi_top, roi_right, roi_bottom = counter.roi

    for s in states:
        if not s.in_roi:
            continue  # only annotate vehicles inside the counting zone
        color = CAR_COLOR if s.stable_class == "car" else TRUCK_COLOR
        x1, y1, x2, y2 = s.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        pts = [p for p in s.trail if p[1] >= roi_top]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, TRAIL_COLOR, 2)
        label = f"ID {s.track_id} {s.stable_class} {s.status}".strip()
        text_color = (255, 255, 255) if s.stable_class == "car" else (0, 0, 0)
        put_text_box(frame, label, (x1, max(20, y1 - 5)), font_scale=0.6,
                     text_color=text_color, box_color=color, alpha=0.8)

    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (255, 255, 255), 2)
    ly, band = counter.line_y, counter.band
    cv2.line(frame, (0, ly), (w, ly), LINE_COLOR, 3)
    cv2.line(frame, (0, ly - band), (w, ly - band), LINE_COLOR, 1)
    cv2.line(frame, (0, ly + band), (w, ly + band), LINE_COLOR, 1)

    put_text_box(frame, f"EXIT: {counter.total_exit}", (20, 40), font_scale=1, box_color=(0, 0, 255))
    put_text_box(frame, f"Cars exit: {counter.exit['car']}", (20, 80))
    put_text_box(frame, f"Trucks exit: {counter.exit['truck']}", (20, 110))
    put_text_box(frame, f"ENTER: {counter.total_enter}", (w - 300, 40), font_scale=1, box_color=(0, 160, 0))
    put_text_box(frame, f"Cars enter: {counter.enter['car']}", (w - 300, 80))
    put_text_box(frame, f"Trucks enter: {counter.enter['truck']}", (w - 300, 110))
    return frame


def to_browser_mp4(src: Path, dst: Path) -> Path:
    """Re-encode to H.264 so browsers can play it. OpenCV's mp4v often won't play in a browser."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            shutil.copyfile(src, dst)
            return dst
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
        check=True,
    )
    return dst


def process_video(
    source: str | Path,
    output: str | Path,
    tracker: Tracker,
    config: CounterConfig | None = None,
    max_seconds: float | None = None,
    max_width: int | None = None,
    show: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[VehicleCounter, np.ndarray | None]:
    """Run the full pipeline. Returns the counter (with counts) and the last annotated frame.

    max_seconds: stop after this much video (useful for the web demo).
    max_width: downscale wider videos first, for speed.
    show: display frames in a window while processing (press q to stop).
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / w) if max_width else 1.0
    w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2  # even sizes for H.264

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_seconds:
        total = min(total, int(max_seconds * fps)) if total else int(max_seconds * fps)

    counter = VehicleCounter(w, h, config)
    output = Path(output)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        last_frame, n = None, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or (total and n >= total):
                    break
                if scale < 1.0:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                states = counter.update(tracker(frame))
                frame = draw_frame(frame, counter, states)
                writer.write(frame)
                last_frame, n = frame, n + 1
                if progress:
                    progress(n, total)
                if show:
                    cv2.imshow("Vehicle counting", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            if show:
                cv2.destroyAllWindows()
        to_browser_mp4(raw, output)
    return counter, last_frame
