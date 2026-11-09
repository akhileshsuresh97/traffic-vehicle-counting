"""Tests for the counting logic and the video pipeline (no YOLOv8 needed).

Run: python -m unittest discover tests
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from counting import CounterConfig, Detection, VehicleCounter, iou_xyxy, merge_truck_trailer  # noqa: E402
from pipeline import process_video  # noqa: E402
from report import save_chart, save_pdf  # noqa: E402

W, H = 1280, 720  # line_y = 633, band = 57, ROI top = 396


def car(tid, cy, cx=600, size=100, cls="car"):
    return Detection((cx - size, cy - size // 2, cx + size, cy + size // 2), cls, tid)


def drive(counter, tid, ys, **kw):
    for y in ys:
        counter.update([car(tid, y, **kw)])


class TestGeometry(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)
        self.assertEqual(iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_truck_trailer_merged_keeps_larger(self):
        tractor = Detection((100, 100, 200, 200), "truck", 1)
        trailer = Detection((150, 90, 400, 210), "truck", 2)
        kept = merge_truck_trailer([tractor, trailer])
        self.assertEqual([d.track_id for d in kept], [2])

    def test_cars_never_merged(self):
        a = Detection((100, 100, 200, 200), "car", 1)
        b = Detection((110, 100, 210, 200), "car", 2)
        self.assertEqual(len(merge_truck_trailer([a, b])), 2)


class TestCounting(unittest.TestCase):
    def setUp(self):
        self.c = VehicleCounter(W, H)

    def test_exit_when_moving_down(self):
        drive(self.c, 1, range(450, 700, 10))
        self.assertEqual((self.c.total_exit, self.c.total_enter), (1, 0))
        self.assertEqual(self.c.exit["car"], 1)

    def test_enter_when_moving_up(self):
        drive(self.c, 1, range(700, 450, -10))
        self.assertEqual((self.c.total_enter, self.c.total_exit), (1, 0))

    def test_counted_once_even_if_jittering_on_line(self):
        drive(self.c, 1, [620, 640, 625, 645, 628, 650, 660, 690])
        self.assertEqual(self.c.total_exit, 1)

    def test_jump_across_line_far_from_it_is_ignored(self):
        drive(self.c, 1, [420, 700])  # 213 px before, 67 px after: outside the ±57 px band
        self.assertEqual(self.c.total_exit, 0)

    def test_outside_roi_not_counted(self):
        drive(self.c, 1, range(450, 700, 10), cx=20)  # centroid left of ROI
        self.assertEqual(self.c.total_exit, 0)

    def test_tiny_boxes_not_counted(self):
        drive(self.c, 1, range(450, 700, 10), size=20)  # 40x20 = 800 px² < 6000
        self.assertEqual(self.c.total_exit, 0)

    def test_other_classes_ignored(self):
        drive(self.c, 1, range(450, 700, 10), cls="bus")
        self.assertEqual(self.c.total_exit, 0)

    def test_class_vote_smooths_flicker(self):
        labels = ["truck"] * 8 + ["car"] * 3
        for y, cls in zip(range(560, 680, 11), labels):
            self.c.update([car(1, y, cls=cls)])
        self.assertEqual(self.c.exit["truck"], 1)
        self.assertEqual(self.c.exit["car"], 0)

    def test_truck_with_trailer_counted_once(self):
        for y in range(450, 700, 10):
            tractor = Detection((500, y - 60, 650, y + 60), "truck", 1)
            trailer = Detection((560, y - 70, 900, y + 70), "truck", 2)
            self.c.update([tractor, trailer])
        self.assertEqual(self.c.total_exit, 1)

    def test_custom_line_position(self):
        c = VehicleCounter(W, H, CounterConfig(line_y=0.7))  # line at 504
        drive(c, 1, range(450, 560, 10))
        self.assertEqual(c.total_exit, 1)


class FakeTracker:
    """Replays scripted vehicle positions, frame by frame, like a tracker would."""

    def __init__(self, script):
        self.script, self.frame = script, 0

    def __call__(self, frame):
        dets = self.script(self.frame)
        self.frame += 1
        return dets


def make_video(path, frames=60, fps=20):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for _ in range(frames):
        out.write(np.full((H, W, 3), 90, np.uint8))
    out.release()


def script(i):
    y = 400 + i * 6  # downward: exits
    return [
        Detection((300, y - 50, 460, y + 50), "car", 1),
        Detection((700, 1100 - y - 50, 860, 1100 - y + 50), "car", 2),  # upward from y=700: enters
        Detection((900, y - 70, 1020, y + 70), "truck", 3),  # tractor
        Detection((950, y - 80, 1200, y + 80), "truck", 4),  # trailer, overlapping
    ]


class TestPipeline(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_video(tmp / "in.mp4")
            counter, last = process_video(tmp / "in.mp4", tmp / "out.mp4", FakeTracker(script))
            self.assertEqual(counter.summary(), {
                "entered": {"total": 1, "car": 1, "truck": 0},
                "exited": {"total": 2, "car": 1, "truck": 1},
            })
            self.assertEqual(last.shape, (H, W, 3))

            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=codec_name,nb_frames", "-of", "json", str(tmp / "out.mp4")],
                capture_output=True, text=True,
            )
            if probe.returncode == 0:
                stream = json.loads(probe.stdout)["streams"][0]
                self.assertEqual(stream["codec_name"], "h264")
                self.assertEqual(int(stream["nb_frames"]), 60)

            self.assertGreater(save_chart(counter, tmp / "c.png").stat().st_size, 1000)
            self.assertTrue(save_pdf(counter, tmp / "r.pdf", "in.mp4").read_bytes().startswith(b"%PDF"))

    def test_max_seconds_and_downscale(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_video(tmp / "in.mp4", frames=60, fps=20)
            _, last = process_video(tmp / "in.mp4", tmp / "out.mp4", FakeTracker(lambda i: []),
                                    max_seconds=1, max_width=640)
            self.assertEqual(last.shape, (360, 640, 3))
            cap = cv2.VideoCapture(str(tmp / "out.mp4"))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 20)


if __name__ == "__main__":
    unittest.main()
