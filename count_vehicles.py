"""Count cars and trucks in a traffic video.

Example:
    python count_vehicles.py traffic.mp4 --model yolov8n.pt --show
"""

import argparse
import json
from pathlib import Path

from counting import CounterConfig
from pipeline import YoloByteTracker, process_video
from report import save_chart, save_pdf


def main() -> None:
    p = argparse.ArgumentParser(description="Detect, track and count vehicles in a video.")
    p.add_argument("video", help="input video file")
    p.add_argument("--model", default="yolov8l.pt",
                   help="YOLOv8 weights (downloaded automatically). yolov8n.pt is fastest on CPU.")
    p.add_argument("--conf", type=float, default=0.5, help="detection confidence threshold")
    p.add_argument("--line", type=float, default=0.88, help="counting line height, fraction of frame (0-1)")
    p.add_argument("--roi-top", type=float, default=0.55, help="top of the counting zone, fraction of frame")
    p.add_argument("--max-seconds", type=float, help="only process the first N seconds")
    p.add_argument("--max-width", type=int, help="downscale wider videos to this width")
    p.add_argument("--out", default="output", help="output folder")
    p.add_argument("--show", action="store_true", help="show frames while processing (press q to stop)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = CounterConfig(line_y=args.line, roi=(0.05, args.roi_top, 0.95, 0.98))
    tracker = YoloByteTracker(args.model, conf=args.conf)

    def progress(n, total):
        if n % 50 == 0:
            print(f"  frame {n}/{total or '?'}", flush=True)

    counter, _ = process_video(
        args.video, out / "annotated.mp4", tracker, config,
        max_seconds=args.max_seconds, max_width=args.max_width, show=args.show, progress=progress,
    )
    save_chart(counter, out / "vehicle_counts.png")
    save_pdf(counter, out / "vehicle_count_report.pdf", Path(args.video).name)
    (out / "counts.json").write_text(json.dumps(counter.summary(), indent=2))

    print(json.dumps(counter.summary(), indent=2))
    print(f"Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
