"""Gradio web demo (runs on Hugging Face Spaces or locally: python app.py)."""

import tempfile
from pathlib import Path

import gradio as gr

from counting import CounterConfig
from pipeline import YoloByteTracker, process_video
from report import save_chart, save_pdf

MODEL = "yolov8s.pt"  # small model: reasonable speed on a free CPU
MAX_SECONDS = 15
MAX_WIDTH = 960


def run(video_path, line_position, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a traffic video first.")
    out = Path(tempfile.mkdtemp())
    # A new tracker per video, so track IDs don't carry over between uploads
    tracker = YoloByteTracker(MODEL, conf=0.5)
    config = CounterConfig(line_y=line_position)

    def report_progress(n, total):
        if total:
            progress(n / total, desc=f"Frame {n} of {total}")

    counter, _ = process_video(
        video_path, out / "annotated.mp4", tracker, config,
        max_seconds=MAX_SECONDS, max_width=MAX_WIDTH, progress=report_progress,
    )
    chart = save_chart(counter, out / "vehicle_counts.png")
    pdf = save_pdf(counter, out / "vehicle_count_report.pdf", Path(video_path).name)
    summary = (
        f"**Entered:** {counter.total_enter} (cars {counter.enter['car']}, trucks {counter.enter['truck']})  \n"
        f"**Exited:** {counter.total_exit} (cars {counter.exit['car']}, trucks {counter.exit['truck']})"
    )
    return str(out / "annotated.mp4"), summary, str(chart), str(pdf)


with gr.Blocks(title="Traffic vehicle counting") as demo:
    gr.Markdown(
        "# Traffic vehicle detection, tracking and counting\n"
        "YOLOv8 detects cars and trucks, ByteTrack follows them across frames, and each vehicle is "
        "counted once when it crosses the green line: upwards counts as ENTER, downwards as EXIT. "
        f"Works best with a fixed camera looking along a road. Only the first {MAX_SECONDS} seconds "
        "are processed, which takes about a minute on this free CPU."
    )
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Traffic video", sources=["upload"])
            line = gr.Slider(0.5, 0.95, value=0.88, step=0.01,
                             label="Counting line position (fraction of frame height, from the top)")
            button = gr.Button("Count vehicles", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Annotated video")
            counts = gr.Markdown()
    with gr.Row():
        chart_out = gr.Image(label="Counts by class and direction")
        pdf_out = gr.File(label="PDF report")
    button.click(run, inputs=[video_in, line], outputs=[video_out, counts, chart_out, pdf_out])
    gr.Markdown(
        "Source code: [github.com/akhileshsuresh97/traffic-vehicle-counting]"
        "(https://github.com/akhileshsuresh97/traffic-vehicle-counting)"
    )

if __name__ == "__main__":
    demo.launch()
