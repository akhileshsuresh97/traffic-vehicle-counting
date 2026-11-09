"""Turn part of the annotated video into a small GIF for the README.

Example:
    python make_gif.py output/annotated.mp4 docs/demo.gif --start 2 --duration 6
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video")
    p.add_argument("gif")
    p.add_argument("--start", type=float, default=0, help="start time in seconds")
    p.add_argument("--duration", type=float, default=6, help="length in seconds")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
    filters = (f"fps={args.fps},scale={args.width}:-1:flags=lanczos,split[a][b];"
               "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(args.start), "-t", str(args.duration),
                    "-i", args.video, "-filter_complex", filters, args.gif], check=True)
    size_mb = Path(args.gif).stat().st_size / 1e6
    print(f"Saved {args.gif} ({size_mb:.1f} MB)")
    if size_mb > 8:
        print("Tip: GIFs over ~8 MB load slowly on GitHub. Try a lower --width, --fps or --duration.")


if __name__ == "__main__":
    main()
