import os
import re
import cv2
import math
import argparse
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector
from ultralytics import YOLO

# Target Video/Audio specifications
VIDEO_EXTENSIONS = {".avi", ".flv", ".mkv", ".mp4", ".webm", ".wmv"}


@dataclass
class Config:
    """All tunables for a run. Built once from CLI args and shared read-only
    across the (single-threaded) analysis stage and the parallel render stage."""
    # Detection / grouping
    model: str
    device: object                 # 'cpu', 'mps', or an int CUDA index
    min_seconds: float
    max_seconds: float
    max_grace: float               # allow a shot to run this far past max before a hard cut
    similarity: float              # histogram correlation to merge A/B angles in one room
    # Human filter
    person_conf: float
    min_area_ratio: float          # fraction of frame a person must fill
    sample_frames: int
    min_human_ratio: float         # fraction of sampled frames that must contain a human
    # Encoding
    video_codec: str
    video_bitrate: str
    audio_codec: str
    audio_bitrate: str
    resolution: str                # bounding box "WxH"; long edge caps output, aspect preserved, no pad
    fps: int
    loudnorm: str                  # ffmpeg loudnorm args, e.g. "I=-14:TP=-1.5:LRA=11"
    # Runtime
    jobs: int

    @property
    def min_human_frames(self):
        """Required human-positive samples, tracking the sample count."""
        return max(1, math.ceil(self.min_human_ratio * self.sample_frames))


# Thread-safe logging so concurrent render workers don't interleave lines.
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg)


def format_tc(seconds):
    """Formats float seconds as HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def sanitize_filename(name):
    name, ext = os.path.splitext(name)
    sanitized = re.sub(r'[^A-Za-z0-9_-]', '_', name)
    return re.sub(r'_+', '_', sanitized).strip('_') + ext


def resolve_device(use_gpu):
    """Default to CPU; opt into GPU (CUDA, else Apple MPS) only when requested."""
    if not use_gpu:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    print("  ! --gpu requested but no CUDA/MPS device is available; falling back to CPU.")
    return "cpu"


def group_shots_into_scenes(cap, scene_list, cfg):
    """Merges rapid cuts using minimum duration, visual similarity, and hard/soft max limits."""
    if not scene_list:
        return []

    min_duration = cfg.min_seconds
    max_duration = cfg.max_seconds
    grace = cfg.max_grace

    # 1. HARD CUT PRE-PROCESSING
    # Chop up any excessively long single continuous shots before grouping. A shot is only
    # hard-cut when it runs past max + grace; within the grace window it is allowed to reach
    # its natural end so we don't slice mid-shot unless truly needed.
    processed_cuts = []
    for raw_start, raw_end in scene_list:
        s_sec = raw_start.get_seconds()
        e_sec = raw_end.get_seconds()

        while (e_sec - s_sec) > max_duration + grace:
            processed_cuts.append((s_sec, s_sec + max_duration))
            s_sec += max_duration
        processed_cuts.append((s_sec, e_sec))

    def get_hist(start_sec, end_sec):
        mid_sec = start_sec + (end_sec - start_sec) / 2.0
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    merged_scenes = []
    current_start = processed_cuts[0][0]
    current_hists = []

    for i in range(len(processed_cuts)):
        start_sec, end_sec = processed_cuts[i]
        hist = get_hist(start_sec, end_sec)

        if not current_hists:
            current_hists.append(hist)
            continue

        current_duration = start_sec - current_start

        is_similar = False
        if hist is not None:
            for prev_hist in current_hists[-3:]:
                if prev_hist is not None:
                    similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if similarity >= cfg.similarity:
                        is_similar = True
                        break

        # 2. SOFT CUT LOGIC
        # Once we've reached max, cut at this camera change (a natural boundary) rather than
        # merging further. The artificial hard cut above is the only mid-shot cut.
        if current_duration >= max_duration and current_duration >= min_duration:
            merged_scenes.append((current_start, start_sec))
            current_start = start_sec
            current_hists = [hist]
        # Keep merging if it visually matches OR if we haven't reached the user's minimum seconds
        elif is_similar or current_duration < min_duration:
            current_hists.append(hist)
        else:
            merged_scenes.append((current_start, start_sec))
            current_start = start_sec
            current_hists = [hist]

    merged_scenes.append((current_start, processed_cuts[-1][1]))
    return merged_scenes


def has_human(cap, start_sec, end_sec, model, cfg):
    """Samples frames to verify prominent human presence, strictly checking boundaries."""
    duration = end_sec - start_sec
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_area = width * height if (width and height) else (1920 * 1080)

    n = cfg.sample_frames
    # Sample frames evenly across the clip
    sample_timestamps = [start_sec + duration * (i + 1) / (n + 1) for i in range(n)]

    # Read every sample first, then run YOLO once on the whole batch (much faster than
    # N separate inference calls, especially on GPU).
    frames = []
    valid_idx = []
    for i, ts in enumerate(sample_timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
            valid_idx.append(i)

    human_hits = [False] * n
    if frames:
        results = model(frames, device=cfg.device, verbose=False)
        for res, i in zip(results, valid_idx):
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id == 0 and conf >= cfg.person_conf:
                    xyxy = box.xyxy[0].tolist()
                    box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

                    if (box_area / total_area) >= cfg.min_area_ratio:
                        human_hits[i] = True
                        break

    # 1. Total Density Check: Does the clip have enough humans overall?
    if sum(human_hits) < cfg.min_human_frames:
        return False

    # 2. Strict Edge Check: Kills clips that fade into intros/outros/logos.
    # At least one of the first two samples AND one of the last two samples MUST have a human.
    if not any(human_hits[:2]):
        return False  # Failed intro check
    if not any(human_hits[-2:]):
        return False  # Failed outro check

    return True


def process_scene(input_file, start_time, duration, output_file, cfg):
    tmp_output = output_file.with_suffix('.tmp.mp4')
    # `resolution` is a *bounding box*, not a fixed output canvas: cap the long
    # edge at max(w, h) while preserving each clip's own aspect ratio (portrait
    # stays portrait, landscape stays landscape) and never upscaling. Do NOT pad
    # to a fixed frame — padding bakes black bars into the pixels, so a portrait
    # clip would ship as a letterboxed landscape file and the player (a read-only
    # consumer that cannot crop baked-in bars) shows it as a horizontal strip
    # with the content boxed inside on a portrait phone. The `min(edge, iw/ih)`
    # box caps the long edge without upscaling; `force_divisible_by=2` keeps the
    # even dimensions H.264/H.265 4:2:0 requires. Commas inside min() are escaped
    # (`\,`) so ffmpeg's filtergraph parser doesn't read them as filter separators.
    w, h = cfg.resolution.lower().split('x')
    long_edge = max(int(w), int(h))
    vf = (
        f"scale=w=min({long_edge}\\,iw):h=min({long_edge}\\,ih):"
        f"force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"fps={cfg.fps}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(input_file),
        "-c:v", cfg.video_codec,
        "-b:v", cfg.video_bitrate,
        "-maxrate", cfg.video_bitrate,
        "-bufsize", "6000k",
        "-vf", vf,
        "-c:a", cfg.audio_codec,
        "-b:a", cfg.audio_bitrate,
        "-af", f"loudnorm={cfg.loudnorm}",
        "-f", "mp4",
        "-movflags", "+faststart",
        "-loglevel", "error",
        "-stats",
        str(tmp_output)
    ]
    subprocess.run(cmd, check=True)
    tmp_output.rename(output_file)


def render_job(video_path, start_sec, duration_sec, out_filepath, cfg, s_idx, total):
    """Runs in a worker thread. Each ffmpeg call is an isolated subprocess writing its own
    temp file before an atomic rename, so concurrent jobs never interfere."""
    try:
        process_scene(video_path, start_sec, duration_sec, out_filepath, cfg)
        log(f"    + Finished Scene {s_idx}/{total}: {out_filepath.name}")
        return True
    except subprocess.CalledProcessError:
        log(f"    ! Error processing scene {s_idx}. FFmpeg failed.")
        return False


def build_config(args):
    grace = args.max_grace if args.max_grace is not None else 0.15 * args.max_seconds
    if getattr(args, 'encoder', None):
        if args.encoder == "h264":
            args.video_codec = "libx264"
        elif args.encoder == "hevc":
            args.video_codec = "libx265"
        elif args.encoder == "av1":
            args.video_codec = "libsvtav1"

    return Config(
        model=args.model,
        device=resolve_device(args.gpu),
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        max_grace=grace,
        similarity=args.similarity,
        person_conf=args.person_conf,
        min_area_ratio=args.min_area_ratio,
        sample_frames=args.sample_frames,
        min_human_ratio=args.min_human_ratio,
        video_codec=args.video_codec,
        video_bitrate=args.video_bitrate,
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
        resolution=args.resolution,
        fps=args.fps,
        loudnorm=args.loudnorm,
        jobs=max(1, args.jobs),
    )


_list_lock = threading.Lock()

class VideoProgress:
    def __init__(self, list_path, rel_path, total_scenes):
        self.list_path = list_path
        self.rel_path = rel_path
        self.total = total_scenes
        self.completed = 0
        self.failed = 0
        self.lock = threading.Lock()

    def scene_done(self, future):
        with self.lock:
            try:
                success = future.result()
            except Exception:
                success = False
            
            if success:
                self.completed += 1
            else:
                self.failed += 1
            
            if self.completed + self.failed == self.total:
                if self.failed == 0 and self.list_path:
                    with _list_lock:
                        with open(self.list_path, 'a', encoding='utf-8') as f:
                            f.write(str(self.rel_path) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Batch process scenes with medium/close-up humans.")
    parser.add_argument("-i", "--input", required=True, help="Input folder")
    parser.add_argument("-o", "--output", required=True, help="Output bucket folder")
    parser.add_argument("--processed-list", type=str, default=None,
                        help="Path to a text file containing a list of relative file paths that have already been processed.")

    # Durations
    parser.add_argument("-m", "--min-seconds", type=float, default=45.0,
                        help="Minimum duration in seconds for a clip (default: 45.0)")
    parser.add_argument("-x", "--max-seconds", type=float, default=300.0,
                        help="Maximum approximate duration in seconds for a clip (default: 300.0)")
    parser.add_argument("--max-grace", type=float, default=None,
                        help="Seconds a shot may run past --max-seconds to reach a natural cut "
                             "before a hard cut is forced (default: 15%% of --max-seconds)")

    # Parallelism / device
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="Concurrent FFmpeg render jobs (default: 1). FFmpeg is itself "
                             "multithreaded, so 2-4 is usually plenty.")
    parser.add_argument("--gpu", action="store_true",
                        help="Run YOLO on GPU (CUDA, else Apple MPS). Default is CPU.")

    # Human filter
    parser.add_argument("--person-conf", type=float, default=0.65,
                        help="YOLO person confidence threshold (default: 0.65)")
    parser.add_argument("--min-area-ratio", type=float, default=0.20,
                        help="Min fraction of the frame a person must fill (default: 0.20)")
    parser.add_argument("--sample-frames", type=int, default=10,
                        help="Frames sampled per scene for human detection (default: 10)")
    parser.add_argument("--min-human-ratio", type=float, default=0.70,
                        help="Fraction of sampled frames that must contain a human (default: 0.70)")

    # Grouping / model
    parser.add_argument("--similarity", type=float, default=0.55,
                        help="Histogram correlation to merge camera angles in one scene (default: 0.55)")
    parser.add_argument("--model", default="yolo11s.pt",
                        help="YOLO model name or path (default: yolo11s.pt)")

    # Encoding
    parser.add_argument("--encoder", choices=["h264", "hevc", "av1"], default=None,
                        help="Simplified flag to choose the video encoder. Maps to libx264, libx265 (default), or libsvtav1. Overrides --video-codec if set.")
    parser.add_argument("--video-codec", default="libx265", help="FFmpeg video codec (default: libx265)")
    parser.add_argument("--video-bitrate", default="3000k", help="Target video bitrate (default: 3000k)")
    parser.add_argument("--audio-codec", default="aac", help="FFmpeg audio codec (default: aac)")
    parser.add_argument("--audio-bitrate", default="128k", help="Target audio bitrate (default: 128k)")
    parser.add_argument("--resolution", default="1920x1080", help="Bounding box WxH; the long edge (max of W,H) caps output while each clip keeps its own aspect ratio, never upscaled or padded (default: 1920x1080)")
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate (default: 30)")
    parser.add_argument("--loudnorm", default="I=-14:TP=-1.5:LRA=11",
                        help="FFmpeg loudnorm args (default: I=-14:TP=-1.5:LRA=11)")

    args = parser.parse_args()
    cfg = build_config(args)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = set()
    if args.processed_list and os.path.exists(args.processed_list):
        with open(args.processed_list, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    processed_files.add(str(Path(stripped)))

    # Load the detector after parsing so --model / --help are honored without paying the load cost.
    print(f"Loading model '{cfg.model}' on device '{cfg.device}'...")
    detector_model = YOLO(cfg.model)

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(input_dir.rglob(f"*{ext}"))
        videos.extend(input_dir.rglob(f"*{ext.upper()}"))

    print(f"Found {len(videos)} compatible video(s). Rendering with up to {cfg.jobs} parallel job(s).")

    with ThreadPoolExecutor(max_workers=cfg.jobs) as executor:
        for v_idx, video_path in enumerate(videos, start=1):
            try:
                rel_path = video_path.relative_to(input_dir)
            except ValueError:
                rel_path = video_path.name
                
            if str(rel_path) in processed_files:
                print(f"\n[{v_idx}/{len(videos)}] Skipping: {video_path.name} (already processed)")
                continue

            clean_base_name = sanitize_filename(video_path.name)
            base_name_no_ext = os.path.splitext(clean_base_name)[0]

            print(f"\n[{v_idx}/{len(videos)}] Detecting cuts in: {video_path.name}")

            try:
                video = open_video(str(video_path))
                scene_manager = SceneManager()
                scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=15))
                scene_manager.detect_scenes(video, show_progress=True)
                raw_cuts = scene_manager.get_scene_list()
            except Exception as e:
                print(f"Error reading {video_path.name}: {e}")
                continue

            if not raw_cuts:
                raw_cuts = [(video.base_timecode, video.base_timecode + video.duration)]

            print(f"  -> Found {len(raw_cuts)} raw camera cuts. Grouping into logical scenes...")

            cap = cv2.VideoCapture(str(video_path))
            try:
                logical_scenes = group_shots_into_scenes(cap, raw_cuts, cfg)
                total = len(logical_scenes)
                print(f"  -> Grouped into {total} continuous scenes. Scanning for close-ups...")
                
                queued_futures = []

                for s_idx, (start_sec, end_sec) in enumerate(logical_scenes, start=1):
                    out_filename = f"{base_name_no_ext}_scene{s_idx:04d}.mp4"
                    out_filepath = output_dir / out_filename

                    if out_filepath.exists():
                        print(f"    - Skipping Scene {s_idx}/{total} (Already exists)")
                        continue

                    duration_sec = end_sec - start_sec

                    if duration_sec < cfg.min_seconds:
                        print(f"    - Dropping Scene {s_idx}/{total}: "
                              f"[Too short: {duration_sec:.1f}s < {cfg.min_seconds}s]")
                        continue

                    if not has_human(cap, start_sec, end_sec, detector_model, cfg):
                        print(f"    - Dropping Scene {s_idx}/{total}: [No prominent humans found]")
                        continue

                    print(f"    + Queuing Scene {s_idx}/{total}: "
                          f"[{format_tc(start_sec)} to {format_tc(end_sec)}] (Duration: {duration_sec:.1f}s)")

                    future = executor.submit(render_job, video_path, start_sec, duration_sec,
                                             out_filepath, cfg, s_idx, total)
                    queued_futures.append(future)
                
                if not queued_futures:
                    if args.processed_list:
                        with _list_lock:
                            with open(args.processed_list, 'a', encoding='utf-8') as f:
                                f.write(str(rel_path) + '\n')
                else:
                    progress = VideoProgress(args.processed_list, rel_path, len(queued_futures))
                    for f in queued_futures:
                        f.add_done_callback(progress.scene_done)
            finally:
                cap.release()

        # Leaving the `with` block waits for all queued renders to finish.

    print("\n--- Batch Processing Complete! ---")


if __name__ == "__main__":
    main()
