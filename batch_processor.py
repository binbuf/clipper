import os
import re
import cv2
import argparse
import subprocess
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector
from ultralytics import YOLO

# Target Video/Audio specifications
VIDEO_EXTENSIONS = {".avi", ".flv", ".mkv", ".mp4", ".webm", ".wmv"}
VIDEO_CODEC = "libx265"
VIDEO_BITRATE = "3000k"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "128k"

# --- SCENE GROUPING SETTINGS ---
SCENE_SIMILARITY_THRESHOLD = 0.55 # (0.0 to 1.0) Histogram correlation. 0.55 groups A/B camera angles in the same room.

# --- HUMAN DETECTION SETTINGS ---
PERSON_CONF_THRESH = 0.65    # Increased from 0.45 (kills false positives on graphics)
MIN_PERSON_AREA_RATIO = 0.20 # Increased from 0.15 (forces slightly closer shots)
SAMPLE_FRAMES_COUNT = 10     # Increased from 7 for better sampling resolution
MIN_FRAMES_WITH_HUMAN = 7    # 70% of sampled frames must have a human

# Upgrade to the 'small' model. Nano (n) is too easily fooled by abstract logos.
detector_model = YOLO("yolo11s.pt")

def sanitize_filename(name):
    name, ext = os.path.splitext(name)
    sanitized = re.sub(r'[^A-Za-z0-9_-]', '_', name)
    return re.sub(r'_+', '_', sanitized).strip('_') + ext

def group_shots_into_scenes(cap, scene_list, min_duration):
    """Merges rapid cuts into cohesive scenes using minimum duration and visual similarity."""
    if not scene_list:
        return []

    def get_hist(start_tc, end_tc):
        mid_sec = start_tc.get_seconds() + (end_tc.get_seconds() - start_tc.get_seconds()) / 2.0
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    merged_scenes = []
    current_start = scene_list[0][0]
    current_hists = []

    for i in range(len(scene_list)):
        start, end = scene_list[i]
        hist = get_hist(start, end)
        
        if not current_hists:
            current_hists.append(hist)
            continue
            
        current_duration = start.get_seconds() - current_start.get_seconds()
        
        is_similar = False
        if hist is not None:
            for prev_hist in current_hists[-3:]:
                if prev_hist is not None:
                    similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if similarity >= SCENE_SIMILARITY_THRESHOLD:
                        is_similar = True
                        break

        # Keep merging if it visually matches OR if we haven't reached the user's minimum seconds
        if is_similar or current_duration < min_duration:
            current_hists.append(hist)
        else:
            merged_scenes.append((current_start, start))
            current_start = start
            current_hists = [hist]
            
    merged_scenes.append((current_start, scene_list[-1][1]))
    return merged_scenes

def has_human(cap, start_sec, end_sec):
    """Samples frames to verify prominent human presence, strictly checking boundaries."""
    duration = end_sec - start_sec
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_area = width * height if (width and height) else (1920 * 1080)

    # Sample frames evenly across the clip
    sample_timestamps = [start_sec + duration * (i + 1) / (SAMPLE_FRAMES_COUNT + 1) for i in range(SAMPLE_FRAMES_COUNT)]
    
    # Keep track of which specific frames contained humans
    human_hits = []

    for ts in sample_timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            human_hits.append(False)
            continue

        results = detector_model(frame, verbose=False)[0]
        frame_has_human = False

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if cls_id == 0 and conf >= PERSON_CONF_THRESH:
                xyxy = box.xyxy[0].tolist()
                box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                
                if (box_area / total_area) >= MIN_PERSON_AREA_RATIO:
                    frame_has_human = True
                    break 

        human_hits.append(frame_has_human)

    # 1. Total Density Check: Does the clip have enough humans overall?
    if sum(human_hits) < MIN_FRAMES_WITH_HUMAN:
        return False

    # 2. Strict Edge Check: Kills clips that fade into intros/outros/logos.
    # At least one of the first two samples AND one of the last two samples MUST have a human.
    if not any(human_hits[:2]):
        return False  # Failed intro check
    if not any(human_hits[-2:]):
        return False  # Failed outro check

    return True

def process_scene(input_file, start_time, duration, output_file):
    tmp_output = output_file.with_suffix('.tmp.mp4')
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(input_file),
        "-c:v", VIDEO_CODEC,
        "-b:v", VIDEO_BITRATE,
        "-maxrate", VIDEO_BITRATE,
        "-bufsize", "6000k",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-f", "mp4",
        "-movflags", "+faststart",
        "-loglevel", "error",
        "-stats",
        str(tmp_output)
    ]
    subprocess.run(cmd, check=True)
    tmp_output.rename(output_file)

def main():
    parser = argparse.ArgumentParser(description="Batch process scenes with medium/close-up humans.")
    parser.add_argument("-i", "--input", required=True, help="Input folder")
    parser.add_argument("-o", "--output", required=True, help="Output bucket folder")
    # NEW ARGUMENT HERE:
    parser.add_argument("-m", "--min-seconds", type=float, default=12.0, help="Minimum duration in seconds for a clip (default: 12.0)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store the user's config
    MIN_CLIP_DURATION = args.min_seconds

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(input_dir.rglob(f"*{ext}"))
        videos.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(videos)} compatible video(s).")

    for v_idx, video_path in enumerate(videos, start=1):
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
        
        # Pass the configured minimum duration into the grouping function
        logical_scenes = group_shots_into_scenes(cap, raw_cuts, MIN_CLIP_DURATION)
        print(f"  -> Grouped into {len(logical_scenes)} continuous scenes. Scanning for close-ups...")
        
        for s_idx, (start, end) in enumerate(logical_scenes, start=1):
            out_filename = f"{base_name_no_ext}_scene{s_idx:04d}.mp4"
            out_filepath = output_dir / out_filename
            
            if out_filepath.exists():
                print(f"    - Skipping Scene {s_idx}/{len(logical_scenes)} (Already exists)")
                continue
                
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()
            duration_sec = end_sec - start_sec

            # Strict filter: Drop the scene entirely if it's still too short (e.g. at the end of the file)
            if duration_sec < MIN_CLIP_DURATION:
                print(f"    - Dropping Scene {s_idx}/{len(logical_scenes)}: [Too short: {duration_sec:.1f}s < {MIN_CLIP_DURATION}s]")
                continue

            if not has_human(cap, start_sec, end_sec):
                print(f"    - Dropping Scene {s_idx}/{len(logical_scenes)}: [No prominent humans found]")
                continue

            print(f"    + Rendering Scene {s_idx}/{len(logical_scenes)}: [{start.get_timecode()} to {end.get_timecode()}] (Duration: {duration_sec:.1f}s)")
            
            try:
                process_scene(video_path, start_sec, duration_sec, out_filepath)
            except subprocess.CalledProcessError:
                print(f"    ! Error processing scene {s_idx}. FFmpeg failed.")

        cap.release()
                
    print("\n--- Batch Processing Complete! ---")

if __name__ == "__main__":
    main()