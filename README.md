<div align="center">

# 🎬 Clipper

**Automated Video Scene Extraction & Normalization Pipeline**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-yellow.svg)](https://ultralytics.com/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-green.svg)](https://ffmpeg.org/)

</div>

---

**Clipper** is a powerful batch-processing tool that automatically extracts high-quality clips from full-length media. It intelligently detects camera cuts, groups them into logical scenes, filters for prominent human presence using YOLO, and normalizes the output using FFmpeg.

## ✨ Features

- 🎞️ **Intelligent Scene Detection:** Uses `PySceneDetect` with adaptive thresholding to find raw camera cuts.
- 🔗 **Smart Shot Grouping:** Merges rapid cuts into cohesive scenes based on visual similarity (histogram correlation).
- 👤 **Human-Centric Filtering:** Utilizes YOLO11 to ensure extracted clips feature prominent human subjects, strictly filtering out graphics, intros, and logos.
- 🎛️ **Standardized Output:** 
  - **Video:** HEVC (H.265), 1080p, 30fps (pillarboxed/letterboxed to maintain aspect ratio).
  - **Audio:** AAC, EBU R128 loudness normalized.
- 📁 **Batch Processing:** Seamlessly processes entire directories of mixed video formats.
- 🌐 **Social Media:** Outputs media optimized for social media uploads.

## 🛠️ Prerequisites

- **Python 3.8+**
- **FFmpeg:** Must be installed and accessible in your system's PATH.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/binbuf/clipper.git
   cd clipper
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the batch processor by specifying your input directory and output destination.

```bash
python batch_processor.py -i /path/to/raw/videos -o /path/to/output/clips
```

### Options

Every value below is configurable; the defaults match Clipper's intended use (human-centric clips, 45s–5min).

**Core**

| Argument | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | **(Required)** Path to the directory containing source videos. | - |
| `--output` | `-o` | **(Required)** Path to the destination directory for processed clips. | - |
| `--min-seconds` | `-m` | Minimum duration in seconds for an extracted clip. | `45.0` |
| `--max-seconds` | `-x` | Maximum approximate clip duration in seconds. | `300.0` |
| `--max-grace` | | Seconds a shot may run past `--max-seconds` to reach a natural camera cut before a hard cut is forced. | `15%` of max |
| `--jobs` | `-j` | Concurrent FFmpeg render jobs. FFmpeg is itself multithreaded, so `2`–`4` is usually plenty. | `1` |
| `--gpu` | | Run YOLO detection on GPU (CUDA, else Apple MPS). Omit for CPU. | CPU |

**Human filter**

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--person-conf` | YOLO person confidence threshold. | `0.65` |
| `--min-area-ratio` | Minimum fraction of the frame a person must fill. | `0.20` |
| `--sample-frames` | Frames sampled per scene for human detection. | `10` |
| `--min-human-ratio` | Fraction of sampled frames that must contain a human. | `0.70` |

**Grouping & encoding**

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--similarity` | Histogram correlation to merge camera angles in one scene. | `0.55` |
| `--model` | YOLO model name or path. | `yolo11s.pt` |
| `--video-codec` / `--video-bitrate` | Output video codec / target bitrate. | `libx265` / `3000k` |
| `--audio-codec` / `--audio-bitrate` | Output audio codec / target bitrate. | `aac` / `128k` |
| `--resolution` / `--fps` | Output `WxH` / frame rate. | `1920x1080` / `30` |
| `--loudnorm` | FFmpeg `loudnorm` (EBU R128) parameters. | `I=-14:TP=-1.5:LRA=11` |

### Example

```bash
# 3 parallel encodes, YOLO on the GPU, 60s–4min clips
python batch_processor.py -i ./raw_footage -o ./processed_clips -m 60 -x 240 -j 3 --gpu
```

## 🧠 How it Works

1. **Cut Detection:** Scans the entire video using `AdaptiveDetector` to find all raw camera cuts.
2. **Scene Grouping:** Analyzes visual similarity between sequential cuts. Rapid cuts from the same environment are merged until the `min-seconds` threshold is met. Long scenes are ended at the next real camera cut; a mid-shot hard cut is used only when a single unbroken shot runs past `max-seconds` + `max-grace`.
3. **Human Validation:** Samples frames across the proposed clip and runs them through YOLO in a single batched inference. At least `min-human-ratio` of samples must contain a prominent human, and strict first/last boundary checks reject clips that fade into text or logos. Logo-heavy intros and outros contain no prominent human, so they are dropped as "No prominent humans found."
4. **Rendering:** Validated scenes are handed to a pool of up to `--jobs` FFmpeg workers for high-quality HEVC transcoding and audio normalization. Each worker is an isolated subprocess writing to its own temp file, so parallel renders never interfere. Analysis stays single-threaded, so the shared video reader and detector are never contended.