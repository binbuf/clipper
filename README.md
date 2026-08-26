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

| Argument | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | **(Required)** Path to the directory containing source videos. | - |
| `--output` | `-o` | **(Required)** Path to the destination directory for processed clips. | - |
| `--min-seconds` | `-m` | Minimum duration in seconds for an extracted clip. | `12.0` |

### Example

```bash
python batch_processor.py -i ./raw_footage -o ./processed_clips -m 15.0
```

## 🧠 How it Works

1. **Cut Detection:** Scans the entire video using `AdaptiveDetector` to find all raw camera cuts.
2. **Scene Grouping:** Analyzes visual similarity between sequential cuts. Rapid cuts from the same environment are merged until the `min-seconds` threshold is met.
3. **Human Validation:** Samples frames across the proposed clip. At least 70% of samples must contain a prominent human (via YOLO), and strict boundary checks ensure the clip doesn't fade into text or logos.
4. **Rendering:** Validated scenes are passed to FFmpeg for high-quality HEVC transcoding and audio normalization.