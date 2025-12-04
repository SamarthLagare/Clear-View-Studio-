# 💎 Clear View Studio

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-MIT-grey)

**Clear View Studio** is a professional, high-performance media analysis tool built with Python. It features a sleek, dark-themed UI for advanced image processing and video forensics.

---

## ✨ Features

### 📷 Image Studio
A complete workspace for static imagery.
* **Geometry Tools:** Rotate, Mirror (Horizontal/Vertical).
* **Pro Filters:** Black & White, Sepia, Sketch, Invert.
* **Fine Tuning:** Real-time Brightness and Contrast sliders.
* **AI Detection:**
    * Face Detection (Haar Cascades)
    * ORB Feature Extraction (Keypoints)
    * Object & Edge Detection
* **Split View:** Compare Original vs. Processed side-by-side (Vertical, Horizontal, 3-Way).
* **Grid Overlay:** Customizable grid density and color for alignment checks.

### 🎬 Video Lab
A forensics suite for video files.
* **Timeline Analysis:** Scrub through videos frame-by-frame.
* **Frame Editor:** Apply all Image Studio tools to specific video frames.
* **Metadata:** Instant readout of FPS, Resolution, Duration, and Total Frames.
* **Audio Forensics:**
    * Extract audio tracks from video.
    * Visualize **Waveform** and **Frequency Spectrum (FFT)**.
    * Calculate **RMS Energy**, **Decibels (dB)**, and **Sample Rate**.
    * Download extracted audio as `.wav`.

---

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/clear-view-studio.git](https://github.com/your-username/clear-view-studio.git)
    cd clear-view-studio
    ```

2.  **Install dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 📦 Requirements

Create a file named `requirements.txt` and add these libraries:

```text
streamlit
opencv-python-headless
numpy
Pillow
moviepy
