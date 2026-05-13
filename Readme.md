# 🔍 Tracera — Advanced Deepfake Detection System

![Tracera Banner](logo.png)

> **Forensic-grade image analysis to distinguish between human-captured and AI-generated imagery.**

**[📄 Read the Research Paper](https://drive.google.com/file/d/18peiBMI3Rnl-GQCE2b7UPe-TRgHO-c9v/view?usp=drive_link) | [🖼️ View the Project Poster](https://drive.google.com/file/d/1EnHFrBNXQ1yfb_VueXKKngV_ij1M66Hy/view?usp=drive_link)**

Tracera is a high-performance deepfake detection ecosystem powered by **GramNet v3**. It provides a seamless pipeline from state-of-the-art machine learning inference to real-world application via a web dashboard and a browser extension.

---

## 🚀 Key Features

- **High-Fidelity Detection**: Leverages Gram Matrix analysis to identify subtle textural anomalies in AI-generated images.
- **Multi-Vector Support**: Specifically tuned to detect artifacts from GANs (Generative Adversarial Networks) and Diffusion models.
- **Real-Time Analysis**: Instant results through a secure Flask-powered REST API.
- **Browser Extension**: Right-click any image on the web to analyze it directly within your browser.
- **Secure by Design**: Implements rate limiting, file validation, and strict CSP headers.

---

## 🛠️ Tech Stack

### Backend & ML

- **Python 3.10+** — Core logic
- **Flask** — REST API framework
- **PyTorch (CPU)** — Deep learning inference
- **XGBoost & Scikit-learn** — Ensemble classification
- **Pillow** — Image processing

### Frontend & Extension

- **Vanilla JS/CSS** — Ultra-fast, glassmorphic UI
- **Chrome Manifest V3** — Modern browser extension standard

---

## 📦 Project Structure

```text
.
├── app.py                  # Flask backend server & API endpoints
├── inference.py            # GramNet v3 inference engine
├── requirements.txt        # Python dependencies
├── logo.png                # Project branding assets
├── model/                  # Model weights and configuration files
├── static/                 # Web dashboard frontend assets
└── tracera-extension/      # Browser extension source code (Chrome/Edge)
```

---

## ⚙️ Installation & Setup

### 1. Backend Setup

Clone the repository and install dependencies:

```powershell
# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Run the development server:

```powershell
python app.py
```

The API will be available at `http://localhost:7860`.

### 2. Browser Extension Setup

1. Open Chrome/Edge and navigate to `chrome://extensions/`.
2. Enable **Developer mode**.
3. Click **Load unpacked** and select the `tracera-extension` folder.
4. Click the extension icon in your toolbar, enter your API URL (e.g., `http://localhost:7860`), and click **Save**.

---

## 🔬 How it Works (GramNet v3)

Tracera uses a specialized architecture that doesn't just look at pixels, but at the **statistical relationships** between feature maps.

1. **Feature Extraction**: Uses a pre-trained CNN to extract high-level features.
2. **Gram Matrix Analysis**: Computes the correlation between features to capture texture and style.
3. **Eigenvalue Profiling**: Extracts structural signatures from the Gram Matrix.
4. **XGBoost Classification**: A final ensemble layer provides the "Real" vs "Fake" verdict with confidence scores.

---

## 📄 License

© 2026 Tracera. All rights reserved.
Designed for security researchers, journalists, and everyday users to verify digital truth.
