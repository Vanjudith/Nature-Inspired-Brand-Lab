# Nature-Inspired Brand Lab 🌿

An interactive color science application that extracts representative brand color palettes from nature photography and visual inspiration.

Developed as an MVP connecting nature-inspired imagery with a design workflow using perceptual color clustering.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12 / 3.14)
- Git

### 2. Environment Setup

Clone the repository (or navigate to the project directory):
```bash
git clone https://github.com/Vanjudith/Nature-Inspired-Brand-Lab.git
cd Nature-Inspired-Brand-Lab
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Application

Run the Streamlit web application:
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the test suite with `pytest`:
```bash
pytest -v
```

The test suite validates:
- Standard RGB image palette extraction.
- Valid 7-character uppercase HEX formatting (`#RRGGBB`).
- Cluster proportion calculations summing to ~100%.
- Single-color and flat-color images (graceful K-clamping).
- Tiny images (< 50,000 pixels down to 2×2 pixels).
- RGBA / transparency handling (compositing onto white background).
- Grayscale / monochrome conversion.
- Error handling on corrupted and non-image files.
- Deterministic reproducibility under fixed random seeds.
- JSON and CSV export validity and re-parseability.

---

## 🎨 How It Works (Color Science & Architecture)

```
[Inspiration Image]
       │
       ▼
[Image Validation & Normalization (Pillow)]
       │ (Standardizes RGBA / Grayscale / WebP to RGB)
       ▼
[Deterministic Pixel Sampling]
       │ (50,000 pixels baseline for responsiveness & representation)
       ▼
[RGB to CIE L*a*b* Perceptual Conversion]
       │ (scikit-image rgb2lab)
       ▼
[K-Means Clustering in LAB Space]
       │ (scikit-learn KMeans)
       ▼
[Dominance Ranking & Gamut Clipping]
       │ (Sorted by cluster pixel population; clipped to sRGB)
       ▼
[Exportable Palettes & Swatches]
       └── HEX Swatches, RGB, LAB, Cluster %, JSON & CSV Exports
```

### Why CIE LAB instead of RGB?
In standard sRGB space, Euclidean distance between numerical channel values does not match how human vision perceives color differences. For example, two green shades can have a large numerical distance but look nearly identical, while small shifts in orange or blue are immediately noticeable.

This engine transforms sampled pixels into **CIE L\*a\*b\***:
- **L\***: Perceptual Lightness
- **a\***: Green ↔ Red axis
- **b\***: Blue ↔ Yellow axis

K-Means clusters colors based on perceptual distance, resulting in representative palettes that reflect the visual feeling of the original photograph.

---

## ✨ Features

- **Multi-Format Upload**: Supports JPEG, PNG (with transparency handling), and WebP files up to 25 MB.
- **Curated Nature Samples**: Instant one-click testing with high-resolution reference nature images from Pixabay.
- **Representative 5-Color Default**: Extracts an initial 5-color palette, with an adjustable cluster count slider (2 to 10 colors).
- **Cluster Coverage %**: Shows the percentage of sampled pixels corresponding to each color group.
- **Dominance Sorting**: Colors are automatically sorted from most dominant to accent colors.
- **One-Click Export**:
  - Copy individual HEX codes with one click.
  - Copy the full palette as a space-separated HEX string.
  - Download structured JSON with RGB, LAB, counts, and percentages.
  - Download CSV for spreadsheet workflows.
- **Graceful Edge Cases**: Flat images with fewer unique colors than $K$ are automatically clamped without crashing.
- **Diagnostic Evaluation**: Preserves research findings by displaying sample-vs-full mean RGB and image reconstruction Mean Squared Error (MSE).

---

## 📁 Repository Structure

```text
Nature-Inspired-Brand-Lab/
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Build & pytest configuration
├── src/
│   └── brand_lab/
│       ├── __init__.py         # Package exports
│       └── palette.py          # Core image processing & color science pipeline
├── tests/
│   ├── __init__.py
│   └── test_palette.py         # Automated test suite (17 test cases)
├── Docs/
│   └── System_Design.md        # Initial system design and requirements
├── Images/                     # Pixabay nature reference inspiration images
│   ├── image_one.jpg
│   ├── image_two.jpg
│   └── image_three.jpg
├── 01_image_as_data.ipynb      # Research notebook (Experiments 001 & 002)
└── .gitignore                  # Git ignore rules
```

---

## 🔬 Preserved Research

Previous experiments and documentation remain fully preserved:
- `Docs/System_Design.md`: Initial architecture, sampling theory, and hypothesis definitions.
- `01_image_as_data.ipynb`: Exploratory experiments covering Pillow arrays, HSV/LAB conversions, elbow analysis, and initial reconstruction MSE benchmarks.

---

## ⚠️ Known Limitations

1. **Local MVP**: Shipped and verified as a local web application. Not hosted on a public domain.
2. **Spatial Ignorance**: K-Means clusters color values independent of their spatial location. A tiny focal flower and scattered noise pixels with the same color are treated identically.
3. **Cluster Proportions**: Percentages represent proportions of the sampled pixels (50,000 pixels), which closely approximate but do not exactly equal full-image counts.
4. **Zero-Budget Constraints**: No external paid APIs or cloud dependencies are required.

---

## 📜 License

MIT License. Images in `Images/` are sourced from Pixabay under free creative use.
