# Portfolio Case Study: Nature-Inspired Brand Lab

## Executive Summary

**Nature-Inspired Brand Lab** is a local, interactive color-science tool that extracts cohesive, representative brand color palettes from photography and visual inspiration. Built with Python, Streamlit, scikit-learn, scikit-image, and Pillow, the application bridges the gap between raw natural imagery and digital brand identity workflows using perceptual color-space clustering and memory-conscious visual reconstruction.

---

## 1. Project Goal & Problem Statement

Designers and brand strategists frequently turn to nature photography for authentic, organic color inspiration. However, extracting palettes manually using color-picker tools introduces human bias and often over-samples high-contrast accents while missing dominant background tones. Conversely, naive algorithmic approaches using standard sRGB averaging or Euclidean distance in RGB produce muddy, perceptually discordant color swatches because RGB distance does not align with human vision.

**Objective:** Build a self-contained, zero-budget MVP application that:
1. Ingests diverse image formats (JPEG, PNG with transparency compositing, WebP) and curated reference photographs.
2. Extracts a representative $K$-color palette ($K \in [2, 10]$) using perceptual color-space clustering.
3. Provides an interactive side-by-side visual reconstruction preview to evaluate how faithfully the palette reproduces the source image.
4. Quantifies reconstruction fidelity with clearly defined mathematical error metrics.
5. Exports structured palette data in both JSON and CSV formats for immediate design workflow integration.

---

## 2. User Workflow

The application workflow is structured into a streamlined five-stage pipeline:

1. **Ingestion & Selection:**
   - Upload any image up to 25 MB (JPEG, PNG with alpha transparency composited over pure white, or WebP).
   - Alternatively, choose one of three curated nature inspiration presets: *Image 1 (Jellyfish & Blue Water)*, *Image 2 (Translucent Jellyfish on Dark Background)*, or *Image 3 (Blue-and-Yellow Tropical Fish)*.
2. **Parameter Configuration:**
   - Adjust cluster count ($K$) between 2 and 10 colors (default: 5 colors).
   - Configure pixel sample size between 10,000 and 100,000 pixels (default: 50,000 pixels).
3. **Visual Inspection:**
   - View the source image side-by-side with its $K$-color reconstructed preview rendered at identical dimensions.
   - Inspect dominance-sorted color swatch cards detailing HEX codes, RGB tuples, CIE LAB coordinates, and cluster coverage percentages.
4. **Diagnostic Evaluation:**
   - Inspect pixel-level summary statistics comparing sample mean RGB against full-image mean RGB.
   - Review Reconstruction MSE and RGB Vector RMSE to gauge numerical pixel reconstruction fidelity.
5. **Asset Export:**
   - Copy individual HEX codes or the entire palette string with one click.
   - Download machine-readable JSON (with full color-space data and metadata) or tabular CSV.

---

## 3. Key Engineering Decisions & Technical Architecture

### 3.1. Perceptual Clustering (CIE LAB) vs. Display Reproduction (sRGB)
- **Clustering in CIE LAB:** In sRGB space, Euclidean distance between color coordinates does not match human perceptual differences (e.g., small shifts in yellow or cyan appear much larger to the human eye than equivalent shifts in dark greens). The engine normalizes sampled pixels into $[0.0, 1.0]$ and converts them to **CIE L\*a\*b\*** space via `scikit-image.color.rgb2lab`. K-Means is fitted across the perceptual axes ($L^*$ lightness, $a^*$ green-red, and $b^*$ blue-yellow), yielding perceptually balanced centroids.
- **Centroid Gamut Mapping:** LAB centroids are mapped back to sRGB via `scikit-image.color.lab2rgb`, clipped to the $[0.0, 1.0]$ gamut, and rounded to valid 8-bit integers (`0–255`).

### 3.2. Color Space Consistency in Reconstruction & Error Evaluation
- **Assignment & Metric in sRGB:** While clustering operates in LAB space, visual reconstruction and error metrics operate in standard **sRGB** Euclidean distance.
- **Rationale:**
  1. *Mathematical Alignment:* Assigning pixels to the nearest palette color in sRGB space guarantees that the reconstructed image directly minimizes the computed sRGB MSE metric.
  2. *Interactive Performance:* Full-resolution images (often 10–12 megapixels) would require expensive $O(N)$ LAB conversions on every slider adjustment in Streamlit. Computing distances in sRGB avoids this overhead entirely.
  3. *Exact Identity on Pure Images:* If an image is already composed strictly of palette colors, sRGB reconstruction produces zero error ($\text{MSE} = 0.0$) and byte-for-byte identical output arrays.

### 3.3. Deterministic Pixel Sampling
- Processing all 10–15 million pixels of a camera-resolution image during interactive clustering causes unacceptable UI latency.
- The pipeline samples a baseline of 50,000 pixels using a fixed-seed pseudorandom generator (`numpy.random.default_rng(42)`).
- Benchmarking on reference imagery confirms that 50,000 sampled pixels preserve the overall color distribution (e.g., sample mean RGB `[60.13, 59.42, 139.16]` vs. full-image mean RGB `[60.49, 59.69, 138.93]` on `image_one.jpg`) while completing K-Means fitting in under 0.5 seconds.

### 3.4. Memory-Conscious Chunked Processing
- Naive full-image reconstruction across $N$ pixels and $K$ colors using broadcasting would allocate an array of shape $(N, K, 3)$. For a 12-megapixel image with $K=10$, this would require an intermediate memory allocation of $\approx 1.44\text{ GB}$ of RAM.
- **Chunked Implementation:** `reconstruct_image()` and `compute_reconstruction_mse()` iterate across pixels in chunks of 100,000 pixels (`float32`), looping over the $K$ palette centroids.
- **Impact:** Peak working memory per chunk is capped to $< 2\text{ MB}$, eliminating out-of-memory risks regardless of input image dimensions while maintaining $O(N)$ execution time.

### 3.5. Rigorous Metric Definitions
- **Reconstruction MSE:** Mean squared RGB-vector error per pixel:
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N \sum_{c \in \{R,G,B\}} (I_{i,c} - \hat{I}_{i,c})^2 = \frac{1}{N} \sum_{i=1}^N \|\vec{x}_i - \hat{\vec{x}}_i\|_2^2$$
- **RGB Vector RMSE:** The square root of MSE ($\sqrt{\text{MSE}}$).
- **Explicit Disambiguation:** The documentation and UI explicitly clarify that this is the root mean squared *vector* distance per pixel in 3D RGB space, not conventional per-channel RMSE (which would divide by $3N$).

---

## 4. Verification & Quality Assurance

The codebase is protected by an automated test suite containing **23 unit tests** in `tests/test_palette.py`, executed via `pytest`:

| Test Category | Test Count | Behaviors Verified |
|---|---|---|
| **Palette Extraction** | 7 tests | Valid 5-color default, uppercase `#RRGGBB` format, cluster proportions summing to 100%, dominance ordering, single-color K-clamping, tiny images (2×2), and deterministic repeatability under fixed seeds. |
| **Image Loading & Ingestion** | 5 tests | Byte-stream loading, RGBA compositing over white canvas, grayscale-to-RGB conversion, corrupt byte error handling, and oversized file rejections. |
| **Export Validity** | 2 tests | JSON structure and re-parseability; RFC-compliant CSV headers and rows. |
| **Diagnostics & Benchmarks** | 3 tests | End-to-end extraction and positive MSE validation on repository reference images (`image_one.jpg`, `image_two.jpg`, `image_three.jpg`). |
| **Preset Integrity** | 1 test | Validates that all three preset labels exist and map directly to existing on-disk image assets. |
| **Visual Reconstruction** | 5 tests | Output dimensions strictly match source dimensions; every output pixel is an exact palette color; exact zero error ($\text{MSE}=0.0$) on pure palette images; deterministic output; chunking consistency across different chunk sizes (e.g. 37 vs 100,000 px). |

Additionally, headless Streamlit execution was verified using `streamlit.testing.v1.AppTest`, confirming that widget state transitions and reconstruction rendering execute without runtime exceptions.

---

## 5. Architectural Trade-offs & Limitations

1. **Numerical Error vs. Brand Suitability:**
   Reconstruction error measures mathematical pixel fidelity in Euclidean RGB space. A low reconstruction error indicates that palette colors numerically approximate image pixels, but does *not* imply that the palette is aesthetically harmonious, brand-ready, accessible, or emotionally appropriate.
2. **Spatial Ignorance:**
   K-Means clusters color values independently of pixel coordinates. High-frequency background texture and a focal subject of identical hue receive equal clustering priority.
3. **Local Deployment Scope:**
   The MVP is designed and verified as a single-node local web application; distributed processing and persistent cloud storage are out of scope.
4. **Zero-Budget Constraint:**
   The application intentionally relies exclusively on open-source libraries without external paid APIs or GPU acceleration requirements.

---

## 6. Future Opportunities

The following enhancements represent natural extensions of the existing architecture (distinguished from current MVP functionality):

- **Saliency-Weighted Clustering:** Incorporating depth maps or foreground segmentation masks so focal subjects are weighted more heavily than background pixels during sampling.
- **Harmonic Rule Filtering:** Post-processing extracted centroids against classical color harmony rules (analogous, triadic, complementary) to suggest structured brand accent pairings.
- **WCAG Accessibility Analysis:** Automatically computing contrast ratios (e.g., WCAG 2.1 AA/AAA) between extracted dark and light swatches for typography readability.
- **Multi-Image Brand Moodboards:** Aggregating sampled pixel distributions across multiple inspiration images to produce a unified cross-image brand palette.
