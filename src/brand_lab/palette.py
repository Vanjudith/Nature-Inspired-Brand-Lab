"""Core image-to-palette extraction and color science processing module.

Preserves the engineering findings from Experiment 001 and 002:
- RGB representation via Pillow and NumPy
- Deterministic random pixel sampling (50,000 pixels baseline)
- CIE LAB color space conversion via scikit-image for perceptual clustering
- K-Means clustering via scikit-learn
- Conversion back to standard sRGB and validated uppercase HEX format
- Cluster population counts and percentages
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Sequence, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import color
from sklearn.cluster import KMeans

# Default constants based on research and performance benchmarks
DEFAULT_SAMPLE_SIZE: int = 50_000
DEFAULT_CLUSTER_COUNT: int = 5
DEFAULT_RANDOM_STATE: int = 42
MAX_IMAGE_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB


@dataclass(frozen=True)
class PaletteColor:
    """Represents a single extracted color swatch with multi-space values."""

    hex: str
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    count: int
    percentage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "hex": self.hex,
            "rgb": {
                "r": self.rgb[0],
                "g": self.rgb[1],
                "b": self.rgb[2],
            },
            "lab": {
                "l": round(self.lab[0], 2),
                "a": round(self.lab[1], 2),
                "b": round(self.lab[2], 2),
            },
            "pixel_count": self.count,
            "percentage": round(self.percentage, 3),
        }


@dataclass
class PaletteResult:
    """Structured result returned by the palette extraction pipeline."""

    colors: list[PaletteColor]
    total_samples: int
    n_colors_requested: int
    n_colors_returned: int
    clamped: bool
    clamped_reason: str | None
    original_shape: tuple[int, int, int]
    sample_mean_rgb: tuple[float, float, float]
    full_mean_rgb: tuple[float, float, float]

    def hex_list(self) -> list[str]:
        """Returns list of HEX string codes."""
        return [c.hex for c in self.colors]

    def hex_string(self, delimiter: str = " ") -> str:
        """Returns concatenated HEX codes separated by delimiter."""
        return delimiter.join(self.hex_list())


def load_and_validate_image(
    source: Union[str, Path, bytes, BinaryIO],
    max_size_bytes: int = MAX_IMAGE_FILE_SIZE_BYTES,
) -> Image.Image:
    """Loads and validates an image from a path, raw bytes, or file buffer.

    Ensures the image is decoded, checked for size constraints, and
    reliably converted to 3-channel RGB. Handles RGBA/transparency by
    compositing against a clean white canvas.

    Args:
        source: File path, byte buffer, or file-like stream.
        max_size_bytes: Maximum allowed byte size (default 25 MB).

    Returns:
        Validated PIL Image in 'RGB' mode.

    Raises:
        ValueError: If file size exceeds limits, image is zero-sized, or corrupted.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"Image file does not exist: {path}")
        size = path.stat().st_size
        if size > max_size_bytes:
            raise ValueError(
                f"File size ({size / (1024*1024):.2f} MB) exceeds maximum limit "
                f"({max_size_bytes / (1024*1024):.2f} MB)."
            )
        try:
            raw_img = Image.open(path)
            raw_img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Unable to decode image file '{path.name}': {e}") from e

    elif isinstance(source, bytes):
        if len(source) > max_size_bytes:
            raise ValueError(
                f"Data size ({len(source) / (1024*1024):.2f} MB) exceeds limit "
                f"({max_size_bytes / (1024*1024):.2f} MB)."
            )
        try:
            raw_img = Image.open(io.BytesIO(source))
            raw_img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Unable to decode image from byte buffer: {e}") from e

    else:
        # File-like object (e.g. Streamlit UploadedFile or BytesIO)
        content = source.read()
        if len(content) > max_size_bytes:
            raise ValueError(
                f"Uploaded file size ({len(content) / (1024*1024):.2f} MB) exceeds "
                f"limit ({max_size_bytes / (1024*1024):.2f} MB)."
            )
        try:
            raw_img = Image.open(io.BytesIO(content))
            raw_img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Unable to decode uploaded image: {e}") from e

    # Check non-zero dimensions
    if raw_img.width <= 0 or raw_img.height <= 0:
        raise ValueError(f"Invalid image dimensions: {raw_img.size}")

    # Standardize color mode to RGB
    if raw_img.mode in ("RGBA", "LA") or (
        raw_img.mode == "P" and "transparency" in raw_img.info
    ):
        # Composite transparent pixels onto a pure white background
        rgba = raw_img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(background, rgba)
        rgb_img = composed.convert("RGB")
    else:
        rgb_img = raw_img.convert("RGB")

    return rgb_img


def extract_palette(
    image: Image.Image,
    n_colors: int = DEFAULT_CLUSTER_COUNT,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> PaletteResult:
    """Extracts a representative color palette using K-Means in CIE LAB space.

    Pipeline:
    1. Reshapes image to (N, 3) RGB pixels.
    2. Samples up to `sample_size` pixels deterministically using `random_state`.
    3. Detects distinct color count; gracefully clamps K if distinct colors < n_colors.
    4. Converts sampled RGB pixels to CIE LAB color space.
    5. Runs K-Means clustering on LAB coordinates.
    6. Converts cluster centroids back to sRGB and formats valid HEX codes.
    7. Computes cluster pixel counts and percentages.
    8. Returns colors sorted by cluster dominance (highest percentage first).

    Args:
        image: PIL Image (must be in RGB mode or will be auto-converted).
        n_colors: Desired number of palette colors (minimum 1, default 5).
        sample_size: Maximum pixels to sample for clustering (default 50,000).
        random_state: Deterministic random seed for reproducibility.

    Returns:
        PaletteResult containing colors, percentages, and metadata.

    Raises:
        ValueError: If n_colors < 1 or image is empty.
    """
    if n_colors < 1:
        raise ValueError(f"n_colors must be at least 1, got {n_colors}.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    image_array = np.asarray(image, dtype=np.uint8)
    original_shape = image_array.shape  # (H, W, 3)

    if image_array.size == 0 or original_shape[0] == 0 or original_shape[1] == 0:
        raise ValueError("Image contains no pixel data.")

    # Reshape (H, W, 3) -> (N, 3)
    pixels = image_array.reshape(-1, 3)
    total_pixels = len(pixels)

    # Calculate overall image mean RGB
    full_mean_rgb = tuple(float(x) for x in pixels.mean(axis=0))

    # Sample pixels to keep clustering fast and predictable
    if total_pixels <= sample_size:
        sample_pixels = pixels
    else:
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(total_pixels, size=sample_size, replace=False)
        sample_pixels = pixels[sample_indices]

    total_samples = len(sample_pixels)
    sample_mean_rgb = tuple(float(x) for x in sample_pixels.mean(axis=0))

    # Edge case: handle images with very few distinct colors (e.g. solid color)
    unique_colors = np.unique(sample_pixels, axis=0)
    distinct_count = len(unique_colors)

    clamped = False
    clamped_reason = None
    k = n_colors

    if distinct_count < n_colors:
        k = max(1, distinct_count)
        clamped = True
        clamped_reason = (
            f"Image sample contains only {distinct_count} distinct color"
            f"{'s' if distinct_count > 1 else ''}. Requested {n_colors} clusters "
            f"clamped to {k}."
        )

    # Convert sampled pixels from sRGB [0, 255] to normalized [0.0, 1.0] -> LAB
    sample_pixels_norm = sample_pixels.astype(np.float64) / 255.0
    # skimage.color.rgb2lab expects 3D array (1, N, 3)
    sample_lab = color.rgb2lab(sample_pixels_norm.reshape(1, -1, 3)).reshape(-1, 3)

    # Perform K-Means clustering in LAB perceptual color space
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    kmeans.fit(sample_lab)

    lab_centroids = kmeans.cluster_centers_  # shape (k, 3)
    labels = kmeans.labels_  # shape (total_samples,)

    # Calculate cluster populations and percentages
    cluster_counts = np.bincount(labels, minlength=k)
    cluster_percentages = (cluster_counts / total_samples) * 100.0

    # Convert LAB centroids back to sRGB
    rgb_centroids_norm = color.lab2rgb(lab_centroids.reshape(1, -1, 3)).reshape(-1, 3)

    # Clip to valid sRGB [0.0, 1.0] gamut and scale to [0, 255] uint8
    rgb_clipped = np.clip(rgb_centroids_norm, 0.0, 1.0)
    rgb_uint8 = np.clip(np.round(rgb_clipped * 255.0), 0, 255).astype(np.uint8)

    palette_colors: list[PaletteColor] = []
    for i in range(k):
        r, g, b = int(rgb_uint8[i, 0]), int(rgb_uint8[i, 1]), int(rgb_uint8[i, 2])
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        lab_tuple = (
            float(lab_centroids[i, 0]),
            float(lab_centroids[i, 1]),
            float(lab_centroids[i, 2]),
        )
        palette_colors.append(
            PaletteColor(
                hex=hex_code,
                rgb=(r, g, b),
                lab=lab_tuple,
                count=int(cluster_counts[i]),
                percentage=float(cluster_percentages[i]),
            )
        )

    # Sort palette by cluster population descending (most dominant color first)
    palette_colors.sort(key=lambda c: c.count, reverse=True)

    return PaletteResult(
        colors=palette_colors,
        total_samples=total_samples,
        n_colors_requested=n_colors,
        n_colors_returned=len(palette_colors),
        clamped=clamped,
        clamped_reason=clamped_reason,
        original_shape=original_shape,
        sample_mean_rgb=sample_mean_rgb,
        full_mean_rgb=full_mean_rgb,
    )


def export_palette_json(
    result: PaletteResult,
    source_name: str = "image",
    pretty: bool = True,
) -> str:
    """Exports palette results to structured JSON."""
    payload = {
        "source": source_name,
        "sample_size": result.total_samples,
        "requested_colors": result.n_colors_requested,
        "returned_colors": result.n_colors_returned,
        "clamped": result.clamped,
        "clamped_reason": result.clamped_reason,
        "colors": [c.to_dict() for c in result.colors],
        "hex_palette": result.hex_list(),
    }
    return json.dumps(payload, indent=2 if pretty else None)


def export_palette_csv(result: PaletteResult) -> str:
    """Exports palette results to standard CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Rank",
            "HEX",
            "Red",
            "Green",
            "Blue",
            "LAB_L",
            "LAB_a",
            "LAB_b",
            "Pixel_Count",
            "Percentage",
        ]
    )
    for rank, c in enumerate(result.colors, start=1):
        writer.writerow(
            [
                rank,
                c.hex,
                c.rgb[0],
                c.rgb[1],
                c.rgb[2],
                round(c.lab[0], 2),
                round(c.lab[1], 2),
                round(c.lab[2], 2),
                c.count,
                round(c.percentage, 3),
            ]
        )
    return output.getvalue()


def compute_reconstruction_mse(image: Image.Image, result: PaletteResult) -> float:
    """Preserves diagnostic evaluation from Experiment 002.

    Assigns each pixel of the full image to its nearest palette color in RGB
    and computes the Mean Squared Error (MSE).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_array = np.asarray(image, dtype=np.float64)
    pixels = image_array.reshape(-1, 3)

    palette_rgb = np.array([c.rgb for c in result.colors], dtype=np.float64)

    # Compute minimum Euclidean distance squared to palette colors in RGB memory-efficiently
    min_dist_sq = np.full(len(pixels), np.inf, dtype=np.float64)
    for c in palette_rgb:
        dist_sq = np.sum((pixels - c) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    mse = float(np.mean(min_dist_sq))
    return mse
