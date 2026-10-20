"""Comprehensive unit tests for brand_lab palette extraction."""

import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brand_lab.palette import (
    PaletteColor,
    PaletteResult,
    compute_reconstruction_mse,
    export_palette_csv,
    export_palette_json,
    extract_palette,
    load_and_validate_image,
    reconstruct_image,
)

HEX_REGEX = re.compile(r"^#[0-9A-F]{6}$")


@pytest.fixture
def sample_gradient_image() -> Image.Image:
    """Creates a synthetic 100x100 RGB image with rich color variations."""
    w, h = 100, 100
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = [int(255 * x / w), int(255 * y / h), 128]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def single_color_image() -> Image.Image:
    """Creates a 50x50 solid green image."""
    arr = np.full((50, 50, 3), [34, 139, 34], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def tiny_image() -> Image.Image:
    """Creates a 2x2 image (4 pixels total)."""
    arr = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def rgba_image() -> Image.Image:
    """Creates a 50x50 RGBA image with transparent parts."""
    arr = np.zeros((50, 50, 4), dtype=np.uint8)
    # Left half: red opaque
    arr[:, :25] = [200, 40, 40, 255]
    # Right half: blue semi-transparent
    arr[:, 25:] = [40, 40, 200, 128]
    return Image.fromarray(arr, mode="RGBA")


class TestPaletteExtraction:
    def test_default_five_colors_extracted(self, sample_gradient_image: Image.Image):
        """Test 1: Valid image produces requested number of colors."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        assert res.n_colors_returned == 5
        assert len(res.colors) == 5
        assert res.clamped is False

    def test_hex_formatting(self, sample_gradient_image: Image.Image):
        """Test 2: HEX strings follow uppercase #RRGGBB format."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        for c in res.colors:
            assert HEX_REGEX.match(c.hex) is not None, f"Invalid HEX: {c.hex}"

    def test_cluster_proportions_sum_to_100(self, sample_gradient_image: Image.Image):
        """Test 3: Cluster percentages sum to ~100%."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        total_pct = sum(c.percentage for c in res.colors)
        assert pytest.approx(total_pct, rel=1e-3) == 100.0

        total_counts = sum(c.count for c in res.colors)
        assert total_counts == res.total_samples

    def test_colors_sorted_by_dominance(self, sample_gradient_image: Image.Image):
        """Test that returned colors are sorted descending by cluster count."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        counts = [c.count for c in res.colors]
        assert counts == sorted(counts, reverse=True)

    def test_single_color_image_handled_sensibly(self, single_color_image: Image.Image):
        """Test 4: Single-color image clamps gracefully instead of crashing."""
        res = extract_palette(single_color_image, n_colors=5, random_state=42)
        assert res.clamped is True
        assert res.n_colors_returned == 1
        assert len(res.colors) == 1
        assert res.colors[0].percentage == 100.0
        assert res.colors[0].rgb == (34, 139, 34)

    def test_tiny_image_smaller_than_sample_and_k(self, tiny_image: Image.Image):
        """Test 5: Image with fewer pixels than sample size or K."""
        # 4 pixels, request 5 colors
        res = extract_palette(tiny_image, n_colors=5, random_state=42)
        assert res.total_samples == 4
        assert res.clamped is True
        assert res.n_colors_returned == 4
        assert len(res.colors) == 4

    def test_repeatability_under_deterministic_seed(self, sample_gradient_image: Image.Image):
        """Test 7: Repeatable under deterministic settings."""
        res1 = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        res2 = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        assert res1.hex_list() == res2.hex_list()
        assert [c.percentage for c in res1.colors] == [c.percentage for c in res2.colors]


class TestImageLoadingAndValidation:
    def test_load_from_bytes(self, sample_gradient_image: Image.Image):
        """Test loading valid JPEG/PNG from bytes."""
        buf = io.BytesIO()
        sample_gradient_image.save(buf, format="PNG")
        loaded = load_and_validate_image(buf.getvalue())
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)

    def test_load_rgba_converts_with_white_background(self, rgba_image: Image.Image):
        """Test RGBA composite handling."""
        buf = io.BytesIO()
        rgba_image.save(buf, format="PNG")
        loaded = load_and_validate_image(buf.getvalue())
        assert loaded.mode == "RGB"
        assert loaded.size == (50, 50)

    def test_load_grayscale_converts_to_rgb(self):
        """Test grayscale conversion."""
        gray = Image.new("L", (30, 30), color=120)
        buf = io.BytesIO()
        gray.save(buf, format="PNG")
        loaded = load_and_validate_image(buf.getvalue())
        assert loaded.mode == "RGB"

    def test_invalid_image_raises_controlled_error(self):
        """Test 6: Corrupt/invalid image bytes raises controlled ValueError."""
        with pytest.raises(ValueError, match="Unable to decode"):
            load_and_validate_image(b"This is not a real image file content")

    def test_oversized_file_raises_error(self):
        """Test that file exceeding max_size_bytes raises ValueError."""
        dummy_data = b"0" * 1024
        with pytest.raises(ValueError, match="exceeds"):
            load_and_validate_image(dummy_data, max_size_bytes=512)


class TestExportFormats:
    def test_export_json_valid_and_parseable(self, sample_gradient_image: Image.Image):
        """Test 8: JSON export is valid and re-parseable."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        json_str = export_palette_json(res, source_name="test_image.png")
        data = json.loads(json_str)
        assert data["source"] == "test_image.png"
        assert len(data["colors"]) == 5
        assert len(data["hex_palette"]) == 5
        assert data["colors"][0]["hex"] == res.colors[0].hex

    def test_export_csv_valid(self, sample_gradient_image: Image.Image):
        """Test CSV export format."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        csv_str = export_palette_csv(res)
        lines = csv_str.strip().split("\r\n" if "\r\n" in csv_str else "\n")
        assert len(lines) == 6  # 1 header + 5 swatches
        assert "Rank,HEX,Red,Green,Blue" in lines[0]


class TestDiagnosticsAndRealImages:
    def test_real_image_one(self):
        """Tests end-to-end extraction on actual repository image."""
        img_path = Path("Images/image_one.jpg")
        if not img_path.exists():
            pytest.skip("Images/image_one.jpg not found")

        img = load_and_validate_image(img_path)
        res = extract_palette(img, n_colors=5, random_state=42)
        assert res.n_colors_returned == 5
        for c in res.colors:
            assert HEX_REGEX.match(c.hex)

        # Test MSE calculation from Experiment 002
        mse = compute_reconstruction_mse(img, res)
        assert mse > 0

    def test_real_image_two(self):
        """Tests palette extraction on image_two.jpg."""
        img_path = Path("Images/image_two.jpg")
        if not img_path.exists():
            pytest.skip("Images/image_two.jpg not found")

        img = load_and_validate_image(img_path)
        res = extract_palette(img, n_colors=5, random_state=42)
        assert res.n_colors_returned == 5
        assert len(res.colors) == 5
        for c in res.colors:
            assert HEX_REGEX.match(c.hex)
        assert pytest.approx(sum(c.percentage for c in res.colors), rel=1e-3) == 100.0

    def test_real_image_three(self):
        """Tests palette extraction on image_three.jpg."""
        img_path = Path("Images/image_three.jpg")
        if not img_path.exists():
            pytest.skip("Images/image_three.jpg not found")

        img = load_and_validate_image(img_path)
        res = extract_palette(img, n_colors=5, random_state=42)
        assert res.n_colors_returned == 5
        assert len(res.colors) == 5
        for c in res.colors:
            assert HEX_REGEX.match(c.hex)
        assert pytest.approx(sum(c.percentage for c in res.colors), rel=1e-3) == 100.0


class TestCuratedSamplePresets:
    def test_curated_preset_labels_and_files(self):
        """Verifies preset labels and that each maps to the correct existing image file."""
        from app import CURATED_SAMPLES

        expected = {
            "Image 1 (Jellyfish & Blue Water)": "image_one.jpg",
            "Image 2 (Translucent Jellyfish on Dark Background)": "image_two.jpg",
            "Image 3 (Blue-and-Yellow Tropical Fish)": "image_three.jpg",
        }

        assert len(CURATED_SAMPLES) == 3
        for label, expected_filename in expected.items():
            assert label in CURATED_SAMPLES, f"Preset label '{label}' not found in CURATED_SAMPLES"
            img_path = Path(CURATED_SAMPLES[label])
            assert img_path.exists(), f"Image path does not exist: {img_path}"
            assert img_path.name == expected_filename, (
                f"Preset '{label}' mapped to '{img_path.name}', expected '{expected_filename}'"
            )


class TestVisualPaletteReconstruction:
    def test_reconstructed_dimensions_match_input(self, sample_gradient_image: Image.Image):
        """Reconstruction preserves the exact width, height, and mode of the source."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        reconstructed = reconstruct_image(sample_gradient_image, res)

        assert reconstructed.size == sample_gradient_image.size
        assert reconstructed.width == sample_gradient_image.width
        assert reconstructed.height == sample_gradient_image.height
        assert reconstructed.mode == "RGB"

        # Also test rectangular non-square dimensions
        rect_img = Image.new("RGB", (73, 41), color=(100, 150, 200))
        rect_res = extract_palette(rect_img, n_colors=3, random_state=42)
        rect_recon = reconstruct_image(rect_img, rect_res)
        assert rect_recon.size == (73, 41)

    def test_every_output_pixel_is_palette_color(self, sample_gradient_image: Image.Image):
        """Every pixel in the reconstructed image is strictly one of the extracted palette colors."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)
        reconstructed = reconstruct_image(sample_gradient_image, res)

        recon_arr = np.asarray(reconstructed, dtype=np.uint8).reshape(-1, 3)
        unique_recon_colors = {tuple(c) for c in np.unique(recon_arr, axis=0)}
        palette_colors_set = {c.rgb for c in res.colors}

        # The reconstructed colors must be a subset of the palette colors
        assert unique_recon_colors.issubset(palette_colors_set), (
            f"Found colors {unique_recon_colors - palette_colors_set} not present in palette!"
        )

    def test_zero_error_for_image_composed_entirely_of_palette_colors(self):
        """An image composed strictly of palette colors yields 0.0 MSE and identical pixels."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        arr = np.zeros((30, 30, 3), dtype=np.uint8)
        arr[:10, :] = colors[0]
        arr[10:20, :] = colors[1]
        arr[20:, :] = colors[2]

        pure_img = Image.fromarray(arr, mode="RGB")
        res = extract_palette(pure_img, n_colors=3, random_state=42)

        mse = compute_reconstruction_mse(pure_img, res)
        assert pytest.approx(mse, abs=1e-5) == 0.0

        reconstructed = reconstruct_image(pure_img, res)
        recon_arr = np.asarray(reconstructed, dtype=np.uint8)
        assert np.array_equal(arr, recon_arr)

    def test_reconstruction_deterministic_for_identical_input(
        self, sample_gradient_image: Image.Image
    ):
        """Reconstruction is strictly deterministic given identical input and palette."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)

        recon1 = reconstruct_image(sample_gradient_image, res)
        recon2 = reconstruct_image(sample_gradient_image, res)

        arr1 = np.asarray(recon1)
        arr2 = np.asarray(recon2)
        assert np.array_equal(arr1, arr2)

    def test_chunking_consistency_and_edge_cases(
        self, sample_gradient_image: Image.Image, tiny_image: Image.Image, rgba_image: Image.Image
    ):
        """Reconstruction and MSE produce identical results regardless of chunk size."""
        res = extract_palette(sample_gradient_image, n_colors=5, random_state=42)

        # Compare large chunk (default 100k) vs tiny chunk (e.g. 37 pixels)
        recon_default = reconstruct_image(sample_gradient_image, res, chunk_size=100_000)
        recon_chunked = reconstruct_image(sample_gradient_image, res, chunk_size=37)
        assert np.array_equal(np.asarray(recon_default), np.asarray(recon_chunked))

        mse_default = compute_reconstruction_mse(sample_gradient_image, res, chunk_size=100_000)
        mse_chunked = compute_reconstruction_mse(sample_gradient_image, res, chunk_size=37)
        assert pytest.approx(mse_default, rel=1e-6) == mse_chunked

        # Tiny image (4 pixels)
        tiny_res = extract_palette(tiny_image, n_colors=2, random_state=42)
        tiny_recon = reconstruct_image(tiny_image, tiny_res, chunk_size=2)
        assert tiny_recon.size == tiny_image.size

        # RGBA image auto-converts to RGB
        rgba_res = extract_palette(rgba_image, n_colors=3, random_state=42)
        rgba_recon = reconstruct_image(rgba_image, rgba_res)
        assert rgba_recon.mode == "RGB"
        assert rgba_recon.size == rgba_image.size

        # Invalid chunk size
        with pytest.raises(ValueError, match="chunk_size must be at least 1"):
            reconstruct_image(sample_gradient_image, res, chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be at least 1"):
            compute_reconstruction_mse(sample_gradient_image, res, chunk_size=0)
