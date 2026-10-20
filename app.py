"""Nature-Inspired Brand Lab - Interactive Streamlit Application.

Allows designers and creators to upload inspiration images (JPEG, PNG, WebP)
or choose curated nature samples, extracting a representative color palette
using perceptual CIE LAB K-Means clustering.
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

# Add src to path so brand_lab can be imported smoothly
import sys
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from brand_lab.palette import (
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_SAMPLE_SIZE,
    compute_reconstruction_mse,
    export_palette_csv,
    export_palette_json,
    extract_palette,
    load_and_validate_image,
    reconstruct_image,
)

# Page configuration
st.set_page_config(
    page_title="Nature-Inspired Brand Lab",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Nature-Inspired CSS
st.markdown(
    """
    <style>
    /* Nature-inspired theme accent */
    :root {
        --primary-forest: #2D5A4C;
        --secondary-sage: #8EA89D;
        --bg-warm: #FBFBF9;
    }
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1F3B32;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4A5B54;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .swatch-card {
        border-radius: 12px;
        padding: 12px;
        background: #FFFFFF;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #E6EAE7;
        margin-bottom: 16px;
        transition: transform 0.15s ease-in-out;
    }
    .swatch-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .color-block {
        height: 80px;
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.08);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.82rem;
        background-color: #EDF3F0;
        color: #24483C;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


CURATED_SAMPLES: dict[str, Path] = {
    "Image 1 (Jellyfish & Blue Water)": BASE_DIR / "Images" / "image_one.jpg",
    "Image 2 (Translucent Jellyfish on Dark Background)": BASE_DIR / "Images" / "image_two.jpg",
    "Image 3 (Blue-and-Yellow Tropical Fish)": BASE_DIR / "Images" / "image_three.jpg",
}


def get_contrast_text_color(rgb: tuple[int, int, int]) -> str:
    """Calculates relative luminance to pick white or dark text for contrast."""
    r, g, b = [x / 255.0 for x in rgb]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if luminance < 0.45 else "#1A1A1A"


def main():
    st.markdown("<h1 class='main-header'>🌿 Nature-Inspired Brand Lab</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Extract representative brand color palettes from nature photography and visual inspiration using perceptual CIE LAB clustering.</p>",
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Palette Settings")
        n_colors = st.slider(
            "Number of Colors (K)",
            min_value=2,
            max_value=10,
            value=DEFAULT_CLUSTER_COUNT,
            help="Default is 5 representative colors, as validated in system design.",
        )

        sample_size = st.select_slider(
            "Pixel Sample Size",
            options=[10_000, 25_000, 50_000, 100_000],
            value=DEFAULT_SAMPLE_SIZE,
            help="50,000 pixels provides an optimal balance between responsiveness and representativeness.",
        )

        st.markdown("---")
        st.subheader("🖼️ Curated Nature Samples")
        st.write("Test immediately with project reference images:")

        # Initialize uploader key in session state for clean widget resets
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0

        sample_images = CURATED_SAMPLES

        for label, img_path in sample_images.items():
            if img_path.exists():
                if st.button(label, use_container_width=True):
                    st.session_state["active_sample"] = str(img_path)
                    st.session_state["uploader_key"] += 1
                    st.rerun()

        if st.session_state.get("active_sample"):
            active_p = Path(st.session_state["active_sample"])
            st.caption(f"Active preset: **{active_p.name}**")
            if st.button("✕ Clear Sample Image", use_container_width=True):
                st.session_state["active_sample"] = None
                st.rerun()

    # Image Input Section
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploader_k = f"image_uploader_{st.session_state.get('uploader_key', 0)}"
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG, WebP - max 25MB)",
            type=["jpg", "jpeg", "png", "webp"],
            key=uploader_k,
            help="Images are processed locally in your session.",
        )

    # Determine image source
    image_source = None
    source_name = "image"

    if uploaded_file is not None:
        image_source = uploaded_file
        source_name = uploaded_file.name
        # Clear sample if user uploaded a file
        if st.session_state.get("active_sample"):
            st.session_state["active_sample"] = None
    elif st.session_state.get("active_sample"):
        sample_path = Path(st.session_state["active_sample"])
        if sample_path.exists():
            image_source = sample_path
            source_name = sample_path.name

    if image_source is None:
        with col_info:
            st.info("👈 Upload an image or select one of the curated nature samples to extract a brand palette.")
        return

    # Process Image
    try:
        with st.spinner("Decoding and validating image..."):
            pil_image = load_and_validate_image(image_source)

        with st.spinner(f"Clustering {sample_size:,} pixels in CIE LAB space..."):
            palette_result = extract_palette(
                pil_image,
                n_colors=n_colors,
                sample_size=sample_size,
                random_state=42,
            )

    except ValueError as val_err:
        st.error(f"⚠️ Validation Error: {val_err}")
        return
    except Exception as exc:
        st.error(f"❌ Processing Error: Unexpected failure during extraction: {exc}")
        return

    # Results Layout
    st.markdown("---")
    res_col_left, res_col_right = st.columns([1, 1], gap="large")

    with res_col_left:
        st.subheader("Source Image & Reconstruction")

        with st.spinner("Generating palette-reconstructed preview..."):
            reconstructed_image = reconstruct_image(pil_image, palette_result)

        col_orig, col_recon = st.columns(2)
        with col_orig:
            st.image(
                pil_image,
                caption=f"Original ({pil_image.width} × {pil_image.height} px)",
                use_container_width=True,
            )
        with col_recon:
            st.image(
                reconstructed_image,
                caption=f"{palette_result.n_colors_returned}-Color Reconstruction ({reconstructed_image.width} × {reconstructed_image.height} px)",
                use_container_width=True,
            )

        st.caption(
            "💡 **Reconstruction Preview:** Each pixel is mapped to its nearest extracted palette color in sRGB space. "
            "A lower reconstruction error means a closer numerical match to the source, "
            "not necessarily a better brand palette."
        )

        if palette_result.clamped:
            st.warning(f"ℹ️ {palette_result.clamped_reason}")

        # Quick Copy All Palette HEX
        all_hex = palette_result.hex_string(" ")
        st.markdown("**Complete Palette (HEX):**")
        st.code(all_hex, language="text")

        # Export Controls
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            json_data = export_palette_json(palette_result, source_name=source_name)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"palette_{Path(source_name).stem}.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp_col2:
            csv_data = export_palette_csv(palette_result)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"palette_{Path(source_name).stem}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with res_col_right:
        st.subheader(f"Extracted Brand Palette ({palette_result.n_colors_returned} Colors)")
        st.caption(
            "Colors sorted by visual dominance in sampled pixels (highest cluster population first). "
            "Each percentage represents the share of sampled image pixels assigned to that color cluster."
        )

        # Horizontal overview bar
        swatch_cols = st.columns(palette_result.n_colors_returned)
        for idx, color_item in enumerate(palette_result.colors):
            with swatch_cols[idx]:
                text_col = get_contrast_text_color(color_item.rgb)
                st.markdown(
                    f"""
                    <div style="background-color: {color_item.hex}; height: 60px; border-radius: 8px;
                                display: flex; align-items: center; justify-content: center;
                                color: {text_col}; font-weight: 700; font-size: 0.85rem;
                                border: 1px solid rgba(0,0,0,0.1);">
                        #{idx+1}
                    </div>
                    <div style="text-align: center; font-size: 0.75rem; margin-top: 4px; color: #555;">
                        {color_item.percentage:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed Swatch Cards
        for idx, color_item in enumerate(palette_result.colors, start=1):
            text_color = get_contrast_text_color(color_item.rgb)
            with st.container():
                st.markdown(
                    f"""
                    <div class="swatch-card">
                        <div class="color-block" style="background-color: {color_item.hex}; color: {text_color};">
                            {color_item.hex}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                card_c1, card_c2, card_c3 = st.columns([1.5, 2, 1.5])
                with card_c1:
                    st.code(color_item.hex, language="text")
                with card_c2:
                    st.markdown(f"**RGB:** `{color_item.rgb[0]}, {color_item.rgb[1]}, {color_item.rgb[2]}`")
                    st.markdown(f"**LAB:** `{color_item.lab[0]:.1f}, {color_item.lab[1]:.1f}, {color_item.lab[2]:.1f}`")
                with card_c3:
                    st.markdown(f"<span class='metric-badge'>{color_item.percentage:.2f}% coverage</span>", unsafe_allow_html=True)
                    st.caption(f"{color_item.count:,} sampled px")

    # Preservation of Engineering Work & Diagnostics
    with st.expander("🔬 Color Science & Diagnostic Evaluation (Experiments 001 & 002)"):
        st.markdown(
            r"""
            **Why CIE LAB?**  
            Standard Euclidean distance in sRGB does not match human perception of color differences.
            This engine converts pixel data to the **CIE L\*a\*b\*** color space before running K-Means,
            grouping colors perceptually (lightness, green-red, and blue-yellow axes).
            """
        )
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.write("**Pixel Statistics:**")
            st.write(f"- Total Image Pixels: `{palette_result.original_shape[0] * palette_result.original_shape[1]:,}`")
            st.write(f"- Sampled Pixels: `{palette_result.total_samples:,}`")
            st.write(f"- Full Image Mean RGB: `{tuple(round(x, 1) for x in palette_result.full_mean_rgb)}`")
            st.write(f"- Sample Mean RGB: `{tuple(round(x, 1) for x in palette_result.sample_mean_rgb)}`")
        with diag_col2:
            st.write("**Reconstruction Diagnostic (MSE):**")
            with st.spinner("Computing full image reconstruction MSE..."):
                mse = compute_reconstruction_mse(pil_image, palette_result)
                rmse = mse ** 0.5
            st.write(f"- Reconstruction MSE: `{mse:.2f}`")
            st.write(f"- RGB Vector RMSE: `{rmse:.2f}`")
            st.caption(
                f"Diagnostic measure comparing pixel distance between original and {palette_result.n_colors_returned}-color reconstructed image in sRGB space. "
                "Reconstruction MSE is the mean squared RGB-vector error per pixel; "
                "RGB Vector RMSE is the square root of that value (this is not conventional per-channel RMSE). "
                "Note: Lower reconstruction error means a closer numerical match to the source, "
                "not necessarily a better brand palette."
            )


if __name__ == "__main__":
    main()
