# System Design v0.1: Nature-Inspired Brand Lab

## 1. System Goal

The goal of V0.1 is to build a system that extracts a representative color palette from an input image.

The system should reduce the image's many pixel colors to a small set of representative colors while preserving the dominant visual characteristics of the original image.

## 2. Scope

### Included in V0.1

- Accept a single image as input.
- Represent the image as RGB pixel data.
- Sample pixels to reduce computational workload.
- Cluster similar colors using K-Means.
- Extract representative colors from cluster centroids.
- Reconstruct the image using the extracted palette.
- Evaluate the reconstruction using visual inspection and Mean Squared Error (MSE).

### Outside the scope of V0.1

- Recommending additional brand colors.
- Generating palettes based on a desired mood or emotion.
- Interpreting text descriptions of a brand.
- Combining multiple reference images.
- Generating typography, moodboards, or complete brand guidelines.

These features belong to future versions of the project.

## 3. Proposed Architecture

The initial processing pipeline is:

1. Input image
2. Load image as RGB
3. Flatten pixel array
4. Sample pixels
5. Convert sampled pixels from RGB to LAB
6. Experiment with K-Means clustering
7. Evaluate candidate cluster counts
8. Fit the selected clustering configuration
9. Convert LAB centroids back to RGB
10. Generate the representative color palette
11. Reconstruct the image and evaluate the result

## 4. Image Representation

A digital image can be represented as a numerical array in which each pixel contains three color-channel values: red, green, and blue (RGB).

The test image used during initial exploration had an array shape of `(2736, 3648, 3)`.

This corresponds to 2,736 rows, 3,648 columns, and three color channels, representing 9,980,928 pixels.

The pixel values use the `uint8` data type, with channel values ranging from 0 to 255.

## 5. Flattening

The original image array has three dimensions:

`(height, width, channels)`

For color clustering, the image is reshaped into a two-dimensional array:

`(number_of_pixels, 3)`

Each row represents one pixel, and its three columns represent the RGB channels.

Flattening removes the image's spatial arrangement from the representation used for clustering, while preserving each pixel's color values.

## 6. Pixel Sampling

Processing every pixel can be computationally expensive, particularly during experimentation with different clustering configurations.

The initial implementation samples 50,000 pixels from the flattened image.

The purpose of sampling is to reduce computational workload while attempting to preserve the image's color distribution.

An initial representativeness check compares summary statistics, including the mean RGB values of the full image and the sample.

For the test image:

- Full-image mean RGB: approximately `[60.49, 59.69, 138.93]`
- Sample mean RGB: approximately `[60.13, 59.42, 139.16]`

The similarity between these values provides preliminary evidence that the sample captures the image's overall average color.

However, matching means does not prove that the sample preserves every important color, rare accent, or detail. Sampling must be evaluated on additional images.

## 7. Color-Space Selection

RGB is used to load and represent the image and to produce the final displayable palette.

However, numerical distance in RGB does not always correspond closely to perceived color difference.

For the initial clustering experiments, sampled RGB values are converted to CIE LAB.

LAB represents color using:

- **L\***: lightness
- **a\***: the green-to-red dimension
- **b\***: the blue-to-yellow dimension

LAB is used as a candidate representation for perceptually meaningful color grouping.

The choice does not imply that LAB perfectly models human perception. The suitability of the representation must be evaluated through experiments.

HSV was also explored conceptually as a representation that describes hue, saturation, and value. It may be useful for future color manipulation or brand-oriented rules, but it is not the primary representation selected for the current clustering experiment.

## 8. K-Means Clustering

K-Means groups sampled colors into a specified number of clusters.

Each cluster has a centroid, which represents the center of the colors assigned to that cluster in the chosen numerical space.

The initial experiments compared multiple values of K using RGB data. A subsequent experiment used LAB data with five clusters.

The purpose is to determine whether the resulting groups produce a useful, compact representation of the source image.

K-Means does not understand colors semantically. It groups numerical points according to distances in the representation supplied to it.

## 9. Selecting the Number of Clusters

The number of clusters, K, determines how many representative colors the palette contains.

A smaller K produces a simpler palette but may discard meaningful color variation. A larger K can preserve more variation but may produce redundant colors.

The elbow method is used to compare clustering inertia across candidate values of K.

Inertia measures the sum of squared distances between sampled points and their assigned cluster centroids. It generally decreases as K increases, so the smallest inertia alone is not sufficient to select K.

The goal is to identify a practical balance between compact representation and additional detail.

The initial experiments explored K values from 2 to 8. Five clusters were also examined visually and used for reconstruction experiments.

Five clusters are an experimental starting point, not yet a proven optimum for all images.

## 10. Palette Generation

After selecting a clustering configuration, the system obtains the cluster centroids in LAB space.

These centroids are converted back to RGB so they can be displayed as a color palette.

The resulting palette is intended to represent the prominent colors of the source image. It is not yet a complete brand palette: it does not account for a desired emotional mood, accessibility requirements, or specific branding goals.

## 11. Evaluation

The initial evaluation uses two methods.

### Visual inspection

The original image is compared with a reconstructed version in which each pixel is represented by its assigned cluster centroid.

This makes it possible to observe which visual characteristics remain and which details are lost when the image is reduced to a small number of colors.

### Mean Squared Error (MSE)

MSE measures the average squared difference between the original image's RGB channel values and those of the reconstructed image.

The initial five-color RGB reconstruction produced an MSE of approximately 484.20, corresponding to an RMSE of approximately 22 on the 0–255 channel scale.

MSE provides a numerical measure of reconstruction error, but it does not directly measure whether a palette is attractive, perceptually representative, or useful for branding. It must be interpreted alongside visual inspection and further evaluation.

## 12. Initial Technology Choices

The current prototype uses:

- **Python** for implementation.
- **Pillow** for image loading.
- **NumPy** for numerical arrays and pixel manipulation.
- **scikit-image** for color-space conversion.
- **scikit-learn** for K-Means clustering.
- **Matplotlib** for palette and reconstruction visualization.
- **Jupyter Notebook** for exploration and experimentation.
- **VS Code** as the development environment.
- **Git and GitHub** for version control and project documentation.

The project is intended to use free and open-source tools within a zero-budget constraint.

## 13. Assumptions to Test

1. A random sample of 50,000 pixels will adequately represent the color distribution of the images we intend to support.
2. LAB will provide useful groupings for representative palette extraction.
3. K-Means will produce meaningful representative colors for different types of images.
4. A practical value of K can be selected using clustering metrics and visual inspection.
5. Reconstruction error will provide useful supporting evidence when evaluating palette extraction.

These are hypotheses, not established facts.

## 14. Current Limitations

- Initial experiments have focused on one image.
- Sampling representativeness has not been established across diverse images.
- The best color space and cluster count have not been established for all image types.
- Rare but visually important accent colors may be missed.
- K-Means does not preserve spatial relationships between pixels.
- MSE alone cannot establish perceptual or branding quality.
- Transparent images, grayscale images, illustrations, and photographs have not yet been systematically evaluated.

## 15. Next Steps

The next stage will focus on sourcing a small but diverse image dataset, recording image provenance and licensing information, and testing the pipeline across different image types.

The results will be used to revisit the current assumptions and refine the system design before building the integrated prototype.