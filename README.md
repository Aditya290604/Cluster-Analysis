# Astrophysical Cluster Analysis Pipeline

A generalized machine learning pipeline for identifying open star cluster members and mapping stellar populations using raw observational data from the Gaia DR3 archive. 

This project reproduces and scales the membership analysis methodology from recent observational astrophysics literature (Chakraborty et al., 2025).

## Architecture
1. **Data Pipeline (`data_pipeline.py`):** Uses `astroquery` and ADQL to fetch astrometric and photometric data from the Gaia DR3 catalog, applying strict quality filters (e.g., RUWE < 1.4).
2. **Unsupervised Learning (`gmm_model.py`):** Applies a 5-dimensional Gaussian Mixture Model (PMRA, PMDEC, parallax, RA, Dec) to the cluster's core radius to isolate high-probability members.
3. **Supervised Learning (`rf_model.py`):** Trains a Random Forest Classifier on the GMM output to predict cluster membership across a wider tidal radius, followed by a 1.5 * IQR physical parallax filter.
4. **Visualization (`visualizer.py`):** Generates publication-grade Vector Point Diagrams (kinematics) and Color-Magnitude Diagrams (photometry).

## Known Scientific Limitations

This pipeline is highly optimized for **distant, compact, and relatively young open clusters**. It will experience algorithmic or physical failures under the following parameters:

### Algorithmic Limits
*   **The Spatial Variance Trap (Proximity):** The GMM evaluates cluster "tightness" across all 5 dimensions (including RA and Dec)[cite: 3]. If applied to highly proximate clusters (e.g., the Pleiades), the massive spatial spread across the sky will confuse the GMM, causing it to misclassify stationary background space as the cluster. 
*   **Atypical Cluster Shapes:** The final pipeline step utilizes a strict 1.5 * IQR statistical cutoff to remove parallax outliers[cite: 3]. For clusters experiencing severe tidal disruption or highly elongated shapes, this rigid bound may clip valid fringe members (similar to how strict probability limits initially missed known supergiant members in NGC 7419)[cite: 2, 3].

### Observational (Telescope) Limits
*   **Extreme Crowding:** To avoid blended starlight, the pipeline drops any source with a Renormalized Unit Weight Error (RUWE) > 1.4[cite: 3]. If pointed at an extremely dense target, such as a Globular Cluster, the high blending will cause the pipeline to discard the majority of the dataset[cite: 2].
*   **The Faintness Limit:** The pipeline relies on Gaia G-band photometry, which has a physical faintness limit of roughly 20 magnitudes[cite: 2, 3]. For distant clusters (e.g., ~3.6 kpc), stars with masses lower than roughly 1.2 $M_{\odot}$ will not be captured by the telescope, meaning the pipeline cannot map the low-mass population[cite: 2, 3].

## Usage
Set your target cluster coordinates and radius parameters in `main.py` and run:
`python main.py`