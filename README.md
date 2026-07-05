# Astrophysical Cluster Analysis Pipeline

A generalized machine learning pipeline for identifying open star cluster members and mapping stellar populations using raw observational data from the Gaia DR3 archive. 

This project reproduces and scales the membership analysis methodology from recent observational astrophysics literature (Chakraborty et al., 2025).

## Architecture

*   **Data Pipeline (`data_pipeline.py`):** Uses `astroquery` and ADQL to fetch astrometric and photometric data from the Gaia DR3 catalog, applying strict quality filters.
*   **Unsupervised Learning (`gmm_model.py`):** Applies a 5-dimensional Gaussian Mixture Model (PMRA, PMDEC, parallax, RA, Dec) to the cluster's core radius to isolate high-probability members.
*   **Supervised Learning (`rf_model.py`):** Trains a Random Forest Classifier on the GMM output to predict cluster membership across a wider tidal radius.
*   **Visualization (`visualizer.py`):** Generates publication-grade Vector Point Diagrams (kinematics) and Color-Magnitude Diagrams (photometry).

## Conditions for Successful Execution

To run this pipeline without breaking its mathematical and physical logic, the following parameters and conditions must be met:

*   **Distant and Compact Targets:** The target cluster must be relatively distant (like NGC 7419 or Messier 36) so it appears as a tight, concentrated point in the sky[cite: 2, 3]. This ensures its spatial spread is incredibly low, allowing the 5-dimensional GMM to correctly identify the cluster[cite: 3].
*   **Dynamic Parallax Queries:** The ADQL data extraction must use dynamic, cluster-specific minimum and maximum parallax bounds rather than hardcoded limits[cite: 3].
*   **Unrestricted Plotting Axes:** The Vector Point Diagram must use dynamic axis scaling so that clusters with high proper motion are not artificially cropped.
*   **Strict Quality Filters:** The raw data must be filtered to remove null values for proper motion and magnitudes, and must enforce a strict Renormalized Unit Weight Error (RUWE) score to drop blended objects[cite: 3].
*   **Moderate Target Density:** The target cannot be an overwhelmingly dense cluster (like a globular cluster), as intense starlight blending forces the RUWE filter to discard the dataset[cite: 2, 3].
*   **Observable Magnitudes:** The cluster's population must be bright enough to be captured by Gaia's G-band faintness limit[cite: 2, 3].

## Exact Physical and Algorithmic Limits

If a target cluster or the raw data violates these boundaries, the machine learning logic will fail:

*   **The Faintness Boundary:** The pipeline relies on the Gaia telescope's G-band, which has a physical faintness limit of approximately 20 magnitudes[cite: 2, 3].
*   **The Mass Threshold:** For a distant, obscured cluster (e.g., at ~3.6 kpc with an extinction of 5.2 mag), the 20-magnitude limit physically restricts the pipeline to detecting stars with a mass greater than roughly 1.2 solar masses[cite: 2, 3].
*   **The Resolution Limit:** The pipeline strictly requires a RUWE score of < 1.4 to ensure it evaluates clean, unblended stars[cite: 3].
*   **The Core Radius Limit:** The GMM requires a tight central radius to generate its training set, which is constrained to a 5-arcminute core radius in this architecture[cite: 2, 3].
*   **The Probability Cutoffs:** For a star to be classified as a member, it must cross a mathematical probability limit of > 0.9 (90%) in both the unsupervised GMM and supervised Random Forest phases[cite: 2, 3].
*   **The IQR Outlier Bound:** The final data filter strictly limits accepted members to a parallax range defined by 1.5 times the Inter-Quartile Range (IQR)[cite: 3]. This rigid boundary can accidentally exclude true cluster members on the fringes[cite: 2, 3].

## Known Failure Triggers

The pipeline will experience algorithmic failures under the following historical misconfigurations:

*   **Hardcoded ADQL Parallax Bounds:** Using static limits (e.g., 0.1 to 0.8 mas) completely excludes proximate clusters from the raw data download[cite: 2, 3].
*   **Hardcoded Plot Axes:** Limiting the Vector Point Diagram to static bounds (e.g., [-15, 15] mas/yr) crops high-velocity clusters off the visual frame.
*   **The Spatial Variance Trap:** Calculating GMM variance across all 5 dimensions simultaneously (PMRA, PMDEC, parallax, RA, Dec) for highly proximate targets[cite: 2, 3].
*   **Proximity-Induced Spread:** Targeting clusters that are too close to Earth causes a massive spatial spread across the sky, leading to high RA and Dec variance[cite: 2].
*   **Kinematic Misclassification:** When targeting proximate clusters, the massive spatial spread causes the GMM math to misclassify the dense, stationary background space as the tighter target cluster[cite: 3].

## Usage

Set your target cluster coordinates and radius parameters in `main.py` and run:
`python main.py`


## The Configuration Hub (`main.py`)

This pipeline does not rely on hardcoded, single-use limits. Instead, it uses a central configuration hub called the `TARGETS` dictionary. To process a new cluster, you simply add its physical parameters to this dictionary.

```python
TARGETS = {
    "Target_Cluster_Name": {
        "ra": "HH:MM:SS",          # Right Ascension 
        "dec": "+DD:MM:SS",        # Declination
        "core_radius": 5.0,        # Pure kinematic center (arcminutes)
        "full_radius": 15.0,       # Extended tidal halo (arcminutes)
        "min_parallax": 0.1,       # Distance lower bound (mas)
        "max_parallax": 1.0,       # Distance upper bound (mas)
        "max_gmag": 18.0           # Astrometric quality floor (Apparent Magnitude)
    }
}