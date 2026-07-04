import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from astropy.coordinates import SkyCoord
import astropy.units as u

def run_gmm(df: pd.DataFrame, ra_center: str, dec_center: str, core_radius_arcmin: float) -> tuple:
    """
    Isolates the core radius and trains a GMM to find high-probability members.
    """
    # 1. Filter for the 5-arcminute core using Astropy
    center_coord = SkyCoord(f"{ra_center} {dec_center}", unit=(u.hourangle, u.deg))
    star_coords = SkyCoord(df['ra'].values*u.deg, df['dec'].values*u.deg)
    
    separations = center_coord.separation(star_coords).arcminute
    core_df = df[separations <= core_radius_arcmin].copy()
    
    print(f"Sources isolated within {core_radius_arcmin}-arcmin core: {len(core_df)}")
    
    # 2. Prepare features exactly as stated in the paper
    features = ['pmra', 'pmdec', 'parallax', 'ra', 'dec']
    X = core_df[features].values
    
    # Scale the features so large coordinates don't overpower tiny parallaxes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train the Gaussian Mixture Model (2 components: Cluster vs. Field)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_scaled)
    
    # 4. Identify the Star Cluster Component
    # The true cluster will have a tighter spread (lower variance) in proper motion/parallax
    variances = [np.trace(cov) for cov in gmm.covariances_]
    cluster_component = np.argmin(variances)
    
    # 5. Extract probabilities and assign them back to the DataFrame
    probs = gmm.predict_proba(X_scaled)
    core_df['gmm_prob'] = probs[:, cluster_component]
    
    # 6. Filter high-probability members (> 0.9) as specified in the paper
    high_prob_members = core_df[core_df['gmm_prob'] > 0.9].copy()
    
    print(f"High-probability GMM members (> 0.9) found: {len(high_prob_members)}")
    
    return high_prob_members, core_df