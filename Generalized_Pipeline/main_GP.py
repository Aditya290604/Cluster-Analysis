import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# IMPORT THE BACKEND FEDERATED QUERY TOOL
from auto_config import generate_cluster_config

def fetch_gaia_data(config):
    radius = config["full_radius"]
    max_gmag = config["max_gmag"]
    min_plx = config["min_parallax"]
    max_plx = config["max_parallax"]
    
    print(f"\n--- STEP 1: Data Retrieval ({radius}' radius) ---")
    
    coord = SkyCoord(f"{config['ra']} {config['dec']}", unit=(u.hourangle, u.deg))
    ra_deg = coord.ra.deg
    dec_deg = coord.dec.deg

    # BUG FIXED: Added 'parallax BETWEEN {min_plx} AND {max_plx}' to the ADQL query
    query = f"""
    SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius/60.0}))
    AND parallax BETWEEN {min_plx} AND {max_plx}
    AND pmra IS NOT NULL AND pmdec IS NOT NULL
    AND bp_rp IS NOT NULL AND phot_g_mean_mag < {max_gmag}
    AND ruwe < 1.4
    """
    
    job = Gaia.launch_job(query)
    df = job.get_results().to_pandas()
    
    # Calculate distance from center in arcminutes for the GMM core filter
    star_coords = SkyCoord(ra=df['ra'].values*u.deg, dec=df['dec'].values*u.deg)
    df['dist_arcmin'] = coord.separation(star_coords).arcminute
    
    print(f"Total sources retrieved: {len(df)}")
    return df

def run_gmm(df, core_radius):
    print(f"\n--- STEP 2: Gaussian Mixture Model ({core_radius}' core) ---")
    
    # Isolate stars within the 5.0 arcminute physical core
    core_df = df[df['dist_arcmin'] <= core_radius].copy()
    print(f"Sources isolated within {core_radius}-arcmin core: {len(core_df)}")
    
    # 5-Dimensional Kinematic/Spatial Analysis
    features = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    X_core = StandardScaler().fit_transform(core_df[features])
    
    gmm = GaussianMixture(n_components=2, random_state=42)
    core_df['gmm_label'] = gmm.fit_predict(X_core)
    probs = gmm.predict_proba(X_core)
    
    # Identify which component is the cluster (typically tighter PM dispersion)
    comp_0_std = core_df[core_df['gmm_label'] == 0][['pmra', 'pmdec']].std().sum()
    comp_1_std = core_df[core_df['gmm_label'] == 1][['pmra', 'pmdec']].std().sum()
    cluster_label = 0 if comp_0_std < comp_1_std else 1
    
    core_df['gmm_prob'] = probs[:, cluster_label]
    
    high_prob_gmm = len(core_df[core_df['gmm_prob'] > 0.9])
    print(f"High-probability GMM members (> 0.9) found: {high_prob_gmm}")
    
    return core_df, features

def run_rf_and_filter(df, core_df, features):
    print("\n--- STEP 3: Random Forest Classifier ---")
    
    # Train RF on the GMM core findings
    X_train = StandardScaler().fit_transform(core_df[features])
    y_train = (core_df['gmm_prob'] > 0.5).astype(int)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict across the entire extended radius (the halo)
    scaler_full = StandardScaler().fit(core_df[features]) 
    X_full = scaler_full.transform(df[features])
    df['rf_prob'] = rf.predict_proba(X_full)[:, 1]
    
    initial_members = df[df['rf_prob'] > 0.9]
    print(f"Initial RF Members (>90% prob): {len(initial_members)}")
    
    # --- Anchored Parallax IQR Filter ---
    gmm_pure = core_df[core_df['gmm_prob'] > 0.9]
    q1 = gmm_pure['parallax'].quantile(0.25)
    q3 = gmm_pure['parallax'].quantile(0.75)
    iqr = q3 - q1
    min_plx = q1 - 1.5 * iqr
    max_plx = q3 + 1.5 * iqr
    
    final_members = initial_members[(initial_members['parallax'] >= min_plx) & 
                                    (initial_members['parallax'] <= max_plx)]
    
    print(f"Final Members after Parallax IQR filtering: {len(final_members)}")
    return final_members

def plot_results(df, final_members):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Vector Point Diagram (PMRA vs PMDEC)
    ax1.scatter(df['pmra'], df['pmdec'], s=5, c='silver', alpha=0.5, label='All Sources')
    ax1.scatter(final_members['pmra'], final_members['pmdec'], s=15, c='green', label='Predicted Members')
    ax1.set_xlabel('PMRA (mas/yr)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PMDEC (mas/yr)', fontsize=12, fontweight='bold')
    ax1.set_title('Vector Point Diagram (Proper Motion)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Color-Magnitude Diagram (CMD)
    ax2.scatter(df['bp_rp'], df['phot_g_mean_mag'], s=5, c='silver', alpha=0.5, label='All Sources')
    ax2.scatter(final_members['bp_rp'], final_members['phot_g_mean_mag'], s=15, c='green', label='Predicted Members')
    ax2.set_xlabel('BP - RP (Color)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Apparent Magnitude ($m_G$)', fontsize=12, fontweight='bold')
    ax2.set_title('Color-Magnitude Diagram (CMD)', fontsize=14)
    ax2.invert_yaxis()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("==================================================")
    print("   DYNAMIC STELLAR CLUSTER PIPELINE INITIALIZED   ")
    print("==================================================")
    
    target_name = input("Enter the target cluster name (e.g., 'NGC 869', 'Messier 36'): ")
    
    # 1. Ask Auto-Config to build the parameters dynamically
    cluster_config = generate_cluster_config(target_name)
    
    if cluster_config:
        # 2. Feed the dynamic configuration straight into the pipeline
        raw_df = fetch_gaia_data(cluster_config)
        
        # 3. Unsupervised Core Isolation
        core_dataset, model_features = run_gmm(raw_df, cluster_config['core_radius'])
        
        # 4. Supervised Halo Extraction & Filtering
        verified_members = run_rf_and_filter(raw_df, core_dataset, model_features)
        
        # 5. Output
        print(f"\n--- Final Verification ---")
        print(f"Pipeline complete for {target_name.replace(' ', '_')}. Total verified members: {len(verified_members)}")
        print("Launching data visualization...")
        
        plot_results(raw_df, verified_members)
    else:
        print("❌ Pipeline aborted due to missing configuration.")