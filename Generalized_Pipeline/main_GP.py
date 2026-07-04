from data_pipeline import fetch_and_clean_data
from gmm_model import run_gmm
from rf_model import run_rf
from visualizer import plot_results

# ---------------------------------------------------------
# CLUSTER CONFIGURATION HUB
# Easily swap targets by changing the ACTIVE_CLUSTER variable
# ---------------------------------------------------------

TARGETS = {
    "NGC_7419": {
        "ra": "22:54:18.96",
        "dec": "+60:48:50.4",
        "core_radius": 5.0,
        "full_radius": 12.0,
        "min_parallax": 0.1,
        "max_parallax": 0.8
    },
    "Pleiades_M45": {
        "ra": "03:47:24",
        "dec": "+24:07:00",
        "core_radius": 60.0,  
        "full_radius": 120.0,
        "min_parallax": 2.0,  # The Pleiades sits at ~7.3 mas
        "max_parallax": 12.0
    },
    "Messier_36": {
        "ra": "05:36:12",
        "dec": "+34:08:24",
        "core_radius": 5.0,
        "full_radius": 15.0,
        "min_parallax": 0.1,
        "max_parallax": 2.5
    },
    "Messier_47": {
        "ra": "07:36:35",
        "dec": "-14:28:57",
        "core_radius": 10.0,
        "full_radius": 25.0,
        "min_parallax": 0.1,
        "max_parallax": 5.0
    }
}

# ---> SET YOUR TARGET CLUSTER HERE <---
ACTIVE_CLUSTER = "Messier_36"

def main():
    config = TARGETS[ACTIVE_CLUSTER]
    print(f"Initializing Generalized Pipeline for: {ACTIVE_CLUSTER}")
    print(f"Target Coordinates: RA {config['ra']}, Dec {config['dec']}")
    
    # Step 1: Data Fetching
    print(f"\n--- STEP 1: Data Retrieval ({config['full_radius']}' radius) ---")
    df = fetch_and_clean_data(
        config['ra'], 
        config['dec'], 
        config['full_radius'],
        config['min_parallax'],
        config['max_parallax']
    )
    print(f"Total sources retrieved: {len(df)}")
    
    # Step 2: Unsupervised Learning (GMM)
    print(f"\n--- STEP 2: Gaussian Mixture Model ({config['core_radius']}' core) ---")
    gmm_members, core_df = run_gmm(df, config['ra'], config['dec'], config['core_radius'])
    
    # Step 3: Supervised Learning (Random Forest)
    print("\n--- STEP 3: Random Forest Classifier ---")
    final_cluster_members = run_rf(df, core_df)
    
    print("\n--- Final Verification ---")
    print(f"Pipeline complete for {ACTIVE_CLUSTER}. Total verified members: {len(final_cluster_members)}")
    print("Launching data visualization...")

    # Step 4: Visualization
    plot_results(df, final_cluster_members)

if __name__ == "__main__":
    main()