from data_pipeline import fetch_and_clean_data
from gmm_model import run_gmm
from rf_model import run_rf
from visualizer import plot_results

# Hardcoded cluster parameters from the paper
RA_STR = "22:54:18.96"
DEC_STR = "+60:48:50.4"
RADIUS_RF_ARCMIN = 12
RADIUS_GMM_ARCMIN = 5

def main():
    print("Initializing Project: NGC 7419 Reproduction")
    print(f"Target Coordinates: RA {RA_STR}, Dec {DEC_STR}")
    
    # Step 1: Data Fetching
    print("\n--- STEP 1: Data Retrieval ---")
    df = fetch_and_clean_data(RA_STR, DEC_STR, RADIUS_RF_ARCMIN)
    
    print("\n--- Data Pipeline Verification ---")
    print(f"Total sources retrieved in {RADIUS_RF_ARCMIN}': {len(df)}")
    
    # Step 2: Unsupervised Learning (GMM)
    print("\n--- STEP 2: Gaussian Mixture Model (Core Region) ---")
    gmm_members, core_df = run_gmm(df, RA_STR, DEC_STR, RADIUS_GMM_ARCMIN)
    
    # Step 3: Supervised Learning (Random Forest)
    print("\n--- STEP 3: Random Forest Classifier (Full Region) ---")
    final_cluster_members = run_rf(df, core_df)
    
    print("\n--- Final Verification ---")
    print(f"Pipeline complete. Total verified members: {len(final_cluster_members)}")
    print("Launching data visualization...")

    # Step 4: Visualization
    plot_results(df, final_cluster_members)

if __name__ == "__main__":
    main()