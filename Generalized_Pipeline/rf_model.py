import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def run_rf(full_df: pd.DataFrame, core_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains a Random Forest on GMM labels, predicts on the full dataset, 
    and filters outliers using the 1.5 * IQR parallax method.
    """
    features = ['pmra', 'pmdec', 'parallax', 'ra', 'dec']
    
    # 1. Prepare Training Data
    y_train = np.where(core_df['gmm_prob'] > 0.9, 1, 0)
    X_train = core_df[features].values
    
    # 2. Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 3. Predict on the Full Dataset
    X_full = full_df[features].values
    rf_probs = rf.predict_proba(X_full)
    
    full_df['rf_prob'] = rf_probs[:, 1]
    
    # 4. Filter for strict RF probability > 0.90
    rf_members = full_df[full_df['rf_prob'] > 0.90].copy()
    print(f"Initial RF Members (>90% prob): {len(rf_members)}")
    
    # 5. Parallax Outlier Removal (Anchored strictly to the GMM core)
    # We calculate the statistical bounds using ONLY the pure core data
    pure_core = core_df[core_df['gmm_prob'] > 0.9]
    Q1 = pure_core['parallax'].quantile(0.25)
    Q3 = pure_core['parallax'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # We apply that strict core distance bound to the wide Random Forest predictions
    final_members = rf_members[(rf_members['parallax'] >= lower_bound) & 
                               (rf_members['parallax'] <= upper_bound)].copy()

                               
    print(f"Final Members after Parallax IQR filtering: {len(final_members)}")
    
    return final_members