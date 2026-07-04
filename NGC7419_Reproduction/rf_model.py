import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def run_rf(full_df: pd.DataFrame, core_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains a Random Forest on GMM labels and predicts on the full 12-arcmin dataset.
    Filters outliers using the parallax IQR method.
    """
    features = ['pmra', 'pmdec', 'parallax', 'ra', 'dec']
    
    # 1. Prepare the Training Data (from the 5-arcmin core)
    # Assign label 1 to GMM prob > 0.9, else 0
    y_train = np.where(core_df['gmm_prob'] > 0.9, 1, 0)
    X_train = core_df[features].values
    
    # 2. Train the Random Forest Classifier
    # We use random_state=42 for reproducibility
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 3. Predict on the Full 12-arcmin Dataset
    X_full = full_df[features].values
    rf_probs = rf.predict_proba(X_full)
    
    # Probability of being in class 1 (Cluster Member)
    full_df['rf_prob'] = rf_probs[:, 1]
    
    # 4. Filter for strict RF probability > 0.90
    rf_members = full_df[full_df['rf_prob'] > 0.90].copy()
    print(f"Initial RF Members (>90% prob): {len(rf_members)}")
    
    # 5. Parallax Outlier Removal (1.5 * IQR method)
    # As explicitly required by the paper's methodology
    Q1 = rf_members['parallax'].quantile(0.25)
    Q3 = rf_members['parallax'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    final_members = rf_members[(rf_members['parallax'] >= lower_bound) & 
                               (rf_members['parallax'] <= upper_bound)].copy()
    
    print(f"Final Members after Parallax IQR filtering: {len(final_members)}")
    
    return final_members