import matplotlib.pyplot as plt
import pandas as pd

def plot_results(full_df: pd.DataFrame, members_df: pd.DataFrame):
    """
    Generates the Proper Motion Vector Point Diagram and the Color-Magnitude Diagram.
    """
    # Calculate BP-RP color for the CMD
    full_df['bp_rp'] = full_df['BPmag'] - full_df['RPmag']
    members_df['bp_rp'] = members_df['BPmag'] - members_df['RPmag']

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ---------------------------------------------------------
    # Plot 1: Proper Motion Vector Point Diagram (PMRA vs PMDEC)
    # ---------------------------------------------------------
    ax1.scatter(full_df['pmra'], full_df['pmdec'], s=5, color='grey', alpha=0.3, label='All Sources in 12\'')
    ax1.scatter(members_df['pmra'], members_df['pmdec'], s=15, color='green', alpha=0.9, label='Predicted Members')
    
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-15, 15)
    ax1.set_xlabel('PMRA (mas/yr)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PMDEC (mas/yr)', fontsize=12, fontweight='bold')
    ax1.set_title('Vector Point Diagram (Proper Motion)', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---------------------------------------------------------
    # Plot 2: Color-Magnitude Diagram (CMD)
    # ---------------------------------------------------------
    ax2.scatter(full_df['bp_rp'], full_df['Gmag'], s=5, color='grey', alpha=0.3, label='All Sources in 12\'')
    ax2.scatter(members_df['bp_rp'], members_df['Gmag'], s=15, color='green', alpha=0.9, label='Predicted Members')
    
    # In astronomy, brighter magnitudes have lower numerical values, so we invert the Y-axis
    ax2.invert_yaxis()
    
    ax2.set_xlabel('BP - RP (Color)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Apparent Magnitude ($m_G$)', fontsize=12, fontweight='bold')
    ax2.set_title('Color-Magnitude Diagram (CMD)', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()