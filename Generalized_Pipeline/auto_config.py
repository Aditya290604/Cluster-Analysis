import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import warnings

# Suppress messy warnings from missing Astroquery data
warnings.filterwarnings('ignore')

def generate_cluster_config(cluster_name: str):
    """
    Queries SIMBAD and falls back to VizieR to generate the TARGETS dictionary format.
    """
    print(f"📡 Querying SIMBAD database for '{cluster_name}'...")
    
    # Configure SIMBAD
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('plx_value', 'dim_majaxis')
    
    try:
        result_table = custom_simbad.query_object(cluster_name)
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return

    if result_table is None:
        print(f"❌ Error: '{cluster_name}' was not found in the SIMBAD database.")
        return

    # --- 1. COORDINATES (SIMBAD) ---
    ra_col = 'ra' if 'ra' in result_table.columns else 'RA'
    dec_col = 'dec' if 'dec' in result_table.columns else 'DEC'
    ra_val, dec_val = result_table[ra_col][0], result_table[dec_col][0]
    
    if isinstance(ra_val, str):
        coord = SkyCoord(f"{ra_val} {dec_val}", unit=(u.hourangle, u.deg))
    else:
        coord = SkyCoord(ra=ra_val, dec=dec_val, unit=u.deg)
        
    ra_fmt = coord.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True)
    dec_fmt = coord.dec.to_string(unit=u.deg, sep=':', precision=1, pad=True, alwayssign=True)

    # --- 2. EXTRACT SIMBAD DATA OR SET TO NaN ---
    plx = np.nan
    if 'plx_value' in result_table.columns and not np.ma.is_masked(result_table['plx_value'][0]):
        try:
            plx = float(result_table['plx_value'][0])
        except ValueError:
            pass

    dim = np.nan
    if 'dim_majaxis' in result_table.columns and not np.ma.is_masked(result_table['dim_majaxis'][0]):
        try:
            dim = float(result_table['dim_majaxis'][0])
        except ValueError:
            pass

    # --- 3. VIZIER FALLBACK (FEDERATED QUERY) ---
    if np.isnan(plx) or np.isnan(dim):
        print(f"⚠️ Missing parameters in SIMBAD. Attempting VizieR fallback catalogs...")
        v = Vizier(columns=['All'])
        v.ROW_LIMIT = 1
        
        try:
            # Search Cantat-Gaudin (Gaia) and Dias (Historical) open cluster catalogs
            viz_results = v.query_object(cluster_name, catalog=['J/A+A/640/A1/table1', 'B/ocl'])
            
            if viz_results:
                for table_name in viz_results.keys():
                    table = viz_results[table_name]
                    
                    # Rescue Parallax
                    if np.isnan(plx) and 'plx' in table.columns:
                        plx = float(table['plx'][0])
                        print("   -> 🟢 Successfully rescued Parallax from VizieR.")
                        
                    # Rescue Angular Size
                    if np.isnan(dim):
                        if 'r50' in table.columns: # Cantat-Gaudin lists radius in degrees
                            dim = float(table['r50'][0]) * 60.0 * 2.0 
                            print("   -> 🟢 Successfully rescued Size from VizieR (Cantat-Gaudin).")
                        elif 'Diam' in table.columns: # Dias lists diameter in arcmin
                            dim = float(table['Diam'][0])
                            print("   -> 🟢 Successfully rescued Size from VizieR (Dias).")
        except Exception as e:
            print(f"   -> ❌ VizieR fallback failed: {e}")

    # --- 4. FINALIZE PARALLAX BOUNDS ---
    if np.isnan(plx):
        min_par, max_par = 0.1, 1.5
        plx_comment = "# ⚠️ Parallax missing in all databases. Default bounds applied."
    else:
        min_par = max(0.1, round(plx - 0.7, 2))
        max_par = round(plx + 0.7, 2)
        plx_comment = f"# True Parallax: ~{round(plx, 2)} mas (Buffer applied)"

    # --- 5. FINALIZE SPATIAL BOUNDS ---
    if np.isnan(dim):
        full_radius = 15.0
        rad_comment = "# ⚠️ Angular size missing in all databases. Default sweet spot applied."
    else:
        calc_rad = (dim / 2.0) + 5.0
        full_radius = max(15.0, round(calc_rad, 1))
        rad_comment = f"# Database diameter ({dim:.1f}') + 5' halo buffer"

    # --- 6. GENERATE OUTPUT ---
    dict_key = cluster_name.replace(" ", "_")
    
    config_string = f"""
    "{dict_key}": {{
        "ra": "{ra_fmt}",
        "dec": "{dec_fmt}",
        "core_radius": 5.0,        # Standard GMM kinematic core
        "full_radius": {full_radius},       {rad_comment}
        "min_parallax": {min_par},       {plx_comment}
        "max_parallax": {max_par},
        "max_gmag": 18.0           # Astrometric quality floor
    }},"""
    
    print("\n✅ Configuration Generated Successfully:\n")
    print(config_string)

if __name__ == "__main__":
    print("--- Open Cluster Configuration Generator ---")
    target = input("Enter the name of the star cluster (e.g., 'NGC 663', 'Messier 36'): ")
    generate_cluster_config(target)