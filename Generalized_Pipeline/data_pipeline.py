import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

def fetch_and_clean_data(ra_str: str, dec_str: str, radius_arcmin: float, min_parallax: float, max_parallax: float) -> pd.DataFrame:
    """
    Fetches and filters Gaia DR3 data using ADQL based on dynamic cluster constraints.
    """
    # Convert string coordinates to degrees
    coord = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))
    radius_deg = radius_arcmin / 60.0
    
    # ADQL query enforcing dynamic constraints
    query = f"""
    SELECT source_id, ra, dec, parallax, pmra, pmdec, 
           phot_g_mean_mag as Gmag, phot_bp_mean_mag as BPmag, phot_rp_mean_mag as RPmag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius_deg}))
      AND parallax > {min_parallax} AND parallax < {max_parallax}
      AND ruwe < 1.4
      AND pmra IS NOT NULL 
      AND pmdec IS NOT NULL
      AND phot_g_mean_mag IS NOT NULL 
      AND phot_bp_mean_mag IS NOT NULL 
      AND phot_rp_mean_mag IS NOT NULL
    """
    
    # Launch job asynchronously and return as a Pandas DataFrame
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    df = results.to_pandas()
    
    return df