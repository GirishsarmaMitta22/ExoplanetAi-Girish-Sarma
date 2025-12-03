import pandas as pd
import requests
import io

COLUMNS = (
    'pl_name,pl_rade,pl_masse,pl_orbper,pl_eqt,st_teff,st_lum,st_met,st_spectype,st_mass'
)
TAP_URL = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?'
ADQL_QUERY = f"select {COLUMNS} from ps where upper(soltype) like '%CONF%'"
QUERY_PARAMS = {
    'query': ADQL_QUERY,
    'format': 'csv'
}

def collect_exoplanet_data_tap(output_filename='exoplanet_data_raw.csv'):
    
    try:
        response = requests.get(TAP_URL, params=QUERY_PARAMS)
        response.raise_for_status()

        csv_content = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content), comment='#')
        
        df.to_csv(output_filename, index=False)
        
        return df

    except requests.exceptions.RequestException:
        return None
    
if __name__ == "__main__":
    exoplanet_df_tap = collect_exoplanet_data_tap()
    if exoplanet_df_tap is not None:
        pass
