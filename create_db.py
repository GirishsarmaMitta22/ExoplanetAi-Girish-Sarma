import pandas as pd
import sqlite3

def create_database():
    print("Loading CSV...")
    df = pd.read_csv('exoplanet_data_processed.csv')
    
    # Handle duplicate columns (case-insensitive for SQLite)
    df.columns = df.columns.astype(str)
    
    # Identify case-insensitive duplicates
    lower_cols = pd.Series(df.columns).str.lower()
    duplicated_mask = lower_cols.duplicated(keep='first')
    
    if duplicated_mask.any():
        print(f"Dropping {duplicated_mask.sum()} case-insensitive duplicate columns.")
        df = df.loc[:, ~duplicated_mask.values]
    
    print(f"Columns after deduplication: {len(df.columns)}")
    
    print("Connecting to SQLite...")
    conn = sqlite3.connect('exoplanets.db')
    
    print("Writing to database...")
    df.to_sql('exoplanets', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Database created successfully: exoplanets.db")

if __name__ == "__main__":
    create_database()
