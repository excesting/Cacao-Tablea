import pandas as pd
import psycopg2
import os

print("📦 Reading and cleaning CSV files...")

# 1. Combine all datasets in order
csv_files = ["tablea2022.csv", "tablea2023.csv", "tablea2024.csv", "tablea2025.csv"]
dfs = []
for f in csv_files:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f))
    else:
        print(f"⚠️ Warning: Could not find {f}")

if not dfs:
    print("❌ Error: No CSV files found. Make sure they are in the same folder as this script.")
    exit()

master_df = pd.concat(dfs, ignore_index=True)

# 2. Clean numeric columns (remove commas, handle blanks)
numeric_columns = ["Sales (pcs)", "Total Produced", "Net Usable Output"]
for col in numeric_columns:
    master_df[col] = master_df[col].astype(str).str.replace(',', '').astype(float)
    master_df[col] = master_df[col].ffill().fillna(0).astype(int)

# 3. Clean percentages
master_df["Defect Rate"] = master_df["Defect Rate"].astype(str).str.replace('%', '').astype(float) / 100.0
master_df["Defect Rate"] = master_df["Defect Rate"].ffill().fillna(0.0)

# 4. Fill missing events and add required columns
master_df["Event Name"] = master_df["Event Name"].fillna("None")
master_df["time_idx"] = range(len(master_df))
master_df["Branch"] = "Lipa" 

print(f"✅ Data cleaned! Total weeks of history ready to insert: {len(master_df)}")
print("💾 Connecting to Railway cloud database...")

# 5. Connect to your Railway PostgreSQL database
db_url = "postgresql://postgres:AMUBJZqNFccEXkfDGdEJDuyXlmUPWdUy@caboose.proxy.rlwy.net:55551/railway?sslmode=disable"

try:
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    # 6. Clear out any old test data to prevent duplicates
    try:
        cursor.execute("DELETE FROM production_history")
    except psycopg2.errors.UndefinedTable:
        print("❌ Error: 'production_history' table not found.")
        print("Please ensure your Railway app has successfully booted up at least once to create the tables.")
        conn.close()
        exit()

    # 7. Insert the cleaned data row by row
    inserted_count = 0
    for index, row in master_df.iterrows():
        # Create a clean week ID for historical records (e.g., Hist-W0, Hist-W1)
        historical_week_id = f"Hist-W{int(row['time_idx'])}"
        
        # NOTE: PostgreSQL uses %s instead of ? for variable injection
        cursor.execute('''
            INSERT INTO production_history 
            (time_idx, week_id, branch, month, event_name, sales_pcs, total_produced, net_usable_output, defect_rate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            int(row['time_idx']),
            historical_week_id,
            str(row['Branch']),
            str(row['Month']),
            str(row['Event Name']),
            int(row['Sales (pcs)']),
            int(row['Total Produced']),
            int(row['Net Usable Output']),
            float(row['Defect Rate'])
        ))
        inserted_count += 1

    # Commit and close
    conn.commit()
    conn.close()

    print(f"🎉 Success! {inserted_count} historical records have been securely saved to your cloud database.")

except Exception as e:
    print(f"❌ Database connection or insertion error: {e}")