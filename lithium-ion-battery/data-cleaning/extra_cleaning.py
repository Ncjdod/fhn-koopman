import pandas as pd

INPUT_PATH = r"c:\Users\kaank\Desktop\Python Files\cleaned_cycle_data.csv"
OUTPUT_PATH = r"c:\Users\kaank\Desktop\Python Files\cleaned_cycle_data_filtered.csv"

df = pd.read_csv(INPUT_PATH)

mask = (
    ((df["Electrode Type"] == "Calendered") & (df["Specific Charge Capacity (mAh/g)"] < 450)) |
    ((df["Electrode Type"] == "Uncalendered") & (df["Specific Charge Capacity (mAh/g)"] < 300))
)

clean_df = df[mask].copy()
clean_df.to_csv(OUTPUT_PATH, index=False)

print(f"Filtered rows: {len(df) - len(clean_df)}")
print(f"Saved cleaned data to: {OUTPUT_PATH}")