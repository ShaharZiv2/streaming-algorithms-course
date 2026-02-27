"""
Script to clean up duplicate entries in memory tracking file
"""
import pandas as pd

# Load memory tracking data
memory_file = 'datasets/estimations/memory_tracking.csv'
memory_df = pd.read_csv(memory_file)

print(f"Original number of rows: {len(memory_df)}")
print(f"Original unique configurations: {memory_df[['cms_width', 'ams_r', 'iteration']].drop_duplicates().shape[0]}")

# Remove duplicates, keeping the first occurrence
memory_df_clean = memory_df.drop_duplicates(subset=['cms_width', 'ams_r', 'iteration'], keep='first')

print(f"\nCleaned number of rows: {len(memory_df_clean)}")
print(f"Cleaned unique configurations: {memory_df_clean[['cms_width', 'ams_r', 'iteration']].drop_duplicates().shape[0]}")

# Sort by cms_width, ams_r, iteration
memory_df_clean = memory_df_clean.sort_values(['cms_width', 'ams_r', 'iteration']).reset_index(drop=True)

# Save back to file
memory_df_clean.to_csv(memory_file, index=False)

print(f"\nSaved cleaned data to {memory_file}")
print("\nConfigurations in file:")
config_summary = memory_df_clean.groupby(['cms_width', 'ams_r']).size().reset_index(name='count')
print(config_summary)
