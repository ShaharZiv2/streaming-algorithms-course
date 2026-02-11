"""
Script to calculate and add missing memory data for cms_width=2048, ams_r=16
"""
import pandas as pd
from schemas.args import Args
from utils.sketch_manager import SketchManager

# Configuration for the missing data
cms_width = 2048
ams_r = 16
window_size = 10

# Create SketchManager instances to calculate memory
training_args = Args(window_size=window_size, cms_width=cms_width, ams_r=ams_r)
testing_args = Args(window_size=window_size, cms_width=cms_width, ams_r=ams_r)

training_manager = SketchManager(training_args)
testing_manager = SketchManager(testing_args)

# Calculate total memory for training and testing
training_total_memory = training_manager.calculate_total_memory()
testing_total_memory = testing_manager.calculate_total_memory()

print(f"Configuration: CMS Width={cms_width}, AMS r={ams_r}")
print(f"Training total memory: {training_total_memory} bytes ({training_total_memory / (1024**2):.2f} MB)")
print(f"Testing total memory: {testing_total_memory} bytes ({testing_total_memory / (1024**2):.2f} MB)")

# Load existing memory tracking data
memory_file = 'datasets/estimations/memory_tracking.csv'
memory_df = pd.read_csv(memory_file)

# Create new rows for the missing configuration (25 iterations)
new_rows = []
for i in range(25):
    new_rows.append({
        'cms_width': cms_width,
        'ams_r': ams_r,
        'iteration': i,
        'window_size': window_size,
        'training_total_memory_bytes': training_total_memory,
        'testing_total_memory_bytes': testing_total_memory,
        'training_total_memory_mb': training_total_memory / (1024**2),
        'testing_total_memory_mb': testing_total_memory / (1024**2),
    })

# Add new rows to the dataframe
new_df = pd.DataFrame(new_rows)
updated_df = pd.concat([memory_df, new_df], ignore_index=True)

# Sort by cms_width, ams_r, iteration
updated_df = updated_df.sort_values(['cms_width', 'ams_r', 'iteration']).reset_index(drop=True)

# Save back to file
updated_df.to_csv(memory_file, index=False)

print(f"\nAdded {len(new_rows)} rows to {memory_file}")
print("\nUpdated configurations:")
print(updated_df[['cms_width', 'ams_r']].drop_duplicates().sort_values(['cms_width', 'ams_r']))
