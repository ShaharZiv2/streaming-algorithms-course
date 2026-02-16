"""
Generate memory tracking data for existing estimation files.
This script calculates memory usage for each configuration based on the parameters.
"""
import pandas as pd
import glob
import re
from schemas.args import Args
from utils.sketch_manager import SketchManager

def extract_config_from_filename(filename):
    """Extract cms_width, ams_r, and iteration from filename"""
    pattern = r'(?:trng|tstng)_(\d+)_(\d+)_(\d+)\.csv'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None

def calculate_memory_for_config(cms_width, ams_r, window_size=10):
    """Calculate memory usage for a configuration"""
    args = Args(
        window_size=window_size,
        cms_width=cms_width,
        ams_r=ams_r,
    )
    sketch_manager = SketchManager(args)
    return sketch_manager.calculate_total_memory()

def main():
    # Find all training estimation files
    training_files = glob.glob('datasets/estimations/trng_*.csv')

    memory_tracking = []
    configs_seen = set()

    for training_file in sorted(training_files):
        config = extract_config_from_filename(training_file)
        if not config:
            continue

        cms_width, ams_r, iteration = config

        # Check if testing file also exists
        testing_file = training_file.replace('trng_', 'tstng_')

        # Read the files to count windows (rows)
        try:
            training_data = pd.read_csv(training_file)
            testing_data = pd.read_csv(testing_file)
        except:
            continue

        num_training_windows = len(training_data)
        num_testing_windows = len(testing_data)

        # Calculate memory per window
        memory_per_window = calculate_memory_for_config(cms_width, ams_r)

        # Total memory is memory per window times number of windows
        training_total_memory = memory_per_window * num_training_windows
        testing_total_memory = memory_per_window * num_testing_windows

        memory_tracking.append({
            'cms_width': cms_width,
            'ams_r': ams_r,
            'iteration': iteration,
            'window_size': 10,  # default window size
            'training_total_memory_bytes': training_total_memory,
            'testing_total_memory_bytes': testing_total_memory,
            'training_total_memory_mb': training_total_memory / (1024**2),
            'testing_total_memory_mb': testing_total_memory / (1024**2),
            'memory_per_window_bytes': memory_per_window,
            'num_training_windows': num_training_windows,
            'num_testing_windows': num_testing_windows,
        })

        config_key = (cms_width, ams_r)
        if config_key not in configs_seen:
            configs_seen.add(config_key)
            print(f'Processed config: cms_width={cms_width}, ams_r={ams_r}, iteration={iteration}')
            print(f'  Memory per window: {memory_per_window:,} bytes ({memory_per_window / 1024:.2f} KB)')
            print(f'  Training: {num_training_windows} windows, {training_total_memory:,} bytes')
            print(f'  Testing: {num_testing_windows} windows, {testing_total_memory:,} bytes')

    # Save to CSV
    memory_df = pd.DataFrame(memory_tracking)
    memory_df.to_csv('datasets/estimations/memory_tracking.csv', index=False)
    print(f'\nSaved memory tracking for {len(memory_tracking)} configurations to datasets/estimations/memory_tracking.csv')
    print(f'Configurations: {sorted(configs_seen)}')

if __name__ == '__main__':
    main()
