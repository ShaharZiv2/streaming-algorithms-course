import os.path

import pandas as pd

from schemas.args import Args
from utils.arg_utils import parse_arguments
from utils.sketch_manager import SketchManager
from utils.stream import Stream, SourceFiles
from utils.time_utils import func_timer


@func_timer
def process_data_stream(args: Args, source_file: SourceFiles):
    sketch_manager = SketchManager(args)
    training_stream = Stream(source_file, args.window_size)

    window_estimations = []
    total_memory_usage = 0

    for event in training_stream:
        sketch_manager.sketch(event)
        if training_stream.window_done:
            window_estimations.append(sketch_manager.estimate())

            # Calculate memory usage for this window
            window_memory = sketch_manager.calculate_total_memory()
            total_memory_usage += window_memory

            sketch_manager.reset()
            training_stream.reset_window()


    return pd.DataFrame(window_estimations), total_memory_usage


def run_all(args: Args):
    # Initialize memory tracking dataframe
    memory_tracking = []
    memory_tracking_file = 'datasets/estimations/memory_tracking.csv'

    for cms_width in [2048, 8192]:
            for ams_r in [16, 64, 256]:
                for i in range(25):
                    print(f'Running config: cms_width {cms_width} ams_r {ams_r} iteration {i + 1}')
                    training_file_name = f'datasets/estimations/trng_{cms_width}_{ams_r}_{i}.csv'
                    testing_file_name = f'datasets/estimations/tstng_{cms_width}_{ams_r}_{i}.csv'
                    if os.path.exists(training_file_name):
                        continue
                    args = Args(
                        window_size=args.window_size,
                        cms_width=cms_width,
                        ams_r=ams_r,
                    )

                    training_windows_estimations, training_total_memory = process_data_stream(args, SourceFiles.TRAINING_LIGHT)
                    testing_windows_estimations, testing_total_memory = process_data_stream(args, SourceFiles.TESTING_LIGHT)

                    training_windows_estimations.to_csv(training_file_name, index=False)
                    testing_windows_estimations.to_csv(testing_file_name, index=False)

                    # Track memory usage with parameters
                    memory_tracking.append({
                        'cms_width': cms_width,
                        'ams_r': ams_r,
                        'iteration': i,
                        'window_size': args.window_size,
                        'training_total_memory_bytes': training_total_memory,
                        'testing_total_memory_bytes': testing_total_memory,
                        'training_total_memory_mb': training_total_memory / (1024**2),
                        'testing_total_memory_mb': testing_total_memory / (1024**2),
                    })

                    # Save incrementally
                    pd.DataFrame(memory_tracking).to_csv(memory_tracking_file, index=False)

def main():
    args: Args = parse_arguments()
    if args.run_all:
        run_all(args)

    else:
        training_windows_estimations, training_total_memory = process_data_stream(args, SourceFiles.TRAINING)
        testing_windows_estimations, testing_total_memory = process_data_stream(args, SourceFiles.TESTING)

        training_windows_estimations.to_csv('datasets/training_window_estimations.csv', index=False)
        testing_windows_estimations.to_csv('datasets/testing_window_estimations.csv', index=False)

        # Save memory tracking for single run
        memory_data = pd.DataFrame([{
            'cms_width': args.cms_width,
            'ams_r': args.ams_r,
            'window_size': args.window_size,
            'training_total_memory_bytes': training_total_memory,
            'testing_total_memory_bytes': testing_total_memory,
            'training_total_memory_mb': training_total_memory / (1024**2),
            'testing_total_memory_mb': testing_total_memory / (1024**2),
        }])
        memory_data.to_csv('datasets/memory_tracking_single_run.csv', index=False)
        print(f"\nMemory Usage Summary:")
        print(f"  Training: {training_total_memory:,} bytes ({training_total_memory / (1024**2):.2f} MB)")
        print(f"  Testing: {testing_total_memory:,} bytes ({testing_total_memory / (1024**2):.2f} MB)")

if __name__ == '__main__':
    main()
