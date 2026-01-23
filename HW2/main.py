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

    for event in training_stream:
        sketch_manager.sketch(event)
        if training_stream.window_done:
            window_estimations.append(sketch_manager.estimate())

            sketch_manager.reset()
            training_stream.reset_window()

    return pd.DataFrame(window_estimations)


def main():
    args: Args = parse_arguments()
    training_windows_estimations = process_data_stream(args, SourceFiles.TRAINING)
    testing_windows_estimations = process_data_stream(args, SourceFiles.TESTING)

    training_windows_estimations.to_csv('datasets/training_window_estimations.csv', index=False)
    testing_windows_estimations.to_csv('datasets/testing_window_estimations.csv', index=False)

if __name__ == '__main__':
    main()
