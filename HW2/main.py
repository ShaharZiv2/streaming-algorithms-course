import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

from schemas.args import Args
from utils.arg_utils import parse_arguments
from utils.sketch_manager import SketchManager
from utils.stream import Stream, SourceFiles
from utils.time_utils import func_timer


@func_timer
def main():
    args: Args = parse_arguments()
    sketch_manager = SketchManager(args)
    training_stream = Stream(SourceFiles.TRAINING, args.window_size)

    window_estimations = []

    for event in training_stream:
        sketch_manager.sketch(event)
        if training_stream.window_done:
            window_estimations.append(sketch_manager.estimate())

            sketch_manager.reset()
            training_stream.reset_window()

    window_df = pd.DataFrame(window_estimations)
    window_df.to_csv('datasets/window_estimations.csv', index=False)

if __name__ == '__main__':
    main()
