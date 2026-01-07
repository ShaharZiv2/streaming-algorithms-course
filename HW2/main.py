from schemas.args import Args
from utils.arg_utils import parse_arguments
from utils.sketch_manager import SketchManager
from utils.stream import Stream, SourceFiles
from sklearn.ensemble import RandomForestClassifier




def main():
    args: Args = parse_arguments()
    sketch_manager = SketchManager(args)
    for window in Stream(SourceFiles.TRAINING, window_size=args.window_size):
        sketch_manager.sketch(window)
        window_estimation = sketch_manager.estimate()
        # train_model()
        sketch_manager.reset()
        pass


if __name__ == '__main__':
    main()
