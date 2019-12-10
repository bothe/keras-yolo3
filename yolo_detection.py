import argparse

from detectors import detect_in_video, detect_in_image
from yolo import YOLO


def main():
    if FLAGS.save_pb:
        return
    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_in_image(yolo)
    elif "input" in FLAGS:
        detect_in_video(yolo, FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument('--model_path', type=str,
                        help='path to model weight file, default ' + YOLO.get_defaults("model_path"))

    parser.add_argument('--anchors_path', type=str,
                        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path"))

    parser.add_argument('--classes_path', type=str,
                        help='path to class definitions, default ' + YOLO.get_defaults("classes_path"))

    parser.add_argument('--gpu_num', type=int,
                        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num")))

    parser.add_argument('--print_summary', type=bool, default=False,
                        help='Print summary of the models, default ' + str(YOLO.get_defaults("print_summary")))

    parser.add_argument('--save_pb', type=bool, default=False,
                        help='Save the tf model in pb format, default ' + str(YOLO.get_defaults("save_pb")))

    parser.add_argument('--image', default=False, action="store_true",
                        help='Image detection mode, will ignore all positional arguments')
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument("--input", nargs='?', type=str, required=False, default='./path2your_video',
                        help="Video input path")

    parser.add_argument("--output", nargs='?', type=str, default="",
                        help="[Optional] Video output path")

    FLAGS = parser.parse_args()
    yolo = YOLO(**vars(FLAGS))
    main()
