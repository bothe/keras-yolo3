import argparse

from detectors import detect_in_video, detect_in_image
from yolo3.yolo import YOLO


def main():
    if FLAGS.save_pb:
        print("respective pb model file saved in pb_models")
        return
    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            img_path = FLAGS.input
        elif "input" not in FLAGS:
            img_path = input('Input image filename: ')
        detect_in_image(yolo, img_path)
    elif "input" in FLAGS:
        detect_in_video(yolo, FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    # When detecting in VIDEO
    # --model_path k_weights/yolov3-tiny.h5 --anchors_path model_data/tiny_yolo_anchors.txt
    # --input test_data/cold_lake_trafic.mp4
    # When detecting only in IMAGE
    # --model_path k_weights/yolov3-tiny.h5 --anchors_path model_data/tiny_yolo_anchors.txt
    # --image --input test_data/cold_lake_trafic.mp4

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
                        help='Number of GPUs to use, default ' + str(YOLO.get_defaults("gpu_num")))

    parser.add_argument('--print_summary', type=bool, default=False,
                        help='Print summary of the models, default ' + str(YOLO.get_defaults("print_summary")))

    parser.add_argument('--save_pb', type=bool, default=False,
                        help='Save the tf model in pb format, default ' + str(YOLO.get_defaults("save_pb")))

    parser.add_argument('--image', default=False, action="store_true",
                        help='Image detection mode.')
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument("--input", nargs='?', type=str, required=False,
                        help="Video or Image (if with --image) input path")

    parser.add_argument("--output", nargs='?', type=str, default="",
                        help="[Optional] Video output path (currently output frames are being stored in "
                             "test_data/imgs directory)")

    FLAGS = parser.parse_args()
    yolo = YOLO(**vars(FLAGS))
    main()
