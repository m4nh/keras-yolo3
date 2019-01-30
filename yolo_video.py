import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import PIL
import numpy as np
import cv2


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
            t = np.array(image)
            print(t.shape, np.min(t), np.max(t))
        except Exception as e:
            print('Open Error! Try again!', e)
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def detect_webcam(yolo):
    cap = cv2.VideoCapture(0)
    while True:
        ret, live_image = cap.read()
        if ret:
            live_image = cv2.cvtColor(live_image, cv2.COLOR_BGR2RGB)
            try:
                image = PIL.Image.fromarray(live_image)
            except Exception as e:
                print('Open Error! Try again!', e)
                continue
            else:
                arr = np.array(yolo.detect_image(image))
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                cv2.imshow("live", arr)
                cv2.waitKey(1)
    yolo.close_session()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )

    parser.add_argument(
        '--webcam', default=False, action="store_true",
        help='Webcam detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    if FLAGS.webcam:
        detect_webcam(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
