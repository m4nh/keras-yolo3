import sys
import argparse
import numpy as np
from yolo import YOLO, detect_video
from PIL import Image
import cv2


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            detections = []
            r_image = yolo.detect_image(image, detections=detections)
            print("DETECTOINS:", detections)
            r_image.show()
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

    '''
    Command line positional arguments -- for video detection mode
    '''

    parser.add_argument(
        "--images_list",  type=str, required=True,
        help="Images list manifest"
    )

    parser.add_argument(
        "--output_file",  type=str, required=True,
        help="Output file"
    )

    parser.add_argument(
        '--debug', default=False, action="store_true",
        help='Debug mode'
    )

    FLAGS = parser.parse_args()

    yolo = YOLO(**vars(FLAGS))

    f = open(FLAGS.images_list, 'r')
    lines = f.readlines()
    rows = []
    counter = 0
    for l in lines:
        image_path = l.split(' ')[0].replace('\n', '')
        image = Image.open(image_path)
        detections = []
        output = yolo.detect_image(image, detections=detections)

        row = "{}".format(image_path)
        for d in detections:
            box = ",".join(map(str, np.round(d[1]).astype(int).tolist()))
            box += ",{:.2f}".format(d[2])
            chunk = "{},{}".format(d[0], box)
            row += ' '+chunk
        rows.append(row)

        if FLAGS.debug:
            cv2.imshow("img", np.array(output))
            cv2.waitKey(0)
        else:
            print("Percentage: {:.2f}".format(100.0*float(counter)/float(len(lines))))
        counter += 1

    f.close()

    f = open(FLAGS.output_file, 'w')
    for i, r in enumerate(rows):
        f.write(r)
        if i < len(rows)-1:
            f.write('\n')
    f.close()
