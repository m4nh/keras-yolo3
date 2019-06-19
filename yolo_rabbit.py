import sys
import argparse
# from yolo import YOLO, detect_video
from PIL import Image
import PIL
import numpy as np
import cv2
import pika
import json
import pickle
import time
from yolo import YOLO, detect_video
import pprint
from babylon.network import BMessage, SimpleChannel


parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

parser.add_argument(
    '--model_path', type=str,
    help='path to model weight file, default ' + YOLO.get_defaults("model_path")
)

parser.add_argument(
    '--anchors_path', type=str,
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

parser.add_argument(
    "--input", nargs='?', type=str, required=False, default='./path2your_video',
    help="Video input path"
)

parser.add_argument(
    "--output", nargs='?', type=str, default="",
    help="[Optional] Video output path"
)

FLAGS = parser.parse_args()
yolo = YOLO(**vars(FLAGS))


channel = SimpleChannel(topic_name='vision_module')


def mouseCb(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        action = "mouse_left_click" if event == cv2.EVENT_LBUTTONDOWN else "mouse_right_click"
        message = BMessage(action=action)
        message.addField("point", np.array([x,y]))
        channel.publish(message)

cv2.namedWindow("live",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("live",mouseCb)


cap = cv2.VideoCapture(0)
while True:
    ret, live_image = cap.read()
    if ret:
        live_image = cv2.cvtColor(live_image, cv2.COLOR_BGR2RGB)

        image_center = np.array([live_image.shape[1]*0.5, live_image.shape[0]*0.5]).astype(float)
        min_score = 0.1
        super_target_radius = 50
        target_labels = [2]

        try:
            image = PIL.Image.fromarray(live_image)
        except Exception as e:
            print('Open Error! Try again!', e)
            continue
        else:
            detections = yolo.detect_simple(image)

            out_image = live_image.copy()

            feasible_targets = []
            min_distance = 1000000
            min_target = None
            for r in range(detections.shape[0]):

                row = detections[r,:]
                label = int(row[0])
                box = row[1:5]
                score = row[5]
                if score < min_score:
                    continue
                tl = np.array(box[:2])
                br = np.array(box[2:4])
                center = ((tl + br)*0.5).astype(int)
                color = yolo.colors[int(row[0])]
                cv2.circle(out_image, (center[0], center[1]), 50, color, 10)

                if label in target_labels:
                    feasible_targets.append(row)
                    distance = np.linalg.norm(image_center - center)
                    if distance < min_distance:
                        min_distance = distance
                        min_target = row
                    dd = 20
                    cv2.line(out_image, (center[0]-dd, center[1]-dd),(center[0]+dd, center[1]+dd),color, 3)
                    cv2.line(out_image, (center[0]+dd, center[1] - dd), (center[0] - dd, center[1] + dd), color, 3)

            cv2.circle(out_image, (int(image_center[0]),int(image_center[1])),super_target_radius,(0,255,0),2)


            if min_target is not None:

                box = min_target[1:5]
                score = min_target[5]
                tl = np.array(box[:2])
                br = np.array(box[2:4])
                center = ((tl + br) * 0.5).astype(int)
                super_target = False

                if min_distance < super_target_radius:
                    super_target = True

                color = (255,255,255)
                if super_target:
                    color = (0, 255, 0)
                cv2.line(out_image, (center[0] - dd, center[1] - dd), (center[0] + dd, center[1] + dd),color, 8)
                cv2.line(out_image, (center[0] + dd, center[1] - dd), (center[0] - dd, center[1] + dd),color, 8)

                if not super_target:
                    message = BMessage(action="target_acquired")
                    message.addField("target", np.array(center))
                    print("Sending Target:", np.array(center))
                    channel.publish(message)
                else:
                    message = BMessage(action="super_target_acquired")
                    message.addField("target", np.array(center))
                    print("Sending Super Target:", np.array(center))
                    channel.publish(message)
            else:
                message = BMessage(action="no_target_acquired")
                print("Sending No Target:")
                channel.publish(message)


            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)

            cv2.imshow("live", out_image)
            cv2.waitKey(10)
            print("tick")

yolo.close_session()
#
# for i in range(1):
#     arr = np.random.uniform(0,255.0,(2,2,3,3,4,5,1)).astype(np.uint16)
#     message  = BMessage(sender='puno', receiver='gino')
#     message.addField('matrix', arr)
#     message.addField('f', float(2.3))
#     message.addField('i', int(2.3))
#     message.addField('b', True)
#     message.addField('ss', 'siahsaoihs aisi usahsaihu ')
#     print("Sending")
#     pprint.pprint(message.__dict__)
#     channel.publish(message)
#
#
# channel.close()
