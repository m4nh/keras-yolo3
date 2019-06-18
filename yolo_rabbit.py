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

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

args = { "x-max-length": 1 }
channel.queue_declare(queue='hello2', arguments=args)


for i in range(1000):
    arr = np.array([1.1 + i*0.0001] * 10).reshape((2,-1))
    message  = {
        'command': 'pino',
        'data': arr.tolist()
    }
    channel.basic_publish(exchange='', routing_key='hello2', body=json.dumps(message))
    print(json.dumps(message))
    time.sleep(0.1)
connection.close()

