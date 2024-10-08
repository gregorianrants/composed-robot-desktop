from dotenv import load_dotenv
import os
from robonet.Hub import Hub
from robonet.Subscriber import Subscriber
import zmq
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2.aruco as aruco
import math
from robonet.Publisher import Publisher
from ..aruco_location import get_pose

with open('camera_cal.npy','rb') as f:
    camera_matrix = np.load(f)
    camera_distortion = np.load(f)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
marker_size = 100

load_dotenv()

PI_IP = os.getenv("PI_IP")
DESKTOP_IP = os.getenv("DESKTOP_IP")

time.sleep(1)

context = zmq.Context()
subscriber = Subscriber(PI_IP, [{'node':'vision',
                                  'topic':'frame'},
                                  ])

publisher = Publisher(
    hub_ip=PI_IP,
    address=f"tcp://{DESKTOP_IP}",
    node="aruco-location",
    topics=["aruco-location"],
)

subscriber.start()

for (topic,node,bytes) in subscriber.bytes_stream():
    np_array = np.frombuffer(bytes,dtype=np.uint8)
    image = cv2.imdecode(np_array,1)
    image,results =  get_pose(image)
    if(len(results)>0):
        publisher.send_json('aruco-location',{"x":results[0][0], "y": results[0][1], "theta":results[0][2]})
    cv2.imshow('not lost in translation',image)
    cv2.waitKey(1)
   