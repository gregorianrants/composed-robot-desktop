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
from ..computer_vision.aruco_location.markers import Markers

# with open('camera_cal.npy','rb') as f:
#     camera_matrix = np.load(f)
#     camera_distortion = np.load(f)

# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
# marker_size = 100

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

markers = Markers()

subscriber.start()

count = 0

for (topic,node,bytes) in subscriber.bytes_stream():
    np_array = np.frombuffer(bytes,dtype=np.uint8)
    image = cv2.imdecode(np_array,1)
    image =  markers.get_markers(image)
    
    if(len(markers.markers) > 0):
        marker = markers.marker[0]
        c_T_m = marker.c_T_m
        m_T_c = marker.m_T_c
        pitch,yaw,roll = m_T_c.get_pitch_yaw_roll()
        x,y,z,_ = m_T_c.homogeneous_matrix @ np.array([0,0,0,1])
        publisher.send_json('aruco-location',{"x":x, "y": y, "theta":yaw})
        if count%15==0:
            print(math.degrees(pitch),math.degrees(yaw),math.degrees(roll))
            print({'pitch': pitch,'yaw': yaw,'roll': roll})
        count+=1
        
    cv2.imshow('not lost in translation',image)
    cv2.waitKey(1)
   