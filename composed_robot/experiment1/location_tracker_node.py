import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib import animation
import numpy as np
import math
import threading
import time
from robonet.Subscriber import Subscriber
from dotenv import load_dotenv
import os
from ..location_tracker import update

load_dotenv()

PI_IP = os.getenv("PI_IP")

subscriber = Subscriber(
        PI_IP,
        [{"node": "aruco-location", "topic": "aruco-location"}],
    )


subscriber.start()

for topic, node, message in subscriber.json_stream():
    #odometry on robot is done in mm but here we do things in cm
    x = message['x']/10
    y = message['y']/10
    theta = message['theta']
    print(x,y,math.degrees(theta))
    update([x,y],theta)