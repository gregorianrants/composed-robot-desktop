from dotenv import load_dotenv
import os
from robonet.Hub import Hub
from robonet.Publisher import Publisher
import zmq
import time

load_dotenv()
print('fuck helllo')

PI_IP = os.getenv('PI_IP')
DESKTOP_IP = os.getenv('DESKTOP_IP')

time.sleep(2)



publisher = Publisher(
    hub_ip=PI_IP, address=f"tcp://{'0.0.0.0'}", node="motor", topics=["motor-data1","motor-data2"])
# publisher2 = Publisher(
#     hub_ip=PI_IP, address=f"tcp://{DESKTOP_IP}", node="camera", topics=["frame"])

for i in range(100):
    time.sleep(1)
    print("sending it")
    publisher.send_json("motor-data1",{"hello": "there"})
    # publisher2.send_json("frame",{"say": "cheese"})