from dotenv import load_dotenv
import os
from robonet.Hub import Hub
from robonet.Subscriber import Subscriber
import zmq
import time

load_dotenv()

DESKTOP_IP = os.getenv('DESKTOP_IP')

time.sleep(1)


subscriber = Subscriber(DESKTOP_IP,[{'node':'motor','topic':'motor-data1'},{'node':'motor','topic':'motor-data2'}
                               ,{'node': 'camera','topic':'frame'}])
subscriber.start()

for (topic,message) in subscriber.json_stream():
    print(topic,message)