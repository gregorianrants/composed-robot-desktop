from dotenv import load_dotenv
import os
from robonet.Hub import Hub
from robonet.Subscriber import Subscriber
import zmq
import time
import numpy as np
import matplotlib.pyplot as plt


class SpeedMonitor():
    def __init__(self):
        self.speeds = []
        self.times = []
        self.running = False

    def handle_command(self,message):
        command = message['command']
        if command == 'start':
            self.running = True
        elif command == 'stop':
            self.running=False
            self.show()

    def udpate(self,data):
        print('calling update')
        if self.running:
            speed = data["speed_deg/sec"]
            seconds = data['time']
            self.speeds.append(speed)
            self.times.append(seconds)

    def show(self):
        speeds = np.array(self.speeds)
        times = np.array(self.times)
        times = times -times[0]
        plt.plot(times,speeds)
        plt.show()


load_dotenv()

DESKTOP_IP = os.getenv('DESKTOP_IP')

time.sleep(1)

speedMonitor = SpeedMonitor()



context = zmq.Context()
subscriber = Subscriber(context,[{'node':'robot',
                                  'topic':'speed-recorder-command'},
                                  {'node':'robot',
                                  'topic':'motor-data'}
                                  ])

subscriber.add_listener('motor-data',speedMonitor.udpate)
subscriber.add_listener('speed-recorder-command',speedMonitor.handle_command)
subscriber.start()