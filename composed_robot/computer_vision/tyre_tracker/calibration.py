from dotenv import load_dotenv
import os
from robonet.Hub import Hub
from robonet.Subscriber import Subscriber
import zmq
import time
import cv2
import numpy as np
import io
import sys
from .tyre_tracker import track_tyre
from pathlib import Path
from .load_ranges import load_ranges


load_dotenv()

PI_IP = os.getenv('PI_IP')

print(PI_IP)

time.sleep(1)



subscriber = Subscriber(PI_IP,[{'node':'vision','topic':'frame'}])
subscriber.start()

best_so_far_for_yellow_lego = ((23,30),(220,255),(220,255))

count = 0
hue_low = 28
hue_high = 36

sat_low =121
sat_high = 255

val_low = 224
val_high = 255

lower_bound = np.array([hue_low,sat_low,val_low])
upper_bound = np.array([hue_high,sat_high,val_high])

lower_bound,upper_bound = load_ranges()

class TrackBar:
    def __init__(self,name,window,start_value,upper_limit):
        self.name = name
        self.window = window
        self.start_value = start_value
        self.upper_limit = upper_limit
        self.value = start_value
        cv2.createTrackbar(name,'myTracker',start_value,upper_limit,self.update)

    def update(self,val):
        self.value = val
        print(self.name,self.value)

class RangeTrackBars:
    def __init__(self,name,window,low_start,high_start,upper_limit):
        self.lower_bar = TrackBar(f'{name} Low',window,low_start,upper_limit)
        self.upper_bar = TrackBar(f'{name} High',window,high_start,upper_limit)
        self._lower = self.lower_bar.value
        self._upper = self.upper_bar.value

    @property
    def lower(self):
        self._lower = self.lower_bar.value
        return self._lower

    @property
    def upper(self):
        self._upper = self.upper_bar.value
        return self._upper



cv2.namedWindow('myTracker')

hueTracker = RangeTrackBars('Hue','myTracker',low_start=lower_bound[0],high_start=upper_bound[0],upper_limit=179)
rangeTracker = RangeTrackBars('Sat','myTracker',low_start=lower_bound[1],high_start=upper_bound[1],upper_limit=255)
valTracker = RangeTrackBars('Val','myTracker',low_start=lower_bound[2],high_start=upper_bound[2],upper_limit=255)

# while True:
#     k = cv2.waitKey(0) & 0xFF
#     print(k)
#     if k == ord('c'):
#         cv2.destroyAllWindows()
#         break
#     time.sleep(0.001)


count = 0

try: 
    for (topic,node,bytes) in subscriber.bytes_stream():
        count+=1
        array = np.frombuffer(bytes,dtype=np.uint8)
        frame = cv2.imdecode(array,1)
       
        lower_bound = np.array([hueTracker.lower,rangeTracker.lower,valTracker.lower])
        upper_bound = np.array([hueTracker.upper,rangeTracker.upper,valTracker.upper])

        centre = track_tyre(frame,lower_bound,upper_bound)
        if(count%10==0 and centre):
            print(centre)


        height,width,_ = frame.shape
        cv2.line(frame,((width-1)//2,0),((width-1)//2,height-1),(0,0,255),1)
        cv2.line(frame,(0,(height-1)//2),(width-1,(height-1)//2),(0,0,255),1)
       
        cv2.imshow('not lost in translation',frame)
        
        if cv2.waitKey(1)==ord('q'):
            lower = [hueTracker.lower,rangeTracker.lower,valTracker.lower]
            upper = [hueTracker.upper,rangeTracker.upper,valTracker.upper]
            ranges = np.array([[hueTracker.lower,rangeTracker.lower,valTracker.lower],
                              [hueTracker.upper,rangeTracker.upper,valTracker.upper]])
            np.save(Path(__file__).parent/'threshold_ranges',ranges)
            break
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    print('handling interupption')
    cv2.destroyAllWindows()

