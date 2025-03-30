from .tyre_tracker import track_tyre
from robonet.Subscriber import Subscriber
import numpy as np
from dotenv import load_dotenv
import time
import zmq
import os
import cv2

low = np.array([15,226,239])
high = np.array([32,255,255])


load_dotenv()

PI_IP = os.getenv("PI_IP")


print(PI_IP)

context = zmq.Context()
subscriber = Subscriber(PI_IP, [{'node':'vision',
                                  'topic':'frame'},
                                  ])


#print(camera_matrix)

subscriber.start()



time.sleep(1)


for (topic,node,bytes) in subscriber.bytes_stream():
    np_array = np.frombuffer(bytes,dtype=np.uint8)
    image = cv2.imdecode(np_array,1)
    cv2.imshow('image',image)
    centre = track_tyre(image,low,high)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()

    
            
