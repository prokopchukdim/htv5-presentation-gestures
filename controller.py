# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
import time


def brighter(img, value = 30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h,s,v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
	classNames = f.read().split('\n')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize recording movement speed
last_pos = [0,0]
timer = time.perf_counter()
is_right_gesture = False
threshold = 2


while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Get time between frames
    frametime = 1 / cap.get(cv2.CAP_PROP_FPS)

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    frame = brighter(frame)
    
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    results = hands.process(framergb)
    
    # className = ''
    is_left = False
    is_left_fist = False
    is_right = False
    speed = 0
    
    # post process the result
    if results.multi_hand_landmarks:
        landmarks0 = []
        landmarks1 = []

        for i in range(len(results.multi_hand_landmarks)):
            is_left = results.multi_handedness[i].classification[0].label == "Left"
            is_right = results.multi_handedness[i].classification[0].label == "Right"

            avgx = 0
            avgy = 0
            #avgz = 0
            
            for lm in results.multi_hand_landmarks[i].landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                if is_right:
                    avgx += lm.x
                    avgy += lm.y
                    #avgz += lm.z
                
                if i == 0:
                    landmarks0.append([lmx, lmy])
                else:
                    landmarks1.append([lmx, lmy])

            #if(is_left):
##            mpDraw.draw_landmarks(frame, results.multi_hand_landmarks[i], mpHands.HAND_CONNECTIONS, landmark_drawing_spec = mpDraw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))

            

            #detects control gesture
            if is_left and i == 0:
                prediction = model.predict([landmarks0])
                is_left_fist = classNames[np.argmax(prediction)] == 'fist'
            elif is_left and i == 1:
                prediction = model.predict([landmarks1])
                is_left_fist = classNames[np.argmax(prediction)] == 'fist'


##            if time.perf_counter() - timer < threshold:
##                cv2.putText(frame, "Gesture! ", (10, 80), cv2.FONT_HERSHEY_PLAIN, 
##                       1, (255,255,255), 2, cv2.LINE_AA)

            
            #detects movement gesture
            if(is_right and time.perf_counter() - timer >= threshold):
                avgx = avgx / len(results.multi_hand_landmarks[i].landmark)
                avgy = avgy / len(results.multi_hand_landmarks[i].landmark)
                #avgz = avgz / len(results.multi_hand_landmarks[i].landmark)

                if last_pos == [0,0]:
                    last_pos = [avgx,avgy]
                dist_moved = math.sqrt(abs(avgx-last_pos[0])**2 + abs(avgy-last_pos[1])**2)
                speed = dist_moved / frametime

##                if i == 0:
##                    prediction = model.predict([landmarks0])
##                if i == 1:
##                    prediction = model.predict([landmarks1])
##
##                prediction = classNames[np.argmax(prediction)]


##                cv2.putText(frame, "Gesture! ", (10, 80), cv2.FONT_HERSHEY_PLAIN, 
##                       1, (255,255,255), 2, cv2.LINE_AA)
                
                #is_right_gesture = prediction == 'stop' or prediction == 'live long'

##                if speed >= 1:
##                    cv2.putText(frame, "Gesture! ", (10, 80), cv2.FONT_HERSHEY_PLAIN, 
##                       1, (255,255,255), 2, cv2.LINE_AA)
##                    print("Gesture")
##                    timer = time.perf_counter()
                
                last_pos = [avgx,avgy]
    else:
        last_pos == [0,0]

    # draw text on frame

    if is_left_fist:
        if(is_right and speed >= 1):
           print("Gesture")
           timer = time.perf_counter()
        
        cv2.putText(frame, "Control", (10, 40), cv2.FONT_HERSHEY_PLAIN, 
                   1, (255,255,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
