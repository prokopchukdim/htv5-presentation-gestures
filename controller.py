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
    is_gesture = False
    
    # post process the result
    if results.multi_hand_landmarks:
        landmarks0 = []
        landmarks1 = []

        #Cycle through detected hands
        for i in range(len(results.multi_hand_landmarks)):

            #Classify current hand
            is_left = results.multi_handedness[i].classification[0].label == "Left"
            is_right = results.multi_handedness[i].classification[0].label == "Right"

            avgx = 0
            avgy = 0

            #Puts the coordinates of joints into a list readable by the model
            for lm in results.multi_hand_landmarks[i].landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                if is_right:
                    avgx += lm.x
                    avgy += lm.y
                
                if i == 0:
                    landmarks0.append([lmx, lmy])
                else:
                    landmarks1.append([lmx, lmy])


            #draws joints
                    
            #if(is_left):
##            mpDraw.draw_landmarks(frame, results.multi_hand_landmarks[i], mpHands.HAND_CONNECTIONS, landmark_drawing_spec = mpDraw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))

            
            #detects control gesture based on whether hand 1 or hand 2 is being detected 
            if is_left and i == 0:
                prediction = model.predict([landmarks0])
                is_left_fist = classNames[np.argmax(prediction)] == 'fist'
            elif is_left and i == 1:
                prediction = model.predict([landmarks1])
                is_left_fist = classNames[np.argmax(prediction)] == 'fist'

            
            #detects movement gesture if threshold seconds have passed between the last gesture detection and the new one
            if(is_right and time.perf_counter() - timer >= threshold):

                #finds speed of average point on hand
                avgx = avgx / len(results.multi_hand_landmarks[i].landmark)
                avgy = avgy / len(results.multi_hand_landmarks[i].landmark)
                if last_pos == [0,0]:
                    last_pos = [avgx,avgy]
                dist_moved = math.sqrt(abs(avgx-last_pos[0])**2 + abs(avgy-last_pos[1])**2)
                speed = dist_moved / frametime

                #If the speed is over 1, then the user is making a movement gesture
                if speed >= 1:
                    is_gesture = True
                    timer = time.perf_counter()               
                last_pos = [avgx,avgy]
    else:
        last_pos == [0,0]

    #Trigger the if statements if both the control and movement gesture are up.
    if is_left_fist:
        if(is_gesture):
           print("Gesture with control")
##           timer = time.perf_counter()
        
        cv2.putText(frame, "Control", (10, 40), cv2.FONT_HERSHEY_PLAIN, 
                   1, (255,255,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
