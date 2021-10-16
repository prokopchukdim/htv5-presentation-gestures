# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import math

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.4)
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


while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Get time between frames
    frametime = 1 / cap.get(cv2.CAP_PROP_FPS)

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    results = hands.process(framergb)
    
    # className = ''
    is_left = False
    is_left_fist = False
    is_right = False

    

    
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

            if(is_left):
                mpDraw.draw_landmarks(frame, results.multi_hand_landmarks[i], mpHands.HAND_CONNECTIONS, landmark_drawing_spec = mpDraw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))

            if(is_right):
                avgx = avgx / len(results.multi_hand_landmarks[i].landmark)
                avgy = avgy / len(results.multi_hand_landmarks[i].landmark)
                #avgz = avgz / len(results.multi_hand_landmarks[i].landmark)

                if last_pos == [0,0]:
                    last_pos = [avgx,avgy]
                dist_moved = math.sqrt(avgx**2 + avgy**2)
                speed = dist_moved / frametime               


            #detects control gesture
            if is_left and i == 0:
                prediction = model.predict([landmarks0])
                is_left_fist = classNames[np.argmax(prediction)] == 'fist'
            elif is_left and i == 1:
                prediction = model.predict([landmarks1])
                is_left_fist = classNames[np.argmax(prediction)] == 'fist'    
                

    # draw text on frame

    if is_left_fist:
        cv2.putText(frame, "Control", (10, 40), cv2.FONT_HERSHEY_PLAIN, 
                   1, (255,255,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
