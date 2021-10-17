# htv5-presentation-gestures

# About
A gesture recognition script written for the Hack the Valley 5 Hackathon in a 38h period with @katarinasotic. With the goal of improving engagement in videoconferencing using computer vision, this project allows users to move back from their camera and incorporate body language into their presentations without losing fundamental control. In its current state, the Python script uses camera information to determine whether a user needs their slides to be moved forwards or backwards. To trigger these actions, users raise their left fist to enable the program to listen for instructions. They can then swipe with their palm out to the left or to the right to trigger a forwards or backwards slide change. This process allows users to use common body language and hand gestures without accidentally triggering the controls. 

Although this project did not win a prize, it was a great learning experience. [Here](https://devpost.com/software/presentation-gestures-through-computer-vision-pgtcv) is a link to the submission on devpost. 

# Restrictions
Windows allows only one application to use the camera at a time. Since the script uses camera input, it also outputs a virtual camera using pyvirtualcam and Unity Capture which other conferencing apps can then use, and as such only works in Windows. Ideally, this project would be integrated within a conferencing software, so the creation of a virtual camera would be unecessary. Due to the time-restricted nature of the event, the fist-recognition algorithm uses a pre-trained network, which falls within fair-use. Actual deployment of the project will require a re-trained algorithm. 

# Requirements
The project was written in Python 3.8.10, and requires the following packages to be installed: OpenCV 2, NumPy, MediaPipe, Tensorflow, pynput, pyvirtualcam (with Unity Capture, see https://github.com/letmaik/pyvirtualcam for details).

Unity Capture can be installed and uninstalled from the Unity-Capture-master.zip folder (We do not own this work!). 
The project itself can be run from controller.py
