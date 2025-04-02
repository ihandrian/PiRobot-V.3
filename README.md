@@ -4,6 +4,7 @@ https://github.com/user-attachments/assets/c549cace-67d4-4eec-b868-973418a37e95
 
 This project implements a self-driving robot using Raspberry Pi 3B that can be manually controlled and has the ability to detect and follow people. The robot features a web-based control interface that can be accessed from any device on the same network.
 
 ## Features
 
@@ -12,6 +13,8 @@ This project implements a self-driving robot using Raspberry Pi 3B that can be ma
 - Person detection and tracking using TensorFlow Lite or YOLOv3-Tiny
 - Automatic person following
 - System monitoring (CPU, memory, temperature)
+- Autonomous navigation with obstacle avoidance
+- Data collection for training autonomous behaviors
 - Mobile-friendly interface
 
 ## Hardware Requirements
@@ -42,6 +45,15 @@ I've made all the improvements from [PiRobot V.0](https://github.com/ihandrian/Pi
 - The robot will adjust its speed and direction based on the person's position and distance
 
 
+### 2. Autonomous Navigation
+
+- Added a simple obstacle detection system using computer vision
+- Implemented basic path planning and obstacle avoidance
+- Added data collection for training autonomous behaviors
+- Implemented experience replay with a memory buffer for improved steering predictions
+- Added controls to enable/disable autonomous navigation and data collection
+
+
 ### 2. Fullscreen Video Feed with to enable/disable autonomous navigation and data collection


### 2. Fullscreen Video Feed with Overlay Controls

- Redesigned the interface with a fullscreen video feed
- Added a semi-transparent control panel in the bottom left corner
- Replaced text buttons with arrow icons for more intuitive control
- The control panel has 80% opacity
- Added touch support for mobile devices


### 3. System Information Panel

- Moved system information to the top of the screen
- Made it semi-transparent with white text
- Displays CPU usage, memory usage, temperature, and detection status
- Updates in real-time


### Additional Improvements

- Added visual indicators for person detection (green bounding boxes)
- Added visual indicator for the person being followed (red bounding box)
- Improved error handling for when TensorFlow Lite or YoloV3 is not available
- Added controls to adjust the following speed
- Made the interface fully responsive for different screen sizes


## How to Use
1. Clone this repository. Save all the files to your project directory.
2. Go to your dir "PiRobot-V.1" in your computer.
3. Run virtual environment of your preference, in this case I´m using `venv` to create environment:
    ```plaintext
    python -m venv PiRobot

