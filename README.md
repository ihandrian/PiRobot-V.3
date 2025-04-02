
 This project implements a self-driving robot using Raspberry Pi 3B that can be manually controlled and has the ability to detect and follow person. The robot control based a web-based interface that can be accessed from any device in the same network.
 
 ## Features

 - Person detection and tracking using TensorFlow Lite or YOLOv3-Tiny
 - Automatic person following
 - Adjust its speed and direction based on the person's position and distance
 - System monitoring (CPU, memory, temperature)
 - Autonomous navigation with obstacle avoidance
 - Data collection for training autonomous behaviors
 - Mobile-friendly interface
   
 
## Key Improvements Implemented

### 1. Autonomous Navigation
- Added a simple obstacle detection system using computer vision
- Implemented basic path planning and obstacle avoidance
- Added data collection for training autonomous behaviors
- Implemented experience replay with a memory buffer for improved steering predictions
- Added controls to enable/disable autonomous navigation and data collection
- Implement a combined approach using classical lane detection and behavior cloning from manual steering inputs. This hybrid approach will help the robot learn better navigation skills.

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
    ```
     - Activate:
         - Windows:
          ```plaintext
          PiRobot\Scripts\activate
          ```
         - Linux/macOS:
          ```plaintext
          source PiRobot/bin/activate
          ```
      - Deactivate:
          ```plaintext
          deactivate
          ```
3. Install the required dependencies:

```plaintext
pip3 install -r requirements.txt
```


2. Run the file `main.py`:

```plaintext
python main.py
```

3. Access the control panel at `http://<your_pi_ip>:5002`
4. To use person following:
    - Toggle "Detection" to enable person detection
    - Click "Start Following" to make the robot follow the nearest person
    - Adjust the follow speed as needed

The system is optimized for Raspberry Pi 3B and uses lightweight detection models to ensure smooth performance.
### Future Development

1. **Sensor Integration**: Add classes for different sensors (ultrasonic, IR, etc.)
2. ~~**Autonomous Navigation**: Implement path planning and obstacle avoidance~~
3. ~~**Computer Vision**: Add object detection using TensorFlow Lite or similar~~
4. ~~**Data Logging**: Implement data collection for training autonomous behaviors~~

Support page:
- Paypal https://paypal.me/IrfanHandrian
- Buy me Coffee https://buymeacoffee.com/handrianirv

