import wiringpi as wp
from flask import Flask, render_template, Response, request, jsonify, url_for
import threading
import cv2
import time
import logging
import os
import autonav
import signal
import sys
import numpy as np
import math
from pathlib import Path
from lane_detection import LaneDetector
from behavior_cloning import BehaviorCloner
from hybrid_navigation import HybridNavigator, NavigationMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("robot.log")
    ]
)
logger = logging.getLogger("RobotController")

class MotorController:
    """Handles motor control operations"""
    
    def __init__(self):
        # Setup WiringPi
        wp.wiringPiSetup()
        
        # Pin definitions
        self.MOTOR_PINS = {
            "motor1": {"in1": 0, "in2": 1, "enable": 4},  # GPIO 17, 18, soft PWM 4
            "motor2": {"in1": 2, "in2": 3, "enable": 5}   # GPIO 27, 22, soft PWM 5
        }
        
        # Configure pins and initialize PWM
        for motor, pins in self.MOTOR_PINS.items():
            wp.pinMode(pins["in1"], 1)
            wp.pinMode(pins["in2"], 1)
            wp.softPwmCreate(pins["enable"], 0, 100)
            
        logger.info("Motor controller initialized")
        
    def control_motor(self, motor, direction, speed):
        """Control a specific motor"""
        pins = self.MOTOR_PINS[motor]
        wp.softPwmWrite(pins["enable"], int(speed * 100))
        
        if direction == "forward":
            wp.digitalWrite(pins["in1"], 0)
            wp.digitalWrite(pins["in2"], 1)
        elif direction == "backward":
            wp.digitalWrite(pins["in1"], 1)
            wp.digitalWrite(pins["in2"], 0)
        else:  # Stop
            wp.digitalWrite(pins["in1"], 0)
            wp.digitalWrite(pins["in2"], 0)
            
    def move(self, direction, speed):
        """Move the robot in a specific direction"""
        if direction == "forward":
            self.control_motor("motor1", "forward", speed)
            self.control_motor("motor2", "forward", speed)
            logger.debug(f"Moving forward at speed {speed}")
        elif direction == "backward":
            self.control_motor("motor1", "backward", speed)
            self.control_motor("motor2", "backward", speed)
            logger.debug(f"Moving backward at speed {speed}")
        elif direction == "left":
            self.control_motor("motor1", "backward", speed)
            self.control_motor("motor2", "forward", speed)
            logger.debug(f"Turning left at speed {speed}")
        elif direction == "right":
            self.control_motor("motor1", "forward", speed)
            self.control_motor("motor2", "backward", speed)
            logger.debug(f"Turning right at speed {speed}")
        elif direction == "stop":
            self.control_motor("motor1", "stop", 0)
            self.control_motor("motor2", "stop", 0)
            logger.debug("Stopping motors")
            
    def cleanup(self):
        """Stop motors and clean up"""
        self.move("stop", 0)
        logger.info("Motors stopped and cleaned up")


class PersonDetector:
    """Handles person detection using TensorFlow Lite"""
    
    def __init__(self):
        self.running = True
        self.detection_enabled = False
        self.follow_mode = False
        self.detections = []
        self.detection_lock = threading.Lock()
        self.model_type = "mobilenet"  # Default model: "mobilenet" or "yolov3"
        
        # Initialize models
        try:
            # Try to import TFLite
            try:
                import tflite_runtime.interpreter as tflite
                self.tflite_available = True
            except ImportError:
                try:
                    # Fall back to TensorFlow if tflite_runtime is not available
                    import tensorflow as tf
                    tflite = tf.lite
                    self.tflite_available = True
                    logger.info("Using TensorFlow's lite module instead of tflite_runtime")
                except ImportError:
                    logger.warning("Neither TensorFlow Lite nor TensorFlow is available. MobileNet detection disabled.")
                    self.tflite_available = False
            
            # Initialize YOLOv3 availability flag
            self.yolov3_available = False
            
            # Path to model files
            model_dir = Path(__file__).parent / "models"
            model_dir.mkdir(exist_ok=True)
            
            # MobileNet model paths
            self.mobilenet_model_path = model_dir / "mobilenet_ssd_v2_coco_quant.tflite"
            self.mobilenet_label_path = model_dir / "coco_labels.txt"
            
            # YOLOv3 model paths
            self.yolov3_model_path = model_dir / "yolov3-tiny.weights"
            self.yolov3_config_path = model_dir / "yolov3-tiny.cfg"
            self.yolov3_names_path = model_dir / "coco.names"
            
            # Download MobileNet model if not exists
            if not self.mobilenet_model_path.exists():
                logger.info("Downloading MobileNet SSD model...")
                self._download_mobilenet_model(
                    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
                    model_dir
                )
            
            # Download YOLOv3 model if not exists
            if not self.yolov3_model_path.exists():
                logger.info("Downloading YOLOv3-tiny model...")
                self._download_yolov3_model(model_dir)
            
            # Initialize TFLite interpreter for MobileNet
            if self.tflite_available:
                self.interpreter = tflite.Interpreter(model_path=str(self.mobilenet_model_path))
                self.interpreter.allocate_tensors()
                
                # Get model details for MobileNet
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.input_shape = self.input_details[0]['shape']
                self.height = self.input_shape[1]
                self.width = self.input_shape[2]
                
                # Load MobileNet labels
                with open(self.mobilenet_label_path, 'r') as f:
                    self.mobilenet_labels = [line.strip() for line in f.readlines()]
            
            # Initialize YOLOv3 model if OpenCV DNN module is available
            try:
                self.yolov3_net = cv2.dnn.readNetFromDarknet(
                    str(self.yolov3_config_path), 
                    str(self.yolov3_model_path)
                )
                # Check if GPU is available
                try:
                    self.yolov3_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.yolov3_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    logger.info("Using CUDA for YOLOv3 inference")
                except:
                    logger.info("CUDA not available, using CPU for YOLOv3 inference")
              
                # Load YOLOv3 class names
                with open(self.yolov3_names_path, 'r') as f:
                    self.yolov3_classes = [line.strip() for line in f.readlines()]
              
                self.yolov3_available = True
                logger.info("YOLOv3 model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing YOLOv3 model: {e}")
                self.yolov3_available = False
              
            logger.info("Person detector initialized successfully")
          
        except ImportError:
            logger.warning("TensorFlow Lite not available. Person detection disabled.")
            self.tflite_available = False
            self.yolov3_available = False
        except Exception as e:
            logger.error(f"Error initializing person detector: {e}")
            self.tflite_available = False
            self.yolov3_available = False
    
    def _download_mobilenet_model(self, url, model_dir):
        """Download and extract MobileNet SSD model files"""
        try:
            import requests
            import zipfile
            import io
            
            # Download the file
            response = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(model_dir)
            
            # Rename files to expected names
            for file in model_dir.glob("*.tflite"):
                file.rename(model_dir / "mobilenet_ssd_v2_coco_quant.tflite")
              
            # Create labels file if not exists
            label_path = model_dir / "coco_labels.txt"
            if not label_path.exists():
                # COCO dataset labels
                coco_labels = [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                ]
                with open(label_path, 'w') as f:
                    for label in coco_labels:
                        f.write(f"{label}\n")
                      
            logger.info("MobileNet SSD model downloaded and extracted successfully")
          
        except Exception as e:
            logger.error(f"Error downloading MobileNet model: {e}")
            raise

    def _download_yolov3_model(self, model_dir):
        """Download YOLOv3-tiny model files"""
        try:
            import requests
          
            # URLs for model files
            weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
            cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
            names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
          
            # Download weights file
            logger.info("Downloading YOLOv3-tiny weights...")
            with open(self.yolov3_model_path, 'wb') as f:
                response = requests.get(weights_url)
                f.write(response.content)
          
            # Download config file
            logger.info("Downloading YOLOv3-tiny config...")
            with open(self.yolov3_config_path, 'wb') as f:
                response = requests.get(cfg_url)
                f.write(response.content)
          
            # Download class names file
            logger.info("Downloading COCO class names...")
            with open(self.yolov3_names_path, 'wb') as f:
                response = requests.get(names_url)
                f.write(response.content)
              
            logger.info("YOLOv3-tiny model files downloaded successfully")
          
        except Exception as e:
            logger.error(f"Error downloading YOLOv3 model files: {e}")
            raise

    def set_model_type(self, model_type):
        """Set detection model type: 'mobilenet' or 'yolov3'"""
        if model_type == "yolov3" and not self.yolov3_available:
            logger.warning("YOLOv3 model not available. Using MobileNet SSD instead.")
            self.model_type = "mobilenet"
        elif model_type == "mobilenet" and not self.tflite_available:
            logger.warning("MobileNet SSD model not available. Person detection disabled.")
            self.model_type = None
        else:
            self.model_type = model_type
          
        logger.info(f"Detection model set to: {self.model_type}")
        return self.model_type
    
    def enable_detection(self, enabled=True):
        """Enable or disable person detection"""
        self.detection_enabled = enabled and (self.tflite_available or self.yolov3_available)
        logger.info(f"Person detection {'enabled' if self.detection_enabled else 'disabled'}")
        return self.detection_enabled
    
    def set_follow_mode(self, enabled=True):
        """Enable or disable person following mode"""
        self.follow_mode = enabled and self.detection_enabled
        logger.info(f"Person following {'enabled' if self.follow_mode else 'disabled'}")
        return self.follow_mode
    
    def detect_persons(self, frame):
        """Detect persons in the given frame using the selected model"""
        if not self.detection_enabled:
            return []
          
        if self.model_type == "mobilenet" and self.tflite_available:
            return self._detect_with_mobilenet(frame)
        elif self.model_type == "yolov3" and self.yolov3_available:
            return self._detect_with_yolov3(frame)
        else:
            return []
          
    def _detect_with_mobilenet(self, frame):
        """Detect persons using MobileNet SSD model"""
        try:
            # Resize and normalize image
            image = cv2.resize(frame, (self.width, self.height))
            image = np.expand_dims(image, axis=0)
          
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], image)
            self.interpreter.invoke()
          
            # Get detection results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
          
            # Filter for persons (class 0 in COCO dataset)
            persons = []
            for i in range(len(scores)):
                if scores[i] >= 0.5 and classes[i] == 0:  # Person class with confidence > 0.5
                    box = boxes[i]
                    persons.append({
                        'box': [
                            int(box[1] * frame.shape[1]),  # xmin
                            int(box[0] * frame.shape[0]),  # ymin
                            int(box[3] * frame.shape[1]),  # xmax
                            int(box[2] * frame.shape[0])   # ymax
                        ],
                        'score': float(scores[i])
                    })
          
            # Update detections
            with self.detection_lock:
                self.detections = persons
              
            return persons
          
        except Exception as e:
            logger.error(f"Error in MobileNet person detection: {e}")
            return []
          
    def _detect_with_yolov3(self, frame):
        """Detect persons using YOLOv3-tiny model"""
        try:
            # Prepare image for YOLOv3
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
          
            # Set input and forward pass
            self.yolov3_net.setInput(blob)
            layer_names = self.yolov3_net.getLayerNames()
            try:
                # OpenCV 4.5.4+
                output_layers = [layer_names[i-1] for i in self.yolov3_net.getUnconnectedOutLayers()]
            except:
                # Older OpenCV versions
                output_layers = [layer_names[i[0]-1] for i in self.yolov3_net.getUnconnectedOutLayers()]
              
            outputs = self.yolov3_net.forward(output_layers)
          
            # Process detections
            persons = []
            conf_threshold = 0.5
            nms_threshold = 0.4
          
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                  
                    # Filter for person class (class 0 in COCO dataset)
                    if confidence > conf_threshold and class_id == 0:  # Person class
                        # YOLO returns center, width, height
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                      
                        # Rectangle coordinates
                        x = max(0, int(center_x - w/2))
                        y = max(0, int(center_y - h/2))
                      
                        persons.append({
                            'box': [x, y, x+w, y+h],
                            'score': float(confidence)
                        })
          
            # Apply non-maximum suppression to remove overlapping boxes
            if persons:
                boxes = np.array([p['box'] for p in persons])
                confidences = np.array([p['score'] for p in persons])
              
                # Convert boxes to the format expected by NMSBoxes
                nms_boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]  # [x, y, w, h]
              
                indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, conf_threshold, nms_threshold)
              
                # Extract final detections
                final_persons = []
                for i in indices.flatten() if len(indices) > 0 else []:
                    final_persons.append(persons[i])
              
                # Update detections
                with self.detection_lock:
                    self.detections = final_persons
              
                return final_persons
          
            return []
          
        except Exception as e:
            logger.error(f"Error in YOLOv3 person detection: {e}")
            return []
    
    def get_follow_target(self):
        """Get the nearest person to follow"""
        with self.detection_lock:
            if not self.detections:
                return None
                
            # Find the largest detection (assuming it's the closest)
            largest_area = 0
            target = None
            
            for person in self.detections:
                box = person['box']
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > largest_area:
                    largest_area = area
                    target = person
                    
            return target
    
    def get_follow_direction(self, frame_width):
        """Calculate direction to move to follow the target person"""
        target = self.get_follow_target()
        if not target:
            return None
            
        box = target['box']
        center_x = (box[0] + box[2]) / 2
        frame_center = frame_width / 2
        
        # Calculate horizontal position relative to center
        position = (center_x - frame_center) / frame_center  # -1 to 1
        
        # Calculate area of bounding box (for distance estimation)
        area = (box[2] - box[0]) * (box[3] - box[1])
        area_ratio = area / (frame_width * frame_width)  # Normalized by frame size
        
        return {
            'position': position,  # -1 (far left) to 1 (far right)
            'area_ratio': area_ratio,  # Approximation of distance
            'box': box
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        logger.info("Person detector cleaned up")


class CameraController:
    """Handles camera operations"""
    
    def __init__(self, person_detector=None):
        self.cameras = []
        self.camera_index = 0
        self.current_camera = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.processed_frame = None
        self.running = True
        self.person_detector = person_detector
        self.detect_cameras()
        
        # Start frame capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        logger.info(f"Camera controller initialized. Found cameras: {self.cameras}")
        
    def detect_cameras(self):
        """Detect available cameras"""
        self.cameras = []
        for i in range(5):  # Check the first 5 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.cameras.append(i)
                    cap.release()
            except Exception as e:
                logger.error(f"Error detecting camera {i}: {e}")
                
        if not self.cameras:
            logger.warning("No cameras detected!")
        
    def set_camera(self, camera_id):
        """Set the active camera"""
        if camera_id in self.cameras:
            self.camera_index = camera_id
            logger.info(f"Switched to camera {camera_id}")
            return True
        else:
            logger.warning(f"Camera {camera_id} not available")
            return False
            
    def _capture_frames(self):
        """Continuously capture frames from the current camera"""
        while self.running:
            try:
                # If camera index changed, reopen camera
                if self.current_camera is None or self.current_camera != self.camera_index:
                    if self.current_camera is not None:
                        cap = cv2.VideoCapture(self.current_camera)
                        cap.release()
                    
                    self.current_camera = self.camera_index
                    cap = cv2.VideoCapture(self.current_camera)
                    
                    if not cap.isOpened():
                        logger.error(f"Failed to open camera {self.current_camera}")
                        time.sleep(1)
                        continue
                
                # Read frame
                success, frame = cap.read()
                if not success:
                    logger.warning(f"Failed to read frame from camera {self.current_camera}")
                    time.sleep(0.1)
                    continue
                    
                # Update latest frame
                with self.frame_lock:
                    self.latest_frame = frame
                    
                    # Process frame for person detection if enabled
                    if self.person_detector and self.person_detector.detection_enabled:
                        processed = frame.copy()
                        
                        # Detect persons
                        persons = self.person_detector.detect_persons(frame)
                        
                        # Draw bounding boxes
                        for person in persons:
                            box = person['box']
                            score = person['score']
                            cv2.rectangle(processed, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(processed, f"Person: {score:.2f}", (box[0], box[1] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Highlight follow target
                        if self.person_detector.follow_mode:
                            target = self.person_detector.get_follow_target()
                            if target:
                                box = target['box']
                                cv2.rectangle(processed, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                                cv2.putText(processed, "TARGET", (box[0], box[1] - 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        self.processed_frame = processed
                    else:
                        self.processed_frame = frame
                    
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in frame capture: {e}")
                time.sleep(1)
                
    def generate_frames(self):
        """Generate frames for streaming"""
        while self.running:
            with self.frame_lock:
                if self.processed_frame is None:
                    time.sleep(0.1)
                    continue
                    
                frame = self.processed_frame.copy()
                
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Error encoding frame: {e}")
                time.sleep(0.1)
                
    def cleanup(self):
        """Clean up camera resources"""
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        logger.info("Camera controller cleaned up")


class PersonFollower:
    """Controls the robot to follow a person"""
    
    def __init__(self, motor_controller, camera_controller, person_detector):
        self.motor_controller = motor_controller
        self.camera_controller = camera_controller
        self.person_detector = person_detector
        self.running = True
        self.follow_thread = None
        self.follow_speed = 0.5  # Default follow speed
        
    def start_following(self):
        """Start the person following thread"""
        if self.follow_thread is None or not self.follow_thread.is_alive():
            self.running = True
            self.follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
            self.follow_thread.start()
            logger.info("Person following started")
            return True
        return False
        
    def stop_following(self):
        """Stop the person following thread"""
        self.running = False
        if self.follow_thread and self.follow_thread.is_alive():
            self.follow_thread.join(timeout=1.0)
            self.motor_controller.move("stop", 0)
            logger.info("Person following stopped")
            return True
        return False
        
    def set_follow_speed(self, speed):
        """Set the following speed"""
        self.follow_speed = max(0.1, min(1.0, float(speed)))
        logger.info(f"Follow speed set to {self.follow_speed}")
        
    def _follow_loop(self):
        """Main loop for person following"""
        while self.running and self.person_detector.follow_mode:
            try:
                # Get frame dimensions
                with self.camera_controller.frame_lock:
                    if self.camera_controller.latest_frame is None:
                        time.sleep(0.1)
                        continue
                    frame_width = self.camera_controller.latest_frame.shape[1]
                
                # Get direction to follow
                direction_info = self.person_detector.get_follow_direction(frame_width)
                
                if direction_info:
                    position = direction_info['position']  # -1 (far left) to 1 (far right)
                    area_ratio = direction_info['area_ratio']  # Approximation of distance
                    
                    # Log the values for debugging
                    logger.debug(f"Follow target: position={position:.2f}, area_ratio={area_ratio:.2f}")
                    
                    # UPDATED FOLLOWING LOGIC:
                    # If person is too far away, move toward them
                    if area_ratio < 0.15:  # Person is far away
                        # If person is significantly off-center, turn toward them first
                        if abs(position) > 0.3:
                            if position < 0:  # Person is to the left
                                logger.debug("Person is far and to the left, turning left")
                                self.motor_controller.move("left", self.follow_speed * 0.8)
                            else:  # Person is to the right
                                logger.debug("Person is far and to the right, turning right")
                                self.motor_controller.move("right", self.follow_speed * 0.8)
                        else:  # Person is mostly centered, move forward
                            logger.debug("Person is far but centered, moving forward")
                            self.motor_controller.move("forward", self.follow_speed)
                    
                    # If person is at a good distance but off-center, adjust position
                    elif area_ratio < 0.4:  # Good distance
                        if abs(position) > 0.2:  # Person is off-center
                            if position < 0:  # Person is to the left
                                logger.debug("Person is at good distance but to the left, turning left")
                                self.motor_controller.move("left", self.follow_speed * 0.6)
                            else:  # Person is to the right
                                logger.debug("Person is at good distance but to the right, turning right")
                                self.motor_controller.move("right", self.follow_speed * 0.6)
                        else:  # Person is centered at good distance
                            # Move forward slowly to maintain distance
                            logger.debug("Person is at good distance and centered, moving forward slowly")
                            self.motor_controller.move("forward", self.follow_speed * 0.3)
                    
                    # If person is too close, back up slightly
                    else:  # Person is too close
                        logger.debug("Person is too close, backing up")
                        self.motor_controller.move("backward", self.follow_speed * 0.4)
                
                else:
                    # No person detected, stop
                    logger.debug("No person detected, stopping")
                    self.motor_controller.move("stop", 0)
                
                time.sleep(0.1)  # Control loop rate
                
            except Exception as e:
                logger.error(f"Error in follow loop: {e}")
                time.sleep(0.5)
                
        # Ensure motors are stopped when exiting
        self.motor_controller.move("stop", 0)
        
    def cleanup(self):
        """Clean up resources"""
        self.stop_following()
        logger.info("Person follower cleaned up")


class RobotWebServer:
    """Web server for robot control"""
    
    def __init__(self, motor_controller, camera_controller, person_detector, hybrid_navigator, person_follower, host="0.0.0.0", port=5002):
        self.app = Flask(__name__)
        self.motor_controller = motor_controller
        self.camera_controller = camera_controller
        self.person_detector = person_detector
        self.hybrid_navigator = hybrid_navigator
        self.person_follower = person_follower
        self.host = host
        self.port = port
        self.server_thread = None
        self.setup_routes()
        
        logger.info(f"Web server initialized on {host}:{port}")
        
    def setup_routes(self):
        """Set up Flask routes"""
        
        @self.app.route("/")
        def home():
            self.camera_controller.detect_cameras()  # Refresh camera list
            return render_template("index.html", 
                                  cameras=self.camera_controller.cameras, 
                                  selected_camera=self.camera_controller.camera_index)
        
        @self.app.route("/control", methods=["POST"])
        def control():
            action = request.form["action"]
            speed = float(request.form["speed"])
            camera_id = request.form.get("camera_id", type=int)
            
            if camera_id is not None:
                self.camera_controller.set_camera(camera_id)
            
            # Record manual steering for training if data collection is enabled
            if self.hybrid_navigator.collecting_data:
                with self.camera_controller.frame_lock:
                    if self.camera_controller.latest_frame is not None:
                        frame = self.camera_controller.latest_frame.copy()
                        self.hybrid_navigator.record_manual_steering(frame, action, speed)
            
            self.motor_controller.move(action, speed)
            return "OK"
        
        @self.app.route("/video_feed")
        def video_feed():
            return Response(self.camera_controller.generate_frames(), 
                           mimetype='multipart/x-mixed-replace; boundary=frame')
                           
        @self.app.route("/detection", methods=["POST"])
        def detection_control():
            action = request.form.get("action")
            
            if action == "enable":
                enabled = self.person_detector.enable_detection(True)
                return jsonify({"status": "ok", "enabled": enabled})
            elif action == "disable":
                enabled = self.person_detector.enable_detection(False)
                return jsonify({"status": "ok", "enabled": enabled})
            else:
                return jsonify({"status": "error", "message": "Invalid action"})
                
        @self.app.route("/follow", methods=["POST"])
        def follow_control():
            action = request.form.get("action")
            
            if action == "start":
                # Enable detection and follow mode
                self.person_detector.enable_detection(True)
                self.person_detector.set_follow_mode(True)
                # Start following
                success = self.person_follower.start_following()
                return jsonify({"status": "ok", "following": success})
            elif action == "stop":
                # Disable follow mode
                self.person_detector.set_follow_mode(False)
                # Stop following
                success = self.person_follower.stop_following()
                return jsonify({"status": "ok", "following": not success})
            elif action == "speed":
                speed = request.form.get("speed", type=float)
                if speed is not None:
                    self.person_follower.set_follow_speed(speed)
                    return jsonify({"status": "ok", "speed": speed})
                else:
                    return jsonify({"status": "error", "message": "Invalid speed"})
            else:
                return jsonify({"status": "error", "message": "Invalid action"})
                
        @self.app.route("/detection_info")
        def detection_info():
            with self.camera_controller.frame_lock:
                if self.camera_controller.latest_frame is None:
                    return jsonify({"info": "No camera feed available"})
                    
            if not self.person_detector.tflite_available and not self.person_detector.yolov3_available:
                return jsonify({"info": "Person detection not available (TFLite and YOLOv3 missing)"})
                
            if not self.person_detector.detection_enabled:
                return jsonify({"info": "Person detection disabled"})
                
            with self.person_detector.detection_lock:
                num_persons = len(self.person_detector.detections)
                
            if self.person_detector.follow_mode:
                target = self.person_detector.get_follow_target()
                if target:
                    return jsonify({
                        "info": f"Following person. {num_persons} person(s) detected.",
                        "following": True,
                        "persons": num_persons
                    })
                else:
                    return jsonify({
                        "info": f"Searching for person to follow. {num_persons} person(s) detected.",
                        "following": False,
                        "persons": num_persons
                    })
            else:
                return jsonify({
                    "info": f"{num_persons} person(s) detected.",
                    "following": False,
                    "persons": num_persons
                })
            
        @self.app.route("/system_info")
        def system_info():
            # Basic system information
            try:
                import psutil
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent
                temperature = None
                
                try:
                    # Try to get Raspberry Pi temperature
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temperature = float(f.read()) / 1000.0
                except:
                    pass
                    
                return jsonify({
                    "cpu": cpu,
                    "memory": memory,
                    "temperature": temperature,
                    "cameras": self.camera_controller.cameras,
                    "active_camera": self.camera_controller.camera_index,
                    "detection_enabled": self.person_detector.detection_enabled,
                    "detection_model": self.person_detector.model_type,
                    "follow_mode": self.person_detector.follow_mode
                })
            except Exception as e:
                logger.error(f"Error getting system info: {e}")
                return jsonify({"error": "Failed to get system information"})

        @self.app.route("/hybrid_navigation", methods=["POST"])
        def hybrid_navigation_control():
            action = request.form.get("action")
            
            if action == "start":
                # Get navigation mode
                mode_str = request.form.get("mode", "HYBRID")
                try:
                    mode = NavigationMode[mode_str]
                except:
                    mode = NavigationMode.HYBRID
                
                # Start navigation
                success = self.hybrid_navigator.start_navigation(mode)
                return jsonify({
                    "status": "ok", 
                    "active": success,
                    "mode": self.hybrid_navigator.mode.name
                })
            elif action == "stop":
                # Stop navigation
                success = self.hybrid_navigator.stop_navigation()
                return jsonify({"status": "ok", "active": not success})
            elif action == "mode":
                # Set navigation mode
                mode_str = request.form.get("mode", "HYBRID")
                try:
                    mode = NavigationMode[mode_str]
                    success = self.hybrid_navigator.set_mode(mode)
                    return jsonify({
                        "status": "ok", 
                        "mode": self.hybrid_navigator.mode.name
                    })
                except:
                    return jsonify({
                        "status": "error", 
                        "message": f"Invalid mode: {mode_str}"
                    })
            elif action == "speed":
                # Set navigation speed
                speed = request.form.get("speed", type=float)
                if speed is not None:
                    self.hybrid_navigator.set_speed(speed)
                    return jsonify({"status": "ok", "speed": speed})
                else:
                    return jsonify({"status": "error", "message": "Invalid speed"})
            elif action == "weights":
                # Set navigation weights
                lane_weight = request.form.get("lane_weight", type=float)
                model_weight = request.form.get("model_weight", type=float)
                
                success = self.hybrid_navigator.set_weights(lane_weight, model_weight)
                return jsonify({
                    "status": "ok", 
                    "lane_weight": self.hybrid_navigator.lane_weight,
                    "model_weight": self.hybrid_navigator.model_weight
                })
            elif action == "toggle_data_collection":
                # Toggle data collection
                enabled = request.form.get("enabled", "").lower() == "true"
                is_enabled = self.hybrid_navigator.toggle_data_collection(enabled)
                return jsonify({"status": "ok", "data_collection": is_enabled})
            elif action == "save_data":
                # Save training data
                success = self.hybrid_navigator.save_training_data()
                return jsonify({
                    "status": "ok" if success else "error",
                    "message": "Training data saved successfully" if success else "Failed to save training data"
                })
            elif action == "train_model":
                # Train model
                epochs = request.form.get("epochs", type=int, default=10)
                success = self.hybrid_navigator.train_model(epochs=epochs)
                return jsonify({
                    "status": "ok" if success else "error",
                    "message": f"Model trained successfully with {epochs} epochs" if success else "Failed to train model"
                })
            else:
                return jsonify({"status": "error", "message": "Invalid action"})

        @self.app.route("/hybrid_navigation_info")
        def hybrid_navigation_info():
            try:
                info = self.hybrid_navigator.get_navigation_info()
                return jsonify(info)
            except Exception as e:
                logger.error(f"Error getting hybrid navigation info: {e}")
                return jsonify({"error": "Failed to get hybrid navigation information"})
        
        @self.app.route("/detection_model", methods=["POST"])
        def detection_model():
            model_type = request.form.get("model_type")
            
            if model_type in ["mobilenet", "yolov3"]:
                active_model = self.person_detector.set_model_type(model_type)
                return jsonify({"status": "ok", "active_model": active_model})
            else:
                return jsonify({"status": "error", "message": "Invalid model type"})

        @self.app.route("/css")
        def css():
            return Response(
                open(os.path.join(os.path.dirname(__file__), 'templates', 'style.css'), 'r').read(),
                mimetype='text/css'
            )
        
    def start(self):
        """Start the web server in a separate thread"""
        self.server_thread = threading.Thread(
            target=lambda: self.app.run(
                host=self.host, 
                port=self.port, 
                debug=False, 
                use_reloader=False,
                threaded=True
            ),
            daemon=True
        )
        self.server_thread.start()
        logger.info(f"Web server running at http://{self.host}:{self.port}")
        
    def cleanup(self):
        """Clean up web server resources"""
        logger.info("Web server shutting down")


class Robot:
    """Main robot class that coordinates all components"""
    
    def __init__(self):
        # Initialize components
        self.person_detector = PersonDetector()
        self.motor_controller = MotorController()
        self.camera_controller = CameraController(self.person_detector)
        
        # Initialize lane detection and behavior cloning
        self.lane_detector = LaneDetector()
        self.behavior_cloner = BehaviorCloner()
        
        # Initialize hybrid navigation
        self.hybrid_navigator = HybridNavigator(
            self.motor_controller,
            self.camera_controller,
            self.lane_detector,
            self.behavior_cloner
        )
        
        # Initialize person following
        self.person_follower = PersonFollower(
            self.motor_controller, 
            self.camera_controller, 
            self.person_detector
        )
        
        # Initialize web server
        self.web_server = RobotWebServer(
            self.motor_controller, 
            self.camera_controller, 
            self.person_detector,
            self.hybrid_navigator,
            self.person_follower,
            "0.0.0.0",  # Add the host parameter explicitly
            5002        # Add the port parameter explicitly
        )
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("Robot initialized")
        logger.info("Robot initialized. Use 'source PiRobot/bin/activate' to activate the virtual environment.")
        
    def start(self):
        """Start all robot components"""
        self.web_server.start()
        logger.info("Robot started")
        logger.info(f"Access the web interface at http://<raspberry_pi_ip>:{self.web_server.port}")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.cleanup()
            
    def signal_handler(self, sig, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {sig}, shutting down...")
        self.cleanup()
        sys.exit(0)
        
    def cleanup(self):
        """Clean up all resources"""
        self.person_follower.cleanup()
        self.camera_controller.cleanup()
        self.hybrid_navigator.cleanup()
        self.person_detector.cleanup()
        self.motor_controller.cleanup()
        self.web_server.cleanup()
        logger.info("Robot shutdown complete")


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    robot = Robot()
    robot.start()

