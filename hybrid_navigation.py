import cv2
import numpy as np
import logging
import time
import threading
from enum import Enum

logger = logging.getLogger("HybridNavigation")

class NavigationMode(Enum):
    MANUAL = 0
    LANE_FOLLOWING = 1
    BEHAVIOR_CLONING = 2
    HYBRID = 3

class HybridNavigator:
    """Combines lane detection and behavior cloning for navigation"""
    
    def __init__(self, motor_controller, camera_controller, lane_detector, behavior_cloner):
        self.motor_controller = motor_controller
        self.camera_controller = camera_controller
        self.lane_detector = lane_detector
        self.behavior_cloner = behavior_cloner
        
        # Navigation parameters
        self.running = False
        self.navigation_thread = None
        self.mode = NavigationMode.MANUAL
        self.speed = 0.4  # Default speed
        self.collecting_data = False
        
        # Hybrid navigation parameters
        self.lane_weight = 0.7  # Weight for lane detection (0-1)
        self.model_weight = 0.3  # Weight for behavior cloning (0-1)
        
        # Last steering commands
        self.last_lane_steering = 0.0
        self.last_model_steering = 0.0
        self.last_hybrid_steering = 0.0
        
        logger.info("Hybrid navigator initialized")
    
    def start_navigation(self, mode=NavigationMode.HYBRID):
        """Start autonomous navigation"""
        if self.navigation_thread is None or not self.navigation_thread.is_alive():
            self.mode = mode
            self.running = True
            self.navigation_thread = threading.Thread(target=self._navigation_loop, daemon=True)
            self.navigation_thread.start()
            logger.info(f"Hybrid navigation started in {mode.name} mode")
            return True
        return False
    
    def stop_navigation(self):
        """Stop autonomous navigation"""
        self.running = False
        if self.navigation_thread and self.navigation_thread.is_alive():
            self.navigation_thread.join(timeout=1.0)
            self.motor_controller.move("stop", 0)
            logger.info("Hybrid navigation stopped")
            return True
        return False
    
    def set_mode(self, mode):
        """Set navigation mode"""
        if not isinstance(mode, NavigationMode):
            try:
                mode = NavigationMode(mode)
            except:
                logger.error(f"Invalid navigation mode: {mode}")
                return False
        
        self.mode = mode
        logger.info(f"Navigation mode set to {mode.name}")
        return True
    
    def set_speed(self, speed):
        """Set navigation speed"""
        self.speed = max(0.1, min(1.0, float(speed)))
        logger.info(f"Navigation speed set to {self.speed}")
        return True
    
    def set_weights(self, lane_weight=None, model_weight=None):
        """Set weights for hybrid navigation"""
        if lane_weight is not None:
            self.lane_weight = max(0.0, min(1.0, float(lane_weight)))
            
        if model_weight is not None:
            self.model_weight = max(0.0, min(1.0, float(model_weight)))
            
        # Normalize weights to sum to 1
        total = self.lane_weight + self.model_weight
        if total > 0:
            self.lane_weight /= total
            self.model_weight /= total
            
        logger.info(f"Navigation weights set to: lane={self.lane_weight:.2f}, model={self.model_weight:.2f}")
        return True
    
    def toggle_data_collection(self, enabled=None):
        """Toggle data collection for training"""
        if enabled is not None:
            self.collecting_data = enabled
        else:
            self.collecting_data = not self.collecting_data
            
        if self.collecting_data:
            self.behavior_cloner.start_collecting()
        else:
            self.behavior_cloner.stop_collecting()
            
        logger.info(f"Data collection {'enabled' if self.collecting_data else 'disabled'}")
        return self.collecting_data
    
    def save_training_data(self):
        """Save collected training data"""
        return self.behavior_cloner.save_data()
    
    def train_model(self, epochs=10):
        """Train the behavior cloning model"""
        return self.behavior_cloner.train_model(epochs=epochs)
    
    def _navigation_loop(self):
        """Main loop for autonomous navigation"""
        while self.running:
            try:
                # Get current frame
                with self.camera_controller.frame_lock:
                    if self.camera_controller.latest_frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.camera_controller.latest_frame.copy()
                
                # Process frame based on navigation mode
                if self.mode == NavigationMode.LANE_FOLLOWING:
                    self._process_lane_following(frame)
                elif self.mode == NavigationMode.BEHAVIOR_CLONING:
                    self._process_behavior_cloning(frame)
                elif self.mode == NavigationMode.HYBRID:
                    self._process_hybrid_navigation(frame)
                else:
                    # Manual mode - do nothing
                    time.sleep(0.1)
                    continue
                
                time.sleep(0.05)  # Control loop rate
                
            except Exception as e:
                logger.error(f"Error in navigation loop: {e}")
                time.sleep(0.5)
                
        # Ensure motors are stopped when exiting
        self.motor_controller.move("stop", 0)
    
    def _process_lane_following(self, frame):
        """Process frame for lane following"""
        # Detect lanes and get steering angle
        processed_frame, steering_angle, confidence = self.lane_detector.detect_lanes(frame)
        
        # Update processed frame
        with self.camera_controller.frame_lock:
            self.camera_controller.processed_frame = processed_frame
        
        # Store last lane steering
        self.last_lane_steering = steering_angle
        
        # Execute steering command if confidence is high enough
        if confidence > 0.3:
            self._execute_steering(steering_angle)
        else:
            # Not confident enough, stop
            self.motor_controller.move("stop", 0)
    
    def _process_behavior_cloning(self, frame):
        """Process frame for behavior cloning"""
        # Predict steering angle from model
        steering_angle = self.behavior_cloner.predict_steering(frame)
        
        # Store last model steering
        self.last_model_steering = steering_angle
        
        # Execute steering command
        self._execute_steering(steering_angle)
        
        # Draw steering indicator on frame
        self._draw_model_steering_indicator(frame, steering_angle)
        
        # Update processed frame
        with self.camera_controller.frame_lock:
            self.camera_controller.processed_frame = frame
    
    def _process_hybrid_navigation(self, frame):
        """Process frame for hybrid navigation (combining lane detection and behavior cloning)"""
        # Get steering from lane detection
        processed_frame, lane_steering, lane_confidence = self.lane_detector.detect_lanes(frame)
        
        # Get steering from behavior cloning
        model_steering = self.behavior_cloner.predict_steering(frame)
        
        # Store last steering values
        self.last_lane_steering = lane_steering
        self.last_model_steering = model_steering
        
        # Combine steering angles based on weights and confidence
        # Adjust lane weight based on confidence
        effective_lane_weight = self.lane_weight * lane_confidence
        
        # Normalize weights
        total_weight = effective_lane_weight + self.model_weight
        if total_weight > 0:
            norm_lane_weight = effective_lane_weight / total_weight
            norm_model_weight = self.model_weight / total_weight
        else:
            norm_lane_weight = 0.5
            norm_model_weight = 0.5
        
        # Calculate hybrid steering
        hybrid_steering = (norm_lane_weight * lane_steering + 
                          norm_model_weight * model_steering)
        
        # Store last hybrid steering
        self.last_hybrid_steering = hybrid_steering
        
        # Execute steering command
        self._execute_steering(hybrid_steering)
        
        # Draw hybrid steering indicator
        self._draw_hybrid_steering_indicator(processed_frame, lane_steering, model_steering, hybrid_steering)
        
        # Update processed frame
        with self.camera_controller.frame_lock:
            self.camera_controller.processed_frame = processed_frame
        
        # Collect training data if enabled
        if self.collecting_data:
            # We're collecting manual steering data, so we don't add samples during autonomous navigation
            pass
    
    def _execute_steering(self, steering_angle):
        """Execute steering command based on steering angle"""
        # Convert steering angle (-1 to 1) to robot commands
        if abs(steering_angle) < 0.1:
            # Go straight
            self.motor_controller.move("forward", self.speed)
        elif steering_angle < 0:
            # Turn left
            self.motor_controller.move("left", self.speed * abs(steering_angle))
        else:
            # Turn right
            self.motor_controller.move("right", self.speed * abs(steering_angle))
    
    def _draw_model_steering_indicator(self, frame, steering_angle):
        """Draw steering indicator for behavior cloning model"""
        height, width = frame.shape[:2]
        
        # Draw steering indicator at the bottom center
        center_x = width // 2
        center_y = height - 50
        
        # Draw background circle
        cv2.circle(frame, (center_x, center_y), 30, (0, 0, 0), -1)
        
        # Calculate indicator position based on steering angle
        indicator_length = 25
        indicator_x = center_x + int(steering_angle * indicator_length)
        
        # Draw steering line
        cv2.line(frame, (center_x, center_y), (indicator_x, center_y), (0, 165, 255), 5)
        
        # Draw model text
        cv2.putText(
            frame,
            "Model",
            (center_x - 25, center_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def _draw_hybrid_steering_indicator(self, frame, lane_steering, model_steering, hybrid_steering):
        """Draw steering indicator for hybrid navigation"""
        height, width = frame.shape[:2]
        
        # Draw steering indicator at the bottom center
        center_x = width // 2
        center_y = height - 50
        
        # Draw background circle
        cv2.circle(frame, (center_x, center_y), 30, (0, 0, 0), -1)
        
        # Calculate indicator positions based on steering angles
        indicator_length = 25
        lane_x = center_x + int(lane_steering * indicator_length)
        model_x = center_x + int(model_steering * indicator_length)
        hybrid_x = center_x + int(hybrid_steering * indicator_length)
        
        # Draw steering lines
        cv2.line(frame, (center_x, center_y), (lane_x, center_y - 10), (0, 255, 0), 3)  # Lane in green
        cv2.line(frame, (center_x, center_y), (model_x, center_y), (0, 165, 255), 3)    # Model in orange
        cv2.line(frame, (center_x, center_y), (hybrid_x, center_y + 10), (255, 0, 0), 5)  # Hybrid in blue
        
        # Draw legend
        cv2.putText(
            frame,
            "Hybrid",
            (center_x - 60, center_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def record_manual_steering(self, frame, steering_command, steering_value):
        """Record manual steering for training"""
        if not self.collecting_data:
            return False
            
        # Convert steering command to angle (-1 to 1)
        steering_angle = 0.0
        
        if steering_command == "left":
            steering_angle = -steering_value
        elif steering_command == "right":
            steering_angle = steering_value
        
        # Get lane detection steering for comparison
        _, lane_steering, _ = self.lane_detector.detect_lanes(frame)
        
        # Add sample to behavior cloner
        self.behavior_cloner.add_sample(frame, steering_angle, lane_steering)
        
        return True
    
    def get_navigation_info(self):
        """Get information about the current navigation state"""
        return {
            "active": self.running,
            "mode": self.mode.name,
            "data_collection": self.collecting_data,
            "buffer_size": self.behavior_cloner.get_buffer_size(),
            "speed": self.speed,
            "lane_weight": self.lane_weight,
            "model_weight": self.model_weight,
            "last_lane_steering": self.last_lane_steering,
            "last_model_steering": self.last_model_steering,
            "last_hybrid_steering": self.last_hybrid_steering
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_navigation()
        # Save any collected data before exiting
        if self.collecting_data and self.behavior_cloner.get_buffer_size() > 0:
            self.behavior_cloner.save_data()
        logger.info("Hybrid navigator cleaned up")

