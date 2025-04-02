import cv2
import numpy as np
import logging
import time
import math

logger = logging.getLogger("LaneDetection")

class LaneDetector:
    """Detects lanes using classical computer vision techniques"""
    
    def __init__(self):
        # Lane detection parameters
        self.roi_height = 0.6  # Region of interest height (percentage from bottom)
        self.min_line_length = 50
        self.max_line_gap = 50
        self.hough_threshold = 20
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.rho = 1
        self.theta = np.pi/180
        
        # Lane tracking
        self.left_lane = []
        self.right_lane = []
        self.lane_history_size = 5
        self.last_steering_angle = 0
        self.steering_smoothing = 0.8  # Higher value = more smoothing
        
        logger.info("Lane detector initialized")
    
    def detect_lanes(self, frame):
        """
        Detect lanes in the given frame
        Returns: (processed_frame, steering_angle, confidence)
        """
        try:
            # Make a copy of the frame for drawing
            processed_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # Define region of interest (bottom portion of the frame)
            roi_y = int(height * (1 - self.roi_height))
            roi = frame[roi_y:height, 0:width]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
            
            # Apply Hough Line Transform
            lines = cv2.HoughLinesP(
                edges,
                self.rho,
                self.theta,
                self.hough_threshold,
                np.array([]),
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            # Process detected lines
            left_lines = []
            right_lines = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate slope
                    if x2 - x1 == 0:  # Avoid division by zero
                        continue
                        
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Filter out horizontal lines
                    if abs(slope) < 0.3:
                        continue
                    
                    # Categorize lines as left or right based on slope
                    if slope < 0:  # Negative slope = left lane
                        left_lines.append(line[0])
                    else:  # Positive slope = right lane
                        right_lines.append(line[0])
            
            # Calculate average left and right lanes
            left_lane = self._average_lane(left_lines)
            right_lane = self._average_lane(right_lines)
            
            # Update lane history
            if left_lane is not None:
                self.left_lane.append(left_lane)
                if len(self.left_lane) > self.lane_history_size:
                    self.left_lane.pop(0)
            
            if right_lane is not None:
                self.right_lane.append(right_lane)
                if len(self.right_lane) > self.lane_history_size:
                    self.right_lane.pop(0)
            
            # Get smoothed lanes
            smoothed_left = self._get_smoothed_lane(self.left_lane)
            smoothed_right = self._get_smoothed_lane(self.right_lane)
            
            # Draw lanes on the frame
            if smoothed_left is not None:
                x1, y1, x2, y2 = smoothed_left
                cv2.line(processed_frame, (x1, y1 + roi_y), (x2, y2 + roi_y), (0, 255, 0), 2)
            
            if smoothed_right is not None:
                x1, y1, x2, y2 = smoothed_right
                cv2.line(processed_frame, (x1, y1 + roi_y), (x2, y2 + roi_y), (0, 255, 0), 2)
            
            # Calculate steering angle
            steering_angle, confidence = self._calculate_steering_angle(
                smoothed_left, smoothed_right, width, height
            )
            
            # Draw steering indicator
            self._draw_steering_indicator(processed_frame, steering_angle, confidence)
            
            # Draw ROI
            cv2.rectangle(processed_frame, (0, roi_y), (width, height), (0, 0, 255), 2)
            
            return processed_frame, steering_angle, confidence
            
        except Exception as e:
            logger.error(f"Error in lane detection: {e}")
            return frame, 0.0, 0.0
    
    def _average_lane(self, lines):
        """Calculate average lane from multiple detected lines"""
        if not lines:
            return None
            
        x1_sum = 0
        y1_sum = 0
        x2_sum = 0
        y2_sum = 0
        
        for line in lines:
            x1, y1, x2, y2 = line
            x1_sum += x1
            y1_sum += y1
            x2_sum += x2
            y2_sum += y2
        
        count = len(lines)
        return [
            int(x1_sum / count),
            int(y1_sum / count),
            int(x2_sum / count),
            int(y2_sum / count)
        ]
    
    def _get_smoothed_lane(self, lane_history):
        """Get smoothed lane from history"""
        if not lane_history:
            return None
            
        x1_sum = 0
        y1_sum = 0
        x2_sum = 0
        y2_sum = 0
        
        for lane in lane_history:
            x1, y1, x2, y2 = lane
            x1_sum += x1
            y1_sum += y1
            x2_sum += x2
            y2_sum += y2
        
        count = len(lane_history)
        return [
            int(x1_sum / count),
            int(y1_sum / count),
            int(x2_sum / count),
            int(y2_sum / count)
        ]
    
    def _calculate_steering_angle(self, left_lane, right_lane, width, height):
        """
        Calculate steering angle based on detected lanes
        Returns: (steering_angle, confidence)
        steering_angle: -1.0 (full left) to 1.0 (full right), 0.0 is center
        confidence: 0.0 to 1.0
        """
        # Default to center with low confidence if no lanes detected
        if left_lane is None and right_lane is None:
            return 0.0, 0.0
        
        # Calculate center point of the frame
        center_x = width // 2
        
        if left_lane is not None and right_lane is not None:
            # Both lanes detected - calculate center of lane
            left_x2 = left_lane[2]
            right_x2 = right_lane[2]
            lane_center = (left_x2 + right_x2) // 2
            
            # Calculate offset from center
            offset = lane_center - center_x
            
            # Calculate steering angle (-1 to 1)
            # Normalize by half width to get value between -1 and 1
            raw_angle = offset / (width / 2)
            
            # Limit to range [-1, 1]
            raw_angle = max(-1.0, min(1.0, raw_angle))
            
            # Apply smoothing
            steering_angle = (self.steering_smoothing * self.last_steering_angle + 
                             (1 - self.steering_smoothing) * raw_angle)
            
            self.last_steering_angle = steering_angle
            
            # High confidence when both lanes detected
            return steering_angle, 0.9
            
        elif left_lane is not None:
            # Only left lane detected - estimate position
            left_x2 = left_lane[2]
            
            # Estimate lane width (assume standard lane width)
            estimated_lane_width = width // 3
            
            # Estimate lane center
            lane_center = left_x2 + estimated_lane_width
            
            # Calculate offset from center
            offset = lane_center - center_x
            
            # Calculate steering angle (-1 to 1)
            raw_angle = offset / (width / 2)
            raw_angle = max(-1.0, min(1.0, raw_angle))
            
            # Apply smoothing
            steering_angle = (self.steering_smoothing * self.last_steering_angle + 
                             (1 - self.steering_smoothing) * raw_angle)
            
            self.last_steering_angle = steering_angle
            
            # Medium confidence with only one lane
            return steering_angle, 0.6
            
        else:  # right_lane is not None
            # Only right lane detected - estimate position
            right_x2 = right_lane[2]
            
            # Estimate lane width
            estimated_lane_width = width // 3
            
            # Estimate lane center
            lane_center = right_x2 - estimated_lane_width
            
            # Calculate offset from center
            offset = lane_center - center_x
            
            # Calculate steering angle (-1 to 1)
            raw_angle = offset / (width / 2)
            raw_angle = max(-1.0, min(1.0, raw_angle))
            
            # Apply smoothing
            steering_angle = (self.steering_smoothing * self.last_steering_angle + 
                             (1 - self.steering_smoothing) * raw_angle)
            
            self.last_steering_angle = steering_angle
            
            # Medium confidence with only one lane
            return steering_angle, 0.6
    
    def _draw_steering_indicator(self, frame, steering_angle, confidence):
        """Draw steering indicator on the frame"""
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
        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.line(frame, (center_x, center_y), (indicator_x, center_y), color, 5)
        
        # Draw confidence text
        cv2.putText(
            frame,
            f"Conf: {confidence:.2f}",
            (center_x - 40, center_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

