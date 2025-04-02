import numpy as np
import cv2
import os
import time
import logging
import pickle
import threading
from collections import deque
from pathlib import Path

logger = logging.getLogger("BehaviorCloning")

class SimpleModel:
    """Simple model for behavior cloning"""
    
    def __init__(self, input_shape=(60, 80, 1)):
        self.input_shape = input_shape
        self.weights = None
        self.bias = 0
        
        # Initialize with simple weights (will be updated during training)
        self.weights = np.zeros((np.prod(input_shape),))
        
        logger.info(f"Simple model initialized with input shape {input_shape}")
    
    def predict(self, image):
        """
        Predict steering angle from image
        Returns: steering_angle (-1.0 to 1.0)
        """
        if self.weights is None:
            return 0.0
            
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Flatten image
        flattened = processed.flatten()
        
        # Simple linear model: y = w*x + b
        steering = np.dot(flattened, self.weights) + self.bias
        
        # Clip to range [-1, 1]
        steering = max(-1.0, min(1.0, steering))
        
        return steering
    
    def train(self, images, steering_angles, epochs=5, learning_rate=0.001):
        """Train the model on collected data"""
        if not images or len(images) != len(steering_angles):
            logger.error("Invalid training data")
            return False
            
        logger.info(f"Training model on {len(images)} samples for {epochs} epochs")
        
        # Initialize weights if not already done
        if self.weights is None:
            sample = self._preprocess_image(images[0])
            self.weights = np.zeros((np.prod(sample.shape),))
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(images)):
                # Preprocess image
                processed = self._preprocess_image(images[i])
                
                # Flatten image
                flattened = processed.flatten()
                
                # Current prediction
                prediction = np.dot(flattened, self.weights) + self.bias
                
                # Calculate error
                error = steering_angles[i] - prediction
                total_loss += error ** 2
                
                # Update weights (gradient descent)
                self.weights += learning_rate * error * flattened
                self.bias += learning_rate * error
            
            # Calculate mean squared error
            mse = total_loss / len(images)
            logger.info(f"Epoch {epoch+1}/{epochs}, MSE: {mse:.6f}")
        
        logger.info("Training completed")
        return True
    
    def _preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize to input shape
        resized = cv2.resize(gray, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize pixel values to [0, 1]
        normalized = resized / 255.0
        
        return normalized
    
    def save(self, filepath):
        """Save model to file"""
        data = {
            'weights': self.weights,
            'bias': self.bias,
            'input_shape': self.input_shape
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Model saved to {filepath}")
        return True
    
    def load(self, filepath):
        """Load model from file"""
        if not os.path.exists(filepath):
            logger.error(f"Model file {filepath} not found")
            return False
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.weights = data['weights']
        self.bias = data['bias']
        self.input_shape = data['input_shape']
        
        logger.info(f"Model loaded from {filepath}")
        return True


class BehaviorCloner:
    """Handles behavior cloning for autonomous navigation"""
    
    def __init__(self, save_dir="behavior_cloning_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Data collection
        self.data_buffer = deque(maxlen=5000)  # Store recent experiences
        self.collecting_data = False
        self.lock = threading.Lock()
        
        # Model
        self.model = SimpleModel()
        self.model_file = self.save_dir / "steering_model.pkl"
        
        # Load model if exists
        if self.model_file.exists():
            self.model.load(str(self.model_file))
        
        logger.info("Behavior cloner initialized")
    
    def start_collecting(self):
        """Start collecting training data"""
        with self.lock:
            self.collecting_data = True
        logger.info("Started collecting training data")
        return True
    
    def stop_collecting(self):
        """Stop collecting training data"""
        with self.lock:
            self.collecting_data = False
        logger.info("Stopped collecting training data")
        return True
    
    def add_sample(self, frame, steering_angle, lane_steering=None):
        """Add a training sample"""
        if not self.collecting_data:
            return False
            
        with self.lock:
            # Store frame and steering angle
            self.data_buffer.append({
                'frame': frame.copy(),
                'steering': steering_angle,
                'lane_steering': lane_steering,
                'timestamp': time.time()
            })
            
        return True
    
    def predict_steering(self, frame):
        """Predict steering angle from current frame"""
        return self.model.predict(frame)
    
    def train_model(self, epochs=10):
        """Train model on collected data"""
        with self.lock:
            if len(self.data_buffer) < 10:
                logger.warning("Not enough training data")
                return False
                
            # Extract training data
            frames = []
            steering_angles = []
            
            for sample in self.data_buffer:
                frames.append(sample['frame'])
                steering_angles.append(sample['steering'])
            
            # Train model
            success = self.model.train(frames, steering_angles, epochs=epochs)
            
            if success:
                # Save model
                self.model.save(str(self.model_file))
                
            return success
    
    def save_data(self):
        """Save collected data to disk"""
        with self.lock:
            if not self.data_buffer:
                logger.warning("No data to save")
                return False
                
            # Create filename with timestamp
            filename = f"training_data_{int(time.time())}.pkl"
            filepath = self.save_dir / filename
            
            # Save data
            with open(filepath, 'wb') as f:
                pickle.dump(list(self.data_buffer), f)
                
            logger.info(f"Saved {len(self.data_buffer)} training samples to {filepath}")
            return True
    
    def load_data(self, filepath):
        """Load training data from file"""
        if not os.path.exists(filepath):
            logger.error(f"Data file {filepath} not found")
            return False
            
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)
            
        with self.lock:
            # Clear current buffer and add loaded data
            self.data_buffer.clear()
            for sample in loaded_data:
                self.data_buffer.append(sample)
                
        logger.info(f"Loaded {len(self.data_buffer)} training samples from {filepath}")
        return True
    
    def get_buffer_size(self):
        """Get the current size of the data buffer"""
        with self.lock:
            return len(self.data_buffer)
    
    def clear_buffer(self):
        """Clear the data buffer"""
        with self.lock:
            self.data_buffer.clear()
        logger.info("Data buffer cleared")
        return True

