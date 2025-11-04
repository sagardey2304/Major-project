"""
Perception Agent - Handles object detection, depth estimation, and environmental sensing
Using calibrated camera for accurate distance estimation
"""
import cv2
import numpy as np
import time
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from .base_agent import BaseAgent, Message, MessageType
from ultralytics import YOLO

@dataclass
class Detection:
    """Represents a detected object with its properties"""
    object_class: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # cx, cy
    distance_meters: float
    warning_level: str  # "critical", "warning", "caution", "safe"
    direction: str  # "left", "center", "right"

class PerceptionAgent(BaseAgent):
    """Agent responsible for environmental perception and object detection"""
    
    def __init__(self, message_bus, config: Dict[str, Any] = None):
        # Ensure config is passed to parent (fixing the error)
        config = config or {}
        super().__init__("PerceptionAgent", message_bus, config)
        
        # Configuration - EXACTLY THE SAME as first code
        self.config = config
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.camera_index = self.config.get('camera_index', 0)  # Default to 0
        self.frame_width = self.config.get('frame_width', 640)
        self.frame_height = self.config.get('frame_height', 480)
        
        # Load calibration data - EXACTLY THE SAME
        self.camera_matrix = None
        self.dist_coeffs = None
        self._load_calibration_data()
        
        # Known widths for different object classes (in meters) - EXACTLY THE SAME
        self.known_widths = {
            'person': 0.45,      # Shoulder width
            'car': 1.8,          # Car width
            'bicycle': 0.7,      # Bicycle width
            'motorcycle': 0.8,   # Motorcycle width
            'bus': 2.5,          # Bus width
            'truck': 2.5,        # Truck width
            'chair': 0.5,        # Chair width
            'default': 0.5       # Default width for unknown objects
        }
        
        # Initialize camera first (try multiple indices if needed) - EXACTLY THE SAME
        self.cap = None
        self._init_camera()
        
        # Initialize models only if camera is working - EXACTLY THE SAME STRUCTURE
        if self.cap and self.cap.isOpened():
            self._init_models()
        else:
            print(f"[{self.name}] Camera initialization failed, skipping model initialization")
            self.send_message(MessageType.SYSTEM_STATUS, {
                'status': 'error',
                'message': 'Camera initialization failed'
            }, priority=3)
            return
        
        # Distance-based warning thresholds (in meters) - EXACTLY THE SAME
        self.distance_thresholds = {
            'critical': 1.5,    # Immediate danger
            'warning': 3.0,     # Warning zone
            'caution': 5.0      # Caution zone
        }
        
        # Detection tracking for real-time validation - EXACTLY THE SAME
        self.recent_detections = {}
        self.detection_timeout = 1.0  # Objects expire after 1 second
        
        # Frame storage for visualization - EXACTLY THE SAME
        self.current_frame = None
        
        # Subscribe to messages - EXACTLY THE SAME
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_message)
        
    def _load_calibration_data(self):
        """Load camera calibration data - IMPROVED PATH HANDLING"""
        # Try multiple possible locations for the calibration file
        possible_paths = [
            "calib_data.npz",  # Current directory
            "./calib_data.npz",  # Current directory explicitly
            "models/calib_data.npz",  # models subdirectory
            "../calib_data.npz",  # Parent directory
        ]
        
        calibration_file = None
        for path in possible_paths:
            if os.path.exists(path):
                calibration_file = path
                break
        
        if calibration_file:
            try:
                data = np.load(calibration_file)
                self.camera_matrix = data["camera_matrix"]
                self.dist_coeffs = data["dist_coeffs"]
                print(f"[{self.name}] Camera calibration data loaded successfully from {calibration_file}")
                print(f"[{self.name}] Camera matrix: {self.camera_matrix}")
                print(f"[{self.name}] Distortion coefficients: {self.dist_coeffs}")
                return
            except Exception as e:
                print(f"[{self.name}] Error loading calibration from {calibration_file}: {e}")
        
        # If we get here, no calibration file was found or loaded successfully
        print(f"[{self.name}] No calibration file found in expected locations, using default calibration")
        self._create_default_calibration()
    
    def _create_default_calibration(self):
        """Create default camera calibration parameters"""
        # Default camera matrix for a typical webcam
        # Assuming focal length of ~800 pixels for 640x480 resolution
        fx = 800.0  # focal length in x direction
        fy = 800.0  # focal length in y direction
        cx = self.frame_width / 2.0   # principal point x
        cy = self.frame_height / 2.0  # principal point y
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Default distortion coefficients (usually small for modern cameras)
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        print(f"[{self.name}] Default calibration created:")
        print(f"[{self.name}]   Focal length: {fx}")
        print(f"[{self.name}]   Principal point: ({cx}, {cy})")
            
    def _init_models(self):
        """Initialize computer vision models - USING ULTRALYTICS YOLO"""
        try:
            # Load YOLOv8s model using ultralytics (SAME AS ORIGINAL CODE)
            self.yolo_model = YOLO("yolov5n.pt")
            print(f"[{self.name}] YOLOv8s model initialized successfully with ultralytics")
            
            # COCO class names - EXACTLY THE SAME as PyTorch version
            self.coco_classes = self.yolo_model.names
            
        except Exception as e:
            print(f"[{self.name}] Error initializing models: {e}")
            self.send_message(MessageType.SYSTEM_STATUS, {
                'status': 'error',
                'message': f'Model initialization failed: {e}'
            }, priority=3)
            
    def _init_camera(self):
        """Initialize camera capture - try multiple indices if needed - EXACTLY THE SAME"""
        max_camera_tries = 3  # Try up to 3 different camera indices
        
        for camera_index in range(max_camera_tries):
            try:
                print(f"[{self.name}] Trying camera index {camera_index}...")
                self.cap = cv2.VideoCapture(camera_index)
                
                if not self.cap.isOpened():
                    print(f"[{self.name}] Camera index {camera_index} not available")
                    if self.cap:
                        self.cap.release()
                    continue
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                
                # Test if camera actually works by reading a frame
                ret, test_frame = self.cap.read()
                if not ret:
                    print(f"[{self.name}] Camera index {camera_index} opened but failed to read frame")
                    self.cap.release()
                    continue
                
                print(f"[{self.name}] Camera initialized successfully at index {camera_index}")
                self.camera_index = camera_index  # Update to the working index
                return
                
            except Exception as e:
                print(f"[{self.name}] Error with camera index {camera_index}: {e}")
                if self.cap:
                    self.cap.release()
        
        # If we get here, no camera worked
        print(f"[{self.name}] Failed to initialize any camera after trying {max_camera_tries} indices")
        self.cap = None
            
    def _estimate_distance(self, object_class: str, bbox_width: int) -> float:
        """Estimate distance using calibrated camera and known object widths - EXACTLY THE SAME"""
        if bbox_width <= 0:
            return float('inf')
            
        # Get known width for this object class
        known_width = self.known_widths.get(object_class, self.known_widths['default'])
        
        # Use focal length from camera matrix (fx)
        focal_length = self.camera_matrix[0, 0]
        
        # Calculate distance using similar triangles: distance = (known_width * focal_length) / perceived_width
        distance = (known_width * focal_length) / bbox_width
        
        return distance
            
    def _get_warning_level(self, distance_meters):
        """Determine warning level based on distance - EXACTLY THE SAME"""
        if distance_meters <= self.distance_thresholds['critical']:
            return "critical"
        elif distance_meters <= self.distance_thresholds['warning']:
            return "warning"
        elif distance_meters <= self.distance_thresholds['caution']:
            return "caution"
        else:
            return "safe"
        
    def _get_direction(self, cx, frame_width):
        """Determine direction based on center x coordinate - EXACTLY THE SAME"""
        left_threshold = frame_width * 0.25  # 25% from left
        right_threshold = frame_width * 0.75  # 75% from left
        
        if cx < left_threshold:
            direction = "left"
        elif cx > right_threshold:
            direction = "right"
        else:
            direction = "center"
            
        return direction
            
    def _validate_real_time_detections(self, detections: List[Detection], current_time: float) -> List[Detection]:
        """Validate detections to ensure they are current and not stale - EXACTLY THE SAME"""
        valid_detections = []
        current_detection_keys = set()
        
        for detection in detections:
            # Create a unique key for each detection based on class and approximate position
            detection_key = f"{detection.object_class}_{detection.center[0]//50}_{detection.center[1]//50}"
            current_detection_keys.add(detection_key)
            
            # Check if this detection is consistent with recent frames
            if detection_key in self.recent_detections:
                # Update existing detection
                self.recent_detections[detection_key] = {
                    'detection': detection,
                    'timestamp': current_time,
                    'count': self.recent_detections[detection_key].get('count', 0) + 1
                }
                
                # Only include detections that have been seen multiple times or are critical
                if (self.recent_detections[detection_key]['count'] >= 2 or 
                    detection.warning_level in ['critical', 'warning']):
                    valid_detections.append(detection)
            else:
                # New detection - add to tracking but only include if critical
                self.recent_detections[detection_key] = {
                    'detection': detection,
                    'timestamp': current_time,
                    'count': 1
                }
                
                # Immediately include critical detections for safety
                if detection.warning_level == 'critical':
                    valid_detections.append(detection)
        
        # Clean up old detections
        expired_keys = [
            key for key, data in self.recent_detections.items() 
            if current_time - data['timestamp'] > self.detection_timeout
        ]
        
        for key in expired_keys:
            del self.recent_detections[key]
            
        return valid_detections
            
    def _detect_objects(self, frame):
        """Perform object detection on frame using calibrated distance estimation - EXACTLY THE SAME AS ORIGINAL"""
        detections = []
        
        try:
            # Undistort frame using calibration data
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # YOLO detection (SAME AS ORIGINAL CODE)
            results = self.yolo_model(undistorted_frame, stream=True)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box
                    cls = int(box.cls[0])  # class id
                    conf = float(box.conf[0])  # confidence
                    
                    if conf < self.confidence_threshold:
                        continue
                        
                    label = self.coco_classes[cls]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    # Calculate bbox width for distance estimation
                    bbox_width = x2 - x1
                    
                    # Estimate distance using calibrated camera (same as your test code)
                    distance_meters = self._estimate_distance(label, bbox_width)
                    
                    # Determine direction
                    direction = self._get_direction(cx, frame.shape[1])
                    
                    # Determine warning level based on distance (same logic as your test code)
                    warning_level = self._get_warning_level(distance_meters)
                    
                    detection = Detection(
                        object_class=label,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        distance_meters=distance_meters,
                        warning_level=warning_level,
                        direction=direction
                    )
                    
                    detections.append(detection)
                
        except Exception as e:
            print(f"[{self.name}] Object detection error: {e}")
            
        return detections
        
    def _run(self):
        """Main perception loop - EXACTLY THE SAME"""
        print(f"[{self.name}] Starting perception loop")
        
        # Check if camera is available
        if not self.cap or not self.cap.isOpened():
            print(f"[{self.name}] No camera available, exiting perception loop")
            self.send_message(MessageType.SYSTEM_STATUS, {
                'status': 'error',
                'message': 'No camera available'
            }, priority=3)
            return
        
        while self._running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"[{self.name}] Failed to read frame from camera {self.camera_index}")
                    time.sleep(0.5)  # Longer delay if camera fails
                    continue
                
                # Store current frame for visualization
                self.current_frame = frame.copy()
                
                # Detect objects
                detections = self._detect_objects(frame)
                
                # Validate detections in real-time
                current_time = time.time()
                valid_detections = self._validate_real_time_detections(detections, current_time)
                
                # Filter detections by warning level for alerts - EXACTLY THE SAME
                critical_detections = [det for det in valid_detections if det.warning_level == "critical"]
                warning_detections = [det for det in valid_detections if det.warning_level == "warning"]
                caution_detections = [det for det in valid_detections if det.warning_level == "caution"]
                
                # Send alerts based on warning level - EXACTLY THE SAME
                for detection in critical_detections:
                    self.send_message(MessageType.OBSTACLE_ALERT, {
                        'object': detection.object_class,
                        'direction': detection.direction,
                        'distance': detection.distance_meters,
                        'warning_level': detection.warning_level,
                        'confidence': detection.confidence,
                        'bbox': detection.bbox,
                        'center': detection.center,
                        'alert_type': 'CRITICAL - IMMEDIATE DANGER'
                    }, priority=3)
                
                for detection in warning_detections:
                    self.send_message(MessageType.OBSTACLE_ALERT, {
                        'object': detection.object_class,
                        'direction': detection.direction,
                        'distance': detection.distance_meters,
                        'warning_level': detection.warning_level,
                        'confidence': detection.confidence,
                        'bbox': detection.bbox,
                        'center': detection.center,
                        'alert_type': 'WARNING - APPROACHING OBSTACLE'
                    }, priority=2)
                
                for detection in caution_detections:
                    self.send_message(MessageType.OBSTACLE_ALERT, {
                        'object': detection.object_class,
                        'direction': detection.direction,
                        'distance': detection.distance_meters,
                        'warning_level': detection.warning_level,
                        'confidence': detection.confidence,
                        'bbox': detection.bbox,
                        'center': detection.center,
                        'alert_type': 'CAUTION - OBJECT DETECTED'
                    }, priority=1)
                
                # Send all detections for navigation processing - EXACTLY THE SAME
                self.send_message(MessageType.SYSTEM_STATUS, {
                    'status': 'perception_update',
                    'detections': [
                        {
                            'object': det.object_class,
                            'direction': det.direction,
                            'distance': det.distance_meters,
                            'warning_level': det.warning_level,
                            'confidence': det.confidence,
                            'bbox': det.bbox,
                            'center': det.center
                        } for det in detections
                    ],
                    'frame_shape': frame.shape,
                    'detection_summary': {
                        'critical': len(critical_detections),
                        'warning': len(warning_detections),
                        'caution': len(caution_detections),
                        'safe': len([det for det in detections if det.warning_level == "safe"])
                    }
                }, priority=1)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[{self.name}] Error in perception loop: {e}")
                time.sleep(0.1)  # Delay on error
                
        # Clean up
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        print(f"[{self.name}] Perception loop stopped")
        
    def handle_message(self, message: Message):
        """Handle incoming messages - FIXED: using message.type"""
        if message.type == MessageType.SYSTEM_STATUS:
            if message.data.get('command') == 'stop':
                self.stop()
                
    def get_current_frame(self):
        """Get current camera frame for visualization - EXACTLY THE SAME"""
        return self.current_frame
        
    def get_current_depth_map(self):
        """Return a simple depth map for compatibility (not used in calibrated mode) - EXACTLY THE SAME"""
        # Return a dummy depth map since we're using calibrated distance estimation
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            return np.ones((h, w), dtype=np.float32)
        return None