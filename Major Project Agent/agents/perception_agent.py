"""
Perception Agent - Handles object detection, depth estimation, and environmental sensing
Using calibrated camera for accurate distance estimation
"""
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from .base_agent import BaseAgent, Message, MessageType

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
        super().__init__("PerceptionAgent", message_bus)
        
        # Configuration
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.camera_index = self.config.get('camera_index', 1)  # Default to 0
        self.frame_width = self.config.get('frame_width', 640)
        self.frame_height = self.config.get('frame_height', 480)
        
        # Load calibration data
        self.camera_matrix = None
        self.dist_coeffs = None
        self._load_calibration_data()
        
        # Known widths for different object classes (in meters)
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
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize camera first (try multiple indices if needed)
        self.cap = None
        self._init_camera()
        
        # Initialize models only if camera is working
        if self.cap and self.cap.isOpened():
            self._init_models()
        else:
            print(f"[{self.agent_name}] Camera initialization failed, skipping model initialization")
            self.send_message(MessageType.SYSTEM_STATUS, {
                'status': 'error',
                'message': 'Camera initialization failed'
            }, priority=3)
            return
        
        # Distance-based warning thresholds (in meters)
        self.distance_thresholds = {
            'critical': 1.0,    # Immediate danger
            'warning': 2.0,     # Warning zone
            'caution': 3.0      # Caution zone
        }
        
        # Detection tracking for real-time validation
        self.recent_detections = {}
        self.detection_timeout = 1.0  # Objects expire after 1 second
        
        # Frame storage for visualization
        self.current_frame = None
        
        # Subscribe to messages
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_message)
        
    def _load_calibration_data(self):
        """Load camera calibration data"""
        try:
            data = np.load("calib_data.npz")
            self.camera_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]
            print(f"[{self.agent_name}] Camera calibration data loaded successfully")
        except Exception as e:
            print(f"[{self.agent_name}] Error loading calibration data: {e}")
            # Fallback to default values if calibration fails
            self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
            self.dist_coeffs = np.zeros(5)
            
    def _init_models(self):
        """Initialize computer vision models"""
        try:
            # Load YOLOv8 model (using nano version for speed as in your test code)
            self.yolo_model = YOLO("yolov8n.pt")
            self.yolo_model.to(self.device).eval()
            self.yolo_model.fuse()
            self.coco_classes = self.yolo_model.names
            
            print(f"[{self.agent_name}] YOLO model initialized successfully")
            
        except Exception as e:
            print(f"[{self.agent_name}] Error initializing models: {e}")
            self.send_message(MessageType.SYSTEM_STATUS, {
                'status': 'error',
                'message': f'Model initialization failed: {e}'
            }, priority=3)
            
    def _init_camera(self):
        """Initialize camera capture - try multiple indices if needed"""
        max_camera_tries = 3  # Try up to 3 different camera indices
        
        for camera_index in range(1,max_camera_tries):
            try:
                print(f"[{self.agent_name}] Trying camera index {camera_index}...")
                self.cap = cv2.VideoCapture(camera_index)
                
                if not self.cap.isOpened():
                    print(f"[{self.agent_name}] Camera index {camera_index} not available")
                    if self.cap:
                        self.cap.release()
                    continue
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                
                # Test if camera actually works by reading a frame
                ret, test_frame = self.cap.read()
                if not ret:
                    print(f"[{self.agent_name}] Camera index {camera_index} opened but failed to read frame")
                    self.cap.release()
                    continue
                
                print(f"[{self.agent_name}] Camera initialized successfully at index {camera_index}")
                self.camera_index = camera_index  # Update to the working index
                return
                
            except Exception as e:
                print(f"[{self.agent_name}] Error with camera index {camera_index}: {e}")
                if self.cap:
                    self.cap.release()
        
        # If we get here, no camera worked
        print(f"[{self.agent_name}] Failed to initialize any camera after trying {max_camera_tries} indices")
        self.cap = None
            
    def _estimate_distance(self, object_class: str, bbox_width: int) -> float:
        """Estimate distance using calibrated camera and known object widths"""
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
        """Determine warning level based on distance"""
        if distance_meters <= self.distance_thresholds['critical']:
            return "critical"
        elif distance_meters <= self.distance_thresholds['warning']:
            return "warning"
        elif distance_meters <= self.distance_thresholds['caution']:
            return "caution"
        else:
            return "safe"
        
    def _get_direction(self, cx, frame_width):
        """Determine direction based on center x coordinate"""
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
        """Validate detections to ensure they are current and not stale"""
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
        """Perform object detection on frame using calibrated distance estimation"""
        detections = []
        
        try:
            # Undistort frame using calibration data
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # YOLO detection (same as your test code)
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
            print(f"[{self.agent_name}] Object detection error: {e}")
            
        return detections
        
    def _run(self):
        """Main perception loop"""
        print(f"[{self.agent_name}] Starting perception loop")
        
        # Check if camera is available
        if not self.cap or not self.cap.isOpened():
            print(f"[{self.agent_name}] No camera available, exiting perception loop")
            self.send_message(MessageType.SYSTEM_STATUS, {
                'status': 'error',
                'message': 'No camera available'
            }, priority=3)
            return
        
        while self._running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"[{self.agent_name}] Failed to read frame from camera {self.camera_index}")
                    time.sleep(0.5)  # Longer delay if camera fails
                    continue
                
                # Store current frame for visualization
                self.current_frame = frame.copy()
                
                # Detect objects
                detections = self._detect_objects(frame)
                
                # Validate detections in real-time
                current_time = time.time()
                valid_detections = self._validate_real_time_detections(detections, current_time)
                
                # Filter detections by warning level for alerts
                critical_detections = [det for det in valid_detections if det.warning_level == "critical"]
                warning_detections = [det for det in valid_detections if det.warning_level == "warning"]
                caution_detections = [det for det in valid_detections if det.warning_level == "caution"]
                
                # Send alerts based on warning level
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
                    }, priority=4)
                
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
                    }, priority=3)
                
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
                    }, priority=2)
                
                # Send all detections for navigation processing
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
                print(f"[{self.agent_name}] Error in perception loop: {e}")
                time.sleep(0.1)  # Delay on error
                
        # Clean up
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        print(f"[{self.agent_name}] Perception loop stopped")
        
    def handle_message(self, message: Message):
        """Handle incoming messages"""
        if message.msg_type == MessageType.SYSTEM_STATUS:
            if message.data.get('command') == 'stop':
                self.stop()
                
    def get_current_frame(self):
        """Get current camera frame for visualization"""
        return self.current_frame
        
    def get_current_depth_map(self):
        """Return a simple depth map for compatibility (not used in calibrated mode)"""
        # Return a dummy depth map since we're using calibrated distance estimation
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            return np.ones((h, w), dtype=np.float32)
        return None