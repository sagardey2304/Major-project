"""
Perception Agent - Optimized for Raspberry Pi with YOLOv5nu.onnx
Uses ONNX runtime for efficient inference
"""
import cv2
import numpy as np
import time
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from .base_agent import BaseAgent, Message, MessageType

# Conditional import for ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available, using OpenCV fallback")

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
    """Agent responsible for environmental perception - Pi Optimized with ONNX"""
    
    def __init__(self, message_bus, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("PerceptionAgent", message_bus, config)
        
        # Pi-optimized configuration
        self.config = config
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.camera_index = self.config.get('camera_index', 0)
        self.frame_width = self.config.get('frame_width', 320)
        self.frame_height = self.config.get('frame_height', 240)
        self.processing_interval = self.config.get('processing_interval', 0.3)
        
        # Load calibration data
        self.camera_matrix = None
        self.dist_coeffs = None
        self._load_calibration_data()
        
        # Known widths for objects
        self.known_widths = {
            'person': 0.45,
            'bicycle': 0.7,
            'car': 1.8,
            'motorcycle': 0.8,
            'bus': 2.5,
            'truck': 2.5,
            'dog': 0.4,
            'cow': 1.2,
            'backpack': 0.3,
            'sports ball': 0.22,
            'baseball bat': 0.08,
            'skateboard': 0.2,
            'tennis racket': 0.3,
            'bottle': 0.08,
            'knife': 0.02,
            'potted plant': 0.3,
            'bed': 1.5,
            'dining table': 1.0,
            'toilet': 0.4,
            'tv': 0.8,
            'laptop': 0.3,
            'cell phone': 0.15,
            'microwave': 0.5,
            'oven': 0.6,
            'toaster': 0.3,
            'sink': 0.5,
            'refrigerator': 0.8,
            'vase': 0.2,
            'hair drier': 0.15,
            'default': 0.5
        }
        
        # Initialize camera
        self.cap = None
        self._init_camera_pi()
        
        # Initialize ONNX model
        if self.cap and self.cap.isOpened():
            self._init_onnx_model()
        else:
            print(f"[{self.name}] Camera initialization failed")
            self.send_message(MessageType.SYSTEM_STATUS, {
                'status': 'error',
                'message': 'Camera initialization failed'
            }, priority=3)
            return
        
        # Distance thresholds
        self.distance_thresholds = {
            'critical': 1.5,
            'warning': 3.0,
            'caution': 5.0
        }
        
        # Detection tracking
        self.recent_detections = {}
        self.detection_timeout = 2.0
        self.last_processing_time = 0
        self.current_frame = None
        
        # Subscribe to messages
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_message)
        
        print(f"[{self.name}] Raspberry Pi Perception Agent (ONNX) initialized")
        
    def _load_calibration_data(self):
        """Load camera calibration data"""
        calibration_file = "calib_data.npz"
        
        if os.path.exists(calibration_file):
            try:
                data = np.load(calibration_file)
                self.camera_matrix = data["camera_matrix"]
                self.dist_coeffs = data["dist_coeffs"]
                print(f"[{self.name}] Calibration loaded")
            except Exception as e:
                print(f"[{self.name}] Error loading calibration: {e}")
                self._create_pi_calibration()
        else:
            self._create_pi_calibration()
    
    def _create_pi_calibration(self):
        """Create Pi-optimized calibration"""
        fx = 400.0
        fy = 400.0
        cx = self.frame_width / 2.0
        cy = self.frame_height / 2.0
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        print(f"[{self.name}] Pi calibration created")
            
    def _init_camera_pi(self):
        """Initialize camera with Pi-specific optimizations"""
        max_camera_tries = 2
        
        for camera_index in range(max_camera_tries):
            try:
                print(f"[{self.name}] Trying camera index {camera_index}...")
                
                self.cap = cv2.VideoCapture(camera_index)
                
                if not self.cap.isOpened():
                    print(f"[{self.name}] Camera index {camera_index} not available")
                    if self.cap:
                        self.cap.release()
                    continue
                
                # Pi-optimized settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test camera
                for _ in range(3):
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"[{self.name}] Camera {camera_index} initialized: {test_frame.shape}")
                        self.camera_index = camera_index
                        return
                    time.sleep(0.1)
                
                self.cap.release()
                
            except Exception as e:
                print(f"[{self.name}] Error with camera {camera_index}: {e}")
                if self.cap:
                    self.cap.release()
        
        print(f"[{self.name}] No camera found")
        self.cap = None

    def _init_onnx_model(self):
        """Initialize YOLOv5nu ONNX model"""
        try:
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX Runtime not available")
            
            # Path to ONNX model
            onnx_model_path = "models/yolov5nu.onnx"
            
            if not os.path.exists(onnx_model_path):
                print(f"[{self.name}] ONNX model not found at {onnx_model_path}")
                self._download_onnx_model()
            
            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']  # Use CPU on Pi
            self.session = ort.InferenceSession(onnx_model_path, providers=providers)
            
            # Get model info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            # YOLOv5 COCO class names
            self.coco_classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
            
            self.model_type = "yolov5nu_onnx"
            print(f"[{self.name}] YOLOv5nu ONNX model loaded successfully")
            print(f"[{self.name}] Input shape: {self.input_shape}")
            
        except Exception as e:
            print(f"[{self.name}] ONNX model failed: {e}")
            # Fallback to OpenCV DNN
            self._init_opencv_fallback()

    def _download_onnx_model(self):
        """Download YOLOv5nu ONNX model if not present"""
        import urllib.request
        model_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5nu.onnx"
        model_path = "models/yolov5nu.onnx"
        
        os.makedirs("models", exist_ok=True)
        print(f"[{self.name}] Downloading YOLOv5nu ONNX model...")
        
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(f"[{self.name}] Model downloaded to {model_path}")
        except Exception as e:
            print(f"[{self.name}] Failed to download model: {e}")
            raise e

    def _init_opencv_fallback(self):
        """Fallback to OpenCV DNN if ONNX fails"""
        try:
            # Try to load MobileNet SSD
            model_path = "models/ssd_mobilenet_v2.caffemodel"
            config_path = "models/ssd_mobilenet_v2.prototxt"
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                self.coco_classes = ['background'] + [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'
                ]
                self.model_type = "ssd_mobilenet"
                print(f"[{self.name}] OpenCV DNN fallback loaded")
            else:
                raise FileNotFoundError("OpenCV model files not found")
                
        except Exception as e:
            print(f"[{self.name}] OpenCV fallback failed: {e}")
            # Ultimate fallback
            self.model_type = "motion"
            self.previous_frame = None
            print(f"[{self.name}] Using basic motion detection")

    def _preprocess_onnx(self, frame):
        """Preprocess frame for ONNX inference"""
        # Resize to model input size (usually 640x640 for YOLOv5)
        input_size = (640, 640)
        resized = cv2.resize(frame, input_size)
        
        # Normalize and convert to RGB
        input_array = resized.astype(np.float32) / 255.0
        input_array = input_array[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
        input_array = np.expand_dims(input_array, axis=0)  # Add batch dimension
        
        return input_array

    def _postprocess_onnx(self, outputs, original_shape):
        """Postprocess ONNX outputs to detections"""
        detections = []
        
        try:
            # YOLOv5 ONNX output format: [batch, num_detections, 85]
            # 85 = [x, y, w, h, conf, class_conf_0, class_conf_1, ... class_conf_79]
            outputs = outputs[0]  # Remove batch dimension
            
            for detection in outputs:
                # Filter by confidence
                confidence = detection[4]
                if confidence < self.confidence_threshold:
                    continue
                
                # Get class with highest confidence
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                # Combined confidence
                total_confidence = confidence * class_confidence
                if total_confidence < self.confidence_threshold:
                    continue
                
                # Get bounding box (x, y, w, h in normalized coordinates)
                x_center, y_center, width, height = detection[0], detection[1], detection[2], detection[3]
                
                # Convert to pixel coordinates
                orig_h, orig_w = original_shape[:2]
                x1 = int((x_center - width/2) * orig_w)
                y1 = int((y_center - height/2) * orig_h)
                x2 = int((x_center + width/2) * orig_w)
                y2 = int((y_center + height/2) * orig_h)
                
                # Ensure coordinates are within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                
                label = self.coco_classes[class_id]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                bbox_width = x2 - x1
                
                # Estimate distance
                distance_meters = self._estimate_distance(label, bbox_width)
                direction = self._get_direction(cx, orig_w)
                warning_level = self._get_warning_level(distance_meters)
                
                detection_obj = Detection(
                    object_class=label,
                    confidence=float(total_confidence),
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    distance_meters=distance_meters,
                    warning_level=warning_level,
                    direction=direction
                )
                
                detections.append(detection_obj)
                
        except Exception as e:
            print(f"[{self.name}] ONNX postprocessing error: {e}")
            
        return detections

    def _detect_objects_onnx(self, frame):
        """Object detection using YOLOv5nu ONNX"""
        detections = []
        
        try:
            # Preprocess
            input_array = self._preprocess_onnx(frame)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_array})
            
            # Postprocess
            detections = self._postprocess_onnx(outputs, frame.shape)
            
        except Exception as e:
            print(f"[{self.name}] ONNX detection error: {e}")
            
        return detections

    def _detect_objects_opencv(self, frame):
        """Fallback detection using OpenCV DNN"""
        detections = []
        
        try:
            blob = cv2.dnn.blobFromImage(
                frame, 0.007843, (300, 300), 127.5
            )
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            for detection in outputs[0, 0]:
                confidence = float(detection[2])
                class_id = int(detection[1])
                
                if confidence > self.confidence_threshold and class_id < len(self.coco_classes):
                    h, w = frame.shape[:2]
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    label = self.coco_classes[class_id]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    bbox_width = x2 - x1
                    
                    distance_meters = self._estimate_distance(label, bbox_width)
                    direction = self._get_direction(cx, frame.shape[1])
                    warning_level = self._get_warning_level(distance_meters)
                    
                    detection_obj = Detection(
                        object_class=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        distance_meters=distance_meters,
                        warning_level=warning_level,
                        direction=direction
                    )
                    
                    detections.append(detection_obj)
                    
        except Exception as e:
            print(f"[{self.name}] OpenCV detection error: {e}")
            
        return detections

    # Keep the same utility methods as before...
    def _estimate_distance(self, object_class: str, bbox_width: int) -> float:
        if bbox_width <= 0:
            return float('inf')
        known_width = self.known_widths.get(object_class, self.known_widths['default'])
        focal_length = self.camera_matrix[0, 0]
        return (known_width * focal_length) / bbox_width
            
    def _get_warning_level(self, distance_meters):
        if distance_meters <= self.distance_thresholds['critical']:
            return "critical"
        elif distance_meters <= self.distance_thresholds['warning']:
            return "warning"
        elif distance_meters <= self.distance_thresholds['caution']:
            return "caution"
        else:
            return "safe"
        
    def _get_direction(self, cx, frame_width):
        left_threshold = frame_width * 0.25
        right_threshold = frame_width * 0.75
        if cx < left_threshold:
            return "left"
        elif cx > right_threshold:
            return "right"
        else:
            return "center"

    def _detect_objects(self, frame):
        """Main detection method"""
        if self.model_type == "yolov5nu_onnx":
            return self._detect_objects_onnx(frame)
        elif self.model_type == "ssd_mobilenet":
            return self._detect_objects_opencv(frame)
        else:
            return []

    # Keep the same message handling and run loop as before...
    def _validate_real_time_detections(self, detections: List[Detection], current_time: float) -> List[Detection]:
        valid_detections = []
        current_detection_keys = set()
        
        for detection in detections:
            detection_key = f"{detection.object_class}_{detection.center[0]//50}_{detection.center[1]//50}"
            current_detection_keys.add(detection_key)
            
            if detection_key in self.recent_detections:
                self.recent_detections[detection_key] = {
                    'detection': detection,
                    'timestamp': current_time,
                    'count': self.recent_detections[detection_key].get('count', 0) + 1
                }
                if (self.recent_detections[detection_key]['count'] >= 2 or 
                    detection.warning_level in ['critical', 'warning']):
                    valid_detections.append(detection)
            else:
                self.recent_detections[detection_key] = {
                    'detection': detection,
                    'timestamp': current_time,
                    'count': 1
                }
                if detection.warning_level == 'critical':
                    valid_detections.append(detection)
        
        expired_keys = [
            key for key, data in self.recent_detections.items() 
            if current_time - data['timestamp'] > self.detection_timeout
        ]
        for key in expired_keys:
            del self.recent_detections[key]
            
        return valid_detections

    def _run(self):
        print(f"[{self.name}] Starting ONNX perception loop")
        
        if not self.cap or not self.cap.isOpened():
            print(f"[{self.name}] No camera available")
            return
        
        while self._running:
            try:
                current_time = time.time()
                if current_time - self.last_processing_time < self.processing_interval:
                    time.sleep(0.05)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                self.current_frame = frame.copy()
                self.last_processing_time = current_time
                
                detections = self._detect_objects(frame)
                valid_detections = self._validate_real_time_detections(detections, current_time)
                
                # Send messages (same as before)
                critical_detections = [det for det in valid_detections if det.warning_level == "critical"]
                warning_detections = [det for det in valid_detections if det.warning_level == "warning"]
                caution_detections = [det for det in valid_detections if det.warning_level == "caution"]
                
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
                
                # ... (similar for warning and caution)
                
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
                    },
                    'model_type': self.model_type
                }, priority=1)
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[{self.name}] Error in perception loop: {e}")
                time.sleep(0.1)
                
        if self.cap:
            self.cap.release()
        print(f"[{self.name}] Perception loop stopped")

    def handle_message(self, message: Message):
        if message.type == MessageType.SYSTEM_STATUS:
            if message.data.get('command') == 'stop':
                self.stop()
                
    def get_current_frame(self):
        return self.current_frame
        
    def get_current_depth_map(self):
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            return np.ones((h, w), dtype=np.float32)
        return None