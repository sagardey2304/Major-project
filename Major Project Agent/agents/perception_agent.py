"""
Perception Agent - Handles object detection, depth estimation, and environmental sensing
"""
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from ultralytics import YOLO
import torch.nn as nn
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
    depth: float
    distance_meters: float
    warning_level: str  # "critical", "warning", "caution", "safe"
    direction: str  # "left", "center", "right"

class ViTEnhancer(nn.Module):
    """Vision Transformer enhancer for feature improvement"""
    def __init__(self, in_channels=32, embed_dim=256, heads=8, num_layers=1):
        super().__init__()
        self.project = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.project(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x

class PerceptionAgent(BaseAgent):
    """Agent responsible for environmental perception and object detection"""
    
    def __init__(self, message_bus, config: Dict[str, Any] = None):
        super().__init__("PerceptionAgent", message_bus)
        
        # Configuration
        self.config = config or {}
        self.proximity_threshold = self.config.get('proximity_threshold', 2.0)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)  # Increased to reduce false positives
        self.camera_index = self.config.get('camera_index', 1)
        self.frame_width = self.config.get('frame_width', 640)
        self.frame_height = self.config.get('frame_height', 480)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models and camera
        self._init_models()
        self._init_camera()
        
        # Distance-based warning thresholds (in meters)
        self.distance_thresholds = self.config.get('distance_thresholds', {
            'critical': 1.5,
            'warning': 3.0,
            'caution': 5.0
        })
        
        # Initialize depth map storage
        self.current_depth_map = None
        
        # Initialize debug counter
        self._debug_counter = 0
        
        # Detection tracking
        self.recent_detections = {}
        self.detection_timeout = 1.0  # Objects expire after 1 second
        
        # Subscribe to messages
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_message)
        
    def _init_models(self):
        """Initialize computer vision models"""
        try:
            # Load YOLOv8 model
            self.yolo_model = YOLO("yolov8s.pt")
            self.yolo_model.to(self.device).eval()
            self.yolo_model.fuse()
            self.coco_classes = self.yolo_model.names
            
            # Load MiDaS depth estimation model
            self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(self.device).eval()
            self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            
            # Initialize ViT Enhancer
            self.enhancer = ViTEnhancer().to(self.device).eval()
            
            print(f"[{self.agent_name}] Models initialized successfully")
            
        except Exception as e:
            print(f"[{self.agent_name}] Error initializing models: {e}")
            self.send_message(MessageType.SYSTEM_STATUS, {
                'status': 'error',
                'message': f'Model initialization failed: {e}'
            }, priority=3)
            
    def _init_camera(self):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_index}")
                
            print(f"[{self.agent_name}] Camera initialized successfully")
            
        except Exception as e:
            print(f"[{self.agent_name}] Error initializing camera: {e}")
            self.send_message(MessageType.SYSTEM_STATUS, {
                'status': 'error',
                'message': f'Camera initialization failed: {e}'
            }, priority=3)
            
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
        left_threshold = frame_width * 0.25
        right_threshold = frame_width * 0.75
        
        if cx < left_threshold:
            direction = "left"
        elif cx > right_threshold:
            direction = "right"
        else:
            direction = "center"
            
        if hasattr(self, '_debug_counter') and self._debug_counter % 30 == 0:
            print(f"[DEBUG] Direction: cx={cx}, frame_width={frame_width}, -> {direction}")
            
        return direction
            
    def _estimate_depth(self, frame, bbox):
        """Estimate depth for a bounding box region"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.midas_transforms(rgb).to(self.device)
            
            with torch.no_grad():
                prediction = self.midas(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
                
            depth_map = prediction.cpu().numpy()
            self.current_depth_map = depth_map
            
            x1, y1, x2, y2 = bbox
            x1c, y1c = int(max(0, x1)), int(max(0, y1))
            x2c, y2c = int(min(frame.shape[1], x2)), int(min(frame.shape[0], y2))
            
            center_x, center_y = (x1c + x2c) // 2, (y1c + y2c) // 2
            region_size = min(20, (x2c - x1c) // 4, (y2c - y1c) // 4)
            
            cx1 = max(0, center_x - region_size)
            cy1 = max(0, center_y - region_size)
            cx2 = min(depth_map.shape[1], center_x + region_size)
            cy2 = min(depth_map.shape[0], center_y + region_size)
            
            depth_region = depth_map[cy1:cy2, cx1:cx2]
            if depth_region.size == 0:
                return 1.0, 8.0
                
            min_depth = np.min(depth_region)
            max_depth = np.max(depth_map)
            relative_depth = min_depth / (max_depth + 1e-8)
            
            if relative_depth < 0.1:
                distance_meters = 0.5
            elif relative_depth < 0.3:
                distance_meters = 1.0 + (relative_depth - 0.1) * 5
            elif relative_depth < 0.6:
                distance_meters = 2.0 + (relative_depth - 0.3) * 10
            else:
                distance_meters = 5.0 + (relative_depth - 0.6) * 25
            
            distance_meters = max(0.3, min(20.0, distance_meters))
            
            return relative_depth, distance_meters
            
        except Exception as e:
            print(f"[{self.agent_name}] Depth estimation error: {e}")
            return 1.0, 8.0
    
    def _detect_objects(self, frame):
        """Perform object detection on frame"""
        detections = []
        
        try:
            with torch.no_grad():
                results = self.yolo_model(frame)[0]
                
            for det in results.boxes.data:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                if conf < self.confidence_threshold:
                    continue
                    
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                label = self.coco_classes[int(cls)]
                
                avg_depth, distance_meters = self._estimate_depth(frame, (x1, y1, x2, y2))
                direction = self._get_direction(cx, frame.shape[1])
                warning_level = self._get_warning_level(distance_meters)
                
                detection = Detection(
                    object_class=label,
                    confidence=float(conf),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(cx, cy),
                    depth=avg_depth,
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
        
        while self._running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"[{self.agent_name}] Failed to read frame")
                    continue
                    
                self._debug_counter += 1
                detections = self._detect_objects(frame)
                
                current_time = time.time()
                
                # Send detections
                for detection in detections:
                    self.send_message(MessageType.OBSTACLE_ALERT, {
                        'object': detection.object_class,
                        'direction': detection.direction,
                        'distance': detection.distance_meters,
                        'warning_level': detection.warning_level,
                        'confidence': detection.confidence,
                        'bbox': detection.bbox,
                        'center': detection.center
                    }, priority=2 if detection.warning_level != "safe" else 1)
                
                self.send_message(MessageType.SYSTEM_STATUS, {
                    'status': 'perception_update',
                    'detections': [d.__dict__ for d in detections],
                    'frame_shape': frame.shape
                }, priority=1)
                
                cv2.waitKey(1)
                
            except Exception as e:
                print(f"[{self.agent_name}] Error in perception loop: {e}")
                
        if hasattr(self, 'cap'):
            self.cap.release()
        print(f"[{self.agent_name}] Perception loop stopped")
        
    def handle_message(self, message: Message) -> None:
        """Handle incoming messages from the message bus"""
        try:
            if message.msg_type == MessageType.SYSTEM_STATUS:
                cmd = message.data.get('command')
                if cmd == 'stop':
                    print(f"[{self.agent_name}] Received stop command")
                    self.stop()
            else:
                pass
        except Exception as e:
            print(f"[{self.agent_name}] Error handling message: {e}")
                
    def get_current_frame(self):
        """Get current camera frame for visualization"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def get_current_depth_map(self):
        """Get current depth map for visualization"""
        return getattr(self, 'current_depth_map', None)
