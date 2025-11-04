"""
Perception Agent - Detect both stationary and moving objects with robust tracking
"""

import cv2
import numpy as np
import time
import os
import threading
from typing import Dict, Any, List

from .base_agent import BaseAgent, MessageBus, MessageType

class PerceptionAgent(BaseAgent):
    """
    Perception Agent that detects both stationary and moving objects
    using multiple detection methods
    """

    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__("PerceptionAgent", message_bus, config)

        # Configuration
        self.camera_index = config.get('camera_index', 0)
        self.frame_width = config.get('frame_width', 640)
        self.frame_height = config.get('frame_height', 480)
        
        # Detection parameters
        self.detection_interval = config.get('detection_interval', 2.0)  # Full detection every 2 seconds
        self.motion_interval = config.get('motion_interval', 0.1)  # Motion detection every 0.1 seconds
        
        # Camera calibration
        self.focal_length = 700
        self.known_widths = {
            'person': 0.5, 'chair': 0.5, 'table': 0.8, 'backpack': 0.3, 
            'bottle': 0.08, 'box': 0.4, 'default': 0.4
        }

        # Object tracking
        self.tracked_objects = {}
        self.object_id_counter = 0
        self.tracking_lock = threading.Lock()
        
        # Detection methods
        self.background_subtractor = None
        self.previous_frame = None
        self.full_detection_time = 0

        # Initialize components
        self.camera = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Performance
        self.fps = 0
        self.frame_count = 0
        self.fps_timer = time.time()

    def _initialize_camera(self):
        """Initialize camera"""
        print(f"[{self.name}] Initializing camera...")
        
        for cam_index in [self.camera_index, 0, 1, 2]:
            try:
                cap = cv2.VideoCapture(cam_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                
                # Test camera
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"[{self.name}] Camera {cam_index} initialized")
                        self.camera = cap
                        # Initialize with first frame
                        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        return True
                    time.sleep(0.1)
                
                cap.release()
            except Exception as e:
                print(f"[{self.name}] Camera {cam_index} failed: {e}")
        
        return False

    def _detect_stationary_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect stationary objects using edge detection and contour analysis"""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # Edge detection for object boundaries
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate edges to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Filter by size - stationary objects are usually larger
                if 1000 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate solidity to filter noise
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    if solidity > 0.3:  # Reasonably solid object
                        distance = self._calculate_accurate_distance(w, 'default')
                        
                        # Only consider objects within reasonable distance
                        if distance <= 8.0:
                            detection = self._create_detection(
                                x, y, w, h, distance, 'stationary', 0.6
                            )
                            detections.append(detection)
                            
        except Exception as e:
            print(f"[{self.name}] Stationary detection error: {e}")
            
        return detections

    def _detect_moving_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect moving objects using frame difference and background subtraction"""
        detections = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Frame difference (fast)
            if self.previous_frame is not None:
                frame_diff = cv2.absdiff(self.previous_frame, gray)
                _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                
                # Clean up threshold
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                # Find contours in frame difference
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 500 < area < 30000:  # Moving objects can be smaller
                        x, y, w, h = cv2.boundingRect(contour)
                        distance = self._calculate_accurate_distance(w, 'person')
                        
                        if distance <= 8.0:
                            detection = self._create_detection(
                                x, y, w, h, distance, 'moving', 0.7
                            )
                            detections.append(detection)
            
            # Method 2: Background subtraction (more robust)
            if self.background_subtractor is None:
                self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=50, varThreshold=16, detectShadows=True
                )
            
            fg_mask = self.background_subtractor.apply(frame)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            
            bg_contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in bg_contours:
                area = cv2.contourArea(contour)
                if 800 < area < 40000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if this is different from frame difference detections
                    overlapping = False
                    for existing in detections:
                        ex, ey, ew, eh = existing['bbox']
                        # Check for overlap
                        if (abs(x - ex) < 50 and abs(y - ey) < 50):
                            overlapping = True
                            break
                    
                    if not overlapping:
                        distance = self._calculate_accurate_distance(w, 'person')
                        if distance <= 8.0:
                            detection = self._create_detection(
                                x, y, w, h, distance, 'moving', 0.8
                            )
                            detections.append(detection)
            
            # Update previous frame
            self.previous_frame = gray.copy()
            
        except Exception as e:
            print(f"[{self.name}] Motion detection error: {e}")
            
        return detections

    def _create_detection(self, x: int, y: int, w: int, h: int, 
                         distance: float, obj_type: str, confidence: float) -> Dict[str, Any]:
        """Create a standardized detection object"""
        center_x = x + w // 2
        frame_center = self.frame_width // 2
        
        # Determine direction
        if center_x < frame_center - 100:
            direction = 'left'
        elif center_x > frame_center + 100:
            direction = 'right'
        else:
            direction = 'center'
        
        # Check if in navigation path
        path_left = frame_center - 150
        path_right = frame_center + 150
        in_path = path_left <= center_x <= path_right
        
        # Determine warning level
        if distance <= 1.5:
            warning_level = 'critical'
            priority = 4
        elif distance <= 3.0:
            warning_level = 'warning'
            priority = 3
        elif distance <= 5.0:
            warning_level = 'caution'
            priority = 2
        else:
            warning_level = 'safe'
            priority = 1
        
        return {
            'class_name': obj_type,
            'confidence': confidence,
            'bbox': [x, y, w, h],
            'center': [center_x, y + h // 2],
            'distance': round(distance, 1),
            'direction': direction,
            'in_path': in_path,
            'warning_level': warning_level,
            'priority': priority,
            'color': [0, 255, 0],
            'detection_type': obj_type,
            'timestamp': time.time()
        }

    def _calculate_accurate_distance(self, bbox_width: int, object_type: str = 'default') -> float:
        """Calculate accurate distance to object"""
        known_width = self.known_widths.get(object_type, self.known_widths['default'])
        
        if bbox_width <= 0:
            return 10.0
            
        distance = (known_width * self.focal_length) / bbox_width
        return max(0.3, min(distance, 10.0))

    def _track_objects(self, new_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Track objects across frames and identify the closest one"""
        current_time = time.time()
        tracked_detections = []
        
        with self.tracking_lock:
            # Mark all existing objects as not seen
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]['seen'] = False
            
            # Match new detections with existing tracked objects
            for detection in new_detections:
                center_x, center_y = detection['center']
                matched = False
                
                # Try to match with existing objects
                for obj_id, tracked_obj in self.tracked_objects.items():
                    if current_time - tracked_obj['last_seen'] < 3.0:  # Only recent objects
                        tracked_center = tracked_obj['center']
                        distance = np.sqrt((center_x - tracked_center[0])**2 + 
                                         (center_y - tracked_center[1])**2)
                        
                        # If close enough, consider it the same object
                        if distance < 50:  # pixels
                            # Update tracked object
                            tracked_obj.update({
                                'center': detection['center'],
                                'bbox': detection['bbox'],
                                'distance': detection['distance'],
                                'direction': detection['direction'],
                                'in_path': detection['in_path'],
                                'warning_level': detection['warning_level'],
                                'priority': detection['priority'],
                                'last_seen': current_time,
                                'seen': True,
                                'confidence': max(tracked_obj['confidence'], detection['confidence'])
                            })
                            detection['tracking_id'] = obj_id
                            matched = True
                            break
                
                # If no match, create new tracked object
                if not matched:
                    new_id = self.object_id_counter
                    self.object_id_counter += 1
                    self.tracked_objects[new_id] = {
                        'center': detection['center'],
                        'bbox': detection['bbox'],
                        'distance': detection['distance'],
                        'direction': detection['direction'],
                        'in_path': detection['in_path'],
                        'warning_level': detection['warning_level'],
                        'priority': detection['priority'],
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'seen': True,
                        'confidence': detection['confidence']
                    }
                    detection['tracking_id'] = new_id
            
            # Remove old objects that haven't been seen
            objects_to_remove = []
            for obj_id, tracked_obj in self.tracked_objects.items():
                if not tracked_obj['seen'] and current_time - tracked_obj['last_seen'] > 5.0:
                    objects_to_remove.append(obj_id)
            
            for obj_id in objects_to_remove:
                del self.tracked_objects[obj_id]
            
            # Create final detections list with tracking info
            for detection in new_detections:
                obj_id = detection.get('tracking_id')
                if obj_id in self.tracked_objects:
                    tracked_obj = self.tracked_objects[obj_id]
                    # Add tracking confidence (how long it's been tracked)
                    track_duration = current_time - tracked_obj['first_seen']
                    tracking_confidence = min(1.0, track_duration / 10.0)  # Max confidence after 10 seconds
                    
                    detection['tracking_confidence'] = tracking_confidence
                    detection['tracking_id'] = obj_id
                    
                    # If tracked for a while, increase confidence
                    if tracking_confidence > 0.5:
                        detection['confidence'] = max(detection['confidence'], 0.8)
                    
                    tracked_detections.append(detection)
        
        return tracked_detections

    def _get_closest_object(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify the closest object from all detections"""
        if not detections:
            return None
        
        # Filter to only objects in path or very close
        relevant_objects = [d for d in detections if d['in_path'] or d['distance'] <= 3.0]
        
        if not relevant_objects:
            # If no objects in path, consider the closest object overall
            relevant_objects = [min(detections, key=lambda x: x['distance'])]
        
        # Find the closest relevant object
        closest = min(relevant_objects, key=lambda x: x['distance'])
        closest['is_closest'] = True
        closest['color'] = [0, 0, 255]  # Red for closest
        
        return closest

    def _run(self):
        """Main perception loop with combined detection"""
        print(f"[{self.name}] Starting perception loop - Detecting ALL objects...")
        
        if not self._initialize_camera():
            print(f"[{self.name}] Camera initialization failed!")
            return
            
        last_full_detection = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Store frame
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Always run motion detection (fast)
                motion_detections = self._detect_moving_objects(frame)
                
                # Run full stationary detection less frequently
                all_detections = motion_detections.copy()
                if current_time - last_full_detection >= self.detection_interval:
                    stationary_detections = self._detect_stationary_objects(frame)
                    all_detections.extend(stationary_detections)
                    last_full_detection = current_time
                
                # Track objects across frames
                tracked_detections = self._track_objects(all_detections)
                
                # Identify closest object
                closest_object = self._get_closest_object(tracked_detections)
                
                # Send updates
                self.send_message(
                    MessageType.SYSTEM_STATUS,
                    {
                        'status': 'perception_update',
                        'detections': tracked_detections,
                        'closest_object': closest_object,
                        'total_objects': len(tracked_detections),
                        'fps': self.fps
                    },
                    priority=2
                )
                
                # Send immediate alert for closest object
                if closest_object and closest_object['distance'] <= 5.0:
                    self.send_message(
                        MessageType.OBSTACLE_ALERT,
                        closest_object,
                        priority=closest_object['priority']
                    )
                
                # Update FPS
                self._update_fps()
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                time.sleep(0.1)
        
        if self.camera:
            self.camera.release()
        print(f"[{self.name}] Perception loop stopped")

    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        if time.time() - self.fps_timer >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_timer = time.time()

    def get_current_frame(self) -> np.ndarray:
        """Get current frame"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_fps(self) -> int:
        return self.fps