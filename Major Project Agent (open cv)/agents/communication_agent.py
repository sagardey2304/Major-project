"""
Communication Agent - Highlight closest object with special bounding box
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Dict, Any, List

from .base_agent import BaseAgent, MessageBus, MessageType, Message

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class CommunicationAgent(BaseAgent):
    """
    Communication Agent that highlights the closest object
    """

    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__("CommunicationAgent", message_bus, config)

        # Configuration
        self.tts_enabled = config.get('tts_enabled', True) and TTS_AVAILABLE
        self.speech_rate = config.get('speech_rate', 150)
        
        # TTS engine
        self.tts_engine = None
        if self.tts_enabled:
            self._initialize_tts()
            
        # Message queue
        self.tts_queue = queue.Queue()
        self.last_announcement = ""
        
        # Current instruction and closest object
        self.current_instruction = ""
        self.closest_object = None
        self.display_lock = threading.Lock()

        # Subscribe to messages
        self.message_bus.subscribe(MessageType.NAVIGATION_UPDATE, self.handle_message)
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_system_message)

    def _initialize_tts(self):
        """Initialize TTS"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self.speech_rate)
            print(f"[{self.name}] TTS initialized")
        except Exception as e:
            print(f"[{self.name}] TTS failed: {e}")
            self.tts_enabled = False

    def _run(self):
        """Main communication loop"""
        print(f"[{self.name}] Communication loop started - Highlighting closest object")
        
        while self._running:
            try:
                # Process TTS queue
                if self.tts_enabled:
                    try:
                        message = self.tts_queue.get(timeout=0.1)
                        self._speak(message)
                    except queue.Empty:
                        pass
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                time.sleep(0.1)
        
        print(f"[{self.name}] Communication loop stopped")

    def handle_message(self, message: Message):
        """Handle navigation updates"""
        if message.type == MessageType.NAVIGATION_UPDATE:
            instruction = message.data.get('instruction', '')
            priority = message.priority
            
            if instruction and instruction != self.last_announcement:
                print(f"[{self.name}] 🎯 {instruction}")
                
                # Update current instruction
                with self.display_lock:
                    self.current_instruction = instruction
                
                # Speak important instructions
                if priority >= 2:
                    self._queue_tts(instruction)
                
                self.last_announcement = instruction

    def handle_system_message(self, message: Message):
        """Handle system status for closest object"""
        if message.type == MessageType.SYSTEM_STATUS:
            data = message.data
            if data.get('status') == 'perception_update':
                with self.display_lock:
                    self.closest_object = data.get('closest_object')

    def _queue_tts(self, message: str):
        """Queue message for TTS"""
        if self.tts_enabled:
            self.tts_queue.put(message)

    def _speak(self, message: str):
        """Speak message"""
        try:
            if self.tts_engine:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[{self.name}] TTS error: {e}")

    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Create display with highlighted closest object"""
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Draw all detection boxes
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                
                # Color based on whether it's the closest object
                if detection.get('is_closest', False):
                    # HIGHLIGHT closest object with thick red box
                    color = (0, 0, 255)  # Bright red
                    thickness = 4
                    
                    # Draw highlighted bounding box
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                    
                    # Draw distance label with background
                    label = f"CLOSEST: {detection.get('distance', 0):.1f}m"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(display_frame, (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), color, -1)
                    cv2.putText(display_frame, label, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Draw direction arrow
                    direction = detection.get('direction', 'center')
                    self._draw_direction_indicator(display_frame, x + w//2, y, direction)
                    
                else:
                    # Regular objects with thinner boxes
                    color_map = {
                        'critical': (0, 0, 255),
                        'warning': (0, 165, 255),  
                        'caution': (0, 255, 255),
                        'safe': (0, 255, 0)
                    }
                    color = color_map.get(detection.get('warning_level', 'safe'), (0, 255, 0))
                    thickness = 2
                    
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                    
                    # Simple distance label for regular objects
                    label = f"{detection.get('distance', 0):.1f}m"
                    cv2.putText(display_frame, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw navigation path
        center_x = width // 2
        path_left = center_x - 150
        path_right = center_x + 150
        cv2.rectangle(display_frame, (path_left, 0), (path_right, height), (255, 255, 255), 2)
        
        # Draw UI elements
        self._draw_distance_zones(display_frame)
        self._draw_instruction(display_frame)
        self._draw_closest_object_info(display_frame)
        
        # Draw FPS
        cv2.putText(display_frame, f"FPS: {self._get_fps()}", (width - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display_frame

    def _draw_direction_indicator(self, frame: np.ndarray, x: int, y: int, direction: str):
        """Draw direction indicator for closest object"""
        if direction == 'left':
            # Left arrow
            points = np.array([[x, y-20], [x-15, y], [x, y+20]], np.int32)
            cv2.fillPoly(frame, [points], (0, 0, 255))
        elif direction == 'right':
            # Right arrow  
            points = np.array([[x, y-20], [x+15, y], [x, y+20]], np.int32)
            cv2.fillPoly(frame, [points], (0, 0, 255))
        else:
            # Center - circle
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

    def _draw_closest_object_info(self, frame: np.ndarray):
        """Display information about the closest object"""
        with self.display_lock:
            closest_obj = self.closest_object
        
        if closest_obj:
            distance = closest_obj.get('distance', 0)
            direction = closest_obj.get('direction', 'unknown')
            in_path = closest_obj.get('in_path', False)
            
            info_text = f"Closest: {distance}m {direction}"
            if in_path:
                info_text += " (IN PATH)"
            
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _draw_distance_zones(self, frame: np.ndarray):
        """Draw distance zones"""
        zones = [
            ('🚨 CRITICAL <1.5m', (0, 0, 255)),
            ('⚠️ WARNING <3.0m', (0, 165, 255)),
            ('🟡 CAUTION <5.0m', (0, 255, 255)),
            ('✅ SAFE >5.0m', (0, 255, 0))
        ]
        
        for i, (text, color) in enumerate(zones):
            y = 60 + i * 25
            cv2.putText(frame, text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_instruction(self, frame: np.ndarray):
        """Draw current navigation instruction"""
        with self.display_lock:
            instruction = self.current_instruction
        
        if instruction:
            # Draw instruction background
            y_start = frame.shape[0] - 100
            cv2.rectangle(frame, (0, y_start), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            
            # Draw instruction text (split if too long)
            words = instruction.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + word) < 40:
                    current_line += " " + word
                else:
                    lines.append(current_line.strip())
                    current_line = word
            if current_line:
                lines.append(current_line.strip())
            
            for i, line in enumerate(lines):
                y = y_start + 30 + i * 25
                cv2.putText(frame, line, (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _get_fps(self) -> int:
        """Simple FPS counter"""
        if not hasattr(self, 'fps_counter'):
            self.fps_counter = 0
            self.fps_timer = time.time()
            self.last_fps = 0
            
        self.fps_counter += 1
        if time.time() - self.fps_timer >= 1.0:
            self.last_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = time.time()
            
        return self.last_fps

    def send_user_message(self, message: str, priority: int = 1):
        """Send user message"""
        self.send_message(
            MessageType.USER_COMMUNICATION,
            {'message': message},
            priority=priority
        )