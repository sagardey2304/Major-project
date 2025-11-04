"""
Communication Agent - Proper movement and distance change communication
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
    Communication Agent that properly reports movement and distance changes
    """

    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__("CommunicationAgent", message_bus, config)

        # Configuration
        self.tts_enabled = config.get('tts_enabled', True) and TTS_AVAILABLE
        self.speech_rate = config.get('speech_rate', 160)
        self.movement_report_interval = config.get('movement_report_interval', 2.0)
        
        # State tracking
        self.last_reported_distance = None
        self.last_reported_direction = None
        self.last_movement_report = 0
        self.last_object_state = ""
        self.current_instruction = ""
        self.closest_object = None
        self.display_lock = threading.Lock()

        # TTS engine
        self.tts_engine = None
        if self.tts_enabled:
            self._initialize_tts()
            
        # Message queues
        self.tts_queue = queue.Queue()
        self.urgent_queue = queue.Queue()

        # Subscribe to messages
        self.message_bus.subscribe(MessageType.NAVIGATION_UPDATE, self.handle_navigation_message)
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_system_message)
        self.message_bus.subscribe(MessageType.OBSTACLE_ALERT, self.handle_obstacle_message)

    def _initialize_tts(self):
        """Initialize TTS"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self.speech_rate)
            print(f"[{self.name}] TTS initialized for movement reporting")
        except Exception as e:
            print(f"[{self.name}] TTS failed: {e}")
            self.tts_enabled = False

    def _run(self):
        """Main communication loop with movement reporting"""
        print(f"[{self.name}] Starting communication with movement reporting...")
        
        while self._running:
            try:
                current_time = time.time()
                
                # Process urgent messages first
                if not self.urgent_queue.empty():
                    try:
                        message = self.urgent_queue.get_nowait()
                        self._speak(message)
                    except queue.Empty:
                        pass
                
                # Process regular messages
                if not self.tts_queue.empty():
                    try:
                        message = self.tts_queue.get_nowait()
                        self._speak(message)
                    except queue.Empty:
                        pass
                
                # Continuous movement and distance reporting
                self._report_movement_and_changes(current_time)
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                time.sleep(0.1)
        
        print(f"[{self.name}] Communication loop stopped")

    def handle_navigation_message(self, message: Message):
        """Handle navigation instructions"""
        instruction = message.data.get('instruction', '')
        priority = message.priority
        
        if instruction and instruction != self.current_instruction:
            print(f"[{self.name}] 🎯 {instruction}")
            
            with self.display_lock:
                self.current_instruction = instruction
            
            if priority >= 3:
                self.urgent_queue.put(instruction)
            else:
                self.tts_queue.put(instruction)

    def handle_system_message(self, message: Message):
        """Handle system status with movement information"""
        if message.type == MessageType.SYSTEM_STATUS:
            data = message.data
            if data.get('status') == 'perception_update':
                with self.display_lock:
                    self.closest_object = data.get('closest_object')
                    
                    # Check for movement in the environment
                    if data.get('movement_detected', False):
                        print(f"[{self.name}] Movement detected in scene")

    def handle_obstacle_message(self, message: Message):
        """Handle obstacle alerts with movement context"""
        obstacle_data = message.data
        distance = obstacle_data.get('distance', 0)
        direction = obstacle_data.get('direction', 'ahead')
        is_moving = obstacle_data.get('is_moving', False)
        movement_type = obstacle_data.get('movement_type', 'stationary')
        
        # Create context-aware alert message
        if is_moving:
            if movement_type == 'moving':
                alert_msg = f"Moving object {direction} at {distance} meters"
            elif movement_type == 'changing_distance':
                if distance < self.last_reported_distance if self.last_reported_distance else distance:
                    alert_msg = f"Object approaching {direction} now at {distance} meters"
                else:
                    alert_msg = f"Object moving away {direction} now at {distance} meters"
            else:
                alert_msg = f"Object {direction} at {distance} meters"
        else:
            alert_msg = f"Stationary object {direction} at {distance} meters"
        
        self.urgent_queue.put(alert_msg)

    def _report_movement_and_changes(self, current_time: float):
        """Report movement and distance changes"""
        with self.display_lock:
            closest_obj = self.closest_object
        
        if not closest_obj:
            # No object detected
            if self.last_object_state != "clear":
                message = "Path clear, no objects detected"
                self.tts_queue.put(message)
                self.last_object_state = "clear"
                self.last_reported_distance = None
                self.last_reported_direction = None
            return
        
        distance = closest_obj.get('distance', 0)
        direction = closest_obj.get('direction', 'ahead')
        is_moving = closest_obj.get('is_moving', False)
        distance_changed = closest_obj.get('distance_changed', False)
        direction_changed = closest_obj.get('direction_changed', False)
        
        # Check if we should report (time-based or change-based)
        should_report = False
        report_message = ""
        
        # Report based on changes
        if distance_changed or direction_changed:
            if is_moving:
                if distance < (self.last_reported_distance or 10):
                    report_message = f"Object moving closer {direction}, now {distance} meters"
                else:
                    report_message = f"Object moving away {direction}, now {distance} meters"
            else:
                report_message = f"Object {direction} at {distance} meters"
            should_report = True
        
        # Report moving objects periodically
        elif is_moving and (current_time - self.last_movement_report >= self.movement_report_interval):
            report_message = f"Moving object {direction} at {distance} meters"
            should_report = True
            self.last_movement_report = current_time
        
        # Report significant distance changes
        elif (self.last_reported_distance is not None and 
              abs(distance - self.last_reported_distance) > 0.5):
            if distance < self.last_reported_distance:
                report_message = f"Object getting closer, now {distance} meters"
            else:
                report_message = f"Object moving away, now {distance} meters"
            should_report = True
        
        # Report if this is a new object state
        current_state = f"{direction}_{distance:.1f}"
        if current_state != self.last_object_state:
            if not should_report:  # Only report if not already reporting
                report_message = f"Object {direction} at {distance} meters"
                should_report = True
        
        # Speak the report if needed
        if should_report and report_message:
            self.tts_queue.put(report_message)
            print(f"[{self.name}] 📢 {report_message}")
            
            # Update tracking state
            self.last_reported_distance = distance
            self.last_reported_direction = direction
            self.last_object_state = current_state

    def _speak(self, message: str):
        """Speak message"""
        if not self.tts_enabled or not self.tts_engine:
            return
        
        try:
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[{self.name}] TTS error: {e}")

    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Create display with movement information"""
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Draw detections with movement info
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                
                # Color based on movement
                if detection.get('is_moving', False):
                    color = (0, 165, 255)  # Orange for moving objects
                    thickness = 3
                elif detection.get('is_closest', False):
                    color = (0, 0, 255)  # Red for closest
                    thickness = 3
                else:
                    color = (0, 255, 0)  # Green for stationary
                    thickness = 1
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                
                # Label with distance and movement info
                distance = detection.get('distance', 0)
                label = f"{distance:.1f}m"
                if detection.get('is_moving', False):
                    label += " MOVING"
                
                cv2.putText(display_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw UI
        self._draw_movement_ui(display_frame)
        
        return display_frame

    def _draw_movement_ui(self, frame: np.ndarray):
        """Draw movement-aware user interface"""
        height, width = frame.shape[:2]
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
        
        with self.display_lock:
            closest_obj = self.closest_object
            instruction = self.current_instruction
        
        # Status text
        if closest_obj:
            distance = closest_obj.get('distance', 0)
            direction = closest_obj.get('direction', 'ahead')
            is_moving = closest_obj.get('is_moving', False)
            
            if is_moving:
                status_text = f"MOVING OBJECT {direction} at {distance}m"
                color = (0, 165, 255)  # Orange
            else:
                status_text = f"OBJECT {direction} at {distance}m"
                color = (0, 0, 255)  # Red
        else:
            status_text = "NO OBJECTS DETECTED"
            color = (0, 255, 0)  # Green
        
        cv2.putText(frame, status_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # FPS
        fps = self._get_fps()
        cv2.putText(frame, f"FPS: {fps}", (width - 80, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Current instruction
        if instruction:
            cv2.rectangle(frame, (0, height-40), (width, height), (0, 0, 0), -1)
            cv2.putText(frame, instruction, (10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _get_fps(self) -> int:
        """Get FPS"""
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
        if priority >= 3:
            self.urgent_queue.put(message)
        else:
            self.tts_queue.put(message)
