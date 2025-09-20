"""
Communication Agent - Handles TTS functionality and user interface
"""
import cv2
import numpy as np
import pyttsx3
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from queue import Queue, Empty

from .base_agent import BaseAgent, Message, MessageType

@dataclass
class AudioMessage:
    """Represents an audio message to be spoken"""
    text: str
    priority: int
    timestamp: float
    voice_settings: Optional[Dict[str, Any]] = None

@dataclass 
class VisualAlert:
    """Represents a visual alert to be displayed"""
    message: str
    alert_type: str  # "info", "warning", "critical", "success"
    duration: float
    position: Tuple[int, int]  # (x, y)
    timestamp: float
    spoken: bool = False  # Track if this alert has been spoken

class CommunicationAgent(BaseAgent):
    """Agent responsible for user communication via TTS and visual feedback"""
    
    def __init__(self, message_bus, config: Dict[str, Any] = None):
        super().__init__("CommunicationAgent", message_bus)
        
        # Configuration
        self.config = config or {}
        self.tts_enabled = self.config.get('tts_enabled', True)
        self.visual_enabled = self.config.get('visual_enabled', True)
        self.speech_rate = self.config.get('speech_rate', 160)
        self.speech_volume = self.config.get('speech_volume', 0.8)
        self.delay_seconds = self.config.get('delay_seconds', 4)
        
        # TTS Setup
        self._init_tts()
        
        # Communication state - tied directly to what's displayed
        self.audio_queue = Queue()
        self.visual_alerts = []
        self.currently_speaking = False
        self.spoken_alert_ids = set()  # Track which visual alerts we've already spoken
        
        # TTS state tracking
        self.last_spoken_time = 0
        self.last_spoken_message = ""
        
        # Status tracking
        self.current_detection_summary = {'critical': 0, 'warning': 0, 'caution': 0, 'safe': 0}
        self.last_status_update = 0
        self.status_update_interval = self.config.get('status_update_interval', 10)  # seconds
        
        # Visual display settings
        self.display_config = self.config.get('display', {
            'width': 640,
            'height': 480,
            'font_scale': 0.8,
            'font_thickness': 2,
            'alert_duration': 4.0
        })
        
        # Subscribe to messages
        self.message_bus.subscribe(MessageType.OBSTACLE_ALERT, self.handle_message)
        self.message_bus.subscribe(MessageType.NAVIGATION_UPDATE, self.handle_message)
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_message)
        self.message_bus.subscribe(MessageType.USER_COMMUNICATION, self.handle_message)
        
        # Start TTS worker thread
        self.tts_worker_thread = threading.Thread(target=self._tts_worker)
        self.tts_worker_running = True
        self.tts_worker_thread.start()
        
        print(f"[{self.agent_name}] Communication Agent initialized")
        
    def _init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            if self.tts_enabled:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.speech_rate)
                self.tts_engine.setProperty('volume', self.speech_volume)
                
                # Get available voices
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Prefer female voice if available
                    female_voice = next((v for v in voices if 'female' in v.name.lower()), None)
                    if female_voice:
                        self.tts_engine.setProperty('voice', female_voice.id)
                        
                print(f"[{self.agent_name}] TTS engine initialized")
            else:
                self.tts_engine = None
                print(f"[{self.agent_name}] TTS disabled")
                
        except Exception as e:
            print(f"[{self.agent_name}] Error initializing TTS: {e}")
            self.tts_engine = None
            
    def _tts_worker(self):
        """Worker thread for processing TTS queue"""
        while self.tts_worker_running:
            try:
                # Get message from queue with timeout
                try:
                    audio_msg = self.audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                    
                if audio_msg and self.tts_engine:
                    # Check if we should speak this message
                    if self._should_speak(audio_msg):
                        self._speak_message(audio_msg)
                        self.last_spoken_message = audio_msg.text
                        self.last_spoken_time = time.time()
                        
            except Exception as e:
                print(f"[{self.agent_name}] Error in TTS worker: {e}")
                
    def _should_speak(self, audio_msg: AudioMessage) -> bool:
        """Determine if message should be spoken based on timing and content"""
        current_time = time.time()
        
        # Always speak high priority messages
        if audio_msg.priority >= 3:
            return True
            
        # Check timing constraint
        time_since_last = current_time - self.last_spoken_time
        if time_since_last < self.delay_seconds:
            # Don't speak if same message was recent
            if audio_msg.text == self.last_spoken_message:
                return False
                
        return True
        
    def _speak_message(self, audio_msg: AudioMessage):
        """Speak a message using TTS"""
        try:
            if self.tts_engine:
                # Apply voice settings if provided
                if audio_msg.voice_settings:
                    for prop, value in audio_msg.voice_settings.items():
                        self.tts_engine.setProperty(prop, value)
                        
                # Speak the message
                self.tts_engine.say(audio_msg.text)
                self.tts_engine.runAndWait()
                
                # Reset to default settings
                self.tts_engine.setProperty('rate', self.speech_rate)
                self.tts_engine.setProperty('volume', self.speech_volume)
                
        except Exception as e:
            print(f"[{self.agent_name}] Error speaking message: {e}")
            
    def _create_visual_alert(self, message: str, alert_type: str, duration: float = None) -> VisualAlert:
        """Create a visual alert"""
        if duration is None:
            duration = self.display_config['alert_duration']
            
        # Position based on alert type
        position_map = {
            'critical': (10, 30),
            'warning': (10, 60),
            'info': (10, 90),
            'success': (10, 120)
        }
        
        position = position_map.get(alert_type, (10, 30))
        
        return VisualAlert(
            message=message,
            alert_type=alert_type,
            duration=duration,
            position=position,
            timestamp=time.time()
        )
        
    def _process_obstacle_alert(self, alert_data: Dict[str, Any]):
        """Process obstacle alert - ONLY create visual alerts, NO automatic TTS"""
        obj = alert_data.get('object', 'obstacle')
        direction = alert_data.get('direction', 'ahead')
        distance = alert_data.get('distance', 0)
        warning_level = alert_data.get('warning_level', 'safe')
        alert_type_msg = alert_data.get('alert_type', '')
        
        # Skip safe objects - they don't need alerts
        if warning_level == 'safe':
            return
            
        # Create message based on warning level
        if warning_level == 'critical':
            message = f"Stop! {obj} very close {direction}!"
            alert_type = 'critical'
            duration = 2.0
        elif warning_level == 'warning':
            message = f"Warning! {obj} approaching {direction}!"
            alert_type = 'warning'
            duration = 3.0
        elif warning_level == 'caution':
            message = f"Caution, {obj} detected {direction}"
            alert_type = 'warning'
            duration = 4.0
        else:
            return
        
        # ONLY create visual alert - NO TTS here
        visual_alert = self._create_visual_alert(f"{alert_type_msg}: {message}", alert_type, duration)
        visual_alert.spoken = False  # Mark as not spoken yet
        self.visual_alerts.append(visual_alert)
    
    def _process_navigation_update(self, nav_data: Dict[str, Any]):
        """Process navigation update for communication"""
        instruction_type = nav_data.get('instruction_type', 'proceed')
        direction = nav_data.get('direction', 'forward')
        reason = nav_data.get('reason', '')
        priority = nav_data.get('priority', 1)
        
        # Create appropriate message based on instruction type
        if instruction_type == 'stop':
            message = "Stop immediately!"
            audio_priority = 4
            alert_type = 'critical'
        elif instruction_type == 'avoid':
            message = f"Move {direction} to avoid obstacle"
            audio_priority = 3
            alert_type = 'warning'
        elif instruction_type == 'turn':
            message = f"Turn {direction}"
            audio_priority = 2
            alert_type = 'info'
        else:  # proceed
            message = "Path clear, proceed"
            audio_priority = 1
            alert_type = 'success'
            
        # Only speak navigation updates if they're important enough
        if audio_priority >= 2:
            audio_msg = AudioMessage(
                text=message,
                priority=audio_priority,
                timestamp=time.time()
            )
            self.audio_queue.put(audio_msg)
            
        # Always create visual alert for navigation updates
        visual_alert = self._create_visual_alert(message, alert_type, duration=3.0)
        self.visual_alerts.append(visual_alert)
        
    def _update_visual_display(self, frame: np.ndarray) -> np.ndarray:
        """Update frame with visual alerts and information"""
        if not self.visual_enabled or frame is None:
            return frame
            
        display_frame = frame.copy()
        current_time = time.time()
        
        # Remove expired alerts
        self.visual_alerts = [
            alert for alert in self.visual_alerts 
            if current_time - alert.timestamp < alert.duration
        ]
        
        # Display active alerts
        for alert in self.visual_alerts:
            self._draw_alert(display_frame, alert)
            
        # Draw system status
        self._draw_system_status(display_frame)
        
        return display_frame
        
    def _draw_alert(self, frame: np.ndarray, alert: VisualAlert):
        """Draw a visual alert on the frame"""
        # Color mapping for alert types
        color_map = {
            'critical': (0, 0, 255),    # Red
            'warning': (0, 165, 255),   # Orange
            'info': (255, 255, 0),      # Yellow
            'success': (0, 255, 0)      # Green
        }
        
        color = color_map.get(alert.alert_type, (255, 255, 255))
        x, y = alert.position
        
        # Calculate text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.display_config['font_scale']
        thickness = self.display_config['font_thickness']
        
        (text_w, text_h), baseline = cv2.getTextSize(alert.message, font, font_scale, thickness)
        
        # Draw background rectangle
        bg_color = (0, 0, 0)  # Black background
        cv2.rectangle(frame, (x-5, y-text_h-10), (x+text_w+10, y+10), bg_color, -1)
        
        # Draw border based on alert type
        border_thickness = 3 if alert.alert_type == 'critical' else 1
        cv2.rectangle(frame, (x-5, y-text_h-10), (x+text_w+10, y+10), color, border_thickness)
        
        # Draw text
        cv2.putText(frame, alert.message, (x, y), font, font_scale, color, thickness)
        
    def _speak_active_alerts(self):
        """Check active visual alerts and speak any that haven't been spoken yet"""
        current_time = time.time()
        
        for alert in self.visual_alerts:
            # Only process alerts that are still active and haven't been spoken
            if (not alert.spoken and 
                current_time - alert.timestamp < alert.duration):
                
                # Extract the actual message (remove prefixes like "CRITICAL - IMMEDIATE DANGER: ")
                message = alert.message
                if ": " in message:
                    message = message.split(": ", 1)[1]
                    
                # Determine priority and voice settings based on alert type
                if alert.alert_type == 'critical':
                    priority = 4
                    voice_settings = {'rate': self.speech_rate + 50}
                elif alert.alert_type == 'warning':
                    priority = 3
                    voice_settings = {'rate': self.speech_rate + 20}
                elif alert.alert_type == 'info':
                    priority = 2
                    voice_settings = None
                else:
                    priority = 1
                    voice_settings = None
                
                # Queue the audio message
                audio_msg = AudioMessage(
                    text=message,
                    priority=priority,
                    timestamp=current_time,
                    voice_settings=voice_settings
                )
                self.audio_queue.put(audio_msg)
                
                # Mark as spoken
                alert.spoken = True
        
    def _draw_system_status(self, frame: np.ndarray):
        """Draw system status information"""
        # Draw timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw agent status
        status_text = f"Communication: {'Active' if self._running else 'Stopped'}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self._running else (0, 0, 255), 1)
                   
    def send_user_message(self, message: str, priority: int = 2, voice_settings: Dict[str, Any] = None):
        """Send a message to the user via TTS and visual display"""
        # Audio message
        audio_msg = AudioMessage(
            text=message,
            priority=priority,
            timestamp=time.time(),
            voice_settings=voice_settings
        )
        self.audio_queue.put(audio_msg)
        
        # Visual alert
        alert_type = 'critical' if priority >= 4 else 'warning' if priority >= 3 else 'info'
        visual_alert = self._create_visual_alert(message, alert_type)
        self.visual_alerts.append(visual_alert)
        
    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None) -> np.ndarray:
        """Process frame with visual overlays"""
        if frame is None:
            return None
            
        # Update visual display
        processed_frame = self._update_visual_display(frame)
        
        # Draw detection overlays if provided
        if detections:
            self._draw_detections(processed_frame, detections)
            
        return processed_frame
        
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """Draw detection bounding boxes and labels"""
        for detection in detections:
            bbox = detection.get('bbox')
            if not bbox:
                continue
                
            x1, y1, x2, y2 = bbox
            obj_class = detection.get('object', 'unknown')
            confidence = detection.get('confidence', 0.0)
            warning_level = detection.get('warning_level', 'safe')
            distance = detection.get('distance', 0.0)
            
            # Color based on warning level
            color_map = {
                'critical': (0, 0, 255),    # Red
                'warning': (0, 165, 255),   # Orange
                'caution': (0, 255, 255),   # Yellow
                'safe': (0, 255, 0)         # Green
            }
            
            color = color_map.get(warning_level, (255, 255, 255))
            thickness = 3 if warning_level == 'critical' else 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with distance
            label = f"{obj_class} {distance:.1f}m ({confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            
            # Draw warning level indicator
            if warning_level != 'safe':
                level_text = warning_level.upper()
                cv2.putText(frame, level_text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    def _run(self):
        """Main communication loop"""
        print(f"[{self.agent_name}] Starting communication loop")
        
        while self._running:
            try:
                # Communication agent primarily responds to messages
                # Visual processing happens when process_frame is called
                
                # Check for periodic status updates independently
                self._check_periodic_status()
                
                # Check and speak any active visual alerts that haven't been spoken
                self._speak_active_alerts()
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[{self.agent_name}] Error in communication loop: {e}")
                
        # Clean up
        self.tts_worker_running = False
        if self.tts_worker_thread.is_alive():
            self.tts_worker_thread.join(timeout=1.0)
            
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
                
        print(f"[{self.agent_name}] Communication loop stopped")
        
    def _check_periodic_status(self):
        """Check if it's time for a periodic status update"""
        current_time = time.time()
        
        # Only check for status updates at proper time intervals
        if current_time - self.last_status_update >= self.status_update_interval:
            # Check current detection summary
            total_detections = sum(self.current_detection_summary.values())
            critical = self.current_detection_summary.get('critical', 0)
            warning = self.current_detection_summary.get('warning', 0)
            caution = self.current_detection_summary.get('caution', 0)
            safe = self.current_detection_summary.get('safe', 0)
            
            # Only provide status if there are no immediate threats
            if critical == 0 and warning == 0:
                if caution > 0:
                    message = f"{caution} objects in caution zone, path generally clear"
                    priority = 1
                elif total_detections == 0:
                    message = "No obstacles detected, path clear"
                    priority = 1
                elif safe > 0:
                    message = f"Area clear, {safe} objects detected at safe distances"
                    priority = 1
                else:
                    message = "Environment scanning active, no immediate threats"
                    priority = 1
                    
                # Only send status update if it's different from the last one
                if message != self.last_spoken_message:
                    # Send status update
                    audio_msg = AudioMessage(
                        text=message,
                        priority=priority,
                        timestamp=current_time
                    )
                    self.audio_queue.put(audio_msg)
                    
                    # Update status display
                    visual_alert = self._create_visual_alert(f"STATUS: {message}", 'info', duration=3.0)
                    self.visual_alerts.append(visual_alert)
                
            # Update the last status check time regardless of whether we sent a message
            self.last_status_update = current_time
    
    def handle_message(self, message: Message):
        """Handle incoming messages"""
        try:
            if message.msg_type == MessageType.OBSTACLE_ALERT:
                self._process_obstacle_alert(message.data)
                
            elif message.msg_type == MessageType.NAVIGATION_UPDATE:
                self._process_navigation_update(message.data)
                
            elif message.msg_type == MessageType.USER_COMMUNICATION:
                data = message.data
                self.send_user_message(
                    data.get('message', ''),
                    data.get('priority', 2),
                    data.get('voice_settings')
                )
                
            elif message.msg_type == MessageType.SYSTEM_STATUS:
                data = message.data
                if data.get('command') == 'stop':
                    self.stop()
                elif data.get('status') == 'error':
                    error_msg = data.get('message', 'System error occurred')
                    self.send_user_message(f"Error: {error_msg}", priority=3)
                elif data.get('status') == 'perception_update':
                    # Update detection summary
                    detection_summary = data.get('detection_summary', {})
                    if detection_summary:
                        self.current_detection_summary = detection_summary
                    
        except Exception as e:
            print(f"[{self.agent_name}] Error handling message: {e}")
            
    def get_current_alerts(self) -> List[VisualAlert]:
        """Get current active visual alerts"""
        current_time = time.time()
        return [
            alert for alert in self.visual_alerts
            if current_time - alert.timestamp < alert.duration
        ]
        
    def clear_alerts(self):
        """Clear all visual alerts"""
        self.visual_alerts.clear()
        
    def set_tts_enabled(self, enabled: bool):
        """Enable or disable TTS"""
        self.tts_enabled = enabled
        if not enabled and self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass