"""
Communication Agent - Handles TTS functionality and user interface
(continuous speech + scenario-change feedback + lagged TTS + bounding boxes)
"""
import time
import threading
import re
from queue import Queue, Empty
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
import pyttsx3

from .base_agent import BaseAgent, Message, MessageType


@dataclass
class AudioMessage:
    text: str
    priority: int
    timestamp: float
    voice_settings: Optional[Dict[str, Any]] = None


@dataclass
class VisualAlert:
    message: str
    alert_type: str
    duration: float
    position: Tuple[int, int]
    timestamp: float
    spoken: bool = False


class CommunicationAgent(BaseAgent):
    def __init__(self, message_bus, config: Dict[str, Any] = None):
        super().__init__("CommunicationAgent", message_bus)

        # Config
        self.config = config or {}
        self.tts_enabled = self.config.get("tts_enabled", True)
        self.visual_enabled = self.config.get("visual_enabled", True)
        self.speech_rate = self.config.get("speech_rate", 160)
        self.speech_volume = self.config.get("speech_volume", 0.8)
        self.delay_seconds = self.config.get("delay_seconds", 4)
        self.status_update_interval = self.config.get("status_update_interval", 10)

        self.display_config = self.config.get(
            "display",
            {
                "width": 640,
                "height": 480,
                "font_scale": 0.8,
                "font_thickness": 2,
                "alert_duration": 4.0,
            },
        )

        # TTS setup
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self._default_voice = None
        self._default_rate = self.speech_rate
        self._default_volume = self.speech_volume
        self._init_tts()

        # Queues
        self.audio_queue: "Queue[AudioMessage]" = Queue()
        self.visual_alerts: List[VisualAlert] = []

        # State
        self.last_spoken_time = time.time()
        self.last_spoken_message = ""
        self.last_detection_state = None
        self.current_detection_summary = {"critical": 0, "warning": 0, "caution": 0, "safe": 0}
        self.last_status_update = 0.0

        # Subscribe
        self.message_bus.subscribe(MessageType.OBSTACLE_ALERT, self.handle_message)
        self.message_bus.subscribe(MessageType.NAVIGATION_UPDATE, self.handle_message)
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_message)
        self.message_bus.subscribe(MessageType.USER_COMMUNICATION, self.handle_message)

        # Worker
        self.tts_worker_running = True
        self.tts_worker_thread = threading.Thread(target=self._tts_worker, name="TTSWorker")
        self.tts_worker_thread.start()

        print(f"[{self.agent_name}] Communication Agent initialized")

    # -------------------- TTS --------------------
    def _init_tts(self):
        """Initialize TTS engine"""
        try:
            if not self.tts_enabled:
                self.tts_engine = None
                return
            engine = pyttsx3.init()
            print("[TTS] Engine initialized successfully")
            with self.tts_lock:
                engine.setProperty("rate", self.speech_rate)
                engine.setProperty("volume", self.speech_volume)
                voices = engine.getProperty("voices")
                if voices:
                    chosen_voice = next(
                        (v for v in voices if "female" in getattr(v, "name", "").lower()), voices[0]
                    )
                    engine.setProperty("voice", chosen_voice.id)
                    self._default_voice = chosen_voice.id
                self.tts_engine = engine
        except Exception as e:
            print(f"[TTS] Error initializing TTS: {e}")
            self.tts_engine = None

    def _natural_pause(self, text: str):
        if re.search(r'[.!?]$', text.strip()):
            time.sleep(0.3)
        elif re.search(r'[,:;]$', text.strip()):
            time.sleep(0.15)

    def _tts_worker(self):
        """Background TTS worker to speak messages with optional delay"""
        speech_delay = 2.0  # lag for non-critical messages
        while self.tts_worker_running:
            try:
                audio_msg: AudioMessage = self.audio_queue.get(timeout=0.1)
                if not self.tts_engine or not self.tts_enabled:
                    continue
                if self._should_speak(audio_msg):
                    with self.tts_lock:
                        if audio_msg.priority >= 4:  # critical interrupt
                            try:
                                self.tts_engine.stop()
                            except Exception:
                                pass
                        else:
                            time.sleep(speech_delay)

                        self.tts_engine.say(audio_msg.text)
                        self._natural_pause(audio_msg.text)
                        self.tts_engine.runAndWait()  # ensures audio is played

                    self.last_spoken_message = audio_msg.text
                    self.last_spoken_time = time.time()
            except Empty:
                continue
            except Exception as e:
                print(f"[TTS] Worker error: {e}")

        # Stop engine when exiting
        try:
            if self.tts_engine:
                with self.tts_lock:
                    self.tts_engine.stop()
        except Exception:
            pass

    def _should_speak(self, audio_msg: AudioMessage) -> bool:
        now = time.time()
        if audio_msg.priority >= 3:
            return True
        if now - self.last_spoken_time < self.delay_seconds:
            return False
        return True

    # -------------------- Scenario --------------------
    def _check_periodic_status(self):
        current_time = time.time()
        detection_state = tuple(sorted(self.current_detection_summary.items()))
        if detection_state != self.last_detection_state:
            self.last_detection_state = detection_state
            critical = self.current_detection_summary.get("critical", 0)
            warning = self.current_detection_summary.get("warning", 0)
            caution = self.current_detection_summary.get("caution", 0)
            safe = self.current_detection_summary.get("safe", 0)
            if critical > 0:
                message, priority = f"{critical} critical obstacle(s)! Stop!", 4
            elif warning > 0:
                message, priority = f"{warning} obstacle(s) in warning zone.", 3
            elif caution > 0:
                message, priority = f"{caution} object(s) caution zone.", 2
            elif safe > 0:
                message, priority = f"{safe} safe objects detected.", 1
            else:
                message, priority = "No obstacles detected.", 1
            self.audio_queue.put(AudioMessage(message, priority, current_time))
            self.visual_alerts.append(self._create_visual_alert(f"STATUS: {message}", "info", 3.0))
            return
        if current_time - self.last_status_update >= self.status_update_interval:
            self.last_status_update = current_time
            self.audio_queue.put(AudioMessage("System monitoring active.", 1, current_time))

    # -------------------- Alerts / Drawing --------------------
    def _create_visual_alert(self, message: str, alert_type: str, duration: float = None) -> VisualAlert:
        if duration is None:
            duration = self.display_config.get("alert_duration", 4.0)
        pos_map = {"critical": (10, 30), "warning": (10, 60), "info": (10, 90), "success": (10, 120)}
        return VisualAlert(message, alert_type, duration, pos_map.get(alert_type, (10, 30)), time.time())

    def _process_obstacle_alert(self, data: Dict[str, Any]):
        obj = data.get("object", "obstacle")
        direction = data.get("direction", "ahead")  # 'ahead', 'left', 'right', 'back'
        lvl = data.get("warning_level", "safe")
        if lvl == "safe":
            return

        msg_prefix = "Obstacle behind! " if direction == "back" else ""

        if lvl == "critical":
            msg, typ, dur = f"{msg_prefix}Stop! {obj} very close {direction}!", "critical", 2.0
        elif lvl == "warning":
            msg, typ, dur = f"{msg_prefix}Warning! {obj} approaching {direction}!", "warning", 3.0
        elif lvl == "caution":
            msg, typ, dur = f"{msg_prefix}Caution, {obj} detected {direction}", "warning", 4.0
        else:
            return

        alert = self._create_visual_alert(msg, typ, dur)
        alert.spoken = False
        self.visual_alerts.append(alert)
        # Push to TTS
        self.audio_queue.put(AudioMessage(msg, 3 if lvl != "safe" else 1, time.time()))

    def _process_navigation_update(self, data: Dict[str, Any]):
        t = data.get("instruction_type", "proceed")
        d = data.get("direction", "forward")
        if t == "stop":
            msg, pr, typ = "Stop immediately!", 4, "critical"
        elif t == "avoid":
            msg, pr, typ = f"Move {d} to avoid obstacle", 3, "warning"
        elif t == "turn":
            msg, pr, typ = f"Turn {d}", 2, "info"
        else:
            msg, pr, typ = "Path clear, proceed", 1, "success"
        if pr >= 2:
            self.audio_queue.put(AudioMessage(msg, pr, time.time()))
        self.visual_alerts.append(self._create_visual_alert(msg, typ, 3.0))

    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None) -> np.ndarray:
        if frame is None:
            return None
        processed = self._update_visual_display(frame)
        if detections:
            self._draw_detections(processed, detections)
        return processed

    def _update_visual_display(self, frame: np.ndarray) -> np.ndarray:
        if not self.visual_enabled:
            return frame
        current_time = time.time()
        self.visual_alerts = [a for a in self.visual_alerts if current_time - a.timestamp < a.duration]
        disp = frame.copy()
        for a in self.visual_alerts:
            self._draw_alert(disp, a)
        return disp

    def _draw_alert(self, frame: np.ndarray, alert: VisualAlert):
        colors = {"critical": (0, 0, 255), "warning": (0, 165, 255), "info": (255, 255, 0), "success": (0, 255, 0)}
        color = colors.get(alert.alert_type, (255, 255, 255))
        x, y = alert.position
        cv2.putText(frame, alert.message, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    float(self.display_config["font_scale"]), color, int(self.display_config["font_thickness"]))

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            obj_class = det.get("object", "unknown")
            confidence = det.get("confidence", 0.0)
            warning_level = det.get("warning_level", "safe")
            distance = det.get("distance", 0.0)
            color_map = {"critical": (0, 0, 255), "warning": (0, 165, 255),
                         "caution": (0, 255, 255), "safe": (0, 255, 0)}
            color = color_map.get(warning_level, (255, 255, 255))
            thickness = 3 if warning_level == "critical" else 2
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{obj_class} {distance:.1f}m ({confidence:.2f})"
            cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            if warning_level != "safe":
                cv2.putText(frame, warning_level.upper(), (x1, min(h - 1, y2 + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # -------------------- Run Loop --------------------
    def _run(self):
        print(f"[{self.agent_name}] Starting communication loop")
        while getattr(self, "_running", True):
            try:
                self._check_periodic_status()
                time.sleep(0.1)
            except Exception as e:
                print(f"[{self.agent_name}] Error in loop: {e}")
        self.tts_worker_running = False
        if self.tts_worker_thread.is_alive():
            self.tts_worker_thread.join(timeout=1.0)
        with self.tts_lock:
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except Exception:
                    pass
        print(f"[{self.agent_name}] Communication loop stopped")

    # -------------------- Message Handling --------------------
    def handle_message(self, message: Message):
        try:
            if message.msg_type == MessageType.OBSTACLE_ALERT:
                self._process_obstacle_alert(message.data)
            elif message.msg_type == MessageType.NAVIGATION_UPDATE:
                self._process_navigation_update(message.data)
            elif message.msg_type == MessageType.USER_COMMUNICATION:
                d = message.data
                self.send_user_message(d.get("message", ""), d.get("priority", 2), d.get("voice_settings"))
            elif message.msg_type == MessageType.SYSTEM_STATUS:
                d = message.data
                if d.get("command") == "stop":
                    self.stop()
                elif d.get("status") == "error":
                    self.send_user_message(f"Error: {d.get('message', 'System error')}", priority=3)
                elif d.get("status") == "perception_update":
                    if d.get("detection_summary"):
                        self.current_detection_summary = d["detection_summary"]
        except Exception as e:
            print(f"[{self.agent_name}] Error handling message: {e}")

    # -------------------- Send User Message --------------------
    def send_user_message(self, message: str, priority: int = 2, voice_settings: Dict[str, Any] = None):
        """Send a message to the user via TTS and visual display"""
        audio_msg = AudioMessage(
            text=message,
            priority=priority,
            timestamp=time.time(),
            voice_settings=voice_settings
        )
        self.audio_queue.put(audio_msg)
        alert_type = "critical" if priority >= 4 else "warning" if priority >= 3 else "info"
        self.visual_alerts.append(self._create_visual_alert(message, alert_type))
