"""
Multi-Agent Navigation System for Visually Impaired
Improved main: robust imports, safe device selection, camera auto-detection,
graceful failure when agents are missing.
"""
import cv2
import numpy as np
import time
import sys
import signal
import os
from typing import Dict, Any

from config import config_manager

# ==== Safe Agent Imports ====
AGENTS_AVAILABLE = True
try:
    from agents.base_agent import MessageBus, Message, MessageType
    from agents.perception_agent import PerceptionAgent
    from agents.navigation_agent import NavigationAgent
    from agents.communication_agent import CommunicationAgent
except Exception as e:
    print(f"[main] ERROR: Could not import agents package: {e}")
    print("[main] TIP: Ensure you have an 'agents/' folder with __init__.py and required files.")
    AGENTS_AVAILABLE = False


def _detect_device():
    """Detect torch device availability safely."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[main] PyTorch device detected: {device}")
        return device
    except Exception:
        print("[main] PyTorch not available. Using CPU only.")
        return "cpu"


def find_working_camera_index(preferred_index: int = 1, max_search: int = 5) -> int:
    """Find a working camera index, fallback to search if preferred fails."""
    def _is_open(idx):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            cap.release()
            return ret or ok
        cap.release()
        return False

    if preferred_index is not None and isinstance(preferred_index, int):
        if _is_open(preferred_index):
            print(f"[camera] Using preferred camera index: {preferred_index}")
            return preferred_index

    for idx in range(max_search):
        if _is_open(idx):
            print(f"[camera] Found working camera at index: {idx}")
            return idx

    print(f"[camera] No working camera detected in range 0..{max_search - 1}")
    return -1


class NavigationSystem:
    """Main system that coordinates all agents"""

    def __init__(self):
        self.message_bus = None
        self.perception_agent = None
        self.navigation_agent = None
        self.communication_agent = None

        # Load configuration
        self.config = config_manager.config

        # Detect and select camera
        preferred_cam = getattr(self.config.perception, "camera_index", 0)
        working_cam = find_working_camera_index(preferred_cam, max_search=6)
        if working_cam >= 0:
            self.config.perception.camera_index = working_cam
        else:
            print("[main] WARNING: No available camera detected; perception agent may fail.")

        self.running = False
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        self.current_frame = None
        self.current_detections = []
        

        # Handle OS signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.device = _detect_device()

    def _signal_handler(self, signum, frame):
        print("\n[main] Received shutdown signal. Stopping system...")
        self.stop()

    def initialize_agents(self):
        """Initialize all agents (graceful failure if missing)."""
        if not AGENTS_AVAILABLE:
            print("[main] Cannot initialize agents: 'agents' package not available.")
            return False

        try:
            print("[main] Initializing Multi-Agent Navigation System...")
            self.message_bus = MessageBus()
            self.perception_agent = PerceptionAgent(self.message_bus, config_manager.get_perception_config())
            self.navigation_agent = NavigationAgent(self.message_bus, config_manager.get_navigation_config())
            self.communication_agent = CommunicationAgent(self.message_bus, config_manager.get_communication_config())

            # Subscribe to system messages
            self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self._handle_system_message)
            self.message_bus.subscribe(MessageType.OBSTACLE_ALERT, self._handle_obstacle_alert)
            self.message_bus.subscribe(MessageType.NAVIGATION_UPDATE, self._handle_navigation_update)

            print("[main] All agents initialized successfully")
            return True

        except Exception as e:
            print(f"[main] ERROR initializing agents: {e}")
            return False

    def start(self):
        if not self.initialize_agents():
            print("[main] Failed to initialize agents. Exiting.")
            return False

        try:
            if self.message_bus:
                self.message_bus.start()
            if self.perception_agent:
                self.perception_agent.start()
            if self.navigation_agent:
                self.navigation_agent.start()
            if self.communication_agent:
                self.communication_agent.start()

            self.running = True
            print("[main] Multi-Agent Navigation System started successfully")

            if self.communication_agent:
                try:
                    self.communication_agent.send_user_message("Multi-Agent Navigation System activated", priority=2)
                except Exception:
                    pass

            self._run_visualization_loop()
            return True

        except Exception as e:
            print(f"[main] Error starting system: {e}")
            self.stop()
            return False

    def stop(self):
        if self.running:
            print("[main] Stopping Multi-Agent Navigation System...")
        self.running = False

        try:
            if self.perception_agent:
                self.perception_agent.stop()
            if self.navigation_agent:
                self.navigation_agent.stop()
            if self.communication_agent:
                self.communication_agent.stop()
            if self.message_bus:
                self.message_bus.stop()
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[main] Multi-Agent Navigation System stopped")

    # --- Visualization loop and helpers ---
    def _run_visualization_loop(self):
        print("[main] Starting visualization loop... Press 'q' to quit")
        try:
            cv2.namedWindow(self.config.window_names['main'], cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"[main] Could not create OpenCV windows: {e}")

        while self.running:
            try:
                if self.perception_agent:
                    frame = self.perception_agent.get_current_frame()
                    if frame is not None:
                        self.current_frame = frame
                    

                display_frame = self.current_frame
                if display_frame is not None and self.communication_agent:
                    display_frame = self.communication_agent.process_frame(display_frame, self.current_detections)

                if display_frame is not None:
                    if self.current_detections:
                        self._draw_distance_zones(display_frame)
                    if self.config.show_fps:
                        self._draw_fps(display_frame)
                    cv2.imshow(self.config.window_names['main'], display_frame)

                

                self._update_fps()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[main] User requested quit")
                    break
                elif key == ord('s'):
                    self._save_screenshot()
                elif key == ord('c') and self.communication_agent:
                    self.communication_agent.clear_alerts()
                elif key == ord('h'):
                    self._show_help()

            except Exception as e:
                print(f"[main] Error in visualization loop: {e}")

        self.stop()

    def _draw_distance_zones(self, frame):
        try:
            zone_colors = {'critical': (0, 0, 255), 'warning': (0, 165, 255),
                           'caution': (0, 255, 255), 'safe': (0, 255, 0)}
            zones = [('critical', '≤1m', 'CRITICAL'), ('warning', '2≤m', 'WARNING'),
                     ('caution', '≤3m', 'CAUTION'), ('safe', '>3m', 'SAFE')]
            zone_counts = {z: 0 for z in zone_colors}
            for d in self.current_detections:
                zone_counts[d.get('warning_level', 'safe')] += 1
            y = 50
            for zone_name, dist, label in zones:
                color = zone_colors[zone_name]
                count = zone_counts[zone_name]
                cv2.rectangle(frame, (10, y), (160, y + 30), color, -1 if count > 0 else 2)
                text = f"{label} {dist}" + (f" ({count})" if count > 0 else "")
                cv2.putText(frame, text, (15, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255) if count > 0 else color, 2)
                y += 40
        except Exception:
            pass

    def _draw_fps(self, frame):
        cv2.putText(frame, f"FPS: {self.current_fps}", (frame.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _update_fps(self):
        self.fps_counter += 1
        now = time.time()
        if now - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = now

    def _save_screenshot(self):
        if self.current_frame is not None:
            filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, self.current_frame)
            print(f"[main] Screenshot saved as {filename}")

    def _show_help(self):
        print("""
        Keyboard Commands:
        q - Quit
        s - Save screenshot
        c - Clear alerts
        h - Show help
        """)
        if self.communication_agent:
            self.communication_agent.send_user_message("Help displayed in console", priority=1)

    def _handle_system_message(self, message):
        self.current_detections = message.data.get('detections', [])

    def _handle_obstacle_alert(self, message):
        if self.config.debug_mode:
            print(f"[main] OBSTACLE ALERT: {message.data}")

    def _handle_navigation_update(self, message):
        if self.config.debug_mode:
            print(f"[main] NAV UPDATE: {message.data}")


def main():
    print("Multi-Agent Navigation System for Visually Impaired")
    print("=" * 50)
    try:
        nav_system = NavigationSystem()
        if not nav_system.start():
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user")
    except Exception as e:
        print(f"[main] Unexpected error: {e}")
        sys.exit(1)
    print("[main] System shutdown complete")


if __name__ == "__main__":
    main()
