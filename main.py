"""
Multi-Agent Navigation System for Visually Impaired
Main application that coordinates Perception, Navigation, and Communication agents
"""
import cv2
import numpy as np
import time
import sys
import signal
from typing import Dict, Any, List

from agents.base_agent import MessageBus, Message, MessageType
from agents.perception_agent import PerceptionAgent
from agents.navigation_agent import NavigationAgent
from agents.communication_agent import CommunicationAgent
from config import config_manager

class NavigationSystem:
    """Main system that coordinates all agents"""
    
    def __init__(self):
        # Initialize message bus
        self.message_bus = MessageBus()
        
        # Load configuration
        self.config = config_manager.config
        
        # Initialize agents
        self.perception_agent = None
        self.navigation_agent = None
        self.communication_agent = None
        
        # System state
        self.running = False
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # Frame storage for visualization
        self.current_frame = None
        self.current_detections = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nReceived shutdown signal. Stopping system...")
        self.stop()
        
    def initialize_agents(self):
        """Initialize all agents"""
        try:
            print("Initializing Multi-Agent Navigation System...")
            
            # Create agents
            self.perception_agent = PerceptionAgent(
                self.message_bus, 
                config_manager.get_perception_config()
            )
            
            self.navigation_agent = NavigationAgent(
                self.message_bus,
                config_manager.get_navigation_config()
            )
            
            self.communication_agent = CommunicationAgent(
                self.message_bus,
                config_manager.get_communication_config()
            )
            
            # Subscribe to system messages for visualization
            self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self._handle_system_message)
            self.message_bus.subscribe(MessageType.OBSTACLE_ALERT, self._handle_obstacle_alert)
            self.message_bus.subscribe(MessageType.NAVIGATION_UPDATE, self._handle_navigation_update)
            
            print("All agents initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing agents: {e}")
            return False
            
    def start(self):
        """Start the navigation system"""
        if not self.initialize_agents():
            print("Failed to initialize agents. Exiting.")
            return False
            
        try:
            # Start message bus
            self.message_bus.start()
            
            # Start all agents
            self.perception_agent.start()
            self.navigation_agent.start()
            self.communication_agent.start()
            
            self.running = True
            print("Multi-Agent Navigation System started successfully")
            
            # Send startup message
            self.communication_agent.send_user_message(
                "Multi-Agent Navigation System activated. Using calibrated distance estimation.", 
                priority=2
            )
            
            # Run main visualization loop
            self._run_visualization_loop()
            
            return True
            
        except Exception as e:
            print(f"Error starting system: {e}")
            self.stop()
            return False
            
    def stop(self):
        """Stop the navigation system"""
        print("Stopping Multi-Agent Navigation System...")
        self.running = False
        
        # Stop agents
        if self.perception_agent:
            self.perception_agent.stop()
        if self.navigation_agent:
            self.navigation_agent.stop()
        if self.communication_agent:
            self.communication_agent.stop()
            
        # Stop message bus
        if self.message_bus:
            self.message_bus.stop()
            
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        print("Multi-Agent Navigation System stopped")
        
    def _run_visualization_loop(self):
        """Main visualization loop"""
        print("Starting visualization loop... Press 'q' to quit")
        print("System now using calibrated distance estimation with known object widths")
        
        # Create windows
        cv2.namedWindow(self.config.window_names['main'], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.window_names['main'], 1280, 720)
        
        if self.config.show_depth_map:
            cv2.namedWindow(self.config.window_names['depth'], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.config.window_names['depth'], 640, 480)
            
        while self.running:
            try:
                # Get current frame from perception agent
                if self.perception_agent:
                    frame = self.perception_agent.get_current_frame()
                    if frame is not None:
                        self.current_frame = frame
                    
                # Process frame with communication agent for visual overlays
                if self.current_frame is not None and self.communication_agent:
                    display_frame = self.communication_agent.process_frame(
                        self.current_frame, 
                        self.current_detections
                    )
                    
                    # Draw distance zones visualization
                    if self.current_detections:
                        self._draw_distance_zones(display_frame)
                    
                    # Draw FPS if enabled
                    if self.config.show_fps:
                        self._draw_fps(display_frame)
                    
                    # Draw calibration status
                    self._draw_calibration_status(display_frame)
                        
                    # Show main window
                    cv2.imshow(self.config.window_names['main'], display_frame)
                    
                # Show depth map placeholder if enabled
                if self.config.show_depth_map:
                    depth_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(depth_placeholder, "CALIBRATED MODE", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(depth_placeholder, "Using calibrated distance", (30, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(depth_placeholder, "estimation with known", (30, 190), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(depth_placeholder, "object widths", (30, 230), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(depth_placeholder, "Depth map not used", (30, 280), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    cv2.imshow(self.config.window_names['depth'], depth_placeholder)
                    
                # Update FPS counter
                self._update_fps()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User requested quit")
                    break
                elif key == ord('s'):
                    self._save_screenshot()
                elif key == ord('c'):
                    if self.communication_agent:
                        self.communication_agent.clear_alerts()
                elif key == ord('h'):
                    self._show_help()
                elif key == ord('d'):
                    # Toggle depth map display
                    self.config.show_depth_map = not self.config.show_depth_map
                    if not self.config.show_depth_map:
                        cv2.destroyWindow(self.config.window_names['depth'])
                    else:
                        cv2.namedWindow(self.config.window_names['depth'], cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(self.config.window_names['depth'], 640, 480)
                    
            except Exception as e:
                print(f"Error in visualization loop: {e}")
                time.sleep(0.1)  # Prevent rapid error looping
                
        self.stop()
        
    def _draw_calibration_status(self, frame):
        """Draw calibration status information"""
        status_text = "CALIBRATED DISTANCE ESTIMATION"
        cv2.putText(frame, status_text, (frame.shape[1] - 400, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    def _draw_distance_zones(self, frame):
        """Draw distance-based warning zones visualization"""
        h, w = frame.shape[:2]
        
        # Draw distance zone indicators on the left side
        zone_height = 30
        zone_width = 150
        start_y = 50
        
        # Zone colors (BGR format)
        zone_colors = {
            'critical': (0, 0, 255),    # Red
            'warning': (0, 165, 255),   # Orange  
            'caution': (0, 255, 255),   # Yellow
            'safe': (0, 255, 0)         # Green
        }
        
        # Zone labels and thresholds
        zones = [
            ('critical', '≤1.5m', 'CRITICAL'),
            ('warning', '≤3.0m', 'WARNING'), 
            ('caution', '≤5.0m', 'CAUTION'),
            ('safe', '>5.0m', 'SAFE')
        ]
        
        # Count detections in each zone
        zone_counts = {'critical': 0, 'warning': 0, 'caution': 0, 'safe': 0}
        for detection in self.current_detections:
            warning_level = detection.get('warning_level', 'safe')
            zone_counts[warning_level] = zone_counts.get(warning_level, 0) + 1
        
        # Draw zones
        for i, (zone_name, distance_text, label) in enumerate(zones):
            y = start_y + i * (zone_height + 10)
            color = zone_colors[zone_name]
            count = zone_counts[zone_name]
            
            # Draw zone rectangle
            cv2.rectangle(frame, (10, y), (10 + zone_width, y + zone_height), color, 2)
            
            # Fill rectangle if there are detections in this zone
            if count > 0:
                cv2.rectangle(frame, (10, y), (10 + zone_width, y + zone_height), color, -1)
                text_color = (255, 255, 255) if zone_name == 'critical' else (0, 0, 0)
            else:
                text_color = color
            
            # Draw text
            text = f"{label} {distance_text}"
            if count > 0:
                text += f" ({count})"
            
            cv2.putText(frame, text, (15, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Draw title
        cv2.putText(frame, "Distance Zones:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    def _draw_fps(self, frame):
        """Draw FPS counter on frame"""
        cv2.putText(frame, f"FPS: {self.current_fps}", 
                   (frame.shape[1] - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                   
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = current_time
            
    def _save_screenshot(self):
        """Save current frame as screenshot"""
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            print(f"Screenshot saved as {filename}")
            
            if self.communication_agent:
                self.communication_agent.send_user_message(
                    f"Screenshot saved", priority=1
                )
                
    def _show_help(self):
        """Show help information"""
        help_text = """
        Multi-Agent Navigation System - Keyboard Commands:
        'q' - Quit the application
        's' - Save screenshot
        'c' - Clear visual alerts
        'd' - Toggle depth map display
        'h' - Show this help
        
        System Mode: Calibrated Distance Estimation
        Using camera calibration data and known object widths
        for accurate distance measurement.
        """
        print(help_text)
        
        if self.communication_agent:
            self.communication_agent.send_user_message(
                "Help displayed in console. Using calibrated distance mode.", priority=1
            )
            
    def _handle_system_message(self, message: Message):
        """Handle system status messages for visualization"""
        try:
            data = message.data
            status = data.get('status')
            
            if status == 'perception_update':
                self.current_detections = data.get('detections', [])
                
        except Exception as e:
            print(f"Error handling system message: {e}")
            
    def _handle_obstacle_alert(self, message: Message):
        """Handle obstacle alerts for logging/debugging"""
        if self.config.debug_mode:
            data = message.data
            print(f"OBSTACLE ALERT: {data.get('object')} at {data.get('direction')} - {data.get('distance'):.1f}m")
            
    def _handle_navigation_update(self, message: Message):
        """Handle navigation updates for logging/debugging"""
        if self.config.debug_mode:
            data = message.data
            print(f"NAV UPDATE: {data.get('instruction_type')} - {data.get('reason')}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'running': self.running,
            'fps': self.current_fps,
            'agents': {
                'perception': self.perception_agent is not None and self.perception_agent._running,
                'navigation': self.navigation_agent is not None and self.navigation_agent._running,
                'communication': self.communication_agent is not None and self.communication_agent._running
            },
            'mode': 'calibrated_distance'
        }
        
        # Add agent-specific status
        if self.navigation_agent:
            status['navigation_status'] = self.navigation_agent.get_current_status()
            
        return status

def main():
    """Main entry point"""
    print("Multi-Agent Navigation System for Visually Impaired")
    print("=" * 50)
    print("Mode: Calibrated Distance Estimation")
    print("Using camera calibration data and known object widths")
    print("=" * 50)
    
    try:
        # Create and start system
        nav_system = NavigationSystem()
        success = nav_system.start()
        
        if not success:
            print("Failed to start navigation system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
        
    print("System shutdown complete")

if __name__ == "__main__":
    main()