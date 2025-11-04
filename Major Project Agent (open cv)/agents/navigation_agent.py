"""
Navigation Agent - Provide clear instructions for both stationary and moving objects
"""

import time
import threading
from typing import Dict, Any, List

from .base_agent import BaseAgent, MessageBus, MessageType, Message

class NavigationAgent(BaseAgent):
    """
    Navigation Agent that communicates clearly about closest obstacles
    """

    def __init__(self, message_bus: MessageBus, config: Dict[str, Any]):
        super().__init__("NavigationAgent", message_bus, config)

        # Navigation parameters
        self.safe_distance = config.get('safe_distance', 2.0)
        self.update_frequency = config.get('update_frequency', 1.5)  # Less frequent to avoid spam
        
        # State
        self.closest_object = None
        self.all_objects = []
        self.last_guidance_time = 0
        self.last_instruction = ""
        self.object_lock = threading.Lock()

        # Subscribe to messages
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_message)

    def _run(self):
        """Main navigation loop"""
        print(f"[{self.name}] Navigation loop started - Clear communication for all objects")
        
        while self._running:
            try:
                current_time = time.time()
                
                # Generate guidance
                if current_time - self.last_guidance_time >= self.update_frequency:
                    self._generate_comprehensive_guidance()
                    self.last_guidance_time = current_time
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                time.sleep(0.1)
        
        print(f"[{self.name}] Navigation loop stopped")

    def handle_message(self, message: Message):
        """Handle perception updates"""
        if message.type == MessageType.SYSTEM_STATUS:
            data = message.data
            if data.get('status') == 'perception_update':
                with self.object_lock:
                    self.closest_object = data.get('closest_object')
                    self.all_objects = data.get('detections', [])

    def _generate_comprehensive_guidance(self):
        """Generate clear guidance about the environment"""
        with self.object_lock:
            closest_obj = self.closest_object
            all_objects = self.all_objects
        
        if not closest_obj:
            # No objects detected
            if len(all_objects) == 0:
                instruction = "Path clear. Continue straight ahead."
                self._send_instruction(instruction, 'clear', 1)
            else:
                # Objects detected but none are close or in path
                instruction = "Objects around you but path is clear. Continue forward."
                self._send_instruction(instruction, 'info', 1)
            return
        
        distance = closest_obj.get('distance', 999)
        direction = closest_obj.get('direction', 'center')
        in_path = closest_obj.get('in_path', False)
        obj_type = closest_obj.get('detection_type', 'object')
        
        # Generate specific instructions
        if not in_path and distance > 3.0:
            # Object is nearby but not in direct path
            instruction = f"Object on your {direction} at {distance}m. Path is clear, continue forward."
            self._send_instruction(instruction, 'info', 1)
            
        elif not in_path and distance <= 3.0:
            # Object is close but not in path - still worth mentioning
            instruction = f"Object close on your {direction} at {distance}m. Be aware but path is clear."
            self._send_instruction(instruction, 'caution', 2)
            
        else:
            # Object is in navigation path
            if distance <= 1.5:
                # CRITICAL - Very close emergency
                if direction == 'left':
                    instruction = f"🚨 EMERGENCY! Object very close on LEFT at {distance}m. MOVE RIGHT NOW!"
                elif direction == 'right':
                    instruction = f"🚨 EMERGENCY! Object very close on RIGHT at {distance}m. MOVE LEFT NOW!"
                else:
                    instruction = f"🚨 EMERGENCY! Object DIRECTLY AHEAD at {distance}m. STOP and STEP BACK!"
                
                self._send_instruction(instruction, 'critical', 4)
                
            elif distance <= 2.5:
                # WARNING - Need immediate action
                if direction == 'left':
                    instruction = f"⚠️ Object on left at {distance}m. Move to the RIGHT to avoid."
                elif direction == 'right':
                    instruction = f"⚠️ Object on right at {distance}m. Move to the LEFT to avoid."
                else:
                    instruction = f"⚠️ Object ahead at {distance}m. Move LEFT or RIGHT immediately."
                
                self._send_instruction(instruction, 'warning', 3)
                
            elif distance <= 4.0:
                # CAUTION - Be prepared to maneuver
                if direction == 'left':
                    instruction = f"↪️ Object on left at {distance}m. Be ready to move right."
                elif direction == 'right':
                    instruction = f"↩️ Object on right at {distance}m. Be ready to move left."
                else:
                    instruction = f"🔄 Object ahead at {distance}m. Prepare to move left or right."
                
                self._send_instruction(instruction, 'caution', 2)
                
            else:
                # Safe distance but in path
                instruction = f"Object ahead at {distance}m. Continue with awareness."
                self._send_instruction(instruction, 'info', 1)

    def _send_instruction(self, instruction: str, instruction_type: str, priority: int):
        """Send navigation instruction"""
        if instruction != self.last_instruction:
            self.send_message(
                MessageType.NAVIGATION_UPDATE,
                {
                    'instruction_type': instruction_type,
                    'instruction': instruction,
                    'priority': priority,
                    'timestamp': time.time()
                },
                priority=priority
            )
            self.last_instruction = instruction
            print(f"[{self.name}] 🔊 {instruction}")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current navigation status"""
        with self.object_lock:
            closest_obj = self.closest_object
            all_objects = self.all_objects
        
        if not closest_obj:
            return {
                'status': 'clear', 
                'total_objects': len(all_objects),
                'message': 'No immediate obstacles'
            }
        
        return {
            'status': 'obstacle_detected',
            'closest_distance': closest_obj.get('distance', 999),
            'closest_direction': closest_obj.get('direction', 'unknown'),
            'in_path': closest_obj.get('in_path', False),
            'total_objects': len(all_objects),
            'message': f"Closest: {closest_obj.get('direction')} at {closest_obj.get('distance')}m"
        }