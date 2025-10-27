"""
Navigation Agent - Processes spatial information and provides directional guidance
"""
import time
import math
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, Message, MessageType

class Direction(Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    FORWARD = "forward"
    BACKWARD = "backward"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class NavigationInstruction:
    """Represents a navigation instruction"""
    instruction_type: str  # "turn", "avoid", "proceed", "stop"
    direction: Direction
    distance: float
    reason: str
    priority: Priority
    confidence: float

@dataclass
class SpatialMap:
    """Represents the current spatial understanding"""
    obstacles: List[Dict[str, Any]]
    clear_paths: List[Direction]
    recommended_direction: Direction
    risk_level: int  # 1-4 scale
    timestamp: float

class NavigationAgent(BaseAgent):
    """Agent responsible for navigation planning and spatial analysis"""
    
    def __init__(self, message_bus, config: Dict[str, Any] = None):
        super().__init__("NavigationAgent", message_bus)
        
        # Configuration
        self.config = config or {}
        self.safe_distance = self.config.get('safe_distance', 2.0)  # meters
        self.warning_distance = self.config.get('warning_distance', 3.0)  # meters
        self.path_width = self.config.get('path_width', 1.0)  # meters
        self.update_frequency = self.config.get('update_frequency', 0.5)  # seconds
        
        # State tracking
        self.current_spatial_map: Optional[SpatialMap] = None
        self.last_detections = []
        self.navigation_history = []
        self.last_instruction_time = 0
        
        # Risk assessment parameters
        self.risk_thresholds = self.config.get('risk_thresholds', {
            'low': 3.0,      # > 3m clear
            'medium': 2.0,   # 2-3m obstacles
            'high': 1.0,     # 1-2m obstacles
            'critical': 0.5  # < 1m obstacles
        })
        
        # Subscribe to messages
        self.message_bus.subscribe(MessageType.OBSTACLE_ALERT, self.handle_message)
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_message)
        
        print(f"[{self.agent_name}] Navigation Agent initialized")
        
    def _assess_risk_level(self, detections: List[Dict[str, Any]]) -> int:
        """Assess the current risk level based on detections"""
        if not detections:
            return 1  # Low risk - no obstacles
            
        # Find closest obstacle in path
        path_obstacles = [det for det in detections if det.get('in_path', False)]
        
        if not path_obstacles:
            return 1  # Low risk - no obstacles in path
            
        min_distance = min(det.get('distance', float('inf')) for det in path_obstacles)
        
        if min_distance > self.risk_thresholds['low']:
            return 1  # Low
        elif min_distance > self.risk_thresholds['medium']:
            return 2  # Medium
        elif min_distance > self.risk_thresholds['high']:
            return 3  # High
        else:
            return 4  # Critical
            
    def _analyze_clear_paths(self, detections: List[Dict[str, Any]]) -> List[Direction]:
        """Analyze which directions are clear for movement"""
        # Count obstacles by direction
        direction_obstacles = {
            Direction.LEFT: [],
            Direction.CENTER: [],
            Direction.RIGHT: []
        }
        
        for det in detections:
            if det.get('in_path', False):
                direction = det.get('direction', 'center')
                if direction == 'left':
                    direction_obstacles[Direction.LEFT].append(det)
                elif direction == 'right':
                    direction_obstacles[Direction.RIGHT].append(det)
                else:
                    direction_obstacles[Direction.CENTER].append(det)
        
        # Determine clear paths
        clear_paths = []
        
        for direction, obstacles in direction_obstacles.items():
            if not obstacles:
                clear_paths.append(direction)
            else:
                # Check if obstacles are far enough
                min_distance = min(obs.get('distance', 0) for obs in obstacles)
                if min_distance > self.safe_distance:
                    clear_paths.append(direction)
                    
        return clear_paths
        
    def _recommend_direction(self, clear_paths: List[Direction], current_detections: List[Dict[str, Any]]) -> Direction:
        """Recommend the best direction to move"""
        if not clear_paths:
            # No clear paths - recommend stopping or backing up
            return Direction.BACKWARD
            
        # Prefer center if available
        if Direction.CENTER in clear_paths:
            return Direction.CENTER
            
        # If center is blocked, prefer the side with fewer/farther obstacles
        left_score = self._calculate_path_score(Direction.LEFT, current_detections)
        right_score = self._calculate_path_score(Direction.RIGHT, current_detections)
        
        if Direction.LEFT in clear_paths and Direction.RIGHT in clear_paths:
            return Direction.LEFT if left_score > right_score else Direction.RIGHT
        elif Direction.LEFT in clear_paths:
            return Direction.LEFT
        elif Direction.RIGHT in clear_paths:
            return Direction.RIGHT
        else:
            return Direction.FORWARD  # Default fallback
            
    def _calculate_path_score(self, direction: Direction, detections: List[Dict[str, Any]]) -> float:
        """Calculate a score for a path direction (higher is better)"""
        direction_str = direction.value
        relevant_detections = [det for det in detections if det.get('direction') == direction_str]
        
        if not relevant_detections:
            return 100.0  # Perfect score for clear path
            
        # Score based on distance and number of obstacles
        total_distance = sum(det.get('distance', 0) for det in relevant_detections)
        avg_distance = total_distance / len(relevant_detections)
        
        # Penalize for number of obstacles and reward for distance
        score = avg_distance * 10 - len(relevant_detections) * 5
        return max(0, score)
        
    def _create_navigation_instruction(self, spatial_map: SpatialMap) -> NavigationInstruction:
        """Create navigation instruction based on spatial analysis"""
        risk_level = spatial_map.risk_level
        recommended_dir = spatial_map.recommended_direction
        
        # Determine instruction type and priority
        if risk_level == 4:  # Critical
            if recommended_dir == Direction.BACKWARD:
                instruction_type = "stop"
                reason = "Critical obstacle ahead - stop immediately"
                priority = Priority.CRITICAL
            else:
                instruction_type = "avoid"
                reason = f"Avoid obstacle by moving {recommended_dir.value}"
                priority = Priority.CRITICAL
        elif risk_level == 3:  # High
            instruction_type = "avoid" 
            reason = f"Obstacle detected - move {recommended_dir.value}"
            priority = Priority.HIGH
        elif risk_level == 2:  # Medium
            instruction_type = "turn"
            reason = f"Adjust path - turn {recommended_dir.value}"
            priority = Priority.MEDIUM
        else:  # Low
            instruction_type = "proceed"
            reason = "Path is clear - proceed forward"
            priority = Priority.LOW
            
        # Calculate confidence based on number of clear paths and detection confidence
        clear_path_ratio = len(spatial_map.clear_paths) / 3.0
        confidence = min(0.9, clear_path_ratio * 0.8 + 0.2)
        
        return NavigationInstruction(
            instruction_type=instruction_type,
            direction=recommended_dir,
            distance=0.0,  # Not used for current implementation
            reason=reason,
            priority=priority,
            confidence=confidence
        )
        
    def _should_send_instruction(self, instruction: NavigationInstruction) -> bool:
        """Determine if instruction should be sent based on timing and priority"""
        current_time = time.time()
        time_since_last = current_time - self.last_instruction_time
        
        # Always send critical instructions
        if instruction.priority == Priority.CRITICAL:
            return True
            
        # Send high priority if enough time has passed
        if instruction.priority == Priority.HIGH and time_since_last > 1.0:
            return True
            
        # Send medium/low priority based on update frequency
        if time_since_last > self.update_frequency:
            return True
            
        return False
        
    def _process_detections(self, detections: List[Dict[str, Any]]):
        """Process detection data and update spatial understanding"""
        try:
            # Assess current situation
            risk_level = self._assess_risk_level(detections)
            clear_paths = self._analyze_clear_paths(detections)
            recommended_direction = self._recommend_direction(clear_paths, detections)
            
            # Create spatial map
            spatial_map = SpatialMap(
                obstacles=detections,
                clear_paths=clear_paths,
                recommended_direction=recommended_direction,
                risk_level=risk_level,
                timestamp=time.time()
            )
            
            self.current_spatial_map = spatial_map
            
            # Generate navigation instruction
            instruction = self._create_navigation_instruction(spatial_map)
            
            # Send instruction if appropriate
            if self._should_send_instruction(instruction):
                self.send_message(MessageType.NAVIGATION_UPDATE, {
                    'instruction_type': instruction.instruction_type,
                    'direction': instruction.direction.value,
                    'reason': instruction.reason,
                    'priority': instruction.priority.value,
                    'confidence': instruction.confidence,
                    'risk_level': risk_level,
                    'clear_paths': [path.value for path in clear_paths]
                }, priority=instruction.priority.value)
                
                self.last_instruction_time = time.time()
                
            # Store in navigation history
            self.navigation_history.append({
                'timestamp': time.time(),
                'spatial_map': spatial_map,
                'instruction': instruction
            })
            
            # Keep only recent history (last 50 entries)
            if len(self.navigation_history) > 50:
                self.navigation_history.pop(0)
                
        except Exception as e:
            print(f"[{self.agent_name}] Error processing detections: {e}")
            
    def _run(self):
        """Main navigation loop"""
        print(f"[{self.agent_name}] Starting navigation loop")
        
        while self._running:
            try:
                # Navigation agent primarily responds to messages
                # The main processing happens in handle_message
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[{self.agent_name}] Error in navigation loop: {e}")
                
        print(f"[{self.agent_name}] Navigation loop stopped")
        
    def handle_message(self, message: Message):
        """Handle incoming messages"""
        try:
            if message.msg_type == MessageType.OBSTACLE_ALERT:
                # Handle critical obstacle alerts
                obstacle_data = message.data
                
                # Send immediate navigation instruction for critical obstacles
                self.send_message(MessageType.NAVIGATION_UPDATE, {
                    'instruction_type': 'avoid',
                    'direction': self._get_avoidance_direction(obstacle_data),
                    'reason': f"Immediate avoidance needed - {obstacle_data.get('object', 'obstacle')} detected",
                    'priority': Priority.CRITICAL.value,
                    'confidence': 0.9,
                    'obstacle': obstacle_data
                }, priority=Priority.CRITICAL.value)
                
            elif message.msg_type == MessageType.SYSTEM_STATUS:
                data = message.data
                
                if data.get('status') == 'perception_update':
                    # Process perception updates
                    detections = data.get('detections', [])
                    self.last_detections = detections
                    self._process_detections(detections)
                    
                elif data.get('command') == 'stop':
                    self.stop()
                    
        except Exception as e:
            print(f"[{self.agent_name}] Error handling message: {e}")
            
    def _get_avoidance_direction(self, obstacle_data: Dict[str, Any]) -> str:
        """Get the best avoidance direction for an obstacle"""
        obstacle_direction = obstacle_data.get('direction', 'center')
        
        if obstacle_direction == 'left':
            return 'right'
        elif obstacle_direction == 'right':
            return 'left'
        else:  # center
            # Choose based on current spatial understanding
            if self.current_spatial_map:
                clear_paths = self.current_spatial_map.clear_paths
                if Direction.RIGHT in clear_paths:
                    return 'right'
                elif Direction.LEFT in clear_paths:
                    return 'left'
            return 'backward'  # Default to backing up
            
    def get_current_status(self) -> Dict[str, Any]:
        """Get current navigation status"""
        if self.current_spatial_map:
            return {
                'risk_level': self.current_spatial_map.risk_level,
                'clear_paths': [path.value for path in self.current_spatial_map.clear_paths],
                'recommended_direction': self.current_spatial_map.recommended_direction.value,
                'obstacle_count': len(self.current_spatial_map.obstacles),
                'last_update': self.current_spatial_map.timestamp
            }
        return {'status': 'no_data'}
