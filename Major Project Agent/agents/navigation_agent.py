"""
Navigation Agent - Enhanced with ReAct capabilities while maintaining original interface
"""
import time
import math
import json
import re
import requests
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

@dataclass
class ReActStep:
    """Represents a single ReAct reasoning step"""
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str

@dataclass
class NavigationTool:
    """Represents a tool available to the ReAct agent"""
    name: str
    description: str
    function: callable

class OllamaClient:
    """Client for communicating with Ollama LLM"""
    def __init__(self, model="llama2", base_url="http://localhost:11434", timeout=30):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
    
    def generate(self, prompt: str) -> str:
        """Generate response from Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            raise Exception(f"Ollama generation failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class NavigationAgent(BaseAgent):
    """Agent responsible for navigation planning and spatial analysis - Enhanced with ReAct"""
    
    def __init__(self, message_bus, config: Dict[str, Any] = None):
        super().__init__("NavigationAgent", message_bus)
        
        # Configuration
        self.config = config or {}
        self.safe_distance = self.config.get('safe_distance', 2.0)  # meters
        self.warning_distance = self.config.get('warning_distance', 3.0)  # meters
        self.path_width = self.config.get('path_width', 1.0)  # meters
        self.update_frequency = self.config.get('update_frequency', 0.5)  # seconds
        
        # ReAct/LLM Configuration
        self.use_react = self.config.get('use_react', True)  # Enable/disable ReAct
        self.ollama_model = self.config.get('ollama_model', 'gpt-oss')
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
        self.use_fallback = self.config.get('use_fallback', True)
        
        # Initialize LLM client
        self.llm_client = None
        self.react_enabled = False
        if self.use_react:
            self._initialize_llm()
        
        # State tracking
        self.current_spatial_map: Optional[SpatialMap] = None
        self.last_detections = []
        self.navigation_history = []
        self.react_history = []  # New: Store ReAct reasoning history
        self.last_instruction_time = 0
        
        # Risk assessment parameters
        self.risk_thresholds = self.config.get('risk_thresholds', {
            'low': 3.0,      # > 3m clear
            'medium': 2.0,   # 2-3m obstacles
            'high': 1.0,     # 1-2m obstacles
            'critical': 0.5  # < 1m obstacles
        })
        
        # Initialize ReAct tools (if enabled)
        if self.react_enabled:
            self.tools = self._initialize_react_tools()
        
        # Subscribe to messages
        self.message_bus.subscribe(MessageType.OBSTACLE_ALERT, self.handle_message)
        self.message_bus.subscribe(MessageType.SYSTEM_STATUS, self.handle_message)
        
        mode = "ReAct-enhanced" if self.react_enabled else "Rule-based"
        print(f"[{self.agent_name}] Navigation Agent initialized in {mode} mode")
    
    def _initialize_llm(self):
        """Initialize LLM client with fallback handling"""
        try:
            self.llm_client = OllamaClient(
                model=self.ollama_model,
                base_url=self.ollama_url
            )
            
            if self.llm_client.is_available():
                # Test with simple prompt
                test_response = self.llm_client.generate("Hello")
                if test_response:
                    self.react_enabled = True
                    print(f"[{self.agent_name}] ReAct mode enabled - LLM connected")
                else:
                    raise Exception("Empty response from LLM")
            else:
                raise Exception("Ollama service not available")
                
        except Exception as e:
            print(f"[{self.agent_name}] LLM initialization failed: {e}")
            if self.use_fallback:
                print(f"[{self.agent_name}] Continuing with rule-based navigation")
                self.react_enabled = False
            else:
                raise e
    
    def _initialize_react_tools(self) -> List[NavigationTool]:
        """Initialize available tools for ReAct agent"""
        return [
            NavigationTool(
                name="assess_risk_level",
                description="Assess risk level (1-4) based on obstacle detections",
                function=self._assess_risk_level
            ),
            NavigationTool(
                name="analyze_clear_paths", 
                description="Analyze which directions are clear for movement",
                function=self._analyze_clear_paths
            ),
            NavigationTool(
                name="get_obstacle_details",
                description="Get detailed information about obstacles",
                function=self._get_obstacle_details
            ),
            NavigationTool(
                name="calculate_path_score",
                description="Calculate safety score for a direction",
                function=self._calculate_path_score
            ),
            NavigationTool(
                name="check_emergency_stop",
                description="Check if emergency stop is needed",
                function=self._check_emergency_stop
            )
        ]
        
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

    def _get_obstacle_details(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed obstacle information"""
        if not detections:
            return {"total_obstacles": 0, "closest_distance": float('inf'), "summary": "No obstacles"}
        
        distances = [det.get('distance', 0) for det in detections]
        obstacle_summary = []
        
        for det in detections[:3]:  # Limit to 3 for brevity
            distance = det.get('distance', 0)
            direction = det.get('direction', 'unknown')
            object_type = det.get('object', 'obstacle')
            obstacle_summary.append(f"{object_type} at {distance:.1f}m {direction}")
        
        return {
            "total_obstacles": len(detections),
            "closest_distance": min(distances),
            "average_distance": sum(distances) / len(distances),
            "summary": "; ".join(obstacle_summary)
        }
    
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
    
    def _check_emergency_stop(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if emergency stop is needed"""
        critical_obstacles = [
            det for det in detections 
            if det.get('distance', float('inf')) < self.risk_thresholds['critical'] and det.get('in_path', False)
        ]
        
        return {
            "emergency_needed": len(critical_obstacles) > 0,
            "critical_count": len(critical_obstacles),
            "reason": f"Critical obstacles within {self.risk_thresholds['critical']}m" if critical_obstacles else "Safe"
        }

    # ORIGINAL RULE-BASED METHODS (kept as fallback)
    def _recommend_direction(self, clear_paths: List[Direction], current_detections: List[Dict[str, Any]]) -> Direction:
        """Recommend the best direction to move (fallback method)"""
        if not clear_paths:
            return Direction.BACKWARD
            
        if Direction.CENTER in clear_paths:
            return Direction.CENTER
            
        left_score = self._calculate_path_score(Direction.LEFT, current_detections)
        right_score = self._calculate_path_score(Direction.RIGHT, current_detections)
        
        if Direction.LEFT in clear_paths and Direction.RIGHT in clear_paths:
            return Direction.LEFT if left_score > right_score else Direction.RIGHT
        elif Direction.LEFT in clear_paths:
            return Direction.LEFT
        elif Direction.RIGHT in clear_paths:
            return Direction.RIGHT
        else:
            return Direction.FORWARD
    
    def _create_navigation_instruction(self, spatial_map: SpatialMap) -> NavigationInstruction:
        """Create navigation instruction based on spatial analysis (fallback method)"""
        risk_level = spatial_map.risk_level
        recommended_dir = spatial_map.recommended_direction
        
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
            
        clear_path_ratio = len(spatial_map.clear_paths) / 3.0
        confidence = min(0.9, clear_path_ratio * 0.8 + 0.2)
        
        return NavigationInstruction(
            instruction_type=instruction_type,
            direction=recommended_dir,
            distance=0.0,
            reason=reason,
            priority=priority,
            confidence=confidence
        )

    # REACT METHODS
    def _create_react_prompt(self, detections: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Create ReAct prompt for navigation decision"""
        tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        
        return f"""You are an AI navigation assistant for visually impaired users. Safety is paramount.

CURRENT SITUATION:
- Obstacles: {json.dumps(detections, indent=2) if detections else "None detected"}
- Previous risk: {context.get('previous_risk_level', 'unknown')}
- Last instruction: {context.get('last_instruction', 'none')}

SAFETY RULES:
1. Maintain {self.safe_distance}m from obstacles
2. Emergency stop if obstacles within {self.risk_thresholds['critical']}m
3. When uncertain, choose conservative option

AVAILABLE TOOLS:
{tools_desc}

Use this format:
Thought: [Your reasoning]
Action: [tool name]
Action Input: {{"input": "value"}}
Observation: [Tool result will appear here]

Final Answer: {{"instruction_type": "proceed|turn|avoid|stop", "direction": "forward|left|right|backward", "reason": "explanation", "priority": "low|medium|high|critical", "confidence": 0.0-1.0}}

Begin:"""

    def _parse_react_response(self, response: str) -> Tuple[List[ReActStep], Dict[str, Any]]:
        """Parse LLM ReAct response"""
        steps = []
        final_answer = None
        
        lines = response.split('\n')
        current_step = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Thought:'):
                if current_step and 'thought' in current_step:
                    steps.append(ReActStep(
                        thought=current_step.get('thought', ''),
                        action=current_step.get('action', ''),
                        action_input=current_step.get('action_input', {}),
                        observation=current_step.get('observation', '')
                    ))
                current_step = {'thought': line[8:].strip()}
                
            elif line.startswith('Action:'):
                current_step['action'] = line[7:].strip()
                
            elif line.startswith('Action Input:'):
                try:
                    current_step['action_input'] = json.loads(line[13:].strip())
                except:
                    current_step['action_input'] = {'input': line[13:].strip()}
                    
            elif line.startswith('Observation:'):
                current_step['observation'] = line[12:].strip()
                
            elif line.startswith('Final Answer:'):
                # Extract JSON from final answer
                json_start = response.find('{', response.find('Final Answer:'))
                if json_start != -1:
                    json_end = response.rfind('}') + 1
                    if json_end > json_start:
                        try:
                            final_answer = json.loads(response[json_start:json_end])
                        except:
                            # Fallback parsing
                            final_answer = {
                                "instruction_type": "stop",
                                "direction": "backward", 
                                "reason": "Failed to parse LLM response",
                                "priority": "critical",
                                "confidence": 0.1
                            }
        
        # Add final step
        if current_step and 'thought' in current_step:
            steps.append(ReActStep(
                thought=current_step.get('thought', ''),
                action=current_step.get('action', ''),
                action_input=current_step.get('action_input', {}),
                observation=current_step.get('observation', '')
            ))
        
        return steps, final_answer
    
    def _execute_react_steps(self, steps: List[ReActStep], detections: List[Dict[str, Any]]) -> List[ReActStep]:
        """Execute ReAct steps with tool calls"""
        executed_steps = []
        
        for step in steps:
            if step.action and hasattr(self, 'tools'):
                tool = next((t for t in self.tools if t.name == step.action), None)
                if tool:
                    try:
                        if step.action in ['assess_risk_level', 'analyze_clear_paths', 'get_obstacle_details', 'check_emergency_stop']:
                            result = tool.function(detections)
                        elif step.action == 'calculate_path_score':
                            direction_str = step.action_input.get('input', 'center')
                            direction = Direction(direction_str) if direction_str in [d.value for d in Direction] else Direction.CENTER
                            result = tool.function(direction, detections)
                        else:
                            result = tool.function(detections)
                        
                        step.observation = json.dumps(result) if isinstance(result, dict) else str(result)
                    except Exception as e:
                        step.observation = f"Tool error: {str(e)}"
                else:
                    step.observation = f"Unknown tool: {step.action}"
            
            executed_steps.append(step)
        
        return executed_steps

    def _validate_llm_instruction(self, final_answer: Dict[str, Any], detections: List[Dict[str, Any]]) -> bool:
        """Validate LLM instruction for safety"""
        # Check for critical obstacles
        critical_obstacles = [
            det for det in detections 
            if det.get('distance', float('inf')) < self.risk_thresholds['critical'] and det.get('in_path', False)
        ]
        
        if critical_obstacles:
            instruction_type = final_answer.get('instruction_type', '')
            direction = final_answer.get('direction', '')
            if instruction_type != 'stop' and direction != 'backward':
                print(f"[{self.agent_name}] Safety violation: Critical obstacles but LLM suggested {instruction_type} {direction}")
                return False
        
        return True

    def _create_instruction_from_llm(self, final_answer: Dict[str, Any]) -> NavigationInstruction:
        """Create NavigationInstruction from LLM response"""
        direction_map = {
            'left': Direction.LEFT, 'right': Direction.RIGHT, 'center': Direction.CENTER,
            'forward': Direction.FORWARD, 'backward': Direction.BACKWARD
        }
        priority_map = {
            'low': Priority.LOW, 'medium': Priority.MEDIUM, 
            'high': Priority.HIGH, 'critical': Priority.CRITICAL
        }
        
        return NavigationInstruction(
            instruction_type=final_answer.get('instruction_type', 'stop'),
            direction=direction_map.get(final_answer.get('direction', 'backward'), Direction.BACKWARD),
            distance=0.0,
            reason=final_answer.get('reason', 'LLM navigation decision'),
            priority=priority_map.get(final_answer.get('priority', 'critical'), Priority.CRITICAL),
            confidence=float(final_answer.get('confidence', 0.5))
        )

    def _process_detections(self, detections: List[Dict[str, Any]]):
        """Main processing method - uses ReAct if available, fallback otherwise"""
        if self.react_enabled:
            self._process_detections_react(detections)
        else:
            self._process_detections_fallback(detections)

    def _process_detections_react(self, detections: List[Dict[str, Any]]):
        """Process detections using ReAct reasoning"""
        try:
            # Prepare context
            context = {
                'previous_risk_level': self.current_spatial_map.risk_level if self.current_spatial_map else None,
                'last_instruction': self.navigation_history[-1]['instruction'].instruction_type if self.navigation_history else None
            }
            
            # Generate and execute ReAct
            prompt = self._create_react_prompt(detections, context)
            llm_response = self.llm_client.generate(prompt)
            react_steps, final_answer = self._parse_react_response(llm_response)
            executed_steps = self._execute_react_steps(react_steps, detections)
            
            if final_answer and self._validate_llm_instruction(final_answer, detections):
                instruction = self._create_instruction_from_llm(final_answer)
            else:
                print(f"[{self.agent_name}] LLM decision invalid, using fallback")
                return self._process_detections_fallback(detections)
            
            # Update spatial map
            self.current_spatial_map = SpatialMap(
                obstacles=detections,
                clear_paths=self._analyze_clear_paths(detections),
                recommended_direction=instruction.direction,
                risk_level=self._assess_risk_level(detections),
                timestamp=time.time()
            )
            
            # Send instruction
            if self._should_send_instruction(instruction):
                self.send_message(MessageType.NAVIGATION_UPDATE, {
                    'instruction_type': instruction.instruction_type,
                    'direction': instruction.direction.value,
                    'reason': instruction.reason,
                    'priority': instruction.priority.value,
                    'confidence': instruction.confidence,
                    'risk_level': self.current_spatial_map.risk_level,
                    'clear_paths': [path.value for path in self.current_spatial_map.clear_paths],
                    'reasoning_steps': [step.thought for step in executed_steps]
                }, priority=instruction.priority.value)
                
                self.last_instruction_time = time.time()
            
            # Store history
            self.react_history.append({
                'timestamp': time.time(),
                'steps': executed_steps,
                'final_answer': final_answer,
                'instruction': instruction
            })
            
            self.navigation_history.append({
                'timestamp': time.time(),
                'spatial_map': self.current_spatial_map,
                'instruction': instruction
            })
            
            # Limit history size
            if len(self.react_history) > 20:
                self.react_history.pop(0)
            if len(self.navigation_history) > 50:
                self.navigation_history.pop(0)
                
        except Exception as e:
            print(f"[{self.agent_name}] ReAct processing failed: {e}, using fallback")
            self._process_detections_fallback(detections)

    def _process_detections_fallback(self, detections: List[Dict[str, Any]]):
        """Original rule-based processing (fallback)"""
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
            
            # Keep only recent history
            if len(self.navigation_history) > 50:
                self.navigation_history.pop(0)
                
        except Exception as e:
            print(f"[{self.agent_name}] Error processing detections: {e}")
            
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
        
    def _run(self):
        """Main navigation loop"""
        print(f"[{self.agent_name}] Starting navigation loop")
        
        while self._running:
            try:
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
                detections = [obstacle_data]
                self._process_detections(detections)
                
            elif message.msg_type == MessageType.SYSTEM_STATUS:
                data = message.data
                
                if data.get('status') == 'perception_update':
                    detections = data.get('detections', [])
                    self.last_detections = detections
                    self._process_detections(detections)
                    
                elif data.get('command') == 'stop':
                    self.stop()
                    
        except Exception as e:
            print(f"[{self.agent_name}] Error handling message: {e}")
            
    def get_current_status(self) -> Dict[str, Any]:
        """Get current navigation status"""
        status = {
            'react_enabled': self.react_enabled,
            'llm_model': self.ollama_model if self.react_enabled else None
        }
        
        if self.current_spatial_map:
            status.update({
                'risk_level': self.current_spatial_map.risk_level,
                'clear_paths': [path.value for path in self.current_spatial_map.clear_paths],
                'recommended_direction': self.current_spatial_map.recommended_direction.value,
                'obstacle_count': len(self.current_spatial_map.obstacles),
                'last_update': self.current_spatial_map.timestamp
            })
        else:
            status['status'] = 'no_data'
            
        return status