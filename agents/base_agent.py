"""
Base Agent class and Message Bus system for multi-agent communication
"""
import threading
import queue
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    OBSTACLE_ALERT = "obstacle_alert"
    NAVIGATION_UPDATE = "navigation_update" 
    USER_COMMUNICATION = "user_communication"
    SYSTEM_STATUS = "system_status"
    EMERGENCY = "emergency"

@dataclass
class Message:
    msg_type: MessageType
    sender: str
    data: Dict[str, Any]
    timestamp: float = None
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=emergency
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MessageBus:
    """Central message bus for agent communication"""
    
    def __init__(self):
        self._subscribers: Dict[MessageType, List[Callable]] = {}
        self._message_queue = queue.PriorityQueue()
        self._running = False
        self._worker_thread = None
        
    def subscribe(self, msg_type: MessageType, callback: Callable[[Message], None]):
        """Subscribe to a message type"""
        if msg_type not in self._subscribers:
            self._subscribers[msg_type] = []
        self._subscribers[msg_type].append(callback)
        
    def publish(self, message: Message):
        """Publish a message to the bus"""
        # Priority queue uses negative priority for max-heap behavior
        self._message_queue.put((-message.priority, message.timestamp, message))
        
    def start(self):
        """Start the message bus worker thread"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._message_worker)
        self._worker_thread.start()
        
    def stop(self):
        """Stop the message bus"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join()
            
    def _message_worker(self):
        """Worker thread that processes messages"""
        while self._running:
            try:
                _, _, message = self._message_queue.get(timeout=0.1)
                
                # Deliver message to subscribers
                if message.msg_type in self._subscribers:
                    for callback in self._subscribers[message.msg_type]:
                        try:
                            callback(message)
                        except Exception as e:
                            print(f"Error delivering message: {e}")
                            
            except queue.Empty:
                continue

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_name: str, message_bus: MessageBus):
        self.agent_name = agent_name
        self.message_bus = message_bus
        self._running = False
        self._thread = None
        
    def start(self):
        """Start the agent"""
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        
    def stop(self):
        """Stop the agent"""
        self._running = False
        if self._thread:
            self._thread.join()
            
    def send_message(self, msg_type: MessageType, data: Dict[str, Any], priority: int = 1):
        """Send a message via the message bus"""
        message = Message(msg_type, self.agent_name, data, priority=priority)
        self.message_bus.publish(message)
        
    @abstractmethod
    def _run(self):
        """Main agent loop - to be implemented by subclasses"""
        pass
        
    @abstractmethod
    def handle_message(self, message: Message):
        """Handle incoming messages - to be implemented by subclasses"""
        pass