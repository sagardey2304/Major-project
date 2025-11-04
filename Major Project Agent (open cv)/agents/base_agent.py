"""
Base Agent class and Message Bus for Multi-Agent System
Compatible with OpenCV
"""

import threading
import queue
import time
from enum import Enum
from typing import Dict, Any, Callable, List
from dataclasses import dataclass

class MessageType(Enum):
    """Types of messages that can be sent between agents"""
    OBSTACLE_ALERT = "obstacle_alert"
    NAVIGATION_UPDATE = "navigation_update"
    USER_COMMUNICATION = "user_communication"
    SYSTEM_STATUS = "system_status"
    PERCEPTION_UPDATE = "perception_update"

@dataclass
class Message:
    """Message structure for inter-agent communication"""
    type: MessageType
    sender: str
    data: Dict[str, Any]
    timestamp: float = None
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MessageBus:
    """Central message bus for agent communication"""

    def __init__(self):
        self.subscribers: Dict[MessageType, List[Callable]] = {}
        self.message_queue = queue.PriorityQueue()
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def subscribe(self, message_type: MessageType, callback: Callable):
        """Subscribe to a message type"""
        with self._lock:
            if message_type not in self.subscribers:
                self.subscribers[message_type] = []
            self.subscribers[message_type].append(callback)

    def unsubscribe(self, message_type: MessageType, callback: Callable):
        """Unsubscribe from a message type"""
        with self._lock:
            if message_type in self.subscribers:
                if callback in self.subscribers[message_type]:
                    self.subscribers[message_type].remove(callback)

    def publish(self, message: Message):
        """Publish a message to subscribers"""
        # Priority queue uses negative priority for high-priority-first
        self.message_queue.put((-message.priority, message))

    def start(self):
        """Start the message bus"""
        self._running = True
        self._thread = threading.Thread(target=self._process_messages, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the message bus"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _process_messages(self):
        """Process messages from the queue"""
        while self._running:
            try:
                # Get message with timeout
                _, message = self.message_queue.get(timeout=0.1)

                # Deliver to subscribers
                with self._lock:
                    if message.type in self.subscribers:
                        for callback in self.subscribers[message.type]:
                            try:
                                callback(message)
                            except Exception as e:
                                print(f"[MessageBus] Error in callback: {e}")

                self.message_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[MessageBus] Error processing message: {e}")

class BaseAgent:
    """Base class for all agents"""

    def __init__(self, name: str, message_bus: MessageBus, config: Dict[str, Any]):
        self.name = name
        self.message_bus = message_bus
        self.config = config
        self._running = False
        self._thread = None

    def start(self):
        """Start the agent"""
        print(f"[{self.name}] Starting agent...")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the agent"""
        print(f"[{self.name}] Stopping agent...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        """Main agent loop - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _run()")

    def send_message(self, message_type: MessageType, data: Dict[str, Any], priority: int = 1):
        """Send a message via the message bus"""
        message = Message(
            type=message_type,
            sender=self.name,
            data=data,
            priority=priority
        )
        self.message_bus.publish(message)

    def handle_message(self, message: Message):
        """Handle received messages - to be implemented by subclasses"""
        pass