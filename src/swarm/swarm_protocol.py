
import logging
import threading
import time
import queue
import json
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"

class Message:
    """Message for agent communication"""
    def __init__(self, sender: str, receiver: str, message_type: str, payload: Any, message_id: Optional[str] = None):
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.payload = payload
        self.timestamp = time.time()
        self.message_id = message_id or str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            message_type=data["message_type"],
            payload=data["payload"],
            message_id=data.get("message_id")
        )

class SwarmAgent:
    """Base class for swarm agents"""
    def __init__(self, agent_id: str, swarm: 'SwarmCoordinator'):
        self.agent_id = agent_id
        self.swarm = swarm
        self.status = AgentStatus.IDLE
        self.inbox = queue.Queue()
        self.message_handlers = {}
        self.result = None
        self.error = None
        
    def send_message(self, receiver: str, message_type: str, payload: Any) -> str:
        """Send a message to another agent"""
        message = Message(self.agent_id, receiver, message_type, payload)
        self.swarm.route_message(message)
        return message.message_id
    
    def receive_message(self, message: Message):
        """Receive a message from another agent"""
        self.inbox.put(message)
    
    def register_handler(self, message_type: str, handler: Callable[[Message], None]):
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
    
    def process_messages(self):
        """Process all messages in the inbox"""
        while not self.inbox.empty():
            message = self.inbox.get()
            if message.message_type in self.message_handlers:
                try:
                    self.message_handlers[message.message_type](message)
                except Exception as e:
                    logger.error(f"Error processing message in agent {self.agent_id}: {str(e)}")
            else:
                logger.warning(f"No handler for message type {message.message_type} in agent {self.agent_id}")
    
    def run(self, **kwargs):
        """Main execution method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run method")
    
    def update_status(self, status: AgentStatus):
        """Update agent status"""
        self.status = status
        self.swarm.update_agent_status(self.agent_id, status)

class SwarmCoordinator:
    """Coordinates communication between swarm agents"""
    def __init__(self, config):
        self.config = config
        self.agents = {}
        self.agent_threads = {}
        self.agent_statuses = {}
        self.message_log = []
        self.agent_results = {}
        self.lock = threading.Lock()
    
    def register_agent(self, agent: SwarmAgent):
        """Register an agent with the swarm"""
        with self.lock:
            self.agents[agent.agent_id] = agent
            self.agent_statuses[agent.agent_id] = AgentStatus.IDLE
            logger.info(f"Registered agent: {agent.agent_id}")
    
    def route_message(self, message: Message):
        """Route a message to the appropriate agent"""
        with self.lock:
            self.message_log.append(message.to_dict())
        
        if message.receiver in self.agents:
            self.agents[message.receiver].receive_message(message)
            logger.debug(f"Routed message: {message.message_type} from {message.sender} to {message.receiver}")
        else:
            logger.warning(f"Unknown recipient for message: {message.receiver}")
    
    def broadcast_message(self, sender: str, message_type: str, payload: Any):
        """Broadcast a message to all agents except the sender"""
        for agent_id, agent in self.agents.items():
            if agent_id != sender:
                message = Message(sender, agent_id, message_type, payload)
                agent.receive_message(message)
        logger.debug(f"Broadcasted message: {message_type} from {sender}")
    
    def run_agent(self, agent_id: str, **kwargs):
        """Run an agent in a separate thread"""
        if agent_id not in self.agents:
            logger.error(f"Unknown agent: {agent_id}")
            return
        
        agent = self.agents[agent_id]
        
        def agent_runner():
            try:
                agent.update_status(AgentStatus.RUNNING)
                result = agent.run(**kwargs)
                self.agent_results[agent_id] = result
                agent.update_status(AgentStatus.COMPLETED)
            except Exception as e:
                logger.error(f"Error in agent {agent_id}: {str(e)}")
                agent.error = str(e)
                agent.update_status(AgentStatus.ERROR)
        
        thread = threading.Thread(target=agent_runner)
        self.agent_threads[agent_id] = thread
        thread.start()
        logger.info(f"Started agent: {agent_id}")
    
    def run_agents(self, agent_ids: List[str] = None, **kwargs):
        """Run multiple agents in parallel"""
        if agent_ids is None:
            agent_ids = list(self.agents.keys())
        
        for agent_id in agent_ids:
            self.run_agent(agent_id, **kwargs)
    
    def update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update an agent's status"""
        with self.lock:
            self.agent_statuses[agent_id] = status
            
            # Update streamlit session state if in web context
            try:
                import streamlit as st
                if 'agent_status' in st.session_state:
                    st.session_state.agent_status[agent_id] = status.value
            except Exception:
                pass
    
    def wait_for_agents(self, agent_ids: List[str] = None, timeout: int = None):
        """Wait for agents to complete"""
        if agent_ids is None:
            agent_ids = list(self.agent_threads.keys())
        
        start_time = time.time()
        
        for agent_id in agent_ids:
            if agent_id in self.agent_threads:
                thread = self.agent_threads[agent_id]
                remaining_time = None
                
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_time = max(0, timeout - elapsed)
                
                thread.join(remaining_time)
                
                if thread.is_alive():
                    logger.warning(f"Agent {agent_id} did not complete in time")
                    if agent_id in self.agents:
                        self.agents[agent_id].update_status(AgentStatus.ERROR)
    
    def get_agent_status(self, agent_id: str) -> AgentStatus:
        """Get an agent's status"""
        with self.lock:
            return self.agent_statuses.get(agent_id, AgentStatus.IDLE)
    
    def get_all_statuses(self) -> Dict[str, AgentStatus]:
        """Get all agent statuses"""
        with self.lock:
            return self.agent_statuses.copy()
    
    def get_agent_result(self, agent_id: str) -> Any:
        """Get an agent's result"""
        return self.agent_results.get(agent_id)
    
    def get_message_log(self) -> List[Dict[str, Any]]:
        """Get the message log"""
        with self.lock:
            return self.message_log.copy()
    
    def visualize_swarm(self):
        """Generate a visualization of the swarm"""
        # This would generate a network graph visualization
        # Placeholder implementation
        return {
            "nodes": [{"id": agent_id, "status": status.value} for agent_id, status in self.agent_statuses.items()],
            "edges": [{"from": msg["sender"], "to": msg["receiver"]} for msg in self.message_log if msg["sender"] in self.agent_statuses and msg["receiver"] in self.agent_statuses]
        }
