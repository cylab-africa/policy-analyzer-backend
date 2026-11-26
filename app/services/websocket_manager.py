from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, processing_id: str):
        """Accept a new WebSocket connection and register it"""
        await websocket.accept()
        if processing_id not in self.active_connections:
            self.active_connections[processing_id] = []
        self.active_connections[processing_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, processing_id: str):
        """Remove a WebSocket connection"""
        if processing_id in self.active_connections:
            if websocket in self.active_connections[processing_id]:
                self.active_connections[processing_id].remove(websocket)
            # Clean up empty lists
            if not self.active_connections[processing_id]:
                del self.active_connections[processing_id]
    
    async def send_update(self, processing_id: str, message: dict):
        """Send update to all clients listening to a specific processing_id"""
        if processing_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[processing_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error sending message to client: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, processing_id)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for processing_id in list(self.active_connections.keys()):
            await self.send_update(processing_id, message)
            
manager = ConnectionManager()
def get_websocket_manager():
    """Get the WebSocket connection manager instance."""
    return manager