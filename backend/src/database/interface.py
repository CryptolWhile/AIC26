from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Database(ABC):
    """Abstract base class for all database implementations"""
    
    def __init__(self):
        self.connection = None
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> None:
        """Set up database connection"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Disconnect from the database"""
        pass
        
    @abstractmethod
    def ping(self) -> bool:
        """Check database connection"""
        pass
    
    @abstractmethod
    def create_collection(self, collection_name: str, schema: Any) -> Any:
        """Create new collection/index"""
        pass
    
    @abstractmethod
    def get_collection(self, collection_name: str) -> Any:
        """Get collection/index by name"""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete collection/index"""
        pass
    
    @abstractmethod
    def index_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        """Add data to collection"""
        pass
    
    @abstractmethod
    def delete_data(self, collection_name: str, data_id: str) -> None:
        """Delete data by ID"""
        pass
    
    @abstractmethod
    def update_data(self, collection_name: str, data: Dict[str, Any], data_id: str) -> Any:
        """Update data by ID"""
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query: Dict[str, Any]) -> Any:
        """Search in collection"""
        pass
    
    def ensure_connection(self) -> None:
        """Ensure connection, reconnect if not connected"""
        if not self.is_connected:
            self.connect() 

    