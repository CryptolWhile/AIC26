from abc import ABC, abstractmethod
from typing import Any

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def load_model(self, **kwargs) ->None:
        """Load the embedding model."""
        pass

    @abstractmethod
    def get_embedding_type(self) -> str:
        """Get the type of embedding model."""
        pass