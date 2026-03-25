from abc import ABC, abstractmethod
from typing import Dict, List, Type, Any, Optional, Union, Callable
import pandas as pd


class BaseConnector(ABC):
    """Abstract base class for all connectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the connector."""
        pass

    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> None:
        """
        Initialize the connection to the data source.
        
        Args:
            config: A dictionary containing connection parameters.
        """
        pass

    @abstractmethod
    def query(self, request: str, **kwargs: Any) -> pd.DataFrame:
        """
        Query the data source and return the result as a pandas DataFrame.
        
        Args:
            request: The query string or request identifier.
            **kwargs: Additional parameters for the query.
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return the schema or metadata of the data source.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the connection to the data source is healthy.
        """
        pass


class ConnectorRegistry:
    """Registry for managing and accessing connectors."""
    
    _connectors: Dict[str, Type[BaseConnector]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable[[Type[BaseConnector]], Type[BaseConnector]]:
        """
        Decorator to register a connector class.
        
        Args:
            name: Optional name for the connector. If not provided, the connector's name property is used if available.
        """
        def decorator(connector_cls: Type[BaseConnector]) -> Type[BaseConnector]:
            reg_name = name
            if not reg_name:
                # Try to get the name property from the class if it's set as a class attribute
                # or fallback to the class name
                reg_name = getattr(connector_cls, 'name', None)
                if not isinstance(reg_name, str):
                    reg_name = connector_cls.__name__
            
            cls._connectors[reg_name] = connector_cls
            return connector_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseConnector]]:
        """
        Retrieve a connector class by name.
        """
        return cls._connectors.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """
        List all registered connector names.
        """
        return list(cls._connectors.keys())
