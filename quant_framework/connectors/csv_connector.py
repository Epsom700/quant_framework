"""CSV/Parquet file connector implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from quant_framework.connectors.connectors import BaseConnector, ConnectorRegistry

logger = logging.getLogger(__name__)

@ConnectorRegistry.register("csv")
class CSVConnector(BaseConnector):
    """
    Connector for local CSV and Parquet files.
    
    Usage::
        conn = CSVConnector()
        conn.connect({"file_path": "path/to/data.csv"})
        df = conn.query()
    """

    def __init__(self) -> None:
        self.file_path: Optional[Path] = None
        self._connected = False

    @property
    def name(self) -> str:
        return "csv"

    def connect(self, config: Dict[str, Any]) -> None:
        """
        Connect to a local file.
        
        Args:
            config: Must contain ``file_path``.
        """
        path_str = config.get("file_path")
        if not path_str:
            raise ValueError("config must contain 'file_path'")
            
        path = Path(path_str)
        if not path.exists():
            raise ConnectionError(f"File not found: {path_str}")
            
        self.file_path = path
        self._connected = True
        logger.info(f"Connected to file: {path}")

    def query(self, request: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Read the file into a pandas DataFrame.
        
        Args:
            request: Ignored for this connector.
            **kwargs: Forwarded to ``pd.read_csv`` or ``pd.read_parquet``.
            
        Returns:
            A DataFrame containing the file contents.
        """
        if not self._connected or self.file_path is None:
            raise RuntimeError("Not connected. Call connect(config) first.")
            
        if self.file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(self.file_path, **kwargs)
        else:
            return pd.read_csv(self.file_path, **kwargs)

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the column names of the file.
        """
        if not self._connected or self.file_path is None:
            return {"source": "csv", "columns": []}
            
        try:
            # Read only headers to get schema quickly
            if self.file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(self.file_path, engine="pyarrow")
                columns = list(df.columns)
            else:
                columns = list(pd.read_csv(self.file_path, nrows=0).columns)
            return {"source": "csv", "columns": columns}
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return {"source": "csv", "columns": [], "error": str(e)}

    def health_check(self) -> bool:
        """Return True if the file exists and is a file."""
        if self.file_path is None:
            return False
        return self.file_path.exists() and self.file_path.is_file()
