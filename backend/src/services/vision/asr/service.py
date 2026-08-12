from typing import Optional, Dict, Any, List, Tuple
from src.core import extensions


class ASRService:

    def __init__(self):
        pass
    
    @property
    def elastic_db(self):
        """Get ElasticSearch connection dynamically."""
        return extensions.elastic_db

    def insert(self, data: Dict[str, Any]) -> Tuple[str, int]:
        try:
            result = self.elastic_db.index_data(collection_name="asr", 
                data_list=data)
            if result["total_failed"] > 0:
                return f"Error inserting data: {result['total_failed']} documents failed", 500
            return "Inserted data successfully", 200
        except Exception as e:
            return f"Error inserting data: {str(e)}", 500