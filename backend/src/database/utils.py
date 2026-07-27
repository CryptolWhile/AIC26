from typing import Any, Dict, List, get_args, get_origin
from pydantic import BaseModel, Field
from pymilvus import FieldSchema, DataType

def pydantic_to_milvus_schema(
    model: BaseModel,
    primary_key: str = "id",
    embedding_field: str = "embedding",
    embedding_dim: int = 512
) -> List[FieldSchema]:
    """
    Convert a Pydantic model into Milvus schema (CollectionSchema),
    and automatically add an embedding field.
    Complex types (dict, list) are treated as JSON strings.
    """
    import json

    schema = model.model_json_schema()
    fields = []

    for field, field_schema in schema["properties"].items():
        field_type = field_schema.get("type", "string")
        field_description = field_schema.get("description", "string")

        if field_type == "integer":
            dtype = DataType.INT64
        elif field_type == "number":
            dtype = DataType.FLOAT
        elif field_type == "boolean":
            dtype = DataType.BOOL
        elif field_type in ["array", "object"]:
            dtype = DataType.VARCHAR
        else:  # default string
            dtype = DataType.VARCHAR

        is_primary = (field == primary_key)

        field_obj = FieldSchema(
            name=field,
            dtype=dtype,
            description=field_description,
            is_primary=is_primary,
            auto_id=False,
            max_length=65535 if dtype == DataType.VARCHAR else None  # Increased max_length for JSON data
        )
        fields.append(field_obj)

    fields.append(
        FieldSchema(
            name=embedding_field,
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_dim
        )
    )

    return fields