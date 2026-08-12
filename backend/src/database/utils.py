from typing import Any, Dict, List, get_args, get_origin
from pydantic import BaseModel, Field
from pymilvus import FieldSchema, DataType


def pydantic_to_mongo_schema(model: BaseModel) -> Dict[str, Any]:
    """Convert a Pydantic model into MongoDB JSON Schema."""

    schema = model.model_json_schema()

    properties = {}
    required = schema.get("required", [])

    for field, field_schema in schema["properties"].items():
        field_type = field_schema.get("type", "string")

        if field_type == "array":
            bson_type = "array"
        elif field_type == "boolean":
            bson_type = "bool"
        elif field_type == "integer":
            bson_type = "int"
        elif field_type == "number":
            bson_type = "double"
        elif field_type == "object":
            bson_type = "object"
        else:
            bson_type = "string"

        properties[field] = {"bsonType": bson_type}

    return {
        "bsonType": "object",
        "required": required,
        "properties": properties
    }


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


def pydantic_to_elastic_schema(model: BaseModel) -> Dict[str, Any]:
    """
    Convert a Pydantic model to Elasticsearch index mapping.
    """
    properties = {}

    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        es_type = None

        if "es_type" in field.metadata:
            es_type = field.metadata["es_type"]

        if es_type is None:
            if field_type == str:
                es_type = "text"
            elif field_type == int:
                es_type = "integer"
            elif field_type == float:
                es_type = "float"
            elif field_type == bool:
                es_type = "boolean"
            elif get_origin(field_type) == list:
                inner_type = get_args(field_type)[0]
                if inner_type == str:
                    es_type = "text"
                elif inner_type == int:
                    es_type = "integer"
                elif inner_type == float:
                    es_type = "float"
                else:
                    es_type = "object"
            else:
                es_type = "object"

        properties[field_name] = {"type": es_type}

    return {"mappings": {"properties": properties}}