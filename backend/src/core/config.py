from functools import cached_property

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from pydantic_settings import BaseSettings as PydanticSettings


class BaseSettings(PydanticSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore"
    )


class MongoSettings(BaseSettings):
    MONGODB_URI: str="mongodb://localhost:27017"
    MONGO_DATABASE: str = "AIC"


class RedisSettings(BaseSettings):
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://redis:6379/0"

class FaissSettings(BaseSettings):
    pass

class ElasticSettings(BaseSettings):
    ELASTIC_URI: str = "http://elasticsearch:9200"


class MilvusSettings(BaseSettings):
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: int = 19530


class FaissSettings(BaseSettings):
    FAISS_INDEX_DIR: str = "/data/faiss"
    FAISS_METADATA_FILE: str = "/data/faiss/metadata.json"

class MetadataStoreSettings(BaseSettings):
    METADATA_STORE_DIR: str = "/data/metadata_store"

class OpenAISettings(BaseSettings):
    OPENAI_API_KEY: str


class GeminiSettings(BaseSettings):
    GEMINI_API_KEY: str


class Settings:
    """
    Centralized application settings that aggregates sub-configs.
    """

    @cached_property
    def mongo(self) -> MongoSettings:
        return MongoSettings()

    @cached_property
    def redis(self) -> RedisSettings:
        return RedisSettings()

    @cached_property
    def elastic(self) -> ElasticSettings:
        return ElasticSettings()

    @cached_property
    def milvus(self) -> MilvusSettings:
        return MilvusSettings()

    @cached_property
    def faiss(self) -> FaissSettings:
        return FaissSettings()

    @cached_property
    def metadata_store(self) -> MetadataStoreSettings:
        return MetadataStoreSettings()
    
    @cached_property
    def openai(self) -> OpenAISettings:
        return OpenAISettings()

    @cached_property
    def gemini(self) -> GeminiSettings:
        return GeminiSettings()


def load_settings() -> Settings:
    return Settings()


# class AppConfig(BaseSettings):
#     SECRET_KEY: str = "dev"
#     DATA_DIR: str = "data"
#     DEBUG: bool = True

#     # Weights
#     DIMENSIONS: int = 768
#     VECTOR_WEIGHT: float = 0.6
#     FULL_TEXT_WEIGHT: float = 0.2
#     UNIT_WEIGHT: float = 0.2