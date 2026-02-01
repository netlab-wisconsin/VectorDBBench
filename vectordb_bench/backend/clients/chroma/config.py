from chromadb.config import Settings
from pydantic import SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class ChromaConfig(DBConfig):
    user: str | None = None
    password: SecretStr | None
    host: SecretStr = "localhost"
    port: int = 8000

    def to_dict(self) -> dict:
        config = {
            "host": self.host.get_secret_value(),
            "port": self.port,
        }
        if self.password and self.user:
            config["settings"] = Settings(
                settings=Settings(
                    chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                    chroma_client_auth_credentials=f"{self.user}:{self.password}",
                )
            )
        return config


class ChromaIndexConfig(ChromaConfig, DBCaseConfig):
    metric_type: MetricType = "cosine"
    m: int = 16
    ef_construct: int = 100
    ef_search: int | None = 100

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        if self.metric_type == MetricType.IP:
            return "ip"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        raise ValueError(f"Unsupported metric type: {self.metric_type}")

    # ---- New helpers (metadata-based) ----
    def create_collection_metadata(self) -> dict:
        md: dict = {
            "hnsw:space": self.parse_metric(),
            "hnsw:M": int(self.m),
            "hnsw:construction_ef": int(self.ef_construct),
        }
        if self.ef_search is not None:
            md["hnsw:search_ef"] = int(self.ef_search)
        return md

    def update_search_metadata(self) -> dict:
        if self.ef_search is None:
            return {}
        return {"hnsw:search_ef": int(self.ef_search)}

    # ---- Required by DBCaseConfig (abstract methods) ----
    def index_param(self) -> dict:
        """
        Keep the old interface required by VectorDBBench (DBCaseConfig),
        but we will NOT pass this dict to Chroma's `configuration=...`.
        We'll pass it as metadata in ChromaClient.
        """
        return self.create_collection_metadata()

    def search_param(self) -> dict:
        return self.update_search_metadata()
