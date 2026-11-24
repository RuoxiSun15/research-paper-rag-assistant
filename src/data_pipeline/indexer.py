"""Create and populate an Azure AI Search vector index via the REST API."""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from config import get_config
from src.utils.azure_cli_credential import AzureCliCredentialLite


class IndexingError(Exception):
    """Raised when indexing operations fail."""


class AzureSearchIndexer:
    """Manage the project's hybrid text and vector index."""

    def __init__(self, index_name: str = None):
        config = get_config()
        self.index_name = index_name or config.search_index_name
        self.endpoint = config.search_endpoint.rstrip("/")
        self.api_version = config.search_api_version
        self.api_key = config.search_admin_key
        self.use_azure_cli = config.search_use_azure_cli
        self.credential = AzureCliCredentialLite() if self.use_azure_cli else None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.use_azure_cli:
            token = self.credential.get_token("https://search.azure.com/.default")
            headers["Authorization"] = f"Bearer {token.token}"
        else:
            headers["api-key"] = self.api_key
        return headers

    def _request(self, method: str, path: str, payload: Dict[str, Any] = None,
                 allow_not_found: bool = False) -> Dict[str, Any]:
        separator = "&" if "?" in path else "?"
        url = f"{self.endpoint}{path}{separator}api-version={self.api_version}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers=self._headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
                return json.loads(body.decode("utf-8")) if body else {}
        except urllib.error.HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return {}
            details = exc.read().decode("utf-8", errors="replace")
            raise IndexingError(
                f"Azure Search returned HTTP {exc.code}: {details}"
            ) from exc

    def _index_path(self) -> str:
        return f"/indexes/{urllib.parse.quote(self.index_name)}"

    def _create_index_schema(self) -> Dict[str, Any]:
        return {
            "name": self.index_name,
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
                {"name": "content", "type": "Edm.String", "searchable": True, "analyzer": "en.microsoft"},
                {"name": "content_vector", "type": "Collection(Edm.Single)", "searchable": True, "dimensions": 1536, "vectorSearchProfile": "vector-profile"},
                {"name": "source_file", "type": "Edm.String", "searchable": True, "filterable": True, "facetable": True},
                {"name": "chunk_index", "type": "Edm.Int32", "filterable": True, "sortable": True},
                {"name": "chunk_size", "type": "Edm.Int32", "filterable": True},
                {"name": "content_type", "type": "Edm.String", "filterable": True, "facetable": True},
                {"name": "page_number", "type": "Edm.Int32", "filterable": True, "sortable": True},
                {"name": "formula_latex", "type": "Edm.String", "searchable": True},
                {"name": "figure_caption", "type": "Edm.String", "searchable": True},
                {"name": "figure_visual_summary", "type": "Edm.String", "searchable": True},
                {"name": "metadata", "type": "Edm.String"},
            ],
            "vectorSearch": {
                "algorithms": [{
                    "name": "hnsw-config",
                    "kind": "hnsw",
                    "hnswParameters": {"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"},
                }],
                "profiles": [{"name": "vector-profile", "algorithm": "hnsw-config"}],
            },
        }

    def create_index(self, recreate: bool = False) -> bool:
        existing = self._request("GET", self._index_path(), allow_not_found=True)
        if existing and not recreate:
            return False
        if existing and recreate:
            self._request("DELETE", self._index_path())
        self._request("PUT", self._index_path(), self._create_index_schema())
        return True

    def _prepare_documents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        documents = []
        for chunk in chunks:
            for required in ("chunk_id", "content", "embedding"):
                if required not in chunk:
                    raise ValueError(f"Chunk missing '{required}' field")
            metadata = chunk.get("metadata", {})
            documents.append({
                "@search.action": "upload",
                "id": chunk["chunk_id"],
                "content": chunk["content"],
                "content_vector": chunk["embedding"],
                "source_file": metadata.get("source_file", "unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "chunk_size": metadata.get("chunk_size", len(chunk["content"])),
                "content_type": metadata.get("content_type", "text"),
                "page_number": metadata.get("page_number"),
                "formula_latex": metadata.get("formula_latex"),
                "figure_caption": metadata.get("figure_caption"),
                "figure_visual_summary": metadata.get("figure_visual_summary"),
                "metadata": json.dumps(metadata, default=str),
            })
        return documents

    def index_documents(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {"indexed": 0, "failed": 0, "total": 0}
        documents = self._prepare_documents(chunks)
        indexed = failed = 0
        for offset in range(0, len(documents), 100):
            result = self._request(
                "POST",
                f"/indexes/{urllib.parse.quote(self.index_name)}/docs/index",
                {"value": documents[offset:offset + 100]},
            )
            statuses = result.get("value", [])
            indexed += sum(1 for item in statuses if item.get("status"))
            failed += sum(1 for item in statuses if not item.get("status"))
        return {"indexed": indexed, "failed": failed, "total": len(documents)}

    def get_index_statistics(self) -> Dict[str, Any]:
        index = self._request("GET", self._index_path(), allow_not_found=True)
        if not index:
            return {"index_name": self.index_name, "exists": False, "document_count": 0}
        result = self._request("POST", f"/indexes/{urllib.parse.quote(self.index_name)}/docs/search", {"search": "*", "count": True, "top": 0})
        return {
            "index_name": self.index_name,
            "exists": True,
            "document_count": result.get("@odata.count", 0),
            "fields": [field["name"] for field in index.get("fields", [])],
            "vector_search_enabled": bool(index.get("vectorSearch")),
        }

    def delete_documents_by_source(self, source_file: str) -> int:
        """Delete every indexed chunk belonging to one source PDF."""
        escaped_source = source_file.replace("'", "''")
        result = self._request(
            "POST",
            f"/indexes/{urllib.parse.quote(self.index_name)}/docs/search",
            {
                "search": "*",
                "filter": f"source_file eq '{escaped_source}'",
                "select": "id",
                "top": 1000,
            },
        )
        documents = result.get("value", [])
        if not documents:
            return 0
        self._request(
            "POST",
            f"/indexes/{urllib.parse.quote(self.index_name)}/docs/index",
            {
                "value": [
                    {"@search.action": "delete", "id": document["id"]}
                    for document in documents
                ]
            },
        )
        return len(documents)

    def list_document_sources(self) -> List[Dict[str, Any]]:
        """List indexed PDF names and their number of searchable chunks."""
        result = self._request(
            "POST",
            f"/indexes/{urllib.parse.quote(self.index_name)}/docs/search",
            {
                "search": "*",
                "facets": ["source_file,count:100"],
                "top": 0,
            },
        )
        facets = result.get("@search.facets", {}).get("source_file", [])
        return [
            {"source_file": item["value"], "chunk_count": item["count"]}
            for item in facets
        ]

    def delete_index(self) -> bool:
        existing = self._request("GET", self._index_path(), allow_not_found=True)
        if not existing:
            return False
        self._request("DELETE", self._index_path())
        return True
