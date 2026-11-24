"""Hybrid keyword and vector retrieval using the Azure AI Search REST API."""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from config import get_config
from src.data_pipeline.embedder import Embedder
from src.retrieval.filters import build_source_file_filter
from src.utils.azure_cli_credential import AzureCliCredentialLite


class RetrievalError(Exception):
    """Raised when retrieval fails."""


class AzureSearchRetriever:
    """Retrieve Top-K chunks with hybrid BM25 and vector search."""

    def __init__(self, index_name: str = None, top_k: int = None):
        config = get_config()
        self.index_name = index_name or config.search_index_name
        self.top_k = top_k or config.retrieval_top_k
        self.endpoint = config.search_endpoint.rstrip("/")
        self.api_version = config.search_api_version
        self.api_key = config.search_admin_key
        self.use_azure_cli = config.search_use_azure_cli
        self.credential = AzureCliCredentialLite() if self.use_azure_cli else None
        self.embedder = Embedder()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.use_azure_cli:
            token = self.credential.get_token("https://search.azure.com/.default")
            headers["Authorization"] = f"Bearer {token.token}"
        else:
            headers["api-key"] = self.api_key
        return headers

    def _search(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        name = urllib.parse.quote(self.index_name)
        url = f"{self.endpoint}/indexes/{name}/docs/search?api-version={self.api_version}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8")).get("value", [])
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RetrievalError(
                f"Azure Search returned HTTP {exc.code}: {details}"
            ) from exc

    @staticmethod
    def _process_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        documents = []
        for result in results:
            try:
                metadata = json.loads(result.get("metadata", "{}"))
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            documents.append({
                "id": result.get("id"),
                "content": result.get("content"),
                "source_file": result.get("source_file"),
                "chunk_index": result.get("chunk_index"),
                "content_type": result.get("content_type"),
                "page_number": result.get("page_number"),
                "score": result.get("@search.score", 0.0),
                "metadata": metadata,
            })
        return documents

    def retrieve(self, query: str, top_k: int = None,
                 use_hybrid: bool = True,
                 source_file: str = None) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        k = top_k or self.top_k
        payload: Dict[str, Any] = {
            "search": query if use_hybrid else None,
            "vectorQueries": [{
                "kind": "vector",
                "vector": self.embedder.embed_text(query),
                "k": k,
                "fields": "content_vector",
            }],
            "top": k,
            "select": "id,content,source_file,chunk_index,content_type,page_number,metadata",
        }
        filter_expr = build_source_file_filter(source_file)
        if filter_expr:
            payload["filter"] = filter_expr
        return self._process_results(self._search(payload))

    def retrieve_with_filter(self, query: str, filter_expr: str,
                             top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        payload = {
            "search": query,
            "vectorQueries": [{
                "kind": "vector",
                "vector": self.embedder.embed_text(query),
                "k": k,
                "fields": "content_vector",
            }],
            "filter": filter_expr,
            "top": k,
            "select": "id,content,source_file,chunk_index,content_type,page_number,metadata",
        }
        return self._process_results(self._search(payload))

    def get_context_for_query(self, query: str, top_k: int = None) -> str:
        documents = self.retrieve(query, top_k=top_k)
        if not documents:
            return "No relevant documents found."
        sections = []
        for index, document in enumerate(documents, start=1):
            sections.append(
                f"[Source {index}: {document['source_file']}, "
                f"page {document.get('page_number') or 'unknown'}, "
                f"type {document.get('content_type') or 'text'}]\n"
                f"{document['content']}"
            )
        return "\n\n".join(sections)
