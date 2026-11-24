"""
Embedding Generation Module using Azure AI Foundry.

This module generates embeddings (vector representations) for text chunks
using Azure AI Foundry's deployed embedding model (text-embedding-ada-002).

IMPORTANT: This uses Azure AI Foundry endpoints, NOT Azure OpenAI Service.
"""

import logging
import json
import urllib.error
import urllib.request
from typing import List, Dict, Any
import time

from config import get_config
from src.utils.azure_cli_credential import AzureCliCredentialLite

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class Embedder:
    """
    Generates embeddings using Azure AI Foundry's embedding model.

    CRITICAL NOTE:
        This class uses Azure AI Foundry endpoints, which are different from
        Azure OpenAI Service. Each deployed model in Azure AI Foundry has its
        own endpoint, deployment name, and API key.
    """

    # Embedding dimension for text-embedding-ada-002
    EMBEDDING_DIMENSION = 1536

    # Batch size for embedding API calls (Azure OpenAI limit)
    MAX_BATCH_SIZE = 16
    BATCH_DELAY_SECONDS = 0.1

    def __init__(self):
        """Initialize the Azure AI Foundry embedding client."""
        config = get_config()

        try:
            self.endpoint = config.ai_foundry_embedding_endpoint.rstrip("/")
            self.deployment_name = config.ai_foundry_embedding_deployment
            self.api_version = config.ai_foundry_embedding_api_version
            self.api_key = config.ai_foundry_embedding_key
            self.use_azure_cli = config.ai_foundry_embedding_use_azure_cli
            self.credential = AzureCliCredentialLite() if self.use_azure_cli else None

            logger.info(
                f"Embedder initialized with Azure AI Foundry deployment: {self.deployment_name}"
            )
            logger.info(f"Embedding dimension: {self.EMBEDDING_DIMENSION}")

        except Exception as e:
            logger.error(f"Failed to initialize Embedder: {e}")
            raise EmbeddingError(f"Initialization failed: {e}")

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Call the Azure embeddings REST endpoint without persisting secrets."""
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment_name}"
            f"/embeddings?api-version={self.api_version}"
        )
        headers = {"Content-Type": "application/json"}
        if self.use_azure_cli:
            token = self.credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )
            headers["Authorization"] = f"Bearer {token.token}"
        else:
            headers["api-key"] = self.api_key

        body = json.dumps({
            "input": texts,
            "model": self.deployment_name,
            "dimensions": self.EMBEDDING_DIMENSION,
        }).encode("utf-8")
        for attempt in range(6):
            request = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code != 429 or attempt == 5:
                    raise EmbeddingError(
                        f"Embedding service returned HTTP {exc.code}: {details}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After")
                # Azure's rolling window can still be closed at the exact
                # Retry-After boundary, so leave a small safety margin.
                wait_seconds = (int(retry_after) if retry_after else 60) + 10
                logger.warning(
                    "Embedding rate limit reached; retrying batch in %ss",
                    wait_seconds,
                )
                time.sleep(wait_seconds)
            except urllib.error.URLError as exc:
                if attempt == 5:
                    raise EmbeddingError(
                        f"Embedding service connection failed: {exc}"
                    ) from exc
                wait_seconds = 5 * (attempt + 1)
                logger.warning(
                    "Temporary embedding connection error; retrying in %ss: %s",
                    wait_seconds,
                    exc,
                )
                time.sleep(wait_seconds)

        return [
            item["embedding"]
            for item in sorted(payload["data"], key=lambda item: item["index"])
        ]

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats (dimension: 1536)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            embedding = self._create_embeddings([text])[0]

            # Validate embedding dimension
            if len(embedding) != self.EMBEDDING_DIMENSION:
                raise EmbeddingError(
                    f"Unexpected embedding dimension: {len(embedding)}, "
                    f"expected {self.EMBEDDING_DIMENSION}"
                )

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Processes texts in batches to avoid API limits and improve efficiency.

        Args:
            texts: List of texts to embed
            show_progress: Whether to log progress

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        logger.info(f"Starting batch embedding for {len(texts)} texts")

        all_embeddings = []
        total_batches = (len(texts) + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE

        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i:i + self.MAX_BATCH_SIZE]
            batch_num = (i // self.MAX_BATCH_SIZE) + 1

            if show_progress:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

            try:
                # Call Azure AI Foundry embedding API
                batch_embeddings = self._create_embeddings(batch)

                # Validate all embeddings
                for idx, embedding in enumerate(batch_embeddings):
                    if len(embedding) != self.EMBEDDING_DIMENSION:
                        raise EmbeddingError(
                            f"Unexpected embedding dimension at index {i+idx}: "
                            f"{len(embedding)}, expected {self.EMBEDDING_DIMENSION}"
                        )

                all_embeddings.extend(batch_embeddings)

                # Configurable delay for low-quota Azure deployments.
                if i + self.MAX_BATCH_SIZE < len(texts):
                    time.sleep(self.BATCH_DELAY_SECONDS)

            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                raise EmbeddingError(f"Batch embedding failed at batch {batch_num}: {e}")

        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks.

        Adds 'embedding' field to each chunk dictionary.

        Args:
            chunks: List of chunk dictionaries from TextChunker

        Returns:
            List of chunks with embeddings added

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []

        logger.info(f"Embedding {len(chunks)} chunks")

        try:
            # Extract texts from chunks
            texts = [chunk["content"] for chunk in chunks]

            # Generate embeddings
            embeddings = self.embed_batch(texts)

            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding

            logger.info("Successfully added embeddings to all chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            raise EmbeddingError(f"Chunk embedding failed: {e}")

    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding configuration.

        Returns:
            Dictionary with embedding model information
        """
        config = get_config()

        return {
            "model_deployment": self.deployment_name,
            "embedding_dimension": self.EMBEDDING_DIMENSION,
            "max_batch_size": self.MAX_BATCH_SIZE,
            "endpoint": config.ai_foundry_embedding_endpoint,
            "api_version": config.ai_foundry_embedding_api_version,
            "note": "Using Azure AI Foundry (not Azure OpenAI Service)"
        }


def embed_text(text: str) -> List[float]:
    """
    Convenience function to generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector

    Example:
        >>> embedding = embed_text("Hello, world!")
        >>> print(f"Dimension: {len(embedding)}")
    """
    embedder = Embedder()
    return embedder.embed_text(text)


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to embed multiple chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Chunks with embeddings added

    Example:
        >>> chunks_with_embeddings = embed_chunks(chunks)
        >>> print(chunks_with_embeddings[0].keys())
    """
    embedder = Embedder()
    return embedder.embed_chunks(chunks)


if __name__ == "__main__":
    # Test the embedder
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        embedder = Embedder()

        # Display configuration
        info = embedder.get_embedding_info()
        print("\n" + "=" * 60)
        print("EMBEDDER CONFIGURATION")
        print("=" * 60)
        for key, value in info.items():
            print(f"{key}: {value}")

        # Test single embedding
        test_text = "This is a test sentence for embedding generation."
        print("\n" + "=" * 60)
        print("TESTING SINGLE EMBEDDING")
        print("=" * 60)
        print(f"Text: {test_text}")

        embedding = embedder.embed_text(test_text)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

        # Test batch embedding
        test_texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        print("\n" + "=" * 60)
        print("TESTING BATCH EMBEDDING")
        print("=" * 60)
        print(f"Texts: {len(test_texts)}")

        embeddings = embedder.embed_batch(test_texts)
        print(f"Generated {len(embeddings)} embeddings")
        print(f"All dimensions correct: {all(len(e) == embedder.EMBEDDING_DIMENSION for e in embeddings)}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
