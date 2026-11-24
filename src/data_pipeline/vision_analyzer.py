"""Analyze extracted paper figures with an Azure vision-capable chat model."""

import base64
import json
import logging
import mimetypes
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config import get_config
from src.utils.azure_cli_credential import AzureCliCredentialLite

logger = logging.getLogger(__name__)


class VisionAnalysisError(Exception):
    """Raised when a figure cannot be analyzed."""


def is_meaningful_figure_image(
    image_path: str,
    min_width: int = 200,
    min_height: int = 120,
) -> bool:
    """Exclude tiny PNG decorations such as publisher logos and page buttons."""
    path = Path(image_path)
    if not path.exists():
        return False
    try:
        header = path.read_bytes()[:24]
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            return True
        width, height = struct.unpack(">II", header[16:24])
        return width >= min_width and height >= min_height
    except (OSError, struct.error):
        return False


class VisionAnalyzer:
    """Generate structured descriptions for figures extracted from papers."""

    def __init__(self):
        config = get_config()
        self.deployment_name = config.vision_deployment_name
        self.max_figures = config.vision_max_figures_per_document
        self.endpoint = config.vision_endpoint.rstrip("/")
        self.api_version = config.vision_api_version
        self.api_key = config.vision_api_key
        self.use_azure_cli = config.vision_use_azure_cli
        self.credential = AzureCliCredentialLite() if self.use_azure_cli else None

    @staticmethod
    def _image_data_url(image_path: str) -> str:
        """Encode a local figure as an image data URL."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Figure image not found: {image_path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def analyze_figure(self, figure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze one figure and return a structured, searchable description."""
        image_path = figure.get("image_path")
        if not image_path:
            raise VisionAnalysisError("Figure has no extracted image path")

        caption = figure.get("caption") or "No caption was detected."
        prompt = (
            "Analyze this figure from a research paper. Return JSON only with "
            "these keys: figure_type, summary, key_elements, trends, values, "
            "relationships, limitations. Be precise and do not invent unreadable "
            "labels or values. Use an empty list when a field does not apply.\n\n"
            f"Document caption: {caption}"
        )

        try:
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self._image_data_url(image_path),
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 700,
                "temperature": 0,
            }
            url = (
                f"{self.endpoint}/openai/deployments/{self.deployment_name}"
                f"/chat/completions?api-version={self.api_version}"
            )
            headers = {"Content-Type": "application/json"}
            if self.use_azure_cli:
                token = self.credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )
                headers["Authorization"] = f"Bearer {token.token}"
            else:
                headers["api-key"] = self.api_key

            request = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"] or "{}"
            analysis = json.loads(content)
            analysis["model_deployment"] = self.deployment_name
            return analysis
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise VisionAnalysisError(
                f"Vision service returned HTTP {exc.code}: {details}"
            ) from exc
        except Exception as exc:
            raise VisionAnalysisError(f"Vision analysis failed: {exc}") from exc

    def analyze_figures(
        self,
        figures: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Analyze up to the configured per-document figure limit."""
        analyzed = list(figures or [])
        candidates = [
            figure for figure in analyzed
            if figure.get("image_path")
            and is_meaningful_figure_image(figure["image_path"])
        ]
        for figure in candidates[:self.max_figures]:
            if figure.get("image_path"):
                # Small Azure deployments can briefly return 429 while their
                # per-minute token window resets. Retry with an increasing
                # delay, but do not discard the rest of the document if one
                # figure remains unavailable.
                for attempt in range(3):
                    try:
                        figure["vision_analysis"] = self.analyze_figure(figure)
                        break
                    except VisionAnalysisError as exc:
                        is_rate_limit = "HTTP 429" in str(exc)
                        if not is_rate_limit or attempt == 2:
                            figure["vision_analysis_error"] = str(exc)
                            logger.warning(
                                "Skipping vision analysis for figure %s: %s",
                                figure.get("figure_id"),
                                exc,
                            )
                            break
                        wait_seconds = 20 * (attempt + 1)
                        logger.warning(
                            "Vision rate limit for figure %s; retrying in %ss",
                            figure.get("figure_id"),
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
        return analyzed


def format_vision_analysis(analysis: Dict[str, Any]) -> str:
    """Convert structured vision output to text suitable for embeddings."""
    if not analysis:
        return ""

    lines = [
        f"Figure type: {analysis.get('figure_type', 'unknown')}",
        f"Visual summary: {analysis.get('summary', '')}",
    ]
    for key, label in (
        ("key_elements", "Key elements"),
        ("trends", "Trends"),
        ("values", "Values"),
        ("relationships", "Relationships"),
        ("limitations", "Limitations"),
    ):
        value = analysis.get(key)
        if value:
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value)
            lines.append(f"{label}: {value}")
    return "\n".join(lines)
