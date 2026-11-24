"""Lightweight TokenCredential backed by the signed-in Azure CLI account."""

import json
import subprocess
import time
from typing import Any

from azure.core.credentials import AccessToken


class AzureCliCredentialLite:
    """Fetch and cache an access token using ``az account get-access-token``."""

    def __init__(self):
        self._token: AccessToken | None = None
        self._scope: str | None = None

    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        if not scopes:
            raise ValueError("At least one Azure scope is required")
        scope = scopes[0]
        if (
            self._token
            and self._scope == scope
            and self._token.expires_on > time.time() + 120
        ):
            return self._token

        resource = scope.removesuffix("/.default")
        completed = subprocess.run(
            [
                "az",
                "account",
                "get-access-token",
                "--resource",
                resource,
                "--output",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        expires_on = payload.get("expires_on")
        if expires_on is None:
            expires_on = int(time.time()) + 300

        self._scope = scope
        self._token = AccessToken(payload["accessToken"], int(expires_on))
        return self._token
