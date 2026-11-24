"""Tests for the lightweight Azure CLI token credential."""

import json
from unittest.mock import patch

from src.utils.azure_cli_credential import AzureCliCredentialLite


@patch("src.utils.azure_cli_credential.subprocess.run")
def test_fetches_and_caches_cli_token(run):
    run.return_value.stdout = json.dumps({
        "accessToken": "test-token",
        "expires_on": 4102444800,
    })
    credential = AzureCliCredentialLite()

    first = credential.get_token("https://search.azure.com/.default")
    second = credential.get_token("https://search.azure.com/.default")

    assert first.token == "test-token"
    assert second.token == "test-token"
    assert run.call_count == 1
    assert "https://search.azure.com" in run.call_args.args[0]
