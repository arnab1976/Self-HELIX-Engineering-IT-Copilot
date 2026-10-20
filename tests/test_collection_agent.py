def test_collection_agent_sources_describe_data_and_roles(client):
    response = client.get("/v1/lexora/collect/sources")
    assert response.status_code == 200
    body = response.json()
    assert {source["id"] for source in body["sources"]} == {
        "code", "tickets", "knowledge", "logs", "observability"
    }
    assert {role["name"] for role in body["roles"]} == {
        "Source connector", "Privacy & secret steward", "Licence & quality reviewer", "Dataset registrar"
    }
    assert "HELIX_GITHUB_TOKEN" in body["sources"][0]["environment"]
    assert "data/generated/tickets.csv" in body["sources"][1]["simulation"]
    urls = {source["id"]: source["api_url"] for source in body["sources"]}
    assert urls["code"].startswith("https://api.github.com/repos/")
    assert urls["tickets"].endswith("/api/now/table/incident")
    assert urls["knowledge"].endswith("/wiki/api/v2/spaces/{spaceId}/pages")
    assert urls["logs"].endswith("/loki/api/v1/query_range")
    assert urls["observability"] is None


def test_source_connection_links_do_not_expose_url_credentials():
    from lexora.ingest.collection_agent import _safe_base_url

    url = _safe_base_url(
        "https://collector:private-token@loki.example.test/proxy?token=private-token",
        "https://<your-loki-host>",
    )
    assert url == "https://loki.example.test/proxy"


def test_collection_agent_runs_sanitisation_pipeline_and_registers_dataset(client):
    response = client.post("/v1/lexora/collect/run", json={
        "mode": "simulation",
        "sources": ["knowledge"],
        "sample_text": (
            '# SPDX-License-Identifier: GPL-3.0-only\n'
            'owner = "ava.chen@example.test"\n'
            'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\n'
            'DB_PASSWORD = "Summer2026!"\n'
            'endpoint = "192.168.1.20"\n'
        ),
        "filename": "sample.py",
        "dedupe_threshold": 0.82,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["documents_in"] > result["documents_out"] - 1
    assert result["secrets_redacted"] >= 2
    assert result["pii_masked"] >= 2
    assert result["copyleft_excluded_from_fim"] >= 1
    assert result["dataset_version"].startswith("ds-eng-")
    assert "AKIAIOSFODNN7EXAMPLE" not in result["preview"]
    assert "ava.chen@example.test" not in result["preview"]
    assert "192.168.1.20" not in result["preview"]


def test_collection_agent_reports_missing_live_connector_configuration(client, monkeypatch):
    for name in ("HELIX_SERVICENOW_INSTANCE", "HELIX_SERVICENOW_USER", "HELIX_SERVICENOW_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    response = client.post("/v1/lexora/collect/run", json={
        "mode": "connected", "sources": ["tickets"], "sample_text": ""
    })
    assert response.status_code == 400
    assert "HELIX_SERVICENOW_INSTANCE" in response.json()["detail"]
    assert "Enter credentials in the source dialog or set:" in response.json()["detail"]


def test_collection_agent_disallows_live_metrics_and_traces(client):
    response = client.post("/v1/lexora/collect/run", json={
        "mode": "connected", "sources": ["observability"], "sample_text": ""
    })
    assert response.status_code == 400
    assert "simulation mode only" in response.json()["detail"]


def test_github_connector_uses_server_managed_bearer_token():
    from lexora.ingest.git_connector import GitHubConnector

    connector = GitHubConnector(token="test-token")
    try:
        assert connector.http.headers["Authorization"] == "Bearer test-token"
    finally:
        connector.http.close()


def test_request_credentials_drive_connected_github_fetch(client, monkeypatch):
    from lexora.ingest import collection_agent
    from lexora.ingest.base import Document

    class FakeGitHub:
        def __init__(self, token=None):
            self.token = token
            self.http = type("H", (), {"close": lambda self: None})()

        def files(self, repo, ref="HEAD", limit=300):
            assert self.token == "dialog-token"
            assert repo == "acme/checkout"
            assert ref == "main"
            return [Document("git", "readme.md", "# Checkout\nowner = ava@example.test\n", "doc", repo)]

        def pull_requests(self, repo, limit=10):
            return []

    monkeypatch.setattr(collection_agent, "GitHubConnector", FakeGitHub)
    response = client.post("/v1/lexora/collect/run", json={
        "mode": "connected",
        "sources": ["code"],
        "sample_text": "",
        "credentials": {
            "github_token": "dialog-token",
            "github_repo": "acme/checkout",
            "github_ref": "main",
        },
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["documents_in"] >= 1
    assert "ava@example.test" not in body["preview"]


def test_servicenow_and_confluence_host_normalization():
    from lexora.ingest.collection_agent import _normalize_confluence_site, _normalize_servicenow_instance

    assert _normalize_servicenow_instance("https://acme.service-now.com/nav") == "acme"
    assert _normalize_servicenow_instance("acme") == "acme"
    assert _normalize_confluence_site("https://acme.atlassian.net/wiki") == "acme"
    assert _normalize_confluence_site("acme") == "acme"
