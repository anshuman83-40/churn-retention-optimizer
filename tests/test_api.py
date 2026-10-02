import pytest
from fastapi.testclient import TestClient

from api.main import app
from churn.config import CHURN_MODEL_PATH, UPLIFT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not (CHURN_MODEL_PATH.exists() and UPLIFT_MODEL_PATH.exists()),
    reason="models not trained; run `python -m churn.pipeline` first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_high_risk_customer_gets_offer(client, customer):
    body = client.post("/predict", json=customer).json()
    assert body["risk_level"] == "high"
    assert body["recommendation"] == "send_offer"
    assert body["top_risk_factors"][0]["feature"] == "Contract"


def test_loyal_customer_no_offer(client, loyal_customer):
    body = client.post("/predict", json=loyal_customer).json()
    assert body["churn_probability"] < 0.15
    assert body["recommendation"] == "do_not_send"


def test_batch_and_validation(client, customer, loyal_customer):
    r = client.post("/predict/batch", json=[customer, loyal_customer])
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/predict", json={**customer, "Contract": "Weekly"}).status_code == 422


def test_dashboard_summary(client):
    s = client.get("/api/summary").json()
    assert s["kpis"]["total_customers"] == 7043
    assert abs(sum(seg["share"] for seg in s["segments"]) - 1) < 1e-3
    assert len(s["churn_by_tenure"]["actual"]) == len(s["churn_by_tenure"]["labels"])


def test_customer_table_filter_and_detail(client):
    page = client.get("/api/customers", params={"segment": "persuadable", "size": 5}).json()
    assert page["total"] > 0 and len(page["rows"]) == 5
    assert all(r["segment"] == "Persuadable" for r in page["rows"])
    probs = [r["churn_probability"] for r in page["rows"]]
    assert probs == sorted(probs, reverse=True)
    detail = client.get(f"/api/customers/{page['rows'][0]['customerID']}").json()
    assert detail["recommendation"] == "send_offer" and detail["actions"]
    assert client.get("/api/customers/NOT-A-CUSTOMER").status_code == 404


def test_export_and_frontend(client):
    csv = client.get("/api/customers/export.csv", params={"segment": "loyal"}).text.splitlines()
    assert csv[0].startswith("customerID") and all(",Loyal," in line for line in csv[1:])
    assert "ChurnOpt" in client.get("/").text
