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
