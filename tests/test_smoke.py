from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_managers():
    r = client.get("/managers")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) > 0


def test_succession_endpoint():
    r = client.post("/succession", json=dict(manager_id="E-0001", k=5))
    assert r.status_code == 200
    body = r.json()
    assert body["manager_id"] == "E-0001"
    assert isinstance(body["candidates"], list)
    assert len(body["candidates"]) == 5
    for c in body["candidates"]:
        for f in ("emp_id", "skill_match", "structural_proximity",
                  "readiness_score"):
            assert f in c


def test_synthetic_org_shape():
    from succession_graph.data import make_org

    df, G = make_org()
    assert len(df) == 2_000
    expected = {"emp_id", "manager_id", "role", "level", "skills",
                "tenure_yrs", "performance_rating"}
    assert expected.issubset(set(df.columns))
    # Tree: edges = nodes - 1
    assert G.number_of_edges() == G.number_of_nodes() - 1
