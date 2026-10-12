import json
import signal
from contextlib import contextmanager

import numpy as np
import pytest
from starlette.testclient import TestClient
import crm_app

@pytest.fixture(scope="module")
def client():
    with TestClient(crm_app.app) as c:
        yield c

@contextmanager
def time_limit(seconds):                    # fail instead of waiting forever
    def stop(signum, frame):
        raise TimeoutError(f"no answer from the server after {seconds} s")
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def read_output(ws, name="summary"):
    with time_limit(30):
        for _ in range(50):                   # at most 50 messages...
            message = json.loads(ws.receive_text())
            if name in message.get("values", {}):
                return message["values"][name]
    raise AssertionError(f"output {name} never arrived")

def test_profit_curve():                     # unit tests, computed by hand
    p = crm_app.profit_curve(np.array([1, 0, 1, 0]), fixed=10, per_mail=1, margin=5, n=4)
    assert np.allclose(p, [0, -6, -7, -3, -4])     # index 0: do not launch
    assert crm_app.decision(p, 4).startswith("Do not launch")
    p = crm_app.profit_curve(np.array([1, 0, 1, 0]), fixed=1, per_mail=1, margin=5, n=4)
    assert crm_app.decision(p, 4).startswith("Mail the top 75%")

def test_page_has_the_controls(client):
    page = client.get("/").text
    for name in ("model", "fixed", "per_mail", "margin", "summary", "response",
                 "profit_plot"):
        assert f'id="{name}"' in page

def test_costs_do_not_retrain_but_the_model_does(client):
    crm_app.TRAININGS.clear()
    with client.websocket_connect("/websocket/") as ws:
        ws.send_text(json.dumps({"method": "init", "data": {
            "model": "Decision tree", "fixed": 10_000, "per_mail": 1.5, "margin": 30,
            ".clientdata_output_summary_hidden": False}}))
        first = read_output(ws)
        ws.send_text(json.dumps({"method": "update", "data": {"per_mail": 0.5}}))
        cheaper = read_output(ws)
        assert cheaper != first and crm_app.TRAININGS == ["Decision tree"]
        new_model = {"model": "Nearest neighbours"}
        ws.send_text(json.dumps({"method": "update", "data": new_model}))
        read_output(ws)
        assert crm_app.TRAININGS == ["Decision tree", "Nearest neighbours"]
    print(f"\n  1.50 EUR per brochure: {first}\n  0.50 EUR per brochure: {cheaper}")
