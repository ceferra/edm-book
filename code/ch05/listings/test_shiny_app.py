import json
import pytest
from starlette.testclient import TestClient
import loan_model

@pytest.fixture(scope="module")
def app_module():
    loan_model.train_and_save()            # the model file that the application loads
    import shiny_app
    return shiny_app

@pytest.fixture(scope="module")
def client(app_module):
    with TestClient(app_module.app) as c:
        yield c

def read_output(ws, name="answer"):
    """Read messages from the server until the value of an output arrives."""
    for _ in range(20):
        message = json.loads(ws.receive_text())
        if name in message.get("values", {}):
            return message["values"][name]
    raise AssertionError(f"output {name} never arrived")

def test_answer_text(app_module):                    # unit test of the model wrapper
    low, high = (loan_model.answer_text(app_module.model, income, 0.3)
                 for income in (20, 80))
    assert low.startswith("Probability of repaying: ")
    assert float(high.split(": ")[1]) > float(low.split(": ")[1])

def test_page_has_the_controls(client):              # the user interface is served
    page = client.get("/")
    assert page.status_code == 200
    assert all(f'id="{name}"' in page.text for name in ("income", "debt", "answer"))

def test_output_follows_the_sliders(client, app_module):   # end to end, no browser
    expected = lambda income: loan_model.answer_text(app_module.model, income, 0.3)
    with client.websocket_connect("/websocket/") as ws:
        ws.send_text(json.dumps({"method": "init", "data": {
            "income": 30, "debt": 0.3, ".clientdata_output_answer_hidden": False}}))
        assert read_output(ws) == expected(30)
        ws.send_text(json.dumps({"method": "update", "data": {"income": 80}}))
        assert read_output(ws) == expected(80)
