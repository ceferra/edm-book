import json
from starlette.testclient import TestClient
import placement_app as pa

def read_output(ws, name="total"):
    for _ in range(50):
        message = json.loads(ws.receive_text())
        if name in message.get("values", {}):
            return message["values"][name]
    raise RuntimeError(f"no value for {name}")

def start(ws, n=10):
    ws.send_text(json.dumps({"method": "init", "data": {
        "n": n, ".clientdata_output_total_hidden": False}}))
    return read_output(ws)

with TestClient(pa.app) as client:
    print(f"application started:                  {pa.RUNS}")
    with client.websocket_connect("/websocket/") as user1:
        start(user1)
        print(f"user 1 opens the page:                {pa.RUNS}")
        for n in (50, 200, 1000):                 # user 1 drags the slider
            user1.send_text(json.dumps({"method": "update", "data": {"n": n}}))
            answer = read_output(user1)
        print(f"user 1 moves the slider three times:  {pa.RUNS}")
        with client.websocket_connect("/websocket/") as user2:
            start(user2)
            print(f"user 2 opens the page:                {pa.RUNS}")
print(f"user 1 sees: {answer}")
