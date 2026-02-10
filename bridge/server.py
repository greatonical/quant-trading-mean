# bridge/server.py
# Tiny HTTP bridge for MT5 Executor EA
# Endpoints:
#   GET  /health      -> "ok" (monitoring)
#   GET  /next        -> returns ONE command line for the EA ("key=val;...") or empty body if none
#   POST /enqueue     -> enqueue a command line (text/plain)  [useful for testing]
#   POST /event       -> EA posts events/acks/fills (text/plain "key=val;..."), we log them to CSV

from __future__ import annotations

import os
import csv
import logging
from collections import deque
from datetime import datetime, timezone
from flask import Flask, request, Response
from config.settings import Config

# --- basic logging ---
logging.basicConfig(
    level=os.environ.get("BRIDGE_LOGLEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("mt5-bridge")

# --- config (env with safe defaults) ---
BRIDGE_HOST   = Config.BRIDGE_HOST
BRIDGE_PORT   = Config.BRIDGE_PORT
REPORTS_DIR   = Config.REPORTS_DIR
EVENTS_CSV    = Config.EVENTS_CSV
MAX_QUEUE_LEN = Config.MAX_QUEUE_LEN

os.makedirs(REPORTS_DIR, exist_ok=True)

# --- in-memory FIFO queue of command lines (what EA consumes) ---
_queue: deque[str] = deque(maxlen=MAX_QUEUE_LEN)

# --- CSV init (append mode, create header once) ---
if not os.path.exists(EVENTS_CSV):
    with open(EVENTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_iso", "raw_line"])

app = Flask(__name__)


@app.get("/health")
def health():
    return Response("ok", mimetype="text/plain")


@app.get("/next")
def next_cmd():
    """
    EA polls this. Always return HTTP 200.
    If queue is empty -> return empty body, EA will just skip.
    """
    try:
        line = _queue.popleft()
        log.info(f"Dequeue -> {line}")
        return Response(line, mimetype="text/plain")
    except IndexError:
        # empty queue: still return 200 with empty body (WebRequest likes 200)
        return Response("", mimetype="text/plain")


@app.post("/enqueue")
def enqueue():
    """
    Manual/test enqueue. Body must be text/plain: "key=val;key=val;..."
    Example:
      curl -X POST --data-binary 'cmd=place;order_ref=test1;symbol=EURUSD;side=BUY;lots=0.10' \
           -H 'Content-Type: text/plain' http://127.0.0.1:5000/enqueue
    """
    if request.mimetype != "text/plain":
        return Response("send text/plain body", status=415)
    line = (request.data or b"").decode("utf-8").strip()
    if not line:
        return Response("empty", status=400)
    _queue.append(line)
    log.info(f"Enqueue <- {line}")
    return Response("queued", mimetype="text/plain")


@app.post("/event")
def event():
    """
    EA posts events/acks/fills/rejects/positions/heartbeats here as text/plain "key=val;..."
    We store the raw line in CSV with timestamp for auditing.
    """
    if request.mimetype != "text/plain":
        return Response("send text/plain body", status=415)
    raw = (request.data or b"").decode("utf-8").strip()
    ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with open(EVENTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow([ts_iso, raw])
    log.info(f"EVENT <- {raw}")
    return Response("ok", mimetype="text/plain")


def main():
    log.info(f"Starting MT5 bridge on http://{BRIDGE_HOST}:{BRIDGE_PORT}")
    app.run(host=BRIDGE_HOST, port=BRIDGE_PORT, debug=False)


if __name__ == "__main__":
    main()