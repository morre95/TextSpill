"""Small HTTP/WebSocket peer using only the Python standard library."""
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import struct
import time

ROOT = Path(os.environ["TEST_ROOT"])


def log(value):
    with (ROOT / "deepgram.jsonl").open("a") as out:
        out.write(json.dumps(value) + "\n")


class Peer(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        log({"http": self.path, "wav": body[:4] == b"RIFF", "bytes": len(body),
             "auth": self.headers.get("Authorization") == "Token fake-key"})
        status = int(os.environ.get("TEST_DG_STATUS", "200"))
        body = json.dumps({"results": {"channels": [{"alternatives": [{"transcript": "hela\ntexten"}]}]}}).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def frame(self, value, opcode=1):
        payload = json.dumps(value).encode() if opcode == 1 else value
        header = bytes([0x80 | opcode])
        header += bytes([len(payload)]) if len(payload) < 126 else b"\x7e" + struct.pack("!H", len(payload))
        self.wfile.write(header + payload)
        self.wfile.flush()

    def result(self, text, start, duration, final=True):
        if os.environ.get("TEST_DG_SILENCE"):
            text = ""
        self.frame({"type": "Results", "is_final": final, "speech_final": final,
                    "start": start, "duration": duration,
                    "channel": {"alternatives": [{"transcript": text}]}})

    def do_GET(self):
        log({"ws": self.path, "auth": self.headers.get("Authorization") == "Token fake-key"})
        status = int(os.environ.get("TEST_DG_STATUS", "200"))
        if status != 200:
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        accept = base64.b64encode(hashlib.sha1((self.headers["Sec-WebSocket-Key"] +
            "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        total = 0
        delivered = False
        try:
            while True:
                header = self.rfile.read(2)
                if len(header) != 2:
                    return
                opcode, size = header[0] & 15, header[1] & 127
                if size == 126:
                    size = struct.unpack("!H", self.rfile.read(2))[0]
                elif size == 127:
                    size = struct.unpack("!Q", self.rfile.read(8))[0]
                mask = self.rfile.read(4) if header[1] & 128 else None
                data = self.rfile.read(size)
                if mask:
                    data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
                if opcode == 2:
                    total += len(data)
                    log({"audio": len(data), "raw_pcm": data[:4] != b"RIFF"})
                    if os.environ.get("TEST_DG_DROP"):
                        return
                    if total >= 16000 and not delivered:
                        self.result("preliminärt", 0, .5, False)
                        time.sleep(float(os.environ.get("TEST_DG_DELAY", "0")))
                        self.result("live svenska", 0, .5)
                        self.result("live svenska", 0, .5)  # deliberately duplicated
                        delivered = True
                elif opcode == 1:
                    message = json.loads(data)
                    log(message)
                    if message["type"] == "CloseStream":
                        if os.environ.get("TEST_DG_HANG_CLOSE"):
                            time.sleep(35)
                            return
                        if os.environ.get("TEST_DG_BAD_JSON"):
                            self.wfile.write(b'\x81\x03bad')
                            return
                        start = .5 if delivered else 0
                        if total / 32000 > start:
                            self.result("sista orden", start, total / 32000 - start)
                        self.frame({"type": "Metadata"})
                        self.frame(struct.pack("!H", 1000), opcode=8)
                        return
                elif opcode == 8:
                    return
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    (ROOT / "endpoint").write_text(f"http://127.0.0.1:{server.server_port}/v1/listen")
    server.serve_forever()
