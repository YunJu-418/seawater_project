import json
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import requests

TOKEN_FILE = Path("kakao_token.json")
REDIRECT_URI = "http://localhost:3000"

data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
client_id = str(data.get("rest_api_key", "")).strip()
client_secret = str(data.get("client_secret", "")).strip()

if not client_id:
    raise RuntimeError("rest_api_key is missing")

result = {}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        result["code"] = query.get("code", [""])[0]
        result["error"] = query.get("error", [""])[0]

        body = b"<h2>Kakao authorization received. You may close this window.</h2>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass

params = {
    "client_id": client_id,
    "redirect_uri": REDIRECT_URI,
    "response_type": "code",
    "scope": "talk_message",
}

auth_url = "https://kauth.kakao.com/oauth/authorize?" + urllib.parse.urlencode(params)

server = HTTPServer(("localhost", 3000), Handler)

print("Opening Kakao login in browser...")
webbrowser.open(auth_url)
server.handle_request()
server.server_close()

code = result.get("code", "")
if not code:
    raise RuntimeError("Authorization code was not received: " + result.get("error", ""))

payload = {
    "grant_type": "authorization_code",
    "client_id": client_id,
    "redirect_uri": REDIRECT_URI,
    "code": code,
}

if client_secret:
    payload["client_secret"] = client_secret

response = requests.post(
    "https://kauth.kakao.com/oauth/token",
    data=payload,
    timeout=15,
)

if response.status_code != 200:
    raise RuntimeError(f"Token request failed: {response.status_code} {response.text}")

tokens = response.json()

data["access_token"] = tokens.get("access_token", "")
if tokens.get("refresh_token"):
    data["refresh_token"] = tokens["refresh_token"]

for key in ("expires_in", "refresh_token_expires_in", "scope", "token_type"):
    if key in tokens:
        data[key] = tokens[key]

TOKEN_FILE.write_text(
    json.dumps(data, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("KAKAO_TOKEN_SAVED")