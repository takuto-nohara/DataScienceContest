# プレビュー用の簡易サーバー。GET はこのフォルダのファイルを返し、POST /save/<name>.json は本文をそのファイルに保存する
import http.server
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def do_POST(self):
        m = re.fullmatch(r'/save/([A-Za-z0-9_-]+\.json)', self.path)
        if not m:
            self.send_error(404)
            return
        n = int(self.headers.get('Content-Length', 0))
        with open(os.path.join(ROOT, m.group(1)), 'wb') as f:
            f.write(self.rfile.read(n))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')


http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
