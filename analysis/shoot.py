#!/usr/bin/env python
"""Screenshot every dashboard view with headless Chromium; print console errors.

    python analysis/shoot.py [out_dir]      # default dashboard/backup/
"""
import functools
import http.server
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DASH / "backup"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["intro", "comparison", "md", "metad", "gamd", "remd", "sams", "we", "we2d", "smd", "reference", "samsT"]

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DASH))
handler.log_message = lambda *a, **k: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
    page = b.new_page(viewport={"width": 1600, "height": 3400})
    errs = []
    page.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    for k in KEYS:
        errs.clear()
        page.goto(f"http://127.0.0.1:8766/index.html#{k}")
        page.reload()
        page.wait_for_timeout(3500)
        page.screenshot(path=str(OUT / f"{k}.png"), full_page=True)
        print(k, "errors:", errs[:6])
        if k in ("remd", "smd", "reference") and page.locator('#view-toggle-single button[data-v="grid"]').count():
            page.click('#view-toggle-single button[data-v="grid"]')
            page.wait_for_timeout(4000)
            for _ in range(3):
                page.keyboard.press("ArrowRight")
            page.wait_for_timeout(800)
            page.screenshot(path=str(OUT / f"{k}_grid.png"), full_page=True)
            print(k, "grid errors:", errs[:6])
    b.close()
srv.shutdown()
