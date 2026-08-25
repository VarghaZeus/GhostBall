"""Drive the control panel in a real browser against a running server.

Not a unit test and not run by pytest -- a bring-up tool. ``panel_harness.js``
models a DOM; this uses a real one, with real CSS, real layout and real event
dispatch. The two catch different things, and the tab bug is the argument for
having both: the harness could not see it because it stubbed the selector the
bug was in, and no amount of improving a stub proves the stub is complete.

What only a browser can answer:

* whether an element is *visually* there -- ``getBoundingClientRect`` sees a
  zero-height card that ``hidden`` says nothing about;
* whether a click actually reaches a handler through the real event path;
* whether the CSS produces the layout intended, including the "one tab stretched
  to full width and looked like a submit button" failure, which is purely a
  flex-sizing outcome and invisible to any DOM model.

Usage::

    python -m app.main --mock --port 8000 &
    python tests/browser_check.py http://localhost:8000/
    python tests/browser_check.py http://ghostball.local:8000/ --shots out/

Chrome is driven over the DevTools protocol directly -- no puppeteer, no
selenium, nothing to install beyond a browser that is already on the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
]


def find_browser() -> str | None:
    for name in ("chromium-browser", "chromium", "google-chrome", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for path in CHROME_CANDIDATES:
        if Path(path).is_file():
            return path
    return None


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Browser:
    """A headless Chrome, spoken to over the DevTools protocol."""

    def __init__(self, binary: str) -> None:
        self.binary = binary
        self.port = free_port()
        self.profile = tempfile.mkdtemp(prefix="ghostball-cdp-")
        self.process: subprocess.Popen | None = None
        self.socket = None
        self._id = 0

    async def __aenter__(self) -> Browser:
        self.process = subprocess.Popen(
            [
                self.binary,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                f"--remote-debugging-port={self.port}",
                f"--user-data-dir={self.profile}",
                "--window-size=1400,1000",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await self._connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self.socket is not None:
            await self.socket.close()
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=10)
        shutil.rmtree(self.profile, ignore_errors=True)

    async def _connect(self) -> None:
        import httpx
        import websockets

        deadline = time.monotonic() + 25
        target = None
        while time.monotonic() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://127.0.0.1:{self.port}/json/list")
                pages = [t for t in response.json() if t["type"] == "page"]
                if pages:
                    target = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:  # noqa: BLE001 - the browser is still coming up
                await asyncio.sleep(0.25)
        if target is None:
            raise RuntimeError("Chrome never exposed a debugging target")
        self.socket = await websockets.connect(target, max_size=64 * 1024 * 1024)

    async def send(self, method: str, **params) -> dict:
        self._id += 1
        message_id = self._id
        await self.socket.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            payload = json.loads(await self.socket.recv())
            if payload.get("id") == message_id:
                if "error" in payload:
                    raise RuntimeError(f"{method}: {payload['error']}")
                return payload.get("result", {})

    async def goto(self, url: str) -> None:
        await self.send("Page.enable")
        await self.send("Page.navigate", url=url)
        # Poll for the panel's own readiness rather than a fixed sleep: it
        # fetches several endpoints before it has painted anything.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            ready = await self.eval(
                "document.readyState === 'complete' && "
                "document.getElementById('tabs').children.length > 0"
            )
            if ready:
                await asyncio.sleep(0.8)  # let the first poll paint
                return
            await asyncio.sleep(0.25)
        raise RuntimeError(f"{url} never finished loading")

    async def eval(self, expression: str):
        result = await self.send(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"]["exception"].get("description"))
        return result["result"].get("value")

    async def screenshot(self, path: Path) -> None:
        result = await self.send("Page.captureScreenshot", captureBeyondViewport=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(result["data"]))


#: Reads the *rendered* geometry. `hidden` is a DOM property a model can fake;
#: a zero-width box is a fact about layout, and it is what "the tab looked like
#: a full-width button" actually was.
PROBE = """
(() => {
  const box = (el) => {
    const r = el.getBoundingClientRect();
    return { w: Math.round(r.width), h: Math.round(r.height), visible: r.width > 0 && r.height > 0 };
  };
  const tabs = [...document.getElementById("tabs").children].map((b) => ({
    tab: b.dataset.tab,
    label: b.textContent,
    selected: b.getAttribute("aria-selected") === "true",
    ...box(b),
  }));
  const cards = [...document.querySelectorAll("section.card")].map((c) => ({
    title: c.querySelector("h2").textContent.trim(),
    tab: c.dataset.tab,
    hidden: c.hidden,
    collapsed: c.classList.contains("collapsed"),
    ...box(c),
  }));
  return { tabs, cards, hash: location.hash };
})()
"""


async def run(url: str, shots: Path | None) -> int:
    binary = find_browser()
    if binary is None:
        print("  No Chrome or Edge found; cannot run the browser check.")
        return 2
    print(f"  Browser: {binary}")
    print(f"  Target:  {url}\n")

    failures: list[str] = []

    async with Browser(binary) as browser:
        await browser.goto(url)
        state = await browser.eval(PROBE)

        print("  Tab bar:")
        for tab in state["tabs"]:
            mark = "*" if tab["selected"] else " "
            print(f"   {mark} {tab['label']:<14} {tab['w']:>4}x{tab['h']:<3} visible={tab['visible']}")

        visible_tabs = [t for t in state["tabs"] if t["visible"]]
        if len(visible_tabs) != 4:
            failures.append(f"{len(visible_tabs)} of 4 tabs are visible")

        # The "looked like a button" failure, as a measurement: one tab filling
        # the bar is a flex-sizing outcome no DOM model can see.
        widest = max((t["w"] for t in state["tabs"]), default=0)
        bar_width = await browser.eval(
            "Math.round(document.getElementById('tabs').getBoundingClientRect().width)"
        )
        if widest > bar_width * 0.6:
            failures.append(
                f"a tab is {widest}px of a {bar_width}px bar -- it will read as a button"
            )

        # Every card reachable, by clicking the real buttons.
        print("\n  Cards per tab (via real clicks):")
        seen: dict[str, str] = {}
        for tab in ("play", "setup", "tune", "diagnostics"):
            await browser.eval(
                f"[...document.getElementById('tabs').children]"
                f".find(b => b.dataset.tab === '{tab}').click()"
            )
            await asyncio.sleep(0.35)
            after = await browser.eval(PROBE)

            shown = [c for c in after["cards"] if c["visible"]]
            print(f"   {tab:<12} {', '.join(c['title'] for c in shown) or '(none)'}")
            for card in shown:
                seen[card["title"]] = tab
            if not shown:
                failures.append(f"the {tab} tab shows no cards")
            if len([t for t in after["tabs"] if t["visible"]]) != 4:
                failures.append(f"selecting {tab} hid a tab button")
            if shots:
                await browser.screenshot(shots / f"tab-{tab}.png")

        missing = {c["title"] for c in state["cards"]} - set(seen)
        if missing:
            failures.append(f"unreachable cards: {sorted(missing)}")

        # Collapse, by clicking a real heading.
        print("\n  Collapse:")
        await browser.eval(
            "[...document.getElementById('tabs').children]"
            ".find(b => b.dataset.tab === 'play').click()"
        )
        await asyncio.sleep(0.3)
        # The first *visible* card, not the first in document order. Those are
        # not the same thing once cards are grouped into tabs, and a hidden
        # element measures 0 tall -- which reads as "collapsing did nothing"
        # rather than "you measured the wrong card".
        visible_card = (
            "[...document.querySelectorAll('section.card')]"
            ".find(c => c.getBoundingClientRect().height > 0)"
        )
        before = await browser.eval(
            f"Math.round({visible_card}.querySelector('.card-body')"
            ".getBoundingClientRect().height)"
        )
        await browser.eval(f"{visible_card}.querySelector('h2').click()")
        await asyncio.sleep(0.5)
        after_height = await browser.eval(
            f"Math.round({visible_card}.querySelector('.card-body')"
            ".getBoundingClientRect().height)"
        )
        print(f"   card body {before}px -> {after_height}px")
        if not (before > 0 and after_height < before / 2):
            failures.append(f"collapsing did not shrink the card ({before} -> {after_height})")
        await browser.eval(f"{visible_card}.querySelector('h2').click()")
        await asyncio.sleep(0.4)

        if shots:
            await browser.screenshot(shots / "panel.png")
            print(f"\n  Screenshots in {shots}")

    print()
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        return 1
    print("  All browser checks passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the panel in a real browser.")
    parser.add_argument("url", nargs="?", default="http://localhost:8000/")
    parser.add_argument("--shots", type=Path, default=None, help="write screenshots here")
    args = parser.parse_args(argv)
    return asyncio.run(run(args.url, args.shots))


if __name__ == "__main__":
    sys.exit(main())
