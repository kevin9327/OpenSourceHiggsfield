"""
OpenSourceHiggsfield — Playwright Live 1080p UI Screenshot & Verification Script
=================================================================================
Starts `backend.server:app` on port 8089, launches headless Chromium at 1920x1080,
interacts with all 4 studio tabs and modals, and saves crisp 1080p screenshots to:
  1. `public/screenshots/`
  2. `C:\\Users\\swsz9\\.gemini\\antigravity\\brain\\c9e5e770-b6cf-4fe7-a4b3-7e0ca322c2a8\\`
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

ROOT_DIR = Path(__file__).resolve().parent
SCREENSHOTS_DIR = ROOT_DIR / "public" / "screenshots"
PARENT_ARTIFACT_DIR = Path(r"C:\Users\swsz9\.gemini\antigravity\brain\c9e5e770-b6cf-4fe7-a4b3-7e0ca322c2a8")


def wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.25)
    return False


async def capture_all_screenshots(port: int = 8089) -> list[Path]:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    PARENT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    url = f"http://127.0.0.1:{port}"
    saved_paths: list[Path] = []

    async with async_playwright() as p:
        browser = None
        for ch in ("msedge", "chrome", None):
            try:
                browser = await p.chromium.launch(headless=True, channel=ch) if ch else await p.chromium.launch(headless=True)
                break
            except Exception:
                continue
        if browser is None:
            raise RuntimeError("Could not launch headless Chromium/Edge/Chrome browser")
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1.0,
        )
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(1200)

        # 1. Tab 1: Cinematic Studio (Expand Light & Grade drawers so visual cards + color swatches show!)
        await page.evaluate(
            """() => {
                const dLight = document.getElementById('drawer-light');
                const dGrade = document.getElementById('drawer-grade');
                if (dLight) dLight.classList.remove('hidden');
                if (dGrade) dGrade.classList.remove('hidden');
                const rLight = document.querySelector('.cs-param-row[data-param="light"]');
                if (rLight) rLight.setAttribute('aria-expanded', 'true');
            }"""
        )
        await page.wait_for_timeout(600)
        p1 = SCREENSHOTS_DIR / "ui_tab1_cinematic_studio.png"
        await page.screenshot(path=str(p1), full_page=False)
        saved_paths.append(p1)

        # 2. Tab 2: Re-shoot a video (3D Wireframe Sphere Gizmo + Keyframe Diamonds + Neon-Yellow Wipe)
        await page.click("#tab-btn-reshoot")
        await page.wait_for_timeout(900)
        p2 = SCREENSHOTS_DIR / "ui_tab2_reshoot_3d_sphere.png"
        await page.screenshot(path=str(p2), full_page=False)
        saved_paths.append(p2)

        # 3. Tab 3: Openjutsu Character & VFX Swap Studio
        await page.click("#tab-btn-openjutsu")
        await page.wait_for_timeout(900)
        p3 = SCREENSHOTS_DIR / "ui_tab3_openjutsu.png"
        await page.screenshot(path=str(p3), full_page=False)
        saved_paths.append(p3)

        # 4. Tab 3 + Interactive Trim Video Filmstrip Modal Open
        trim_btn = await page.query_selector("#btn-oj-open-trim") or await page.query_selector("#btn-oj-inline-trim")
        if trim_btn:
            await trim_btn.click()
        else:
            await page.evaluate("() => document.getElementById('modal-trim-video')?.classList.remove('hidden')")
        await page.wait_for_timeout(700)
        p4 = SCREENSHOTS_DIR / "ui_tab3_trim_modal.png"
        await page.screenshot(path=str(p4), full_page=False)
        saved_paths.append(p4)

        # Close trim modal
        await page.evaluate("() => document.getElementById('modal-trim-video')?.classList.add('hidden')")
        await page.wait_for_timeout(250)

        # 5. Tab 4: Workflow & Comfy SDK v0.5.0 Node Graph Workbench
        await page.click("#tab-btn-workflow")
        await page.wait_for_timeout(900)
        p5 = SCREENSHOTS_DIR / "ui_tab4_workflow_sdk.png"
        await page.screenshot(path=str(p5), full_page=False)
        saved_paths.append(p5)

        await browser.close()

    for sp in saved_paths:
        dst = PARENT_ARTIFACT_DIR / sp.name
        shutil.copy2(sp, dst)
        print(f"Saved screenshot: {sp} -> copied to {dst} ({sp.stat().st_size:,} bytes)")

    return saved_paths


def main() -> None:
    port = 8089
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.server:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_port("127.0.0.1", port, timeout=15.0):
            raise RuntimeError(f"Server did not start on port {port}")
        asyncio.run(capture_all_screenshots(port=port))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
