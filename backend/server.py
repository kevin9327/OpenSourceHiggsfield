"""
OpenSourceHiggsfield — FastAPI + Comfy SDK v0.5.0 Web Server (`backend/server.py`)
==================================================================================
Serves the 3-in-1 Director Studio Frontend (`frontend/`) and REST/SSE API endpoints
powered by `comfy-sdk` v0.5.0, `comfy-api-proxy` v0.1.6, and our 5 `OpenHiggsfield*`
PyTorch/OpenCV/FFmpeg custom nodes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .comfy_engine import PUBLIC_DIR, REF_VIDEOS_DIR, ROOT_DIR, WORKFLOWS_DIR, engine
from .mcp_server import call_mcp_tool, handle_mcp_jsonrpc, list_mcp_tools

FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="OpenSourceHiggsfield Director Studio API",
    description=(
        "Open-weights 3-in-1 AI Cinema Studio (Cinematic Studio, 3D Video Re-Shoot, OpenJutsu) "
        "built on ComfyUI, LTX-2.5 22B, FLUX.3, comfy-sdk v0.5.0, comfy-api-proxy v0.1.6, and Comfy MCP."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# System Status, Benchmark Proof & Execution Mode Switch
# ---------------------------------------------------------------------------

@app.get("/favicon.ico")
async def get_favicon() -> Response:
    """Return Comfy neon-yellow SVG icon for browser favicon requests."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#17151C"/>'
        '<circle cx="16" cy="16" r="9" fill="#EAFF45"/></svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    """Return GPU info (RTX 5080 16GB), execution mode, Comfy SDK v0.5.0 status, and workflow list."""
    return engine.get_status()


@app.get("/api/benchmark/proof")
async def get_benchmark_proof() -> dict[str, Any]:
    """Return live hardware telemetry, Sim(3) camera verification, CVGEO1 validation, and 15-repo scorecard."""
    return engine.get_benchmark_proof()


@app.post("/api/mode")
async def set_mode(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Switch between `OPEN_WEIGHTS_LOCAL` (Grand Prize compliant) and `CLOUD_HYBRID`."""
    mode = str(payload.get("mode") or payload.get("execution_mode") or "OPEN_WEIGHTS_LOCAL")
    try:
        return engine.set_execution_mode(mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Pillar 1: Cinematic Studio (`/api/cinematic-studio/generate` & `/api/studio/cinematic`)
# ---------------------------------------------------------------------------

@app.post("/api/cinematic-studio/generate")
@app.post("/api/studio/cinematic")
async def cinematic_studio_generate(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Execute Cinematic Studio (Camera, Lens, Framing, 3D Relight, Film Stock, Color Grade)."""
    try:
        return await engine.run_cinematic_studio(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/studio/cinematic")
async def cinematic_studio_get() -> dict[str, Any]:
    """Execute default Cinematic Studio recipe via GET for instant benchmark verification."""
    return await engine.run_cinematic_studio({})


# ---------------------------------------------------------------------------
# Pillar 2: Re-shoot a Video (`/api/reshoot/*` & `/api/studio/reshoot`)
# ---------------------------------------------------------------------------

@app.get("/api/reshoot/cvgeo")
async def reshoot_cvgeo(
    format: str = Query(default="binary", description="Pass 'json' for JSON summary or 'binary' for raw CVGEO1\\0\\0 stream"),
    frames: int = Query(default=6, ge=1, le=64),
) -> Any:
    """Generate and return a valid `CVGEO1\\0\\0` binary geometry stream or JSON summary (`?format=json`)."""
    if str(format).lower() == "json":
        return engine.generate_reshoot_cvgeo1(num_frames=frames, as_json=True)
    blob = engine.generate_reshoot_cvgeo1(num_frames=frames, as_json=False)
    assert isinstance(blob, bytes)
    return Response(
        content=blob,
        media_type="application/octet-stream",
        headers={"X-CVGEO-Magic": "CVGEO1", "Content-Disposition": 'inline; filename="scene.cvgeo"'},
    )


@app.post("/api/reshoot/analyze")
async def reshoot_analyze(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Estimate metric depth map, 3D point cloud, and spherical camera rig gizmo preview."""
    try:
        return await engine.analyze_reshoot_scene(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/reshoot/generate")
@app.post("/api/studio/reshoot")
async def reshoot_generate(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Warp source video in 3D perspective along keyframed spherical camera trajectory."""
    try:
        return await engine.run_reshoot_generate(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/studio/reshoot")
async def reshoot_studio_get() -> dict[str, Any]:
    """Execute default Re-shoot trajectory via GET for instant benchmark verification."""
    return await engine.run_reshoot_generate({})


# ---------------------------------------------------------------------------
# Pillar 3: OpenJutsu (`/api/openjutsu/generate` & `/api/studio/openjutsu`)
# ---------------------------------------------------------------------------

@app.post("/api/openjutsu/generate")
@app.post("/api/studio/openjutsu")
async def openjutsu_generate(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Execute OpenJutsu character swap with audio preservation and procedural VFX mattes."""
    try:
        return await engine.run_openjutsu_generate(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/studio/openjutsu")
async def openjutsu_studio_get() -> dict[str, Any]:
    """Execute default OpenJutsu character swap via GET for instant benchmark verification."""
    return await engine.run_openjutsu_generate({})


# ---------------------------------------------------------------------------
# Workflows & Comfy SDK v0.5.0 Graph Inspector (`GET /api/workflows`, etc.)
# ---------------------------------------------------------------------------

@app.get("/api/workflows")
async def list_workflows() -> dict[str, Any]:
    """Return manifest of all 6 ComfyUI API-format workflows and their injectable node IDs."""
    manifest = engine.load_manifest()
    return {
        "sdk_version": "0.5.0",
        "proxy_version": "0.1.6",
        "execution_mode": engine.execution_mode,
        "manifest": manifest,
    }


@app.get("/api/workflows/{name}")
async def get_workflow_detail(name: str) -> dict[str, Any]:
    """Return raw `.api.json` graph, open-weights audit, and live Python/TS `comfy-sdk` code."""
    clean = name.replace(".api.json", "").replace(".json", "")
    try:
        graph = engine.load_workflow_json(clean)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    is_open, violations = engine.validate_open_weights_workflow(graph)
    manifest_entry = engine.load_manifest().get("workflows", {}).get(clean, {})
    sdk_snippets = engine.generate_sdk_code_snippet(clean)
    return {
        "id": clean,
        "file": f"{clean}.api.json",
        "meta": manifest_entry,
        "open_weights_compliant": is_open,
        "violations": violations,
        "graph": graph,
        "sdk_code": sdk_snippets,
    }


@app.post("/api/workflows/{name}/execute")
async def execute_workflow_by_name(
    name: str, payload: dict[str, Any] = Body(default_factory=dict)
) -> dict[str, Any]:
    """Execute any of the 6 workflows directly from the Workflow & SDK workbench tab."""
    clean = name.replace(".api.json", "").replace(".json", "")
    if "reshoot" in clean:
        return await engine.run_reshoot_generate(payload)
    if "openjutsu" in clean:
        return await engine.run_openjutsu_generate(payload)
    return await engine.run_cinematic_studio(payload)


# ---------------------------------------------------------------------------
# Comfy MCP & Comfy Local Agent Endpoints (`/api/mcp/tools`, `/api/mcp/call`, `/api/mcp`)
# ---------------------------------------------------------------------------

@app.get("/api/mcp/tools")
async def mcp_list_tools() -> dict[str, Any]:
    """Return the Model Context Protocol (MCP) 6-tool catalog for AI Director agents."""
    return list_mcp_tools()


@app.post("/api/mcp/call")
async def mcp_call_tool(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Execute one of the 6 Comfy MCP tools (`name` + `arguments`) over HTTP."""
    tool_name = str(payload.get("name") or payload.get("tool") or "")
    arguments = payload.get("arguments") or payload.get("params") or {}
    if not tool_name:
        raise HTTPException(status_code=400, detail="Missing 'name' of MCP tool to call.")
    try:
        return await call_mcp_tool(tool_name, arguments)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/mcp")
async def mcp_jsonrpc_endpoint(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Handle standard JSON-RPC 2.0 MCP requests (`initialize`, `tools/list`, `tools/call`)."""
    resp = await handle_mcp_jsonrpc(payload)
    return resp or {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}


# ---------------------------------------------------------------------------
# Job Status & Live SSE Event Stream (`GET /api/jobs/{job_id}`, `/api/events/{job_id}`)
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in engine.jobs_store:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    record = dict(engine.jobs_store[job_id])
    record["events"] = engine.job_events.get(job_id, [])
    return record


@app.get("/api/events/{job_id}")
async def stream_job_events(job_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events (SSE) stream emitting `StatusChange`, `Progress`, and `OutputReady`."""
    if job_id not in engine.jobs_store:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    async def event_generator():
        sent_idx = 0
        while True:
            if await request.is_disconnected():
                break
            events = engine.job_events.get(job_id, [])
            while sent_idx < len(events):
                ev = events[sent_idx]
                sent_idx += 1
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") == "StatusChange" and ev.get("status") in ("completed", "failed", "cancelled"):
                    return
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Static Mounts (`/public`, `/ref_videos`, and `/` for `frontend/`)
# ---------------------------------------------------------------------------

if PUBLIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

if REF_VIDEOS_DIR.exists():
    app.mount("/ref_videos", StaticFiles(directory=str(REF_VIDEOS_DIR)), name="ref_videos")

if WORKFLOWS_DIR.exists():
    app.mount("/workflows", StaticFiles(directory=str(WORKFLOWS_DIR)), name="workflows")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def main() -> None:
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8090, reload=False)


if __name__ == "__main__":
    main()
