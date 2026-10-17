"""
OpenSourceHiggsfield — Model Context Protocol (MCP) & Comfy Local Agent Server (`backend/mcp_server.py`)
========================================================================================================
Implements a standard JSON-RPC 2.0 Model Context Protocol (MCP) server (`initialize`, `tools/list`,
`tools/call`) over both `stdio` (for Claude Desktop, Cursor, and Comfy Local Agent) and HTTP REST/JSON-RPC
(`/api/mcp/tools`, `/api/mcp/call`, `/api/mcp`).

Exposes 6 production tools for autonomous AI Director agents:
  1. `generate_cinematic_shot` — Pillar 1: 6-axis camera/lens/framing/relight/film/grade shot synthesis
  2. `reshoot_video_3d` — Pillar 2: 3D depth unprojection, SE(3) spherical camera orbit, and #FF00FF/inpaint warp
  3. `openjutsu_vfx_swap` — Pillar 3: Single-image character replacement + LTX-2.5 VFX & -14 LUFS audio lock
  4. `export_cvgeo1_geometry` — Comfy-Org `CVGEO1\\0\\0` float16 shuffle2+zlib 3D geometry stream packer/verifier
  5. `verify_sim3_camera_trajectory` — Closed-loop SVD Umeyama `Sim(3)` + CIE LAB `Delta-E` trajectory auditor
  6. `get_hardware_vram_telemetry` — Live NVIDIA Blackwell RTX 5080/5090 `nvidia-smi` & `8n+1` frame budget watchdog
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.comfy_engine import engine, get_hardware_vram_telemetry
from custom_nodes.comfyui_openhiggsfield_nodes import (
    OpenHiggsfieldCameraRig,
    verify_camera_trajectory_sim3,
)

MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_SERVER_INFO = {
    "name": "openhiggsfield-comfy-mcp",
    "version": "1.1.0",
    "description": (
        "OpenSourceHiggsfield 3-in-1 AI Cinema Director Studio MCP Server "
        "(Comfy MCP + Comfy Local Agent + Comfy SDK v0.5.0 + Comfy API Proxy v0.1.6)"
    ),
}

MCP_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "generate_cinematic_shot",
        "description": (
            "Pillar 1 (Cinematic Studio): Direct and render a 2K cinema shot or LTX-2.5 video with "
            "6-axis optical controls (camera sensor/anamorphic lens, framing, 3D surface-normal relighting, "
            "film stock halation, and ACEScg/Kodachrome/Teal-and-Orange print LUT)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Scene description prompt.",
                    "default": "A man with a python over his shoulders stands in a glass conservatory at night.",
                },
                "camera_type": {
                    "type": "string",
                    "enum": ["large_format_anamorphic", "super35_spherical", "imax_65mm", "handheld_16mm"],
                    "default": "large_format_anamorphic",
                },
                "framing": {
                    "type": "string",
                    "enum": ["medium", "low_angle", "close_up", "wide_establishing", "dutch_tilt"],
                    "default": "low_angle",
                },
                "light_preset": {
                    "type": "string",
                    "enum": ["blue_hour", "neon", "golden_hour", "chiaroscuro", "volumetric_fog"],
                    "default": "neon",
                },
                "film_stock": {
                    "type": "string",
                    "enum": ["tungsten_500t", "daylight_250d", "kodak_vision3", "bleach_bypass"],
                    "default": "tungsten_500t",
                },
                "color_grade": {
                    "type": "string",
                    "enum": ["auto", "teal_and_orange", "kodachrome", "noir_silver"],
                    "default": "teal_and_orange",
                },
                "mode": {
                    "type": "string",
                    "enum": ["image", "video"],
                    "default": "image",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "reshoot_video_3d",
        "description": (
            "Pillar 2 (Re-shoot a Video): Unproject video frames into a 3D metric depth point cloud "
            "and re-render along a keyframed SE(3) spherical camera rig trajectory with #FF00FF CrossView "
            "LoRA disocclusion masking or far-depth inpainting, plus optional Matrix Freeze-Time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "rig_preset": {
                    "type": "string",
                    "enum": ["custom_orbit", "bullet_time_360", "dolly_zoom_vertigo", "crane_dive", "fpv_drone_bank"],
                    "default": "custom_orbit",
                },
                "start_azimuth_deg": {"type": "number", "default": 30.0},
                "start_elevation_deg": {"type": "number", "default": 8.0},
                "end_azimuth_deg": {"type": "number", "default": -15.0},
                "end_elevation_deg": {"type": "number", "default": 0.0},
                "fov_deg": {"type": "number", "default": 50.0},
                "disocclusion_fill_mode": {
                    "type": "string",
                    "enum": ["inpaint_far_depth", "magenta_ff00ff"],
                    "default": "inpaint_far_depth",
                },
                "matrix_freeze_time": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "openjutsu_vfx_swap",
        "description": (
            "Pillar 3 (OpenJutsu): Replace a target character in a source video with a single reference "
            "character image using LTX-2.5 22B IC-LoRA + MiniMax H3 + SAM-3 spatiotemporal matting while "
            "preserving original broadcast-normalized (-14 LUFS) audio."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "who_to_replace": {
                    "type": "string",
                    "description": "Natural-language description of the person to replace.",
                    "default": "the guy in the left",
                },
                "jutsu_style": {
                    "type": "string",
                    "enum": [
                        "seamless_character_swap",
                        "neon_chakra_aura",
                        "shadow_clone_burst",
                        "fire_ember_cloak",
                        "cyber_glitch_teleport",
                    ],
                    "default": "seamless_character_swap",
                },
                "quality": {
                    "type": "string",
                    "enum": ["768p", "480p"],
                    "default": "768p",
                },
                "vfx_intensity": {"type": "number", "default": 0.65},
            },
            "required": ["who_to_replace"],
        },
    },
    {
        "name": "export_cvgeo1_geometry",
        "description": (
            "Pack and round-trip verify a Comfy-Org compatible `CVGEO1\\0\\0` binary 3D camera+depth geometry "
            "stream (`float16` depth with byte-transpose `shuffle2` + `zlib` lossless compression)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "num_frames": {"type": "integer", "default": 6, "minimum": 1, "maximum": 64},
                "width": {"type": "integer", "default": 320},
                "height": {"type": "integer", "default": 192},
                "fps": {"type": "number", "default": 24.0},
                "fx_norm": {"type": "number", "default": 0.85},
            },
        },
    },
    {
        "name": "verify_sim3_camera_trajectory",
        "description": (
            "Run closed-loop SVD Umeyama `Sim(3)` 3D camera trajectory alignment and CIE LAB `Delta-E` "
            "color consistency verification on a generated camera rig trajectory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "rig_preset": {
                    "type": "string",
                    "enum": ["custom_orbit", "bullet_time_360", "dolly_zoom_vertigo", "crane_dive", "fpv_drone_bank"],
                    "default": "custom_orbit",
                },
                "frame_count": {"type": "integer", "default": 16},
                "start_azimuth_deg": {"type": "number", "default": 30.0},
                "start_elevation_deg": {"type": "number", "default": 8.0},
                "end_azimuth_deg": {"type": "number", "default": -15.0},
                "end_elevation_deg": {"type": "number", "default": 0.0},
                "fov_deg": {"type": "number", "default": 50.0},
            },
        },
    },
    {
        "name": "get_hardware_vram_telemetry",
        "description": (
            "Query live NVIDIA Blackwell GPU VRAM telemetry (`nvidia-smi`) and compute LTX-2.5 `8n+1` "
            "temporal frame rules and mod-32 spatial resolution budgets."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def list_mcp_tools() -> dict[str, Any]:
    """Return the MCP tools catalog and server metadata."""
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": MCP_SERVER_INFO,
        "tools": MCP_TOOLS_SPEC,
    }


async def call_mcp_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one of the 6 registered Comfy MCP tools and return an MCP-formatted result."""
    args = dict(arguments or {})

    if name == "generate_cinematic_shot":
        data = await engine.run_cinematic_studio(args)
    elif name == "reshoot_video_3d":
        data = await engine.run_reshoot_generate(args)
    elif name == "openjutsu_vfx_swap":
        data = await engine.run_openjutsu_generate(args)
    elif name == "export_cvgeo1_geometry":
        data = engine.generate_reshoot_cvgeo1(
            num_frames=int(args.get("num_frames", 6)),
            width=int(args.get("width", 320)),
            height=int(args.get("height", 192)),
            fps=float(args.get("fps", 24.0)),
            fx_norm=float(args.get("fx_norm", 0.85)),
            as_json=True,
        )
    elif name == "verify_sim3_camera_trajectory":
        rig = OpenHiggsfieldCameraRig()
        traj_str, _, _ = rig.generate_rig(
            rig_preset=str(args.get("rig_preset", "custom_orbit")),
            frame_count=int(args.get("frame_count", 16)),
            fps=24.0,
            start_azimuth_deg=float(args.get("start_azimuth_deg", 30.0)),
            start_elevation_deg=float(args.get("start_elevation_deg", 8.0)),
            end_azimuth_deg=float(args.get("end_azimuth_deg", -15.0)),
            end_elevation_deg=float(args.get("end_elevation_deg", 0.0)),
            fov_deg=float(args.get("fov_deg", 50.0)),
        )
        data = verify_camera_trajectory_sim3(traj_str)
    elif name == "get_hardware_vram_telemetry":
        data = get_hardware_vram_telemetry()
    else:
        raise ValueError(f"Unknown MCP tool: '{name}'. Available tools: {[t['name'] for t in MCP_TOOLS_SPEC]}")

    return {
        "tool": name,
        "isError": False,
        "structuredContent": data,
        "content": [
            {
                "type": "text",
                "text": json.dumps(data, ensure_ascii=False),
            }
        ],
    }


async def handle_mcp_jsonrpc(request_msg: dict[str, Any]) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 MCP message and return a JSON-RPC 2.0 response."""
    req_id = request_msg.get("id")
    method = str(request_msg.get("method", ""))
    params = request_msg.get("params") or {}

    # Notifications have no id and expect no response
    if req_id is None and method.startswith("notifications/"):
        return None

    try:
        if method == "initialize":
            result = {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": MCP_SERVER_INFO,
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": MCP_TOOLS_SPEC}
        elif method == "tools/call":
            tool_name = str(params.get("name", ""))
            tool_args = params.get("arguments") or {}
            result = await call_mcp_tool(tool_name, tool_args)
        elif method == "resources/list":
            result = {"resources": []}
        elif method == "prompts/list":
            result = {"prompts": []}
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": str(exc)},
        }


async def run_stdio_server() -> None:
    """Run the JSON-RPC 2.0 MCP server over standard input/output (`stdio`)."""
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line_str = line.strip()
        if not line_str:
            continue
        try:
            msg = json.loads(line_str)
        except json.JSONDecodeError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        resp = await handle_mcp_jsonrpc(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(run_stdio_server())
