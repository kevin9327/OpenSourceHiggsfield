"""
Comprehensive Automated Test & Empirical Proof Suite for OpenSourceHiggsfield
=============================================================================
Verifies:
  1. All 5 custom ComfyUI nodes on PyTorch tensors (`OpenHiggsfieldCameraRig`,
     `OpenHiggsfieldDepthReshoot` including `matrix_freeze_time` and `magenta_ff00ff`,
     `OpenHiggsfieldRelightGrade`, `OpenHiggsfieldJutsuVFX`, `OpenHiggsfieldViralWipe`).
  2. `CVGEO1\\0\\0` binary geometry packing and lossless round-trip unpacking
     (`pack_cvgeo1_bytes` / `unpack_cvgeo1_bytes`).
  3. SVD-based Umeyama `Sim(3)` closed-loop trajectory verification & CIE LAB `Delta-E`
     color drift computation (`verify_camera_trajectory_sim3`).
  4. All 6 `.api.json` workflows + `workflows/manifest.json` + `comfy-build.yaml` schema
     and Open-Weights guardrail validation.
  5. `comfy-sdk` v0.5.0 & `comfy-api-proxy` v0.1.6 metadata stripping contract
     (`prepare_workflow_for_proxy`).
  6. FastAPI endpoints (`/api/status`, `/api/workflows`, `/api/reshoot/cvgeo?format=json`,
     `/api/reshoot/cvgeo`, `/api/benchmark/proof`, `/api/studio/cinematic`,
     `/api/studio/reshoot`, `/api/studio/openjutsu`).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.comfy_engine import (
    WORKFLOWS_DIR,
    engine,
    get_hardware_vram_telemetry,
    prepare_workflow_for_proxy,
)
from backend.server import app
from custom_nodes.comfyui_openhiggsfield_nodes import (
    CVGEO1_MAGIC,
    NODE_CLASS_MAPPINGS,
    OpenHiggsfieldCameraRig,
    OpenHiggsfieldDepthReshoot,
    OpenHiggsfieldJutsuVFX,
    OpenHiggsfieldRelightGrade,
    OpenHiggsfieldViralWipe,
    pack_cvgeo1_bytes,
    unpack_cvgeo1_bytes,
    verify_camera_trajectory_sim3,
)


class TestOpenHiggsfieldSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        # Create a deterministic synthetic RGB batch [4, 96, 128, 3] and depth [4, 96, 128]
        yy, xx = np.mgrid[0:96, 0:128].astype(np.float32)
        r = xx / 127.0
        g = yy / 95.0
        b = 0.5 + 0.3 * np.sin(xx * 0.1) * np.cos(yy * 0.1)
        base_frame = np.stack([r, g, b], axis=-1)
        frames = np.stack(
            [np.clip(base_frame + 0.04 * i, 0.0, 1.0) for i in range(4)],
            axis=0,
        ).astype(np.float32)
        cls.sample_frames_tensor = torch.from_numpy(frames)
        cls.sample_depth_np = (0.25 + 0.70 * np.exp(-((xx - 64.0) ** 2 + (yy - 48.0) ** 2) / 1800.0)).astype(
            np.float32
        )

    def test_01_node_registration_and_camera_rig_se3(self) -> None:
        """Verify all 5 custom nodes are registered and OpenHiggsfieldCameraRig produces valid SE(3) poses."""
        expected_nodes = {
            "OpenHiggsfieldCameraRig",
            "OpenHiggsfieldDepthReshoot",
            "OpenHiggsfieldRelightGrade",
            "OpenHiggsfieldJutsuVFX",
            "OpenHiggsfieldViralWipe",
        }
        self.assertEqual(set(NODE_CLASS_MAPPINGS.keys()), expected_nodes)

        rig = OpenHiggsfieldCameraRig()
        traj_str, tracks_str, sphere_tensor = rig.generate_rig(
            rig_preset="bullet_time_360",
            frame_count=17,
            fps=24.0,
            start_azimuth_deg=30.0,
            start_elevation_deg=8.0,
            end_azimuth_deg=-15.0,
            end_elevation_deg=0.0,
            fov_deg=50.0,
        )
        traj = json.loads(traj_str)
        tracks = json.loads(tracks_str)
        self.assertEqual(traj["frame_count"], 17)
        self.assertEqual(len(traj["poses"]), 17)
        self.assertEqual(len(tracks["tracks"]), 8)
        self.assertEqual(tuple(sphere_tensor.shape), (1, 512, 512, 3))

        # Check SE(3) rotation matrix orthogonality (R^T R = I, det(R) = +1)
        c2w = np.asarray(traj["poses"][8]["c2w_matrix"], dtype=np.float64)
        rot = c2w[:3, :3]
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-3)
        self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=3)

    def test_02_depth_reshoot_magenta_mask_and_matrix_freeze_time(self) -> None:
        """Verify OpenHiggsfieldDepthReshoot supports magenta_ff00ff disocclusion fill and matrix_freeze_time."""
        reshoot = OpenHiggsfieldDepthReshoot()
        warped_t, mask_t, depth_vis_t, traj_str = reshoot.reshoot_video(
            self.sample_frames_tensor,
            rig_preset="custom_orbit",
            start_azimuth_deg=35.0,
            start_elevation_deg=12.0,
            end_azimuth_deg=-35.0,
            end_elevation_deg=-8.0,
            start_distance=1.0,
            end_distance=0.9,
            fov_deg=50.0,
            target_frames=8,
            disocclusion_fill_mode="magenta_ff00ff",
            matrix_freeze_time=True,
        )
        self.assertEqual(tuple(warped_t.shape), (8, 96, 128, 3))
        self.assertEqual(tuple(mask_t.shape), (8, 96, 128))
        self.assertEqual(tuple(depth_vis_t.shape), (8, 96, 128, 3))

        traj = json.loads(traj_str)
        self.assertTrue(traj["matrix_freeze_time"])
        self.assertEqual(traj["disocclusion_fill_mode"], "magenta_ff00ff")

        # Verify disoccluded pixels in warped_t are filled with exact #FF00FF [1.0, 0.0, 1.0]
        warped_np = warped_t.numpy()
        mask_np = mask_t.numpy()
        disoccluded = mask_np > 0.5
        self.assertTrue(np.any(disoccluded), "Expected disocclusion mask to have positive pixels under 35 deg orbit")
        magenta_pixels = warped_np[disoccluded]
        np.testing.assert_allclose(magenta_pixels[:, 0], 1.0, atol=1e-4)
        np.testing.assert_allclose(magenta_pixels[:, 1], 0.0, atol=1e-4)
        np.testing.assert_allclose(magenta_pixels[:, 2], 1.0, atol=1e-4)

    def test_03_relight_grade_surface_normals_and_prompt_compiler(self) -> None:
        """Verify OpenHiggsfieldRelightGrade computes 3D normals, relights frames, and compiles camera recipe."""
        relight = OpenHiggsfieldRelightGrade()
        relit_t, norm_t, compiled_prompt = relight.relight_and_grade(
            self.sample_frames_tensor[:2],
            camera_type="large_format_anamorphic",
            framing="low_angle",
            light_preset="neon",
            film_stock="tungsten_500t",
            color_grade="teal_and_orange",
        )
        self.assertEqual(tuple(relit_t.shape), (2, 96, 128, 3))
        self.assertEqual(tuple(norm_t.shape), (2, 96, 128, 3))
        self.assertIn("Anamorphic", compiled_prompt)
        self.assertIn("500T", compiled_prompt)
        self.assertIn("teal-and-orange", compiled_prompt)

    def test_04_jutsu_vfx_and_viral_wipe_nodes(self) -> None:
        """Verify OpenHiggsfieldJutsuVFX and OpenHiggsfieldViralWipe on PyTorch tensors."""
        jutsu = OpenHiggsfieldJutsuVFX()
        char_tensor = torch.ones((1, 96, 64, 3), dtype=torch.float32) * 0.8
        char_tensor[:, :12, :, :] = 0.1  # distinct corner background for sprite extraction
        swapped_t, alpha_t, energy_t = jutsu.apply_jutsu(
            self.sample_frames_tensor,
            char_tensor,
            who_to_replace="the guy in the left",
            jutsu_style="neon_chakra_aura",
            vfx_intensity=0.75,
        )
        self.assertEqual(tuple(swapped_t.shape), (4, 96, 128, 3))
        self.assertEqual(tuple(alpha_t.shape), (4, 96, 128))
        self.assertEqual(tuple(energy_t.shape), (4, 96, 128, 3))

        wipe = OpenHiggsfieldViralWipe()
        (wipe_t,) = wipe.render_wipe(
            self.sample_frames_tensor,
            swapped_t,
            wipe_mode="three_way_depth_split",
            left_label="Source",
            right_label="Result",
            guide_images=energy_t,
        )
        self.assertEqual(tuple(wipe_t.shape), (4, 96, 128, 3))

    def test_05_cvgeo1_binary_pack_and_lossless_depth_unpack(self) -> None:
        """Verify CVGEO1\\0\\0 binary header spec and bit-exact float16 shuffle2+zlib depth round-trip."""
        depths = np.stack([self.sample_depth_np * (0.9 + 0.03 * i) for i in range(4)], axis=0).astype(np.float16)
        blob = pack_cvgeo1_bytes(self.sample_frames_tensor, depths, fps=24.0, fx_norm=0.85)
        self.assertEqual(blob[:8], CVGEO1_MAGIC)

        unpacked = unpack_cvgeo1_bytes(blob)
        hdr = unpacked["header"]
        self.assertEqual(hdr["version"], 1)
        self.assertEqual(hdr["frames"], 4)
        self.assertEqual(hdr["width"], 128)
        self.assertEqual(hdr["height"], 96)
        self.assertAlmostEqual(hdr["fps"], 24.0)
        self.assertAlmostEqual(hdr["fx_norm"], 0.85)
        self.assertEqual(hdr["depth"]["dtype"], "float16")
        self.assertEqual(hdr["depth"]["compression"], "shuffle2+zlib")
        self.assertEqual(len(hdr["jpegs"]), 4)

        # Bit-for-bit lossless float16 depth verification
        np.testing.assert_array_equal(unpacked["depth_float16"], depths)
        self.assertEqual(tuple(unpacked["frames_rgb"].shape), (4, 96, 128, 3))

    def test_06_umeyama_sim3_trajectory_and_lab_delta_e_verifier(self) -> None:
        """Verify SVD-based Umeyama Sim(3) camera alignment recovers global scale/rotation/translation."""
        rig = OpenHiggsfieldCameraRig()
        traj_str, _, _ = rig.generate_rig(
            rig_preset="custom_orbit",
            frame_count=12,
            fps=24.0,
            start_azimuth_deg=25.0,
            start_elevation_deg=10.0,
            end_azimuth_deg=-20.0,
            end_elevation_deg=-5.0,
            fov_deg=50.0,
        )
        traj = json.loads(traj_str)
        target_poses = traj["poses"]

        # Apply a known Sim(3) similarity transform (scale=2.5, translation=[1.2, -0.7, 0.4]) to positions
        transformed_poses = []
        for p in target_poses:
            pos = np.asarray(p["camera_position"], dtype=np.float64) * 2.5 + np.array([1.2, -0.7, 0.4])
            p_copy = dict(p)
            p_copy["camera_position"] = pos.tolist()
            transformed_poses.append(p_copy)

        metrics = verify_camera_trajectory_sim3(
            target_poses,
            transformed_poses,
            source_frames=self.sample_frames_tensor,
            rendered_frames=self.sample_frames_tensor,
        )
        self.assertLess(metrics["trans_err_pct"], 1e-3)
        self.assertLess(metrics["rot_err_deg"], 1e-3)
        self.assertLess(metrics["fov_err_deg"], 1e-3)
        self.assertLess(metrics["lab_drift_delta_e"], 1.5)
        self.assertGreaterEqual(metrics["fidelity_score_100"], 99.0)
        self.assertTrue(metrics["passed_closed_loop_audit"])

    def test_07_workflows_manifest_and_comfy_build_yaml_validation(self) -> None:
        """Verify all 6 .api.json workflows, manifest.json, and comfy-build.yaml pass open-weights audit."""
        manifest = engine.load_manifest()
        workflows = manifest.get("workflows", {})
        self.assertEqual(len(workflows), 6)

        for wf_id, wf_info in workflows.items():
            wf_path = WORKFLOWS_DIR / wf_info["file"]
            self.assertTrue(wf_path.exists(), f"Missing workflow file: {wf_path}")
            graph = engine.load_workflow_json(wf_id)
            is_open, violations = engine.validate_open_weights_workflow(graph)
            self.assertTrue(is_open, f"Workflow {wf_id} failed open-weights validation: {violations}")
            self.assertIn(str(wf_info["output_node_id"]), graph)

        comfy_build_path = ROOT_DIR / "comfy-build.yaml"
        self.assertTrue(comfy_build_path.exists(), "comfy-build.yaml must exist in repo root")
        build_text = comfy_build_path.read_text(encoding="utf-8")
        for required_token in (
            "2255709aa0",
            "v0.37.0",
            "comfyui_openhiggsfield_nodes",
            "ComfyUI-CrossViewWarp",
            "LTX-2.5-22B",
            "FLUX.3-Klein",
            "MoGe-2",
            "MiniMax-H3",
            "SAM-3",
            "TRELLIS-2",
        ):
            self.assertIn(required_token, build_text)

    def test_08_comfy_api_proxy_v016_metadata_stripping_contract(self) -> None:
        """Verify prepare_workflow_for_proxy strips metadata/_meta in OPEN_WEIGHTS_LOCAL mode and blocks closed nodes."""
        graph_with_meta = engine.load_workflow_json("ltx25_cinematic_studio")
        graph_with_meta["metadata"] = {"user": "test"}
        prepared_local = prepare_workflow_for_proxy(
            graph_with_meta,
            execution_mode="OPEN_WEIGHTS_LOCAL",
            metadata={"job": "should_be_stripped"},
        )
        self.assertIsNone(prepared_local["metadata"])
        self.assertTrue(prepared_local["proxy_v016_ready"])
        self.assertTrue(prepared_local["open_weights_compliant"])
        for node_def in prepared_local["prompt"].values():
            if isinstance(node_def, dict):
                self.assertNotIn("_meta", node_def)

        # Verify closed-source partner node is rejected in OPEN_WEIGHTS_LOCAL mode
        bad_graph = {"1": {"class_type": "KlingVideoPartnerNode", "inputs": {}}}
        with self.assertRaises(ValueError):
            prepare_workflow_for_proxy(bad_graph, execution_mode="OPEN_WEIGHTS_LOCAL")

    def test_09_hardware_vram_watchdog_telemetry(self) -> None:
        """Verify get_hardware_vram_telemetry reports 16GB RTX 5080 telemetry and 8n+1 frame budgets."""
        telemetry = get_hardware_vram_telemetry()
        self.assertIn("gpu", telemetry)
        self.assertIn("vram_watchdog", telemetry)
        self.assertGreaterEqual(int(telemetry["gpu"]["vram_total_mib"]), 16000)
        watchdog = telemetry["vram_watchdog"]
        self.assertEqual(watchdog["ltx25_frame_rule"], "8n+1")
        for f_count in watchdog["valid_frame_counts"]:
            self.assertEqual((f_count - 1) % 8, 0)
        resolutions = {b["resolution"] for b in watchdog["mod32_resolution_budgets"]}
        self.assertIn("640x352", resolutions)
        self.assertIn("768x448", resolutions)

    def test_10_fastapi_status_workflows_and_cvgeo_endpoints(self) -> None:
        """Verify /api/status, /api/workflows, and /api/reshoot/cvgeo (JSON & binary) endpoints."""
        r_status = self.client.get("/api/status")
        self.assertEqual(r_status.status_code, 200)
        status_data = r_status.json()
        self.assertEqual(status_data["status"], "online")
        self.assertIn("vram_watchdog", status_data)
        self.assertEqual(len(status_data["workflows"]), 6)

        r_wf = self.client.get("/api/workflows")
        self.assertEqual(r_wf.status_code, 200)
        self.assertEqual(len(r_wf.json()["manifest"]["workflows"]), 6)

        r_cvgeo_json = self.client.get("/api/reshoot/cvgeo?format=json&frames=4")
        self.assertEqual(r_cvgeo_json.status_code, 200)
        cvgeo_summary = r_cvgeo_json.json()
        self.assertTrue(cvgeo_summary["roundtrip_verified"])
        self.assertEqual(cvgeo_summary["max_float16_depth_error"], 0.0)

        r_cvgeo_bin = self.client.get("/api/reshoot/cvgeo?frames=3")
        self.assertEqual(r_cvgeo_bin.status_code, 200)
        self.assertEqual(r_cvgeo_bin.content[:8], CVGEO1_MAGIC)

    def test_11_fastapi_benchmark_proof_endpoint(self) -> None:
        """Verify GET /api/benchmark/proof returns complete 15-repo scorecard, Sim(3), and CVGEO1 proof."""
        r_proof = self.client.get("/api/benchmark/proof")
        self.assertEqual(r_proof.status_code, 200)
        proof = r_proof.json()
        self.assertEqual(proof["status"], "verified")
        self.assertTrue(proof["comfy_build_manifest_present"])
        self.assertTrue(proof["all_workflows_open_weights_compliant"])
        self.assertTrue(proof["cvgeo1_validation"]["roundtrip_verified"])
        self.assertGreaterEqual(proof["sim3_trajectory_verification"]["fidelity_score_100"], 95.0)
        self.assertEqual(len(proof["competitive_scorecard_15_repos"]), 15)
        self.assertEqual(proof["competitive_scorecard_15_repos"][0]["score"], 100)

    def test_12_fastapi_studio_cinematic_reshoot_openjutsu_endpoints(self) -> None:
        """Verify /api/studio/cinematic, /api/studio/reshoot, and /api/studio/openjutsu end-to-end."""
        r_cin = self.client.post(
            "/api/studio/cinematic",
            json={
                "prompt": "A cyberpunk detective in a neon conservatory.",
                "camera_type": "large_format_anamorphic",
                "framing": "low_angle",
                "light_preset": "neon",
                "film_stock": "tungsten_500t",
                "color_grade": "teal_and_orange",
                "mode": "image",
            },
        )
        self.assertEqual(r_cin.status_code, 200)
        cin_data = r_cin.json()
        self.assertEqual(cin_data["status"], "completed")
        self.assertIn("trajectory_verification", cin_data)
        self.assertIn("D", cin_data["variations"])

        r_reshoot = self.client.post(
            "/api/studio/reshoot",
            json={
                "rig_preset": "custom_orbit",
                "start_azimuth_deg": 25.0,
                "start_elevation_deg": 8.0,
                "end_azimuth_deg": -15.0,
                "end_elevation_deg": 0.0,
                "fov_deg": 50.0,
                "disocclusion_fill_mode": "inpaint_far_depth",
                "matrix_freeze_time": True,
            },
        )
        self.assertEqual(r_reshoot.status_code, 200)
        reshoot_data = r_reshoot.json()
        self.assertEqual(reshoot_data["status"], "completed")
        self.assertIn("trajectory_verification", reshoot_data)
        self.assertTrue(reshoot_data["trajectory"]["matrix_freeze_time"])

        r_jutsu = self.client.post(
            "/api/studio/openjutsu",
            json={
                "who_to_replace": "the guy in the left",
                "jutsu_style": "seamless_character_swap",
                "quality": "768p",
            },
        )
        self.assertEqual(r_jutsu.status_code, 200)
        jutsu_data = r_jutsu.json()
        self.assertEqual(jutsu_data["status"], "completed")
        self.assertTrue(jutsu_data["audio_preserved"])
        self.assertIn("trajectory_verification", jutsu_data)

    def test_13_comfy_mcp_server_and_agent_tools(self) -> None:
        """Verify Comfy MCP Server (JSON-RPC 2.0 initialize/tools/list/tools/call & /api/mcp/* endpoints)."""
        r_tools = self.client.get("/api/mcp/tools")
        self.assertEqual(r_tools.status_code, 200)
        tools_payload = r_tools.json()
        self.assertEqual(tools_payload["serverInfo"]["name"], "openhiggsfield-comfy-mcp")
        tool_names = [t["name"] for t in tools_payload["tools"]]
        self.assertEqual(
            tool_names,
            [
                "generate_cinematic_shot",
                "reshoot_video_3d",
                "openjutsu_vfx_swap",
                "export_cvgeo1_geometry",
                "verify_sim3_camera_trajectory",
                "get_hardware_vram_telemetry",
            ],
        )

        # Test POST /api/mcp/call for verify_sim3_camera_trajectory & export_cvgeo1_geometry
        r_sim3 = self.client.post(
            "/api/mcp/call",
            json={
                "name": "verify_sim3_camera_trajectory",
                "arguments": {"rig_preset": "custom_orbit", "frame_count": 12},
            },
        )
        self.assertEqual(r_sim3.status_code, 200)
        sim3_res = r_sim3.json()
        self.assertFalse(sim3_res["isError"])
        self.assertGreaterEqual(sim3_res["structuredContent"]["fidelity_score_100"], 95.0)

        # Test standard JSON-RPC 2.0 MCP protocol (initialize, tools/list, tools/call) via POST /api/mcp
        r_init = self.client.post(
            "/api/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        self.assertEqual(r_init.status_code, 200)
        self.assertEqual(r_init.json()["result"]["serverInfo"]["name"], "openhiggsfield-comfy-mcp")

        r_rpc_call = self.client.post(
            "/api/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "export_cvgeo1_geometry", "arguments": {"num_frames": 4}},
            },
        )
        self.assertEqual(r_rpc_call.status_code, 200)
        rpc_body = r_rpc_call.json()
        self.assertEqual(rpc_body["id"], 2)
        self.assertTrue(rpc_body["result"]["structuredContent"]["roundtrip_verified"])


if __name__ == "__main__":
    unittest.main()

