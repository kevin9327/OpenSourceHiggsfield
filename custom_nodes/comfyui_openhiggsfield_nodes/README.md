# ComfyUI-OpenHiggsfield-Nodes (`comfyui-openhiggsfield-nodes`)

Open-source ComfyUI custom node pack implementing the core camera direction, 3D depth-warped video re-shooting, physically-based relighting, film stock grading, and character VFX pipelines for **OpenSourceHiggsfield** (`#OpenSourceHiggsfield`).

## Included Custom Nodes

| Node Class | Display Name | Category | Description |
| :--- | :--- | :--- | :--- |
| `OpenHiggsfieldCameraRig` | **OpenHiggsfield 6-DOF Camera Rig** | `OpenHiggsfield/Director` | Computes $4\times 4$ $SE(3)$ camera trajectory matrices, LTX `LTXVSparseTrackEditor` 2D/3D motion tracks, and 3D spherical gizmo previews for presets (`bullet_time_360`, `snorricam_pov`, `crash_zoom`, `fpv_drone_dive`, `dolly_zoom_vertigo`, `custom_orbit`). |
| `OpenHiggsfieldDepthReshoot` | **OpenHiggsfield 3D Depth Video Re-Shoot** | `OpenHiggsfield/Director` | Warps input images or videos in true 3D perspective using depth maps and keyframed `(azimuth_deg, elevation_deg, distance, fov)` trajectories. Outputs `warped_video`, `disocclusion_magenta_mask` (for LTX-2.5 IC-LoRA Union Control / Inpainting), `depth_preview`, and `trajectory_json`. |
| `OpenHiggsfieldRelightGrade` | **OpenHiggsfield 3D Relight & Film Grade** | `OpenHiggsfield/Director` | Estimates surface normals $\mathbf{N}(x,y)$ and applies physically-based 3D relighting (`blue_hour`, `neon_cyber`, `golden_hour`, `low_key`) + authentic motion-picture film stocks (`tungsten_500t`, `vision3_250d`) and color LUTs (`teal_and_orange`, `kodachrome`, `bleach_bypass`). |
| `OpenHiggsfieldJutsuVFX` | **OpenHiggsfield OpenJutsu Character VFX** | `OpenHiggsfield/Director` | Generates procedural multi-frame energy, chakra, dimensional portal, and smoke shockwave alpha mattes and composites for LTX-2.5 `ic-lora-ingredients` & `ic-lora-alpha-gen` character swaps. |
| `OpenHiggsfieldViralWipe` | **OpenHiggsfield Viral Comparison Wipe** | `OpenHiggsfield/Director` | Renders the signature `#OpenSourceHiggsfield` neon-yellow (`#EAFF45`) split-screen wipe across `Source`, `Depth/Rig Guide`, and `Result` frames with HUD badges. |

## Installation

### Via ComfyUI Manager / Comfy CLI
```bash
comfy node install comfyui-openhiggsfield-nodes
```

### Manual Installation
Symbolic-link or copy `custom_nodes/comfyui_openhiggsfield_nodes` into your `ComfyUI/custom_nodes/` directory.
