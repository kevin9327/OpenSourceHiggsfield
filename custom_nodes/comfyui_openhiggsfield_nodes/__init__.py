"""
ComfyUI-OpenHiggsfield-Nodes
============================
Open-source ComfyUI custom node pack for 6-DOF 3D Camera Control, Depth-Warped Video Re-shooting,
Physically-Based Relighting & Film Grading, OpenJutsu Character/VFX Transformation, and Viral Wipe Previews.
Built for the ComfyUI #OpenSourceHiggsfield Challenge.
"""

from .nodes import (
    CVGEO1_MAGIC,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    OpenHiggsfieldCameraRig,
    OpenHiggsfieldDepthReshoot,
    OpenHiggsfieldJutsuVFX,
    OpenHiggsfieldRelightGrade,
    OpenHiggsfieldViralWipe,
    pack_cvgeo1_bytes,
    unpack_cvgeo1_bytes,
    verify_camera_trajectory_sim3,
)

WEB_DIRECTORY = "./web"

__version__ = "1.1.0"
__all__ = [
    "CVGEO1_MAGIC",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "OpenHiggsfieldCameraRig",
    "OpenHiggsfieldDepthReshoot",
    "OpenHiggsfieldRelightGrade",
    "OpenHiggsfieldJutsuVFX",
    "OpenHiggsfieldViralWipe",
    "pack_cvgeo1_bytes",
    "unpack_cvgeo1_bytes",
    "verify_camera_trajectory_sim3",
]
