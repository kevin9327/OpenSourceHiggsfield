"""
render_cinematic_video.py
Renders the 5-second (120 frames @ 24fps, 1280x720) animated video `cinematic_shot_d.mp4`
for Showcase 1: Cinematic Studio, featuring anamorphic camera push-in, python coil breathing,
and volumetric golden/neon light sweep!
"""
import os
import sys
import math
import time
import subprocess
import numpy as np
import cv2
import imageio_ffmpeg
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_cinematic_studio as CS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "public", "showcase", "cinematic_studio")


def main():
    t0 = time.time()
    CS.clear_scene()
    scene = bpy.context.scene
    # Render 60 keyframes at 800x450 w/ OPTIX denoiser and Lanczos4 super-resolve + cinema postfx to 1280x720
    CS.setup_cycles_gpu(scene, width=800, height=450, samples=28)

    cam_data = bpy.data.cameras.new("StudioCamAnim")
    cam_ob = bpy.data.objects.new("StudioCamAnim", cam_data)
    scene.collection.objects.link(cam_ob)
    scene.camera = cam_ob

    CS.build_conservatory_architecture(scene)
    hero_root, head_pivot, python_ob = CS.build_hero_and_python(scene)
    CS.setup_lighting_preset(scene, "golden_hour")
    CS.configure_camera(scene, cam_ob, "low_angle_anamorphic")

    # Find key lights to animate volumetric light sweep
    sun_back = bpy.data.objects.get("GoldenSunBack")
    rim_left = bpy.data.objects.get("GoldenRimLeft")

    tmp_png = os.path.join(OUT_DIR, "_tmp_anim_d.png")
    n_keys = 60
    key_frames = []

    print("Rendering 60 3D keyframes for cinematic_shot_d.mp4...")
    for k in range(n_keys):
        u = k / (n_keys - 1)
        su = u * u * (3.0 - 2.0 * u)
        t_sec = u * 5.0

        # Anamorphic camera push-in + subtle upward heroic tilt + subtle horizontal parallax
        cam_ob.location = (
            0.05 - 0.14 * su + 0.04 * math.sin(u * math.pi),
            -2.55 + 0.42 * su,
            0.96 - 0.06 * su
        )
        cam_ob.rotation_euler = (
            math.radians(102.5 + 3.2 * su),
            math.radians(0.8 * math.sin(u * math.pi)),
            math.radians(1.2 - 3.5 * su)
        )
        cam_ob.data.dof.focus_distance = abs(cam_ob.location.y)

        # Subtle hero head turn & python breathing motion
        head_pivot.rotation_euler = (
            -0.06 + 0.03 * math.sin(t_sec * 1.5),
            0.03 * math.cos(t_sec * 1.2),
            -0.08 + 0.06 * su
        )
        python_ob.scale = (
            1.0 + 0.015 * math.sin(t_sec * 2.4),
            1.0 + 0.015 * math.cos(t_sec * 2.4),
            1.0 + 0.008 * math.sin(t_sec * 1.8)
        )

        # Sweeping Golden/Neon Conservatory Lights
        if sun_back:
            sun_back.location.x = 1.6 - 2.4 * su
            sun_back.rotation_euler.z = 2.65 + 0.45 * su
        if rim_left:
            rim_left.location.y = 2.5 + 1.5 * math.sin(u * math.pi)

        scene.render.filepath = tmp_png
        bpy.ops.render.render(write_still=True)

        img = cv2.imread(tmp_png)
        img_hd = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
        img_fx = CS.apply_cinema_postfx(img_hd, grade="teal_orange", anamorphic_flare=True)
        key_frames.append(img_fx)

    # Expand 60 keyframes -> 120 frames @ 24fps (5.0s)
    frames_120 = []
    for i in range(len(key_frames)):
        frames_120.append(key_frames[i])
        if i + 1 < len(key_frames):
            frames_120.append(cv2.addWeighted(key_frames[i], 0.5, key_frames[i + 1], 0.5, 0))
        else:
            frames_120.append(key_frames[i].copy())

    out_mp4 = os.path.join(OUT_DIR, "cinematic_shot_d.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", "1280x720",
        "-pix_fmt", "bgr24",
        "-r", "24",
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "17",
        out_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames_120:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()

    if os.path.exists(tmp_png):
        os.remove(tmp_png)
    print(f"Done cinematic_shot_d.mp4 in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
