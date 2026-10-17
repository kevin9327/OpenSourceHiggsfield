"""
render_reshoot.py
Generates all real 3D Blender Cycles (OPTIX GPU) assets for Showcase 2: Re-Shoot 360 & Viral Camera Rigs
Output directory: OpenSourceHiggsfield/public/showcase/reshoot/
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
import mathutils

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "public", "showcase", "reshoot")
os.makedirs(OUT_DIR, exist_ok=True)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_cycles_gpu(scene, width=1280, height=720, samples=28):
    r = scene.render
    r.engine = "CYCLES"
    r.resolution_x = width
    r.resolution_y = height
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.fps = 24
    r.use_persistent_data = True

    cy = scene.cycles
    cy.device = "GPU"
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.025
    cy.use_denoising = True
    cy.denoiser = "OPTIX"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.max_bounces = 5
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 3
    cy.transmission_bounces = 4
    cy.volume_bounces = 0
    cy.sample_clamp_indirect = 8.0
    cy.caustics_reflective = False
    cy.caustics_refractive = False

    vs = scene.view_settings
    vs.view_transform = "AgX"
    try:
        vs.look = "AgX - High Contrast"
    except Exception:
        pass
    vs.exposure = 0.15

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == "OPTIX")


def make_mat(name, base=(0.5, 0.5, 0.5, 1.0), metallic=0.0, roughness=0.35,
             em_color=(0, 0, 0, 1), em_strength=0.0, coat=0.0, bump_scale=0.0, bump_freq=30.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    if em_strength > 0 and "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = em_color
        bsdf.inputs["Emission Strength"].default_value = em_strength
    if bump_scale > 0:
        tc = nodes.new("ShaderNodeTexCoord")
        vor = nodes.new("ShaderNodeTexVoronoi")
        vor.feature = "DISTANCE_TO_EDGE"
        vor.inputs["Scale"].default_value = bump_freq
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_scale
        bump.inputs["Distance"].default_value = 0.015
        links.new(tc.outputs["Object"], vor.inputs["Vector"])
        links.new(vor.outputs["Distance"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_reshoot_scene(scene):
    """
    Builds the Sci-Fi Valkyrie Pilot in purple/chrome cyber-armor touching her visor comms,
    with a massive burning & exploding Dreadnought Starship in the sky behind her!
    """
    # World Sky: Twilight purple-blue storm atmosphere
    world = bpy.data.worlds.new("ReshootWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.14, 0.14, 0.24, 1.0)
    bg.inputs["Strength"].default_value = 1.4

    mat_purple_armor = make_mat("PurpleArmor", base=(0.24, 0.09, 0.44, 1.0), metallic=0.82, roughness=0.18, coat=0.9, bump_scale=0.25, bump_freq=22.0)
    mat_chrome = make_mat("TitaniumChrome", base=(0.72, 0.75, 0.82, 1.0), metallic=0.94, roughness=0.16, bump_scale=0.15, bump_freq=35.0)
    mat_dark_joint = make_mat("CarbonJoint", base=(0.04, 0.045, 0.055, 1.0), metallic=0.5, roughness=0.45, bump_scale=0.3, bump_freq=60.0)
    mat_visor = make_mat("CyberVisor", base=(0.12, 0.45, 0.65, 1.0), metallic=0.88, roughness=0.04, em_color=(0.08, 0.85, 1.0, 1.0), em_strength=1.2, coat=1.0)
    mat_skin = make_mat("PilotSkin", base=(0.58, 0.38, 0.30, 1.0), metallic=0.0, roughness=0.38, coat=0.12)
    mat_cyan_led = make_mat("CyanLED", base=(0.1, 0.95, 1.0, 1.0), em_color=(0.1, 0.92, 1.0, 1.0), em_strength=28.0)
    mat_mag_led = make_mat("MagLED", base=(1.0, 0.15, 0.65, 1.0), em_color=(1.0, 0.15, 0.65, 1.0), em_strength=25.0)

    # 1. HERO PILOT IN FOREGROUND (at origin 0, 0, 0)
    pilot_root = bpy.data.objects.new("PilotRoot", None)
    scene.collection.objects.link(pilot_root)

    # Armored Chest & Torso
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.25, depth=0.55, location=(0, 0, 1.22))
    torso = bpy.context.active_object
    torso.scale = (1.15, 0.72, 1.0)
    torso.parent = pilot_root
    torso.data.materials.append(mat_dark_joint)
    bpy.ops.object.shade_smooth()

    # Sculpted Purple Armor Chest Plates & V-Collar
    for side in (-1, 1):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=0.16, location=(side * 0.12, -0.10, 1.30))
        plate = bpy.context.active_object
        plate.scale = (0.95, 0.55, 1.15)
        plate.rotation_euler = (0.15, side * 0.2, -side * 0.15)
        plate.parent = pilot_root
        plate.data.materials.append(mat_purple_armor)
        bpy.ops.object.shade_smooth()

        # Chrome Shoulder Pauldrons
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=0.15, location=(side * 0.31, 0.0, 1.44))
        pauldron = bpy.context.active_object
        pauldron.scale = (1.15, 0.85, 0.75)
        pauldron.rotation_euler = (0, side * 0.35, 0)
        pauldron.parent = pilot_root
        pauldron.data.materials.append(mat_purple_armor)
        bpy.ops.object.shade_smooth()

        # Chrome trim arch on shoulder
        bpy.ops.mesh.primitive_torus_add(major_radius=0.14, minor_radius=0.016, location=(side * 0.28, 0.0, 1.44), rotation=(0, math.pi / 2 - side * 0.3, 0))
        trim = bpy.context.active_object
        trim.parent = pilot_root
        trim.data.materials.append(mat_chrome)

        # Glowing Cyan Collar Strips
        bpy.ops.mesh.primitive_cube_add(size=0.04, location=(side * 0.11, -0.17, 1.38))
        led = bpy.context.active_object
        led.scale = (0.25, 0.2, 3.2)
        led.rotation_euler = (0.2, -side * 0.45, 0)
        led.parent = pilot_root
        led.data.materials.append(mat_cyan_led)

    # Neck (Armored Ribbed Column)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.068, depth=0.16, location=(0, 0, 1.56))
    neck = bpy.context.active_object
    neck.parent = pilot_root
    neck.data.materials.append(mat_dark_joint)
    bpy.ops.object.shade_smooth()

    # Sculpted Head & Purple Cyber-Helmet + Glowing Visor
    head_root = bpy.data.objects.new("PilotHead", None)
    scene.collection.objects.link(head_root)
    head_root.parent = pilot_root
    head_root.location = (0.0, -0.01, 1.69)

    # Lower Face (Skin: Jaw, Cheeks, Nose, Lips)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24, radius=0.108, location=(0, -0.01, 0.0))
    face = bpy.context.active_object
    face.scale = (0.88, 0.98, 1.12)
    face.parent = head_root
    face.data.materials.append(mat_skin)
    bpy.ops.object.shade_smooth()

    # Nose & Lips
    bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=0.019, radius2=0.007, depth=0.045, location=(0, -0.106, -0.005), rotation=(math.pi / 2 - 0.22, 0, 0))
    nose = bpy.context.active_object
    nose.parent = head_root
    nose.data.materials.append(mat_skin)
    bpy.ops.object.shade_smooth()

    for z_l, r_l in ((-0.042, 0.024), (-0.050, 0.022)):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, radius=r_l, location=(0, -0.095, z_l))
        lip = bpy.context.active_object
        lip.scale = (1.3, 0.5, 0.3)
        lip.parent = head_root
        lip.data.materials.append(mat_skin)
        bpy.ops.object.shade_smooth()

    # Purple Armored Helmet Dome & Cheek Guards
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24, radius=0.122, location=(0, 0.01, 0.045))
    helmet = bpy.context.active_object
    helmet.scale = (0.94, 1.06, 1.05)
    helmet.parent = head_root
    helmet.data.materials.append(mat_purple_armor)
    bpy.ops.object.shade_smooth()

    # Chrome Central Helmet Crest & Ear Pods
    bpy.ops.mesh.primitive_torus_add(major_radius=0.118, minor_radius=0.012, location=(0, 0.01, 0.05), rotation=(0, math.pi / 2, 0))
    crest = bpy.context.active_object
    crest.parent = head_root
    crest.data.materials.append(mat_chrome)

    for side in (-1, 1):
        bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.042, depth=0.03, location=(side * 0.112, 0.01, 0.01), rotation=(0, math.pi / 2, 0))
        ear = bpy.context.active_object
        ear.parent = head_root
        ear.data.materials.append(mat_chrome)

    # Curved Iridescent Cyan-Purple Tactical Visor Shield across Eyes
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.114, depth=0.068, location=(0, -0.015, 0.026))
    visor = bpy.context.active_object
    visor.scale = (0.94, 1.02, 1.0)
    visor.parent = head_root
    visor.data.materials.append(mat_visor)
    bpy.ops.object.shade_smooth()

    # Left Arm (Raised Mechanical/Chrome Hand touching left temple comms, matching reference!)
    # In camera view (looking at +Y), viewer's right side is +X: let's put the raised arm on +X so it matches the reference frame!
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.065, depth=0.34, location=(0.36, -0.04, 1.38), rotation=(0.25, -0.55, -0.2))
    uarm_r = bpy.context.active_object
    uarm_r.parent = pilot_root
    uarm_r.data.materials.append(mat_purple_armor)
    bpy.ops.object.shade_smooth()

    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.055, depth=0.34, location=(0.26, -0.10, 1.56), rotation=(0.35, 0.72, 0.15))
    farm_r = bpy.context.active_object
    farm_r.parent = pilot_root
    farm_r.data.materials.append(mat_chrome)
    bpy.ops.object.shade_smooth()

    # Articulated Cyber-Glove Fingers at Temple (+0.13, -0.06, 1.70)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, radius=0.042, location=(0.14, -0.07, 1.69))
    hand_r = bpy.context.active_object
    hand_r.scale = (0.7, 1.1, 1.2)
    hand_r.parent = pilot_root
    hand_r.data.materials.append(mat_chrome)

    # Other Arm (-X side) angled down heroically
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.068, depth=0.38, location=(-0.34, 0.02, 1.26), rotation=(0.1, 0.25, 0))
    uarm_l = bpy.context.active_object
    uarm_l.parent = pilot_root
    uarm_l.data.materials.append(mat_purple_armor)
    bpy.ops.object.shade_smooth()

    # 2. BACKGROUND: MASSIVE BURNING & EXPLODING CAPITAL STARSHIP + ROCKY CLIFFS
    starship_root = bpy.data.objects.new("CapitalStarship", None)
    scene.collection.objects.link(starship_root)
    # Positioned behind pilot at Y = +14.0m, X = +1.8m, Z = +3.2m, tilted dramatically at 34 degrees
    starship_root.location = (1.65, 13.5, 2.95)
    starship_root.rotation_euler = (math.radians(12.0), math.radians(-34.0), math.radians(-18.0))

    mat_hull = make_mat("StarshipHull", base=(0.11, 0.10, 0.14, 1.0), metallic=0.78, roughness=0.38, bump_scale=0.45, bump_freq=18.0)
    mat_hull_dark = make_mat("StarshipTruss", base=(0.04, 0.04, 0.06, 1.0), metallic=0.85, roughness=0.32)
    mat_fire_core = make_mat("FireCore", base=(1.0, 0.85, 0.35, 1.0), em_color=(1.0, 0.72, 0.18, 1.0), em_strength=65.0)
    mat_fire_mid = make_mat("FireMid", base=(1.0, 0.35, 0.04, 1.0), em_color=(1.0, 0.28, 0.02, 1.0), em_strength=38.0)
    mat_smoke = make_mat("DarkSmoke", base=(0.05, 0.04, 0.05, 1.0), roughness=0.95, bump_scale=0.6, bump_freq=8.0)

    # Main Elongated Dreadnought Hull Wedge & Side Gondolas
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    main_hull = bpy.context.active_object
    main_hull.scale = (7.5, 1.8, 0.85)
    main_hull.parent = starship_root
    main_hull.data.materials.append(mat_hull)

    for z_deck, sx, sy in ((0.65, 4.2, 1.2), (-0.55, 5.0, 1.3), (1.15, 1.8, 0.65)):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-0.6, 0, z_deck))
        deck = bpy.context.active_object
        deck.scale = (sx, sy, 0.45)
        deck.parent = starship_root
        deck.data.materials.append(mat_hull_dark)

    # Greeble Rib Plates & magenta/cyan city lights along the hull
    rng = np.random.default_rng(99)
    for i in range(42):
        gx = rng.uniform(-6.2, 5.8)
        gy = rng.uniform(-1.5, 1.5)
        gz = rng.uniform(-0.6, 0.9)
        bpy.ops.mesh.primitive_cube_add(size=0.35, location=(gx, gy, gz))
        gr = bpy.context.active_object
        gr.scale = (rng.uniform(0.5, 2.8), rng.uniform(0.4, 1.5), rng.uniform(0.4, 1.8))
        gr.parent = starship_root
        if i % 5 == 0:
            gr.data.materials.append(mat_mag_led if i % 10 == 0 else mat_cyan_led)
        else:
            gr.data.materials.append(mat_hull)

    # Blazing Volumetric-style Multi-Layered Explosion Fireballs erupting from forward/upper hull (X = +2.2 .. +5.5)
    explosion_Group = []
    fire_centers = [
        (3.2, -0.4, 0.4, 1.35, mat_fire_core),
        (4.1, -0.6, 0.8, 1.15, mat_fire_core),
        (2.4, -0.3, -0.2, 1.05, mat_fire_mid),
        (3.6, -0.2, 1.2, 1.45, mat_fire_mid),
        (4.8, -0.5, 0.3, 1.25, mat_fire_mid),
        (1.8, -0.5, 0.9, 1.10, mat_smoke),
        (2.9, 0.3, 1.6, 1.55, mat_smoke),
    ]
    for idx, (fx, fy, fz, fr, fmat) in enumerate(fire_centers):
        # Cluster of organic metaball-like displaced spheres for each fireball
        for sub in range(7):
            ox = fx + rng.normal(0, fr * 0.32)
            oy = fy + rng.normal(0, fr * 0.25)
            oz = fz + rng.normal(0, fr * 0.32)
            sr = fr * rng.uniform(0.42, 0.85)
            bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=14, radius=sr, location=(ox, oy, oz))
            fb = bpy.context.active_object
            fb.parent = starship_root
            fb.data.materials.append(fmat)
            bpy.ops.object.shade_smooth()
            explosion_Group.append(fb)

    # Glowing Flying Debris Shards & Escort Frigates in the sky
    debris_objs = []
    for i in range(55):
        dx = rng.uniform(-5.5, 6.5)
        dy = rng.uniform(2.5, 11.5)
        dz = rng.uniform(0.3, 4.8)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=rng.uniform(0.03, 0.14), location=(dx, dy, dz))
        sh = bpy.context.active_object
        sh.scale = (rng.uniform(0.6, 2.8), rng.uniform(0.4, 1.5), rng.uniform(0.3, 1.2))
        sh.rotation_euler = tuple(rng.uniform(0, 6.28, 3))
        sh.data.materials.append(mat_fire_mid if i % 3 == 0 else mat_hull_dark)
        debris_objs.append((sh, sh.location.copy()))

    # Distant Jagged Sci-Fi Mountain Peaks on the horizon (Y = 22m)
    mat_mountain = make_mat("MountainRock", base=(0.06, 0.055, 0.085, 1.0), metallic=0.1, roughness=0.82, bump_scale=0.5, bump_freq=12.0)
    for mx in np.linspace(-18.0, 18.0, 15):
        mh = rng.uniform(3.5, 7.5)
        bpy.ops.mesh.primitive_cone_add(vertices=7, radius1=rng.uniform(3.5, 6.0), radius2=0.1, depth=mh, location=(mx, 22.0 + rng.uniform(-2, 2), mh * 0.35))
        mt = bpy.context.active_object
        mt.rotation_euler = (0, 0, rng.uniform(0, 6.28))
        mt.data.materials.append(mat_mountain)

    # 3. CINEMATIC LIGHTING (Key, Cool Rim, and Orange Explosion Backlight!)
    def add_light(name, loc, rot, energy, color, size=2.0):
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = energy
        ld.color = color
        ld.size = size
        ob = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(ob)
        ob.location = loc
        ob.rotation_euler = rot
        return ob

    add_light("KeyLight", (-1.5, -2.2, 2.1), (math.radians(72), 0, math.radians(-32)), 145.0, (0.85, 0.88, 1.0), size=2.2)
    add_light("PurpleRimLeft", (-2.4, 1.2, 1.9), (math.radians(82), 0, math.radians(-115)), 260.0, (0.68, 0.25, 1.0), size=2.5)
    add_light("ExplosionRimRight", (2.8, 2.5, 2.6), (math.radians(68), 0, math.radians(138)), 520.0, (1.0, 0.45, 0.12), size=3.5)

    return pilot_root, head_root, starship_root, debris_objs


def set_orbit_camera(cam_ob, yaw_deg, pitch_deg, dist=2.05, target=(0.0, 0.0, 1.54), lens=50.0, roll_deg=0.0):
    """Positions camera on a spherical orbit around target with given yaw (deg) and pitch (deg)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    tx, ty, tz = target
    # At yaw=0, pitch=0 -> camera is at (tx, ty - dist, tz) looking +Y
    cx = tx + dist * math.sin(yaw) * math.cos(pitch)
    cy = ty - dist * math.cos(yaw) * math.cos(pitch)
    cz = tz + dist * math.sin(pitch)
    cam_ob.location = (cx, cy, cz)
    direction = mathutils.Vector(target) - cam_ob.location
    quat = direction.to_track_quat("-Z", "Y")
    eul = quat.to_euler()
    eul.y += math.radians(roll_deg)
    cam_ob.rotation_euler = eul
    cam_ob.data.lens = lens
    cam_ob.data.dof.use_dof = True
    cam_ob.data.dof.focus_distance = dist
    cam_ob.data.dof.aperture_fstop = 2.8


def encode_mp4(frames_bgr, out_mp4, fps=24):
    """Encodes a list of BGR numpy frames to H.264 MP4 using imageio_ffmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    h, w = frames_bgr[0].shape[:2]
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "17",
        out_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames_bgr:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def make_depth_warp_hud_frame(frame_bgr, f_idx, n_frames, yaw_deg, pitch_deg):
    """
    Transforms a rendered 3D frame into the signature 'depth_warp_guide.mp4' frame:
    - 3D Depth + Point-Cloud Wireframe Grid + Magenta Disocclusion Zone + Yellow Trajectory HUD!
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130)

    # Create stylized 3D depth/point-cloud base
    depth_tint = cv2.applyColorMap(gray, cv2.COLORMAP_DEEPGREEN)
    base = cv2.addWeighted(frame_bgr, 0.52, depth_tint, 0.48, 0)

    # Overlay cyan 3D mesh wireframe edges
    base[edges > 0] = (255, 240, 60)

    # Draw Magenta Disocclusion Mask Zone along the right silhouette edge as camera orbits
    t_frac = f_idx / max(1, n_frames - 1)
    shift_px = int(abs(yaw_deg - 30.0) * 2.2)
    if shift_px > 4:
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        disocc_mask = (np.abs(grad_x) > 85).astype(np.uint8)
        kernel = np.ones((3, min(28, max(3, shift_px // 3))), np.uint8)
        disocc_band = cv2.dilate(disocc_mask, kernel, iterations=1)
        mag_overlay = base.copy()
        mag_overlay[disocc_band > 0] = (145, 35, 255)  # Vibrant Magenta BGR
        base = cv2.addWeighted(base, 0.65, mag_overlay, 0.35, 0)

    # Draw 3D Point-Cloud Grid dots across frame
    step = 24
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            lum = int(gray[y, x])
            col = (69, 255, 234) if lum > 90 else (180, 160, 80)
            cv2.circle(base, (x, y), 1, col, -1)

    # Draw Bottom-Right 3D Spherical Orbit Widget & Trajectory HUD (matching contact_sheet_reshoot.jpg!)
    cx, cy_hud, r_hud = w - 155, h - 145, 95
    overlay = base.copy()
    cv2.circle(overlay, (cx, cy_hud), r_hud + 12, (24, 20, 32), -1)
    base = cv2.addWeighted(base, 0.35, overlay, 0.65, 0)
    cv2.circle(base, (cx, cy_hud), r_hud, (95, 90, 110), 1, cv2.LINE_AA)
    cv2.ellipse(base, (cx, cy_hud), (r_hud, int(r_hud * 0.38)), 0, 0, 360, (85, 80, 100), 1, cv2.LINE_AA)
    cv2.ellipse(base, (cx, cy_hud), (int(r_hud * 0.38), r_hud), 0, 0, 360, (85, 80, 100), 1, cv2.LINE_AA)

    # Yellow Orbit Trajectory Arc (#EAFF45 -> BGR (69, 255, 234))
    cv2.ellipse(base, (cx, cy_hud), (r_hud, int(r_hud * 0.32)), 0, 15, 165, (69, 255, 234), 3, cv2.LINE_AA)
    # Keyframe diamonds at start (30°/8°) and end (-15°/0°)
    ang_cur = math.radians(90.0 - yaw_deg * 1.4)
    kx = int(cx + r_hud * math.cos(ang_cur))
    ky = int(cy_hud + r_hud * 0.32 * math.sin(ang_cur))
    cv2.circle(base, (kx, ky), 7, (69, 255, 234), -1, cv2.LINE_AA)

    # HUD Telemetry Banner
    t_sec = f_idx / 24.0
    cv2.rectangle(base, (28, 24), (620, 96), (22, 18, 28), -1)
    cv2.rectangle(base, (28, 24), (620, 96), (69, 255, 234), 1)
    cv2.putText(base, f"3D CAMERA RE-SHOOT SOLVER // t={t_sec:0.1f}s", (44, 52),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (69, 255, 234), 1, cv2.LINE_AA)
    cv2.putText(base, f"KEY 0.0s (30 deg / 8 deg) -> KEY 4.4s (-15 deg / 0 deg) | CUR: {yaw_deg:+.1f} / {pitch_deg:+.1f}",
                (44, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 245), 1, cv2.LINE_AA)
    return base


def main():
    t0 = time.time()
    clear_scene()
    scene = bpy.context.scene
    # Render at 640x360 w/ OPTIX and super-resolve to 1280x720 Lanczos + detail sharpen so 600+ frames finish in ~90 seconds!
    setup_cycles_gpu(scene, width=640, height=360, samples=24)

    cam_data = bpy.data.cameras.new("ReshootCam")
    cam_ob = bpy.data.objects.new("ReshootCam", cam_data)
    scene.collection.objects.link(cam_ob)
    scene.camera = cam_ob

    pilot_root, head_root, starship_root, debris_objs = build_reshoot_scene(scene)
    tmp_png = os.path.join(OUT_DIR, "_tmp_rs.png")

    def render_frame_hd():
        scene.render.filepath = tmp_png
        bpy.ops.render.render(write_still=True)
        img = cv2.imread(tmp_png)
        img_hd = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
        # Subtle unsharp mask for crisp 720p edges + warm explosion bloom
        blur = cv2.GaussianBlur(img_hd, (0, 0), 2.0)
        img_hd = cv2.addWeighted(img_hd, 1.25, blur, -0.25, 0)
        return img_hd

    def update_scene_motion(f, fps=24.0):
        t = f / fps
        # Subtle breathing & head turn of the pilot
        head_root.rotation_euler = (math.sin(t * 1.8) * 0.03, math.cos(t * 1.4) * 0.02, math.sin(t * 1.1) * 0.05)
        pilot_root.location.z = math.sin(t * 2.2) * 0.012
        # Starship slowly drifting & debris expanding
        starship_root.location.x = 1.65 - t * 0.08
        starship_root.location.z = 2.95 - t * 0.05
        for idx, (sh, init_loc) in enumerate(debris_objs):
            sh.location.x = init_loc.x + math.sin(t * 1.5 + idx) * 0.18
            sh.location.z = init_loc.z + math.cos(t * 1.3 + idx) * 0.15
            sh.rotation_euler.x += 0.04
            sh.rotation_euler.y += 0.05

    # 1. Render 120 frames (5.0s @ 24fps) for source.mp4, result.mp4, depth_warp_guide.mp4, viral_wipe.mp4
    # Wait: to make rendering super fast (~25 seconds per sequence), we render 60 keyframes (12fps motion) and optically/smoothly interpolate or render 60 frames with 2x temporal smooth interpolation to 120 frames!
    n_keyframes = 60
    source_key = []
    result_key = []
    guide_key = []

    print("Rendering Re-Shoot Source & Result 3D sequences...")
    for k in range(n_keyframes):
        f_idx = k * 2
        t_sec = f_idx / 24.0
        update_scene_motion(f_idx)

        # Source camera: original front medium shot (subtle handheld drift around 12° yaw so pilot is front-left)
        set_orbit_camera(cam_ob, yaw_deg=18.0 + math.sin(t_sec * 1.2) * 1.5, pitch_deg=4.0, dist=1.95)
        src_img = render_frame_hd()
        source_key.append(src_img)

        # Result camera: smoothly orbits from +30° / +8° at 0.0s to -15° / 0° at 4.4s..5.0s, revealing the exploding starship!
        u = min(1.0, t_sec / 4.4)
        smooth_u = u * u * (3.0 - 2.0 * u)
        yaw_res = 30.0 + (-15.0 - 30.0) * smooth_u
        pitch_res = 8.0 + (0.0 - 8.0) * smooth_u
        set_orbit_camera(cam_ob, yaw_deg=yaw_res, pitch_deg=pitch_res, dist=1.95 - 0.12 * math.sin(smooth_u * math.pi))
        res_img = render_frame_hd()
        result_key.append(res_img)

        guide_img = make_depth_warp_hud_frame(res_img, f_idx, 120, yaw_res, pitch_res)
        guide_key.append(guide_img)

    # Upsample 60 keyframes -> 120 frames @ 24fps smoothly
    def expand_to_120(keys):
        out = []
        for i in range(len(keys)):
            out.append(keys[i])
            if i + 1 < len(keys):
                out.append(cv2.addWeighted(keys[i], 0.5, keys[i + 1], 0.5, 0))
            else:
                out.append(keys[i].copy())
        return out

    source_120 = expand_to_120(source_key)
    result_120 = expand_to_120(result_key)
    guide_120 = []
    for f_idx in range(120):
        t_sec = f_idx / 24.0
        u = min(1.0, t_sec / 4.4)
        smooth_u = u * u * (3.0 - 2.0 * u)
        yaw_res = 30.0 + (-15.0 - 30.0) * smooth_u
        pitch_res = 8.0 + (0.0 - 8.0) * smooth_u
        guide_120.append(make_depth_warp_hud_frame(result_120[f_idx], f_idx, 120, yaw_res, pitch_res))

    # Build viral_wipe.mp4 (Neon-yellow #EAFF45 vertical comparison wipe sweeping across Source <-> Guide <-> Result)
    wipe_120 = []
    for f_idx in range(120):
        s_img = source_120[f_idx]
        g_img = guide_120[f_idx]
        r_img = result_120[f_idx]
        h, w = s_img.shape[:2]
        comp = r_img.copy()
        # Sweep wipe bar smoothly back and forth across the 1280px frame
        phase = (f_idx / 119.0) * 2.0 * math.pi
        split1 = int(w * (0.5 + 0.38 * math.sin(phase)))
        split2 = min(w - 1, split1 + 220)
        comp[:, :split1] = s_img[:, :split1]
        comp[:, split1:split2] = g_img[:, split1:split2]
        # Neon Yellow #EAFF45 divider lines (BGR: 69, 255, 234)
        cv2.line(comp, (split1, 0), (split1, h), (69, 255, 234), 4)
        cv2.line(comp, (split2, 0), (split2, h), (69, 255, 234), 4)
        # Pill badges
        cv2.rectangle(comp, (max(12, split1 - 125), 24), (max(120, split1 - 12), 62), (20, 18, 26), -1)
        cv2.putText(comp, "SOURCE", (max(24, split1 - 110), 50), cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 240, 245), 1, cv2.LINE_AA)
        cv2.rectangle(comp, (min(w - 135, split2 + 12), 24), (min(w - 12, split2 + 135), 62), (69, 255, 234), -1)
        cv2.putText(comp, "RESULT", (min(w - 122, split2 + 25), 50), cv2.FONT_HERSHEY_DUPLEX, 0.65, (18, 18, 22), 2, cv2.LINE_AA)
        wipe_120.append(comp)

    encode_mp4(source_120, os.path.join(OUT_DIR, "source.mp4"), fps=24)
    encode_mp4(result_120, os.path.join(OUT_DIR, "result.mp4"), fps=24)
    encode_mp4(guide_120, os.path.join(OUT_DIR, "depth_warp_guide.mp4"), fps=24)
    encode_mp4(wipe_120, os.path.join(OUT_DIR, "viral_wipe.mp4"), fps=24)

    # 2. Render the 3 Viral Higgsfield Camera Rig Presets (4.0s = 96 frames @ 24fps, 1280x720)
    print("Rendering 3 Viral Higgsfield Camera Rig Presets (Bullet-Time, SnorriCam, Crash-Zoom)...")
    n_rig_keys = 48

    # Preset A: rig_bullet_time.mp4 (Frozen time matrix 140-degree sweep around suspended glowing shards & hero)
    bt_keys = []
    update_scene_motion(24)  # Freeze hero & explosion at dramatic peak frame!
    for k in range(n_rig_keys):
        u = k / (n_rig_keys - 1)
        su = u * u * (3 - 2 * u)
        yaw = -65.0 + 130.0 * su
        pitch = 12.0 * math.sin(su * math.pi)
        set_orbit_camera(cam_ob, yaw_deg=yaw, pitch_deg=pitch, dist=1.82 - 0.25 * math.sin(su * math.pi), lens=38.0)
        img = render_frame_hd()
        cv2.putText(img, "RIG PRESET // MATRIX BULLET-TIME 360 ORBIT", (36, 48), cv2.FONT_HERSHEY_DUPLEX, 0.68, (69, 255, 234), 2, cv2.LINE_AA)
        bt_keys.append(img)
    encode_mp4(expand_to_120(bt_keys), os.path.join(OUT_DIR, "rig_bullet_time.mp4"), fps=24)

    # Preset B: rig_snorricam.mp4 (Body-mount POV rig: hero locked dead-center with wide 24mm lens while world spins/tilts wildly)
    sn_keys = []
    for k in range(n_rig_keys):
        u = k / (n_rig_keys - 1)
        update_scene_motion(k * 2)
        yaw = 38.0 * math.sin(u * 2.0 * math.pi)
        pitch = 10.0 * math.cos(u * 2.0 * math.pi)
        roll = 14.0 * math.sin(u * 4.0 * math.pi)
        set_orbit_camera(cam_ob, yaw_deg=yaw, pitch_deg=pitch, dist=1.35, target=(0.0, 0.0, 1.62), lens=26.0, roll_deg=roll)
        img = render_frame_hd()
        cv2.putText(img, "RIG PRESET // SNORRICAM BODY-MOUNT POV LOCK", (36, 48), cv2.FONT_HERSHEY_DUPLEX, 0.68, (69, 255, 234), 2, cv2.LINE_AA)
        sn_keys.append(img)
    encode_mp4(expand_to_120(sn_keys), os.path.join(OUT_DIR, "rig_snorricam.mp4"), fps=24)

    # Preset C: rig_crash_zoom.mp4 (High-speed snap crash-zoom + whip-pan from wide battlefield to extreme visor close-up)
    cz_keys = []
    for k in range(n_rig_keys):
        u = k / (n_rig_keys - 1)
        update_scene_motion(k * 2)
        # Fast exponential snap zoom between u=0.25 and u=0.65
        snap = 1.0 / (1.0 + math.exp(-16.0 * (u - 0.42)))
        dist = 3.4 - 2.15 * snap
        lens = 28.0 + 68.0 * snap
        yaw = -28.0 * (1.0 - snap) + 6.0 * snap
        pitch = -4.0 * (1.0 - snap) + 3.0 * snap
        set_orbit_camera(cam_ob, yaw_deg=yaw, pitch_deg=pitch, dist=dist, target=(0.0, 0.0, 1.60 + 0.08 * snap), lens=lens)
        img = render_frame_hd()
        # Add radial zoom motion blur during the peak of the crash-zoom (0.32 < u < 0.52)
        if 0.30 < u < 0.55:
            blur_z = cv2.GaussianBlur(img, (15, 15), 4.0)
            img = cv2.addWeighted(img, 0.65, blur_z, 0.35, 0)
        cv2.putText(img, f"RIG PRESET // SNAP CRASH-ZOOM ({lens:.0f}mm)", (36, 48), cv2.FONT_HERSHEY_DUPLEX, 0.68, (69, 255, 234), 2, cv2.LINE_AA)
        cz_keys.append(img)
    encode_mp4(expand_to_120(cz_keys), os.path.join(OUT_DIR, "rig_crash_zoom.mp4"), fps=24)

    if os.path.exists(tmp_png):
        os.remove(tmp_png)
    print(f"Done all Re-Shoot 360 & Viral Rig videos in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
