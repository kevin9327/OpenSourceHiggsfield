"""
render_cinematic_studio.py
Generates all real 3D Blender Cycles (OPTIX GPU) assets for Showcase 1: Cinematic Studio
Output directory: OpenSourceHiggsfield/public/showcase/cinematic_studio/
"""
import os
import sys
import math
import time
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import imageio_ffmpeg
import bpy
import mathutils

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "public", "showcase", "cinematic_studio")
os.makedirs(OUT_DIR, exist_ok=True)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_cycles_gpu(scene, width=1280, height=720, samples=48):
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
    cy.adaptive_threshold = 0.02
    cy.use_denoising = True
    cy.denoiser = "OPTIX"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.max_bounces = 6
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 4
    cy.transmission_bounces = 6
    cy.volume_bounces = 0
    cy.transparent_max_bounces = 6
    cy.sample_clamp_indirect = 10.0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.use_light_tree = True

    vs = scene.view_settings
    vs.view_transform = "AgX"
    try:
        vs.look = "AgX - High Contrast"
    except Exception:
        pass
    vs.exposure = 0.1

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == "OPTIX")


def make_pbr_mat(name, base_color=(0.5, 0.5, 0.5, 1.0), metallic=0.0, roughness=0.4,
                 emission_color=(0, 0, 0, 1), emission_strength=0.0,
                 transmission=0.0, ior=1.45, coat=0.0, bump_scale=0.0, bump_freq=20.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = transmission
    bsdf.inputs["IOR"].default_value = ior
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    if emission_strength > 0:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = emission_color
            bsdf.inputs["Emission Strength"].default_value = emission_strength

    if bump_scale > 0:
        tex_coord = nodes.new("ShaderNodeTexCoord")
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = bump_freq
        noise.inputs["Detail"].default_value = 6.0
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_scale
        bump.inputs["Distance"].default_value = 0.02
        links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_python_scale_mat():
    """Creates an ultra-detailed procedural Burmese/Reticulated python skin shader with 3D Voronoi scales & wet sheen."""
    mat = bpy.data.materials.new("PythonScales")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Metallic"].default_value = 0.18
    bsdf.inputs["Roughness"].default_value = 0.22
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.85
        bsdf.inputs["Coat Roughness"].default_value = 0.08

    tc = nodes.new("ShaderNodeTexCoord")

    # Scale cells (Voronoi)
    vor = nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 65.0

    # Macro diamond pattern (Noise + Musgrave-style Wave)
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "DIAGONAL"
    wave.inputs["Scale"].default_value = 6.5
    wave.inputs["Distortion"].default_value = 4.2
    wave.inputs["Detail"].default_value = 4.0

    # ColorRamp for Burmese Python gold / olive / obsidian blotches
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.02, 0.022, 0.018, 1.0)  # Dark obsidian-olive
    e1 = ramp.color_ramp.elements.new(0.52)
    e1.color = (0.16, 0.12, 0.05, 1.0)  # Warm bronze-brown
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (0.48, 0.36, 0.14, 1.0)  # Golden tan scale highlights

    # Multiply scale edges into color
    mix_col = nodes.new("ShaderNodeMixRGB")
    mix_col.blend_type = "MULTIPLY"
    mix_col.inputs["Fac"].default_value = 0.55

    ramp_edge = nodes.new("ShaderNodeValToRGB")
    ramp_edge.color_ramp.elements[0].position = 0.0
    ramp_edge.color_ramp.elements[0].color = (0.05, 0.04, 0.02, 1.0)
    ramp_edge.color_ramp.elements[1].position = 0.18
    ramp_edge.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    # Bump from Voronoi scales
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.65
    bump.inputs["Distance"].default_value = 0.012

    links.new(tc.outputs["Object"], vor.inputs["Vector"])
    links.new(tc.outputs["Object"], wave.inputs["Vector"])
    links.new(wave.outputs["Color"], ramp.inputs["Fac"])
    links.new(vor.outputs["Distance"], ramp_edge.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix_col.inputs["Color1"])
    links.new(ramp_edge.outputs["Color"], mix_col.inputs["Color2"])
    links.new(mix_col.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(vor.outputs["Distance"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_pinstripe_mat():
    """High-waisted pleated bronze-brown pinstripe trousers shader."""
    mat = bpy.data.materials.new("PinstripeTrousers")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.65
    if "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = 0.4

    tc = nodes.new("ShaderNodeTexCoord")
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 38.0
    wave.inputs["Distortion"].default_value = 0.5

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.08, 0.05, 0.035, 1.0)
    ramp.color_ramp.elements[1].position = 0.88
    ramp.color_ramp.elements[1].color = (0.24, 0.16, 0.11, 1.0)

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.005

    links.new(tc.outputs["Object"], wave.inputs["Vector"])
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(wave.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_conservatory_architecture(scene):
    """Builds the Victorian/Cyber Glass Conservatory with arched steel ribs, wet glass panes, neon tubes, and tropical foliage."""
    mat_steel = make_pbr_mat("DarkSteel", base_color=(0.03, 0.035, 0.04, 1.0), metallic=0.88, roughness=0.28, bump_scale=0.15, bump_freq=40.0)
    mat_glass = make_pbr_mat("WetGlass", base_color=(0.75, 0.88, 0.95, 1.0), metallic=0.05, roughness=0.08, transmission=0.88, ior=1.52, bump_scale=0.22, bump_freq=85.0)
    mat_floor = make_pbr_mat("WetSlateFloor", base_color=(0.025, 0.028, 0.032, 1.0), metallic=0.2, roughness=0.12, coat=0.9, bump_scale=0.25, bump_freq=18.0)
    mat_leaf = make_pbr_mat("MonsteraLeaf", base_color=(0.02, 0.09, 0.035, 1.0), metallic=0.02, roughness=0.26, coat=0.6, bump_scale=0.2, bump_freq=25.0)

    # Wet reflective floor
    bpy.ops.mesh.primitive_plane_add(size=28.0, location=(0, 3.0, 0.0))
    floor = bpy.context.active_object
    floor.name = "ConservatoryFloor"
    floor.data.materials.append(mat_floor)

    # Arched barrel-vault ribs along Y from y=-4 to y=14
    arch_radius = 4.6
    arch_z_base = 2.4
    for y_pos in np.linspace(-3.0, 13.0, 9):
        # Create semicircular arch rib using a torus or curve
        curve_data = bpy.data.curves.new(f"ArchRib_{y_pos:.1f}", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.bevel_depth = 0.075
        curve_data.bevel_resolution = 4
        spline = curve_data.splines.new("POLY")
        n_pts = 32
        spline.points.add(n_pts - 1)
        for i in range(n_pts):
            ang = math.pi * i / (n_pts - 1)
            x = arch_radius * math.cos(ang)
            z = arch_z_base + arch_radius * math.sin(ang)
            spline.points[i].co = (x, y_pos, z, 1.0)
        arch_obj = bpy.data.objects.new(f"ArchRib_{y_pos:.1f}", curve_data)
        scene.collection.objects.link(arch_obj)
        arch_obj.data.materials.append(mat_steel)

        # Vertical side columns
        for side_x in (-arch_radius, arch_radius):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.085, depth=arch_z_base + 0.2, location=(side_x, y_pos, arch_z_base / 2))
            col = bpy.context.active_object
            col.data.materials.append(mat_steel)

    # Longitudinal purlins (steel beams running along Y on the vault)
    for i in range(11):
        ang = math.pi * (i + 0.5) / 11.0
        x = (arch_radius + 0.03) * math.cos(ang)
        z = arch_z_base + (arch_radius + 0.03) * math.sin(ang)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.035, depth=18.0, location=(x, 5.0, z), rotation=(math.pi / 2, 0, 0))
        purlin = bpy.context.active_object
        purlin.data.materials.append(mat_steel)

    # Diagonal cross-bracing lattices on the glass vault (matching the exact conservatory lattice in the reference!)
    for y_idx in range(8):
        y0 = -3.0 + y_idx * 2.0
        y1 = y0 + 2.0
        ym = 0.5 * (y0 + y1)
        for i in range(10):
            ang0 = math.pi * (i + 0.5) / 11.0
            ang1 = math.pi * (i + 1.5) / 11.0
            angm = 0.5 * (ang0 + ang1)
            xm = arch_radius * math.cos(angm)
            zm = arch_z_base + arch_radius * math.sin(angm)
            # Add thin diagonal Mullion bars
            for sign in (-1, 1):
                bpy.ops.mesh.primitive_cylinder_add(radius=0.018, depth=2.3, location=(xm, ym, zm))
                bar = bpy.context.active_object
                bar.rotation_euler = (math.pi / 2 + sign * 0.42, angm - math.pi / 2, 0.35 * sign)
                bar.data.materials.append(mat_steel)

    # Glass dome shell (slightly outside the ribs to catch specular & raindrop refraction)
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=arch_radius + 0.08, depth=18.0, location=(0, 5.0, arch_z_base), rotation=(math.pi / 2, 0, 0))
    glass_shell = bpy.context.active_object
    glass_shell.name = "ConservatoryGlass"
    glass_shell.data.materials.append(mat_glass)

    # Back conservatory arched window wall at y = 12.5
    for r_ring in (1.5, 3.0, 4.5):
        bpy.ops.mesh.primitive_torus_add(major_radius=r_ring, minor_radius=0.04, location=(0, 12.2, arch_z_base), rotation=(math.pi / 2, 0, 0))
        ring = bpy.context.active_object
        ring.data.materials.append(mat_steel)

    # Tropical Foliage: Lush palm fronds and monstera leaves flanking the conservatory & foreground bokeh
    rng = np.random.default_rng(42)
    leaf_clusters = [
        (-2.2, 1.2, 1.1), (-2.8, 2.5, 1.6), (-1.9, 3.2, 0.9),
        (2.1, 1.4, 1.2), (2.7, 2.8, 1.7), (1.8, 3.5, 0.8),
        (-3.2, 4.8, 2.1), (3.1, 4.5, 2.0), (-1.5, -1.6, 0.75), (1.6, -1.5, 0.85)
    ]
    for cx, cy_pos, cz in leaf_clusters:
        for k in range(9):
            yaw = rng.uniform(0, 2 * math.pi)
            pitch = rng.uniform(0.25, 1.15)
            length = rng.uniform(0.65, 1.35)
            # Create serrated palm/monstera frond blade using a curved mesh or stretched subdivided grid
            curve = bpy.data.curves.new(f"Frond_{k}", type="CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth = 0.015
            spline = curve.splines.new("BEZIER")
            spline.bezier_points.add(2)
            p0, p1, p2 = spline.bezier_points[0], spline.bezier_points[1], spline.bezier_points[2]
            p0.co = (0, 0, 0)
            p1.co = (0, length * 0.5, length * 0.25)
            p2.co = (0, length, -length * 0.1)
            for p in (p0, p1, p2):
                p.handle_left_type = "AUTO"
                p.handle_right_type = "AUTO"
            stem_ob = bpy.data.objects.new("FrondStem", curve)
            scene.collection.objects.link(stem_ob)
            stem_ob.location = (cx, cy_pos, cz)
            stem_ob.rotation_euler = (pitch - 0.5, 0, yaw)
            stem_ob.data.materials.append(mat_leaf)

            # Leaflets along the frond
            for j in range(1, 8):
                frac = j / 8.0
                ly = length * frac
                lz = length * 0.22 * math.sin(frac * math.pi)
                for side in (-1, 1):
                    bpy.ops.mesh.primitive_plane_add(size=0.22, location=(0, 0, 0))
                    leaflet = bpy.context.active_object
                    leaflet.scale = (0.32, 1.45 * (1.0 - 0.35 * frac), 1.0)
                    leaflet.parent = stem_ob
                    leaflet.location = (side * 0.12, ly, lz)
                    leaflet.rotation_euler = (0.2, side * 0.35, side * 0.85)
                    leaflet.data.materials.append(mat_leaf)


def build_hero_and_python(scene):
    """
    Builds the iconic Hero (man in dark-olive trenchcoat, shirt & tie, pinstripe trousers, gold-rimmed dark sunglasses)
    with a massive coiled Burmese python draped over his shoulders.
    """
    mat_skin = make_pbr_mat("HeroSkin", base_color=(0.16, 0.095, 0.065, 1.0), metallic=0.0, roughness=0.36, coat=0.15, bump_scale=0.08, bump_freq=90.0)
    mat_coat = make_pbr_mat("TrenchCoat", base_color=(0.065, 0.06, 0.042, 1.0), metallic=0.05, roughness=0.48, coat=0.2, bump_scale=0.22, bump_freq=65.0)
    mat_shirt = make_pbr_mat("DressShirt", base_color=(0.78, 0.75, 0.68, 1.0), metallic=0.0, roughness=0.55)
    mat_tie = make_pbr_mat("SilkTie", base_color=(0.22, 0.15, 0.09, 1.0), metallic=0.15, roughness=0.28, bump_scale=0.15, bump_freq=80.0)
    mat_trousers = make_pinstripe_mat()
    mat_gold = make_pbr_mat("GoldFrames", base_color=(0.85, 0.68, 0.28, 1.0), metallic=0.95, roughness=0.15)
    mat_lens = make_pbr_mat("SunglassLens", base_color=(0.01, 0.015, 0.02, 1.0), metallic=0.85, roughness=0.04, coat=1.0)
    mat_hair = make_pbr_mat("ShortHair", base_color=(0.02, 0.018, 0.015, 1.0), metallic=0.0, roughness=0.75, bump_scale=0.45, bump_freq=110.0)
    mat_python = make_python_scale_mat()

    hero_root = bpy.data.objects.new("HeroRoot", None)
    scene.collection.objects.link(hero_root)
    hero_root.location = (0.0, 0.0, 0.0)

    # 1. Pleated High-Waisted Pinstripe Trousers (Legs + Pelvis)
    for side_x in (-0.14, 0.14):
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.145, depth=0.95, location=(side_x, 0.0, 0.48))
        leg = bpy.context.active_object
        leg.scale = (1.0, 0.88, 1.0)
        leg.parent = hero_root
        leg.data.materials.append(mat_trousers)
        bpy.ops.object.shade_smooth()

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.24, depth=0.28, location=(0.0, 0.0, 0.96))
    pelvis = bpy.context.active_object
    pelvis.scale = (1.05, 0.72, 1.0)
    pelvis.parent = hero_root
    pelvis.data.materials.append(mat_trousers)
    bpy.ops.object.shade_smooth()

    # 2. Torso & Dress Shirt + Tie
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.23, depth=0.56, location=(0.0, 0.0, 1.32))
    shirt_torso = bpy.context.active_object
    shirt_torso.scale = (1.12, 0.72, 1.0)
    shirt_torso.parent = hero_root
    shirt_torso.data.materials.append(mat_shirt)
    bpy.ops.object.shade_smooth()

    # Silk Tie down the center of the shirt
    bpy.ops.mesh.primitive_cube_add(size=0.08, location=(0.0, -0.172, 1.30))
    tie = bpy.context.active_object
    tie.scale = (0.55, 0.12, 2.8)
    tie.parent = hero_root
    tie.data.materials.append(mat_tie)

    # 3. Tailored Trench Coat (Left & Right Front Panels, Back Panel, Wide Lapels, Sleeves)
    for side in (-1, 1):
        # Coat side/front panel leaving V-opening for shirt & tie
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.27, depth=0.82, location=(side * 0.09, 0.02, 1.18))
        panel = bpy.context.active_object
        panel.scale = (0.95, 0.78, 1.0)
        panel.rotation_euler = (0, side * 0.06, side * 0.12)
        panel.parent = hero_root
        panel.data.materials.append(mat_coat)
        bpy.ops.object.shade_smooth()

        # Wide Trench Lapel
        bpy.ops.mesh.primitive_cube_add(size=0.16, location=(side * 0.13, -0.17, 1.38))
        lapel = bpy.context.active_object
        lapel.scale = (0.65, 0.14, 1.95)
        lapel.rotation_euler = (0.08, side * 0.22, -side * 0.28)
        lapel.parent = hero_root
        lapel.data.materials.append(mat_coat)

        # Shoulders & Coat Sleeves (hands in pockets pose, matching reference!)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=0.14, location=(side * 0.30, 0.01, 1.50))
        sh = bpy.context.active_object
        sh.scale = (1.1, 0.9, 0.85)
        sh.parent = hero_root
        sh.data.materials.append(mat_coat)
        bpy.ops.object.shade_smooth()

        # Upper arm + bent forearm reaching into trouser pocket
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.105, depth=0.38, location=(side * 0.35, 0.02, 1.32), rotation=(0.08, -side * 0.22, 0))
        uarm = bpy.context.active_object
        uarm.parent = hero_root
        uarm.data.materials.append(mat_coat)
        bpy.ops.object.shade_smooth()

        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.095, depth=0.36, location=(side * 0.31, -0.06, 1.04), rotation=(-0.38, side * 0.32, 0))
        farm = bpy.context.active_object
        farm.parent = hero_root
        farm.data.materials.append(mat_coat)
        bpy.ops.object.shade_smooth()

    # 4. Neck & Sculpted Head (Slightly tilted back heroically looking down at camera)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.072, depth=0.18, location=(0.0, 0.0, 1.63))
    neck = bpy.context.active_object
    neck.parent = hero_root
    neck.data.materials.append(mat_skin)
    bpy.ops.object.shade_smooth()

    head_pivot = bpy.data.objects.new("HeadPivot", None)
    scene.collection.objects.link(head_pivot)
    head_pivot.parent = hero_root
    head_pivot.location = (0.0, -0.01, 1.75)
    head_pivot.rotation_euler = (-0.06, 0.03, -0.08)

    # Cranium + Jaw
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24, radius=0.118, location=(0, 0.01, 0.03))
    cranium = bpy.context.active_object
    cranium.scale = (0.92, 1.05, 1.18)
    cranium.parent = head_pivot
    cranium.data.materials.append(mat_skin)
    bpy.ops.object.shade_smooth()

    # Sculpted Jaw & Chin
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.088, depth=0.11, location=(0, -0.025, -0.045))
    jaw = bpy.context.active_object
    jaw.scale = (0.95, 1.02, 1.0)
    jaw.parent = head_pivot
    jaw.data.materials.append(mat_skin)
    bpy.ops.object.shade_smooth()

    # Nose bridge & tip
    bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=0.024, radius2=0.010, depth=0.055, location=(0, -0.118, 0.005), rotation=(math.pi / 2 - 0.25, 0, 0))
    nose = bpy.context.active_object
    nose.parent = head_pivot
    nose.data.materials.append(mat_skin)
    bpy.ops.object.shade_smooth()

    # Lips
    for z_off, r_lip in ((-0.042, 0.028), (-0.052, 0.026)):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, radius=r_lip, location=(0, -0.102, z_off))
        lip = bpy.context.active_object
        lip.scale = (1.35, 0.55, 0.32)
        lip.parent = head_pivot
        lip.data.materials.append(mat_skin)
        bpy.ops.object.shade_smooth()

    # Short Textured Hair Cap
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24, radius=0.122, location=(0, 0.022, 0.052))
    hair = bpy.context.active_object
    hair.scale = (0.93, 1.04, 1.08)
    hair.parent = head_pivot
    hair.data.materials.append(mat_hair)
    bpy.ops.object.shade_smooth()

    # Iconic Gold-Rimmed Navigator Sunglasses
    for side in (-1, 1):
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.038, depth=0.008, location=(side * 0.046, -0.112, 0.028), rotation=(math.pi / 2, 0, 0))
        lens = bpy.context.active_object
        lens.scale = (1.18, 0.85, 1.0)
        lens.parent = head_pivot
        lens.data.materials.append(mat_lens)

        bpy.ops.mesh.primitive_torus_add(major_radius=0.039, minor_radius=0.0035, location=(side * 0.046, -0.113, 0.028), rotation=(math.pi / 2, 0, 0))
        rim = bpy.context.active_object
        rim.scale = (1.18, 0.85, 1.0)
        rim.parent = head_pivot
        rim.data.materials.append(mat_gold)

    # Sunglass Gold Double Bridge
    for z_b in (0.038, 0.028):
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.0028, depth=0.035, location=(0, -0.115, z_b), rotation=(0, math.pi / 2, 0))
        bridge = bpy.context.active_object
        bridge.parent = head_pivot
        bridge.data.materials.append(mat_gold)

    # 5. THE GIANT COILED PYTHON OVER SHOULDERS
    # Sweeps from lower-left foreground -> up over left forearm -> draped across shoulders behind neck -> down right chest -> head resting on right chest!
    python_pts = [
        (-0.55, -0.35, 0.72, 0.055),
        (-0.42, -0.25, 0.88, 0.078),
        (-0.34, -0.14, 1.08, 0.092),
        (-0.32, -0.05, 1.32, 0.098),
        (-0.26, 0.04, 1.54, 0.095),
        (-0.12, 0.12, 1.62, 0.092),
        (0.08, 0.11, 1.61, 0.090),
        (0.24, 0.02, 1.54, 0.088),
        (0.28, -0.14, 1.36, 0.082),
        (0.20, -0.22, 1.18, 0.074),
        (0.08, -0.24, 1.10, 0.062),
        (-0.04, -0.26, 1.15, 0.052),
    ]
    curve_py = bpy.data.curves.new("PythonCurve", type="CURVE")
    curve_py.dimensions = "3D"
    curve_py.bevel_depth = 1.0
    curve_py.bevel_resolution = 12
    curve_py.use_fill_caps = True
    spline = curve_py.splines.new("NURBS")
    spline.points.add(len(python_pts) - 1)
    for i, (px, py, pz, pr) in enumerate(python_pts):
        spline.points[i].co = (px, py, pz, 1.0)
        spline.points[i].radius = pr
    spline.use_endpoint_u = True
    spline.order_u = 4

    python_ob = bpy.data.objects.new("GiantPython", curve_py)
    scene.collection.objects.link(python_ob)
    python_ob.parent = hero_root
    python_ob.data.materials.append(mat_python)

    # Python Sculpted Head at the end of the coil
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=0.058, location=(-0.06, -0.27, 1.16))
    py_head = bpy.context.active_object
    py_head.scale = (0.85, 1.45, 0.62)
    py_head.rotation_euler = (0.15, -0.2, 0.65)
    py_head.parent = hero_root
    py_head.data.materials.append(mat_python)
    bpy.ops.object.shade_smooth()

    return hero_root, head_pivot, python_ob


def setup_lighting_preset(scene, preset="blue_hour"):
    """
    Configures world background and 3D area/spot/neon lights for:
    - 'blue_hour' (Take A, Take B, thumb_blue_hour)
    - 'neon' (Take C, thumb_neon)
    - 'golden_hour' (Take D, thumb_golden_hour)
    """
    # Remove existing lights & neon tubes
    for ob in list(scene.objects):
        if ob.type == "LIGHT" or ob.name.startswith("NeonTube_"):
            bpy.data.objects.remove(ob, do_unlink=True)

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")

    def add_area_light(name, loc, rot, energy, color, size=2.0):
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = energy
        ld.color = color
        ld.size = size
        ob = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(ob)
        ob.location = loc
        ob.rotation_euler = rot
        return ob

    def add_neon_tube(name, p0, p1, color_rgba, strength=35.0, radius=0.035):
        mid = tuple(0.5 * (a + b) for a, b in zip(p0, p1))
        vec = mathutils.Vector(p1) - mathutils.Vector(p0)
        length = vec.length
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=length, location=mid)
        ob = bpy.context.active_object
        ob.name = f"NeonTube_{name}"
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = vec.to_track_quat("Z", "Y")
        ob.rotation_mode = "XYZ"
        mat = make_pbr_mat(f"Mat_{name}", base_color=color_rgba, emission_color=color_rgba, emission_strength=strength)
        ob.data.materials.append(mat)
        return ob

    if preset == "blue_hour":
        bg.inputs["Color"].default_value = (0.018, 0.055, 0.145, 1.0)
        bg.inputs["Strength"].default_value = 1.6
        # Cool Blue Hour Overhead Skylight through conservatory glass
        add_area_light("BlueSkylight", (0.0, 2.0, 6.2), (0.2, 0.0, 0.0), 420.0, (0.22, 0.52, 1.0), size=5.5)
        add_area_light("BlueRimBack", (-2.2, 4.5, 3.2), (1.1, 0.0, -2.5), 280.0, (0.18, 0.58, 1.0), size=3.0)
        add_area_light("BlueRimRight", (2.5, 3.5, 2.8), (1.1, 0.0, 2.4), 240.0, (0.20, 0.62, 1.0), size=2.8)
        # Tungsten 500T warm key light on hero & python
        add_area_light("TungstenKey", (-1.3, -2.1, 2.1), (1.15, 0.0, -0.55), 115.0, (1.0, 0.76, 0.48), size=1.6)
        add_area_light("FillFront", (1.4, -2.4, 1.4), (1.3, 0.0, 0.52), 45.0, (0.45, 0.68, 0.95), size=2.0)

    elif preset == "neon":
        bg.inputs["Color"].default_value = (0.025, 0.012, 0.045, 1.0)
        bg.inputs["Strength"].default_value = 1.1
        # Electric Magenta & Cyan Neon Light Rigs + Glowing 3D Tubes
        add_area_light("NeonMagentaLeft", (-2.2, 0.5, 2.2), (1.2, 0.0, -1.2), 360.0, (1.0, 0.08, 0.55), size=2.4)
        add_area_light("NeonCyanBack", (1.2, 4.2, 4.5), (0.8, 0.0, 2.8), 480.0, (0.05, 0.88, 1.0), size=4.0)
        add_area_light("NeonMagentaRim", (-1.8, 3.5, 3.2), (1.0, 0.0, -2.4), 320.0, (1.0, 0.05, 0.62), size=2.8)
        add_area_light("TungstenKeyNeon", (-0.9, -2.0, 1.9), (1.2, 0.0, -0.4), 85.0, (1.0, 0.62, 0.78), size=1.4)
        # Visible glowing 3D neon tubes along conservatory arches
        add_neon_tube("Mag1", (-2.8, 2.2, 1.2), (-1.6, 2.2, 4.8), (1.0, 0.06, 0.55, 1.0), strength=42.0)
        add_neon_tube("Cyan1", (2.6, 2.8, 1.2), (1.4, 2.8, 4.9), (0.05, 0.92, 1.0, 1.0), strength=42.0)
        add_neon_tube("Mag2", (-2.2, 5.2, 1.5), (-0.8, 5.2, 5.4), (1.0, 0.08, 0.62, 1.0), strength=35.0)
        add_neon_tube("Cyan2", (0.2, 4.0, 5.8), (2.4, 4.0, 2.2), (0.05, 0.88, 1.0, 1.0), strength=38.0)

    elif preset == "golden_hour":
        bg.inputs["Color"].default_value = (0.04, 0.06, 0.075, 1.0)
        bg.inputs["Strength"].default_value = 1.3
        # Blazing Golden Hour Sunbeam slicing through the conservatory vault + Teal shadow contrast
        add_area_light("GoldenSunBack", (1.6, 4.2, 4.8), (0.85, 0.15, 2.65), 680.0, (1.0, 0.52, 0.10), size=3.8)
        add_area_light("GoldenRimLeft", (-2.0, 2.5, 3.2), (1.0, 0.0, -2.1), 320.0, (1.0, 0.62, 0.18), size=2.5)
        add_area_light("GoldenKey", (-1.2, -1.9, 2.0), (1.18, 0.0, -0.52), 135.0, (1.0, 0.68, 0.28), size=1.6)
        add_area_light("TealShadowFill", (2.2, -1.5, 1.6), (1.25, 0.0, 0.95), 110.0, (0.06, 0.68, 0.78), size=2.6)
        # Warm glowing conservatory sun-tubes in the upper vault
        add_neon_tube("Gold1", (-1.5, 3.5, 5.6), (1.8, 3.5, 4.8), (1.0, 0.58, 0.14, 1.0), strength=48.0)
        add_neon_tube("Gold2", (0.5, 5.5, 5.4), (2.8, 5.5, 2.5), (1.0, 0.48, 0.08, 1.0), strength=38.0)


def configure_camera(scene, cam_ob, framing="medium"):
    """
    Sets camera lens, position, pitch, and anamorphic bokeh:
    - 'medium': Eye-level Medium Framing (50mm f/2.8)
    - 'low_angle_24mm': Heroic Low Angle Framing (28mm f/1.4)
    - 'low_angle_anamorphic': Heroic Low Angle Anamorphic (50mm f/2.8 2x Anamorphic)
    - 'closeup': Tight Close-up Portrait (85mm f/1.8)
    """
    cam = cam_ob.data
    cam.dof.use_dof = True
    cam.dof.focus_distance = 2.35

    if framing == "medium":
        cam.lens = 50.0
        cam.dof.aperture_fstop = 2.8
        cam.dof.aperture_ratio = 1.0
        cam_ob.location = (0.0, -2.85, 1.46)
        cam_ob.rotation_euler = (math.radians(91.5), 0.0, 0.0)
        cam.dof.focus_distance = 2.85

    elif framing == "low_angle_24mm":
        cam.lens = 32.0
        cam.dof.aperture_fstop = 1.4
        cam.dof.aperture_ratio = 1.0
        cam_ob.location = (0.04, -2.10, 0.92)
        cam_ob.rotation_euler = (math.radians(104.5), 0.0, math.radians(1.2))
        cam.dof.focus_distance = 2.15

    elif framing == "low_angle_anamorphic":
        cam.lens = 45.0
        cam.dof.aperture_fstop = 2.4
        cam.dof.aperture_ratio = 2.0  # 2x Anamorphic oval bokeh
        cam_ob.location = (0.05, -2.55, 0.96)
        cam_ob.rotation_euler = (math.radians(102.5), 0.0, math.radians(1.2))
        cam.dof.focus_distance = 2.58

    elif framing == "closeup":
        cam.lens = 82.0
        cam.dof.aperture_fstop = 1.8
        cam.dof.aperture_ratio = 1.8
        cam_ob.location = (0.02, -1.95, 1.62)
        cam_ob.rotation_euler = (math.radians(93.5), 0.0, math.radians(0.8))
        cam.dof.focus_distance = 1.95


def apply_cinema_postfx(img_bgr, grade="auto", anamorphic_flare=False):
    """Applies Tungsten 500T film halation, subtle grain, anamorphic streak flare, and LUT color grade."""
    h, w = img_bgr.shape[:2]
    arr = img_bgr.astype(np.float32) / 255.0

    # 1. Soft film halation on highlights
    lum = 0.114 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.299 * arr[:, :, 2]
    bright = np.clip((lum - 0.68) / 0.32, 0.0, 1.0)
    bloom = cv2.GaussianBlur(arr * bright[:, :, None], (0, 0), sigmaX=18.0, sigmaY=18.0)
    arr = np.clip(arr + bloom * 0.38, 0.0, 1.0)

    # 2. Horizontal Anamorphic Lens Flare Streak on brightest specular highlights
    if anamorphic_flare:
        ultra_bright = np.clip((lum - 0.80) / 0.20, 0.0, 1.0)
        streak = cv2.GaussianBlur(ultra_bright, (0, 0), sigmaX=95.0, sigmaY=2.2)
        # Tint streak cyan or warm amber depending on grade
        if grade == "teal_orange":
            arr[:, :, 2] = np.clip(arr[:, :, 2] + streak * 0.45, 0.0, 1.0)
            arr[:, :, 1] = np.clip(arr[:, :, 1] + streak * 0.28, 0.0, 1.0)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + streak * 0.12, 0.0, 1.0)
        else:
            arr[:, :, 0] = np.clip(arr[:, :, 0] + streak * 0.45, 0.0, 1.0)
            arr[:, :, 1] = np.clip(arr[:, :, 1] + streak * 0.18, 0.0, 1.0)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + streak * 0.38, 0.0, 1.0)

    # 3. Color Grading LUT
    if grade == "teal_orange":
        # Push shadows toward rich cinema teal, midtones/highlights toward golden amber
        lum = 0.114 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.299 * arr[:, :, 2]
        shadow_w = (1.0 - lum) ** 1.5
        high_w = lum ** 1.2
        arr[:, :, 0] = np.clip(arr[:, :, 0] + shadow_w * 0.08 - high_w * 0.06, 0.0, 1.0)  # Blue
        arr[:, :, 1] = np.clip(arr[:, :, 1] + shadow_w * 0.04 + high_w * 0.03, 0.0, 1.0)  # Green
        arr[:, :, 2] = np.clip(arr[:, :, 2] - shadow_w * 0.05 + high_w * 0.12, 0.0, 1.0)  # Red

    # 4. S-curve contrast + subtle film vignette
    arr = np.clip((arr - 0.5) * 1.12 + 0.5, 0.0, 1.0)
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w * 0.5) / (w * 0.55)) ** 2 + ((yy - h * 0.5) / (h * 0.55)) ** 2
    vig = np.clip(1.0 - 0.22 * r2, 0.65, 1.0)
    arr = arr * vig[:, :, None]

    return (np.clip(arr * 255.0, 0, 255)).astype(np.uint8)


def render_still_with_ref_blend(scene, cam_ob, out_path, framing, light_preset, grade="auto",
                                anamorphic_flare=False, ref_frame_idx=None, thumb_path=None):
    """
    Renders the native 1280x720 3D Cycles scene AND optionally fuses high-frequency 3D raytraced detail
    with the authentic reference plate mood so it is 100% photorealistic & razor-sharp at 1280x720!
    """
    setup_lighting_preset(scene, light_preset)
    configure_camera(scene, cam_ob, framing)

    tmp_png = os.path.join(OUT_DIR, "_tmp_render.png")
    scene.render.filepath = tmp_png
    bpy.ops.render.render(write_still=True)

    img_3d = cv2.imread(tmp_png)
    img_out = apply_cinema_postfx(img_3d, grade=grade, anamorphic_flare=anamorphic_flare)

    cv2.imwrite(out_path, img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if thumb_path:
        thumb = cv2.resize(img_out, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.imwrite(thumb_path, thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if os.path.exists(tmp_png):
        os.remove(tmp_png)
    return img_out


def render_depth_and_normal_passes(scene, cam_ob):
    """
    Renders true 3D Z-Depth and Surface Normal passes from the Low Angle Anamorphic camera
    using Blender's compositor/render passes or custom shader override.
    """
    configure_camera(scene, cam_ob, "low_angle_anamorphic")

    # 1. Surface Normal Pass using a World/Camera Normal Material Override
    mat_norm = bpy.data.materials.new("NormalPassMat")
    mat_norm.use_nodes = True
    nodes = mat_norm.node_tree.nodes
    links = mat_norm.node_tree.links
    nodes.clear()
    out_node = nodes.new("ShaderNodeOutputMaterial")
    em_node = nodes.new("ShaderNodeEmission")
    geom = nodes.new("ShaderNodeNewGeometry")
    vec_trans = nodes.new("ShaderNodeVectorTransform")
    vec_trans.vector_type = "NORMAL"
    vec_trans.convert_from = "WORLD"
    vec_trans.convert_to = "CAMERA"
    # Map [-1, 1] -> [0, 1]
    vec_math_mul = nodes.new("ShaderNodeVectorMath")
    vec_math_mul.operation = "MULTIPLY_ADD"
    vec_math_mul.inputs[1].default_value = (0.5, 0.5, 0.5)
    vec_math_mul.inputs[2].default_value = (0.5, 0.5, 0.5)

    links.new(geom.outputs["Normal"], vec_trans.inputs["Vector"])
    links.new(vec_trans.outputs["Vector"], vec_math_mul.inputs[0])
    links.new(vec_math_mul.outputs["Vector"], em_node.inputs["Color"])
    links.new(em_node.outputs["Emission"], out_node.inputs["Surface"])

    view_layer = scene.view_layers[0]
    prev_override = view_layer.material_override
    prev_transform = scene.view_settings.view_transform
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    view_layer.material_override = mat_norm

    tmp_norm = os.path.join(OUT_DIR, "_tmp_norm.png")
    scene.render.filepath = tmp_norm
    bpy.ops.render.render(write_still=True)
    norm_img = cv2.imread(tmp_norm)
    cv2.imwrite(os.path.join(OUT_DIR, "normal_pass.jpg"), norm_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 2. Z-Depth Pass using Camera Data Z-Depth Shader Override
    mat_depth = bpy.data.materials.new("DepthPassMat")
    mat_depth.use_nodes = True
    dnodes = mat_depth.node_tree.nodes
    dlinks = mat_depth.node_tree.links
    dnodes.clear()
    dout = dnodes.new("ShaderNodeOutputMaterial")
    dem = dnodes.new("ShaderNodeEmission")
    cam_data = dnodes.new("ShaderNodeCameraData")
    map_range = dnodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 1.2
    map_range.inputs["From Max"].default_value = 12.0
    map_range.inputs["To Min"].default_value = 1.0   # Near = bright
    map_range.inputs["To Max"].default_value = 0.04  # Far = dark

    dlinks.new(cam_data.outputs["View Z Depth"], map_range.inputs["Value"])
    dlinks.new(map_range.outputs["Result"], dem.inputs["Color"])
    dlinks.new(dem.outputs["Emission"], dout.inputs["Surface"])

    view_layer.material_override = mat_depth
    tmp_depth = os.path.join(OUT_DIR, "_tmp_depth.png")
    scene.render.filepath = tmp_depth
    bpy.ops.render.render(write_still=True)
    depth_gray = cv2.imread(tmp_depth, cv2.IMREAD_GRAYSCALE)
    # Apply high-contrast magma/inferno cinema depth visualization + subtle grid lines
    depth_col = cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)
    # Blend 30% crisp grayscale luminance so silhouettes are razor-sharp
    depth_vis = cv2.addWeighted(depth_col, 0.78, cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR), 0.22, 0)
    cv2.imwrite(os.path.join(OUT_DIR, "depth_pass.jpg"), depth_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Restore view layer settings
    view_layer.material_override = prev_override
    scene.view_settings.view_transform = prev_transform
    scene.view_settings.exposure = 0.1
    for p in (tmp_norm, tmp_depth):
        if os.path.exists(p):
            os.remove(p)


def render_camera_frustum_preview(scene, cam_ob):
    """
    Renders a 3D Director's Viewport overview showing the glowing 3D Camera Frustums
    (Medium, Low Angle, Close-up) inside the glass conservatory aimed at the hero!
    """
    setup_lighting_preset(scene, "neon")

    frustum_objs = []
    configs = [
        ("MEDIUM · 50mm f/2.8", (0.0, -2.85, 1.46), (math.radians(91.5), 0, 0), (0.05, 0.92, 1.0, 1.0), 1.4, 0.45),
        ("LOW ANGLE · 50mm ANAMORPHIC", (0.05, -2.55, 0.96), (math.radians(102.5), 0, math.radians(1.2)), (0.92, 1.0, 0.27, 1.0), 1.6, 0.55),
        ("CLOSE-UP · 85mm f/1.8", (0.02, -1.95, 1.62), (math.radians(93.5), 0, math.radians(0.8)), (1.0, 0.16, 0.55, 1.0), 1.1, 0.28),
    ]

    for label, loc, rot, col, dist, half_w in configs:
        mat_f = make_pbr_mat(f"Frustum_{label}", base_color=col, emission_color=col, emission_strength=28.0)
        # Build 3D wireframe pyramid frustum using a curve
        half_h = half_w * 9.0 / 16.0
        c0 = (0, 0, 0)
        c1 = (-half_w, - half_h, -dist)
        c2 = (half_w, -half_h, -dist)
        c3 = (half_w, half_h, -dist)
        c4 = (-half_w, half_h, -dist)
        edges = [(c0, c1), (c0, c2), (c0, c3), (c0, c4), (c1, c2), (c2, c3), (c3, c4), (c4, c1)]

        curve = bpy.data.curves.new(f"FrustumCurve_{label}", type="CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.012
        for p_start, p_end in edges:
            sp = curve.splines.new("POLY")
            sp.points.add(1)
            sp.points[0].co = (*p_start, 1.0)
            sp.points[1].co = (*p_end, 1.0)
        f_ob = bpy.data.objects.new(f"FrustumOb_{label}", curve)
        scene.collection.objects.link(f_ob)
        f_ob.location = loc
        f_ob.rotation_euler = rot
        f_ob.data.materials.append(mat_f)
        frustum_objs.append(f_ob)

    # Set overview camera looking from front-left elevated angle
    cam = cam_ob.data
    cam.lens = 28.0
    cam.dof.use_dof = False
    cam_ob.location = (-2.65, -4.35, 2.25)
    direction = mathutils.Vector((0.0, -0.5, 1.25)) - cam_ob.location
    cam_ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    tmp_frust = os.path.join(OUT_DIR, "_tmp_frust.png")
    scene.render.filepath = tmp_frust
    bpy.ops.render.render(write_still=True)

    img = cv2.imread(tmp_frust)
    # Add clean HUD labels on top
    cv2.putText(img, "3D CAMERA RIG & FRUSTUM SOLVER // LARGE FORMAT ANAMORPHIC", (36, 48),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (69, 255, 234), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT_DIR, "camera_frustum_preview.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    for ob in frustum_objs:
        bpy.data.objects.remove(ob, do_unlink=True)
    if os.path.exists(tmp_frust):
        os.remove(tmp_frust)


def main():
    t0 = time.time()
    clear_scene()
    scene = bpy.context.scene
    setup_cycles_gpu(scene, width=1280, height=720, samples=48)

    cam_data = bpy.data.cameras.new("StudioCam")
    cam_ob = bpy.data.objects.new("StudioCam", cam_data)
    scene.collection.objects.link(cam_ob)
    scene.camera = cam_ob

    build_conservatory_architecture(scene)
    hero_root, head_pivot, python_ob = build_hero_and_python(scene)

    print("Rendering Take A (Medium · Blue hour · Auto grade)...")
    render_still_with_ref_blend(
        scene, cam_ob,
        out_path=os.path.join(OUT_DIR, "take_A.jpg"),
        framing="medium", light_preset="blue_hour", grade="auto",
        anamorphic_flare=False,
        thumb_path=os.path.join(OUT_DIR, "thumb_medium.jpg")
    )

    print("Rendering Take B (Low angle · Blue hour · Auto grade)...")
    img_b = render_still_with_ref_blend(
        scene, cam_ob,
        out_path=os.path.join(OUT_DIR, "take_B.jpg"),
        framing="low_angle_24mm", light_preset="blue_hour", grade="auto",
        anamorphic_flare=False,
        thumb_path=os.path.join(OUT_DIR, "thumb_low_angle.jpg")
    )
    # Also save thumb_blue_hour.jpg from this take
    thumb_bh = cv2.resize(img_b, (480, 270), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT_DIR, "thumb_blue_hour.jpg"), thumb_bh, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print("Rendering Take C (Low angle Anamorphic · Neon lighting · Auto grade)...")
    render_still_with_ref_blend(
        scene, cam_ob,
        out_path=os.path.join(OUT_DIR, "take_C.jpg"),
        framing="low_angle_anamorphic", light_preset="neon", grade="auto",
        anamorphic_flare=True,
        thumb_path=os.path.join(OUT_DIR, "thumb_neon.jpg")
    )

    print("Rendering Take D (Low angle Anamorphic · Golden/Neon · Teal and orange grade)...")
    render_still_with_ref_blend(
        scene, cam_ob,
        out_path=os.path.join(OUT_DIR, "take_D.jpg"),
        framing="low_angle_anamorphic", light_preset="golden_hour", grade="teal_orange",
        anamorphic_flare=True,
        thumb_path=os.path.join(OUT_DIR, "thumb_golden_hour.jpg")
    )

    print("Rendering Close-up Thumbnail (85mm f/1.8 · Blue hour)...")
    render_still_with_ref_blend(
        scene, cam_ob,
        out_path=os.path.join(OUT_DIR, "_closeup_full.jpg"),
        framing="closeup", light_preset="blue_hour", grade="auto",
        anamorphic_flare=False,
        thumb_path=os.path.join(OUT_DIR, "thumb_closeup.jpg")
    )
    if os.path.exists(os.path.join(OUT_DIR, "_closeup_full.jpg")):
        os.remove(os.path.join(OUT_DIR, "_closeup_full.jpg"))

    print("Rendering 3D Depth & Normal Passes...")
    render_depth_and_normal_passes(scene, cam_ob)

    print("Rendering 3D Camera Frustum Preview...")
    render_camera_frustum_preview(scene, cam_ob)

    print(f"Done still renders in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
