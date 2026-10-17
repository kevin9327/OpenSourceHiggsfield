"""
OpenSourceHiggsfield — Official <2-Minute Competition Demo Video Generator (`build_demo_video.py`)
==================================================================================================
Generates the 84-second (1920x1080 @ 24fps, H.264 + AAC 48kHz stereo audio) submission demo video at:
  `public/outputs/OpenHiggsfield_Submission_Demo_1080p.mp4`

Features:
  - Neural English voiceover narration via `edge-tts` (`en-US-ChristopherNeural` / `en-US-GuyNeural`)
    mixed with a synthesized 110 BPM lo-fi studio hip-hop / electronic background beat.
  - 6 chapters covering the Hook, Why It's Innovative (True 3D Math + Custom PyTorch Nodes + LTX-2.5 DFR),
    Pillar 1 (Cinematic Studio), Pillar 2 (Re-shoot a Video + 3D Sphere Gizmo), Pillar 3 (OpenJutsu + Trim Modal),
    and Pillar 4 (Dual-Surface `comfy-sdk` v0.5.0 `OPEN_WEIGHTS_LOCAL` vs `CLOUD_HYBRID` + Closing Card).
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import wave
from pathlib import Path

import cv2
import edge_tts
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT_DIR / "public"
SCREENSHOTS_DIR = PUBLIC_DIR / "screenshots"
SHOWCASE_DIR = PUBLIC_DIR / "showcase"
OUTPUTS_DIR = PUBLIC_DIR / "outputs"
TEMP_AUDIO_DIR = OUTPUTS_DIR / "_demo_audio_tmp"

WIDTH = 1920
HEIGHT = 1080
FPS = 24

CHAPTERS = [
    {
        "id": "ch1_hook",
        "start": 0.0,
        "end": 9.0,
        "title": "01 / THE CHALLENGE HOOK",
        "subtitle": "ComfyUI #OpenSourceHiggsfield Competition",
        "narration": (
            "Higgsfield promised to open-source their platform. They only dropped a training skeleton. "
            "So we built OpenHiggsfield Studio on the Comfy Developer Platform."
        ),
    },
    {
        "id": "ch2_innovative",
        "start": 9.0,
        "end": 24.0,
        "title": "02 / WHY IT'S INNOVATIVE: TRUE 3D CAMERA & RELIGHT MATH",
        "subtitle": "5 Custom PyTorch ComfyUI Nodes + LTX-2.5 22B DFR + FLUX.3 + comfy-sdk v0.5.0",
        "narration": (
            "Why is OpenHiggsfield innovative? Instead of faking camera control with prompt strings, "
            "our five custom PyTorch ComfyUI nodes compute real four-by-four S E 3 camera trajectories, "
            "3D depth point-cloud projections with magenta disocclusion masks, and surface-normal relighting, "
            "orchestrated via comfy-sdk zero-point-five and comfy-api-proxy."
        ),
    },
    {
        "id": "ch3_cinematic_studio",
        "start": 24.0,
        "end": 40.0,
        "title": "03 / PILLAR 1: CINEMATIC STUDIO",
        "subtitle": "6-Row DoP Stack · Large Format Anamorphic 50mm f/2.8 · 3D Normal Relighting · Film LUTs",
        "narration": (
            "Pillar one: Cinematic Studio. Direct every shot with a six-row Director of Photography stack, "
            "picking Large Format Anamorphic 50 millimeter glass, low-angle framing, neon or blue-hour 3D relighting, "
            "Kodak Tungsten 500T film halation, and Teal-and-Orange LUTs across four live shot variations."
        ),
    },
    {
        "id": "ch4_reshoot",
        "start": 40.0,
        "end": 57.0,
        "title": "04 / PILLAR 2: RE-SHOOT A VIDEO & VIRAL CAMERA RIGS",
        "subtitle": "Interactive 3D Wireframe Sphere Gizmo · Keyframe Diamonds ◆ · Source vs Result Wipe",
        "narration": (
            "Pillar two: Re-shoot a Video. Drag the camera along an interactive 3D wireframe sphere "
            "from plus thirty degrees yaw to minus fifteen degrees, or trigger one-click viral rigs like "
            "Bullet-Time 360, SnorriCam P O V, and Crash Zoom. Our depth-reshoot node warps the video "
            "in 3D perspective with a live neon-yellow Source-to-Result comparison wipe."
        ),
    },
    {
        "id": "ch5_openjutsu",
        "start": 57.0,
        "end": 72.0,
        "title": "05 / PILLAR 3: OPENJUTSU CHARACTER & VFX SWAP",
        "subtitle": "Interactive Filmstrip Trim Modal · Single-Image Character Sheet · Audio-Synced LTX Swap",
        "narration": (
            "Pillar three: OpenJutsu, Comfy's teased character swap app, built and open-sourced today. "
            "Trim any five to fifteen second clip in the interactive filmstrip modal, drop in a single character sheet, "
            "and swap the performer with procedural chakra and portal VFX while locking original audio sync."
        ),
    },
    {
        "id": "ch6_dual_sdk_closing",
        "start": 72.0,
        "end": 84.0,
        "title": "06 / DUAL-SURFACE SDK ENGINE & 100% OPEN-WEIGHTS COMPLIANCE",
        "subtitle": "OPEN_WEIGHTS_LOCAL (comfy-api-proxy :8189) ↔ CLOUD_HYBRID (Comfy Cloud & Router)",
        "narration": (
            "With one click, switch between Open-Weights Local mode via comfy-api-proxy for one hundred percent "
            "Grand Prize and RTX 5090 open-source compliance, or Cloud Hybrid mode. "
            "OpenHiggsfield Studio: cinema direction, finally open source."
        ),
    },
]

TOTAL_DURATION = int(CHAPTERS[-1]["end"])
TOTAL_FRAMES = TOTAL_DURATION * FPS


def _get_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if mono:
        candidates = ["C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"]
    elif bold:
        candidates = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/calibrib.ttf"]
    else:
        candidates = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/calibri.ttf"]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


async def generate_narration_track() -> Path:
    """Synthesize per-chapter voiceovers via `edge-tts` and mix with a 110 BPM background studio beat."""
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    master_wav = TEMP_AUDIO_DIR / "master_demo_audio.wav"
    if master_wav.exists() and master_wav.stat().st_size > 100000:
        return master_wav
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    sr = 44100
    total_samples = TOTAL_DURATION * sr

    # 1. Synthesize subtle 110 BPM studio electronic/hip-hop instrumental bed
    t_arr = np.arange(total_samples, dtype=np.float32) / float(sr)
    bpm = 110.0
    beat_hz = bpm / 60.0
    beat_phase = (t_arr * beat_hz) % 1.0
    bar_phase = (t_arr * (beat_hz / 4.0)) % 1.0

    # Warm sub-bass progression (D minor -> Bb major -> G minor -> A7)
    chord_freqs = np.select(
        [bar_phase < 0.25, bar_phase < 0.50, bar_phase < 0.75],
        [73.42, 58.27, 98.00],
        default=110.00,
    )
    bass = np.sin(2.0 * np.pi * chord_freqs * t_arr) * np.exp(-beat_phase * 3.5) * 0.055
    kick = np.sin(2.0 * np.pi * (120.0 * np.exp(-beat_phase * 18.0) + 45.0) * t_arr) * np.exp(-beat_phase * 9.5) * 0.065
    pad = (
        np.sin(2.0 * np.pi * chord_freqs * 2.0 * t_arr) * 0.018
        + np.sin(2.0 * np.pi * chord_freqs * 3.0 * t_arr) * 0.012
    )
    bed_mono = np.clip(bass + kick + pad, -0.25, 0.25)
    mix_stereo = np.stack([bed_mono, bed_mono], axis=-1)

    # 2. Generate TTS WAV per chapter (try edge-tts first, fallback to offline Windows SAPI5 SpVoice)
    for ch in CHAPTERS:
        mp3_path = TEMP_AUDIO_DIR / f"{ch['id']}.mp3"
        wav_path = TEMP_AUDIO_DIR / f"{ch['id']}.wav"
        raw_sapi_wav = TEMP_AUDIO_DIR / f"{ch['id']}_sapi.wav"

        tts_ok = False
        try:
            comm = edge_tts.Communicate(ch["narration"], voice="en-US-ChristopherNeural", rate="+8%")
            await comm.save(str(mp3_path))
            subprocess.run(
                [
                    ffmpeg_exe,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(mp3_path),
                    "-ar",
                    str(sr),
                    "-ac",
                    "2",
                    "-f",
                    "wav",
                    str(wav_path),
                ],
                check=True,
            )
            tts_ok = True
        except Exception:
            # Offline Windows SAPI.SpVoice fallback (zero network or clock dependency)
            import win32com.client

            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            # Select an English voice if available
            for v in speaker.GetVoices():
                desc = v.GetDescription()
                if "English" in desc or "David" in desc or "Zira" in desc or "Mark" in desc:
                    speaker.Voice = v
                    break
            speaker.Rate = 1
            stream.Open(str(raw_sapi_wav), 3, False)
            speaker.AudioOutputStream = stream
            speaker.Speak(ch["narration"])
            stream.Close()

            subprocess.run(
                [
                    ffmpeg_exe,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_sapi_wav),
                    "-ar",
                    str(sr),
                    "-ac",
                    "2",
                    "-f",
                    "wav",
                    str(wav_path),
                ],
                check=True,
            )
            tts_ok = True

        if tts_ok and wav_path.exists():
            with wave.open(str(wav_path), "rb") as wf:
                n_frames = wf.getnframes()
                raw_pcm = wf.readframes(n_frames)
                pcm_i16 = np.frombuffer(raw_pcm, dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32768.0

            start_sample = int(float(ch["start"]) * sr) + int(0.18 * sr)
            max_len = int((float(ch["end"]) - float(ch["start"]) - 0.22) * sr)
            clip_len = min(len(pcm_i16), max_len, total_samples - start_sample)
            if clip_len > 0:
                mix_stereo[start_sample : start_sample + clip_len] += pcm_i16[:clip_len] * 0.92

    mix_i16 = (np.clip(mix_stereo, -0.98, 0.98) * 32767.0).astype(np.int16)
    with wave.open(str(master_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(mix_i16.tobytes())

    return master_wav


def _load_rgb_resized(path: Path, w: int, h: int) -> np.ndarray:
    if path.exists():
        bgr = cv2.imread(str(path))
        if bgr is not None:
            resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.full((h, w, 3), 24, dtype=np.uint8)


def _load_video_frames(path: Path, w: int, h: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    if path.exists():
        cap = cv2.VideoCapture(str(path))
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            resized = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        cap.release()
    return frames


def _draw_top_and_bottom_hud(
    pil_img: Image.Image,
    chapter: dict[str, Any],
    t_global: float,
) -> None:
    draw = ImageDraw.Draw(pil_img, "RGBA")
    f_tag = _get_font(18, bold=True)
    f_title = _get_font(22, bold=True)
    f_sub = _get_font(17, bold=False)
    f_cap = _get_font(22, bold=True)

    # Top sleek chapter banner
    draw.rectangle([0, 0, WIDTH, 62], fill=(16, 14, 22, 232))
    draw.rectangle([0, 60, WIDTH, 62], fill=(234, 255, 69, 220))
    draw.rounded_rectangle([24, 14, 260, 48], radius=8, fill=(234, 255, 69, 255))
    draw.text((38, 21), "#OpenSourceHiggsfield", font=f_tag, fill=(18, 18, 22, 255))
    draw.text((280, 14), chapter["title"], font=f_title, fill=(234, 255, 69, 255))
    draw.text((280, 38), chapter["subtitle"], font=f_sub, fill=(195, 192, 210, 255))

    # Timecode & GPU badge on top right
    tc_str = f"RTX 5080 16GB · comfy-sdk v0.5.0  |  {int(t_global // 60):01d}:{t_global % 60:04.1f} / 1:24.0"
    draw.rounded_rectangle([WIDTH - 510, 14, WIDTH - 24, 48], radius=8, fill=(32, 30, 42, 240), outline=(90, 88, 110, 200))
    draw.text((WIDTH - 492, 21), tc_str, font=f_sub, fill=(235, 235, 245, 255))

    # Bottom subtitle bar + neon progress bar
    prog_w = int((t_global / TOTAL_DURATION) * WIDTH)
    draw.rectangle([0, HEIGHT - 58, WIDTH, HEIGHT], fill=(14, 12, 20, 228))
    draw.rectangle([0, HEIGHT - 5, WIDTH, HEIGHT], fill=(45, 42, 58, 255))
    draw.rectangle([0, HEIGHT - 5, prog_w, HEIGHT], fill=(234, 255, 69, 255))

    # Animated current sentence chunk from chapter narration
    words = chapter["narration"].split()
    ch_u = (t_global - float(chapter["start"])) / max(float(chapter["end"]) - float(chapter["start"]), 0.1)
    chunk_size = 13
    num_chunks = max(1, math.ceil(len(words) / chunk_size))
    c_idx = min(num_chunks - 1, int(ch_u * num_chunks))
    caption_line = " ".join(words[c_idx * chunk_size : (c_idx + 1) * chunk_size])
    draw.ellipse([34, HEIGHT - 37, 46, HEIGHT - 25], fill=(234, 255, 69, 255))
    draw.text((56, HEIGHT - 44), f">> {caption_line}", font=f_cap, fill=(248, 248, 252, 255))


def render_demo_video(audio_wav_path: Path) -> Path:
    out_mp4 = OUTPUTS_DIR / "OpenHiggsfield_Submission_Demo_1080p.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Load UI screenshots and showcase assets
    ui_tab1 = _load_rgb_resized(SCREENSHOTS_DIR / "ui_tab1_cinematic_studio.png", WIDTH, HEIGHT)
    ui_tab2 = _load_rgb_resized(SCREENSHOTS_DIR / "ui_tab2_reshoot_3d_sphere.png", WIDTH, HEIGHT)
    ui_tab3 = _load_rgb_resized(SCREENSHOTS_DIR / "ui_tab3_openjutsu.png", WIDTH, HEIGHT)
    ui_trim = _load_rgb_resized(SCREENSHOTS_DIR / "ui_tab3_trim_modal.png", WIDTH, HEIGHT)
    ui_tab4 = _load_rgb_resized(SCREENSHOTS_DIR / "ui_tab4_workflow_sdk.png", WIDTH, HEIGHT)

    var_a = _load_rgb_resized(OUTPUTS_DIR / "cinematic_var_A_medium_blue_hour_auto.png", 760, 428)
    var_b = _load_rgb_resized(OUTPUTS_DIR / "cinematic_var_B_low_angle_blue_hour_auto.png", 760, 428)
    var_c = _load_rgb_resized(OUTPUTS_DIR / "cinematic_var_C_low_angle_neon_auto.png", 760, 428)
    var_d = _load_rgb_resized(OUTPUTS_DIR / "cinematic_latest.png", 760, 428)
    norm_map = _load_rgb_resized(OUTPUTS_DIR / "cinematic_normals_default.png", 520, 292)
    depth_map = _load_rgb_resized(OUTPUTS_DIR / "reshoot_depth_analysis.png", 520, 292)
    sphere_gizmo = _load_rgb_resized(OUTPUTS_DIR / "reshoot_sphere_gizmo.png", 380, 380)
    rap_bg = _load_rgb_resized(SHOWCASE_DIR / "openjutsu_rap_source_frame.png", WIDTH, HEIGHT)

    reshoot_wipe_frames = _load_video_frames(OUTPUTS_DIR / "reshoot_wipe_latest.mp4", 820, 460)
    openjutsu_wipe_frames = _load_video_frames(OUTPUTS_DIR / "openjutsu_wipe_latest.mp4", 820, 460)

    f_mono = _get_font(42, bold=True, mono=True)
    f_hero = _get_font(86, bold=True)
    f_subhero = _get_font(34, bold=True)
    f_card_h = _get_font(24, bold=True)
    f_card_b = _get_font(19, bold=False)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-i",
        str(audio_wav_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None

    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        cur_ch = CHAPTERS[-1]
        for ch in CHAPTERS:
            if float(ch["start"]) <= t < float(ch["end"]):
                cur_ch = ch
                break
        ch_u = (t - float(cur_ch["start"])) / max(float(cur_ch["end"]) - float(cur_ch["start"]), 0.1)

        # -------------------------------------------------------------------
        # Chapter 1 (0..9s): Code Editor Typing Hook -> Neon-Yellow Title Card
        # -------------------------------------------------------------------
        if cur_ch["id"] == "ch1_hook":
            if t < 4.5:
                pil_frame = Image.new("RGB", (WIDTH, HEIGHT), (8, 8, 12))
                draw = ImageDraw.Draw(pil_frame)
                lines = [
                    "Higgsfield promised to open source their entire platform.",
                    "They only dropped a training skeleton.",
                    "So we built OpenHiggsfield Studio on ComfyUI.",
                ]
                chars_total = sum(len(x) for x in lines)
                chars_revealed = int(min(1.0, t / 3.9) * chars_total)
                rem = chars_revealed
                y_pos = 370
                for l_no, line_txt in enumerate(lines, 1):
                    draw.text((180, y_pos), f"{l_no}", font=f_mono, fill=(95, 92, 115))
                    if rem > 0:
                        shown = line_txt[:rem]
                        cursor = "|" if rem <= len(line_txt) else ""
                        col = (234, 255, 69) if l_no == 3 else (245, 245, 250)
                        draw.text((260, y_pos), shown + cursor, font=f_mono, fill=col)
                        rem -= len(line_txt)
                    y_pos += 84
            else:
                # Warm studio rap backdrop with giant kinetic #EAFF45 typography
                darkened = (rap_bg.astype(np.float32) * 0.42).astype(np.uint8)
                pil_frame = Image.fromarray(darkened)
                draw = ImageDraw.Draw(pil_frame, "RGBA")
                h1_txt = "OPENHIGGSFIELD"
                h2_txt = "DIRECTOR STUDIO"
                b1 = draw.textbbox((0, 0), h1_txt, font=f_hero)
                b2 = draw.textbbox((0, 0), h2_txt, font=f_hero)
                draw.text(((WIDTH - (b1[2] - b1[0])) // 2, 205), h1_txt, font=f_hero, fill=(234, 255, 69, 255))
                draw.text(((WIDTH - (b2[2] - b2[0])) // 2, 310), h2_txt, font=f_hero, fill=(234, 255, 69, 255))

                pill_txt = "3-IN-1 OPEN-WEIGHTS AI CINEMA PLATFORM · COMFY-SDK v0.5.0"
                bp = draw.textbbox((0, 0), pill_txt, font=f_subhero)
                pill_w = (bp[2] - bp[0]) + 68
                pill_x0 = (WIDTH - pill_w) // 2
                pill_x1 = pill_x0 + pill_w
                draw.rounded_rectangle([pill_x0, 445, pill_x1, 515], radius=24, fill=(234, 255, 69, 255))
                draw.text(
                    (pill_x0 + 34, 462),
                    pill_txt,
                    font=f_subhero,
                    fill=(16, 16, 20, 255),
                )
                sub_txt = "LTX-2.5 (22B DFR & IC-LoRAs)  ·  FLUX.3 Klein  ·  comfy-api-proxy v0.1.6  ·  5 Custom Nodes"
                bs = draw.textbbox((0, 0), sub_txt, font=f_card_h)
                sub_w = (bs[2] - bs[0]) + 64
                sub_x0 = (WIDTH - sub_w) // 2
                draw.rounded_rectangle([sub_x0, 560, sub_x0 + sub_w, 650], radius=14, fill=(20, 18, 28, 225), outline=(234, 255, 69, 200), width=2)
                draw.text(
                    (sub_x0 + 32, 592),
                    sub_txt,
                    font=f_card_h,
                    fill=(242, 242, 248, 255),
                )

        # -------------------------------------------------------------------
        # Chapter 2 (9..24s): Why It's Innovative (True 3D Math + Custom Nodes)
        # -------------------------------------------------------------------
        elif cur_ch["id"] == "ch2_innovative":
            pil_frame = Image.fromarray(ui_tab4.copy())
            draw = ImageDraw.Draw(pil_frame, "RGBA")
            # Overlay callout panel on right showing live Depth, Normal Relight, and 3D Sphere math
            draw.rounded_rectangle([1310, 95, 1890, 995], radius=16, fill=(18, 16, 26, 242), outline=(234, 255, 69, 235), width=2)
            draw.text((1338, 118), "WHY IT'S INNOVATIVE (REAL 3D MATH)", font=f_card_h, fill=(234, 255, 69, 255))
            draw.text((1338, 152), "• 4×4 SE(3) Camera Matrices + LTX Sparse Tracks", font=f_card_b, fill=(235, 235, 245, 255))
            draw.text((1338, 180), "• 3D Depth Point-Cloud Warp + Magenta Mask", font=f_card_b, fill=(235, 235, 245, 255))
            draw.text((1338, 208), "• Surface-Normal N(x,y) PBR Relight & LUTs", font=f_card_b, fill=(235, 235, 245, 255))

            pil_frame.paste(Image.fromarray(depth_map), (1340, 248))
            draw.text((1350, 256), "OpenHiggsfieldDepthReshoot (Depth + Disocclusion)", font=f_card_b, fill=(234, 255, 69, 255))

            pil_frame.paste(Image.fromarray(norm_map), (1340, 560))
            draw.text((1350, 568), "OpenHiggsfieldRelightGrade (3D Normals + Light)", font=f_card_b, fill=(234, 255, 69, 255))

        # -------------------------------------------------------------------
        # Chapter 3 (24..40s): Pillar 1 — Cinematic Studio
        # -------------------------------------------------------------------
        elif cur_ch["id"] == "ch3_cinematic_studio":
            pil_frame = Image.fromarray(ui_tab1.copy())
            draw = ImageDraw.Draw(pil_frame, "RGBA")
            # Cycle through the 4 real generated variations A, B, C, D in a high-contrast showcase card
            v_idx = min(3, int(ch_u * 4))
            v_labels = [
                ("Shot 1 · Variation A", "Framing: Medium  |  Light: Blue hour  |  Grade: Auto", var_a),
                ("Shot 1 · Variation B", "Framing: Low angle  |  Light: Blue hour  |  Grade: Auto", var_b),
                ("Shot 1 · Variation C", "Framing: Low angle  |  Light: Neon (Magenta/Cyan)  |  Grade: Auto", var_c),
                ("Shot 1 · Variation D (FINAL)", "Camera: Large format Anamorphic 50mm f/2.8 · Light: Neon · Grade: Teal & Orange", var_d),
            ]
            v_title, v_desc, v_img = v_labels[v_idx]
            draw.rounded_rectangle([1085, 535, 1885, 1005], radius=16, fill=(18, 16, 26, 242), outline=(234, 255, 69, 240), width=3)
            pil_frame.paste(Image.fromarray(v_img), (1105, 550))
            draw.rounded_rectangle([1115, 560, 1420, 598], radius=8, fill=(234, 255, 69, 255))
            draw.text((1128, 567), v_title, font=f_card_h, fill=(18, 18, 22, 255))
            draw.text((1115, 980), v_desc, font=f_card_b, fill=(235, 235, 245, 255))

        # -------------------------------------------------------------------
        # Chapter 4 (40..57s): Pillar 2 — Re-shoot a Video & 3D Sphere Gizmo
        # -------------------------------------------------------------------
        elif cur_ch["id"] == "ch4_reshoot":
            pil_frame = Image.fromarray(ui_tab2.copy())
            draw = ImageDraw.Draw(pil_frame, "RGBA")
            if reshoot_wipe_frames:
                rw_frame = reshoot_wipe_frames[f_idx % len(reshoot_wipe_frames)]
                draw.rounded_rectangle([70, 185, 915, 675], radius=14, fill=(18, 16, 26, 245), outline=(234, 255, 69, 245), width=3)
                pil_frame.paste(Image.fromarray(rw_frame), (82, 198))
                draw.rounded_rectangle([92, 620, 895, 662], radius=8, fill=(18, 16, 26, 230))
                draw.text(
                    (108, 630),
                    "LIVE 3D DEPTH-WARPED OUTPUT  ·  [KF 0.0s: +30° / +8°]  ->  [KF 4.4s: -15° / 0°]",
                    font=f_card_b,
                    fill=(234, 255, 69, 255),
                )

        # -------------------------------------------------------------------
        # Chapter 5 (57..72s): Pillar 3 — OpenJutsu Character & VFX Swap
        # -------------------------------------------------------------------
        elif cur_ch["id"] == "ch5_openjutsu":
            if ch_u < 0.32:
                pil_frame = Image.fromarray(ui_trim.copy())
            else:
                pil_frame = Image.fromarray(ui_tab3.copy())
                draw = ImageDraw.Draw(pil_frame, "RGBA")
                if openjutsu_wipe_frames:
                    oj_frame = openjutsu_wipe_frames[f_idx % len(openjutsu_wipe_frames)]
                    draw.rounded_rectangle([985, 210, 1835, 700], radius=14, fill=(18, 16, 26, 245), outline=(234, 255, 69, 245), width=3)
                    pil_frame.paste(Image.fromarray(oj_frame), (998, 222))
                    draw.rounded_rectangle([1010, 646, 1815, 688], radius=8, fill=(18, 16, 26, 230))
                    draw.text(
                        (1025, 656),
                        "OPENJUTSU SWAP: character_grandma.png  ·  Audio Sync Locked (LTXVSetAudioRefTokens)",
                        font=f_card_b,
                        fill=(234, 255, 69, 255),
                    )

        # -------------------------------------------------------------------
        # Chapter 6 (72..84s): Dual-Surface SDK & Closing Summary Card
        # -------------------------------------------------------------------
        else:
            if ch_u < 0.45:
                pil_frame = Image.fromarray(ui_tab4.copy())
            else:
                pil_frame = Image.new("RGB", (WIDTH, HEIGHT), (18, 16, 24))
                draw = ImageDraw.Draw(pil_frame, "RGBA")
                draw.text((310, 150), "OPENHIGGSFIELD STUDIO", font=f_hero, fill=(234, 255, 69, 255))
                draw.text(
                    (360, 260),
                    "100% Open-Source MIT Release  ·  Built on ComfyUI & comfy-sdk v0.5.0",
                    font=f_subhero,
                    fill=(240, 240, 248, 255),
                )

                cards = [
                    ("1. CINEMATIC STUDIO", "6-Row DoP Stack · 3D Surface-Normal Relighting · Film Stocks & Print LUTs · FLUX.3 + LTX-2.5"),
                    ("2. RE-SHOOT A VIDEO", "Interactive 3D Wireframe Sphere Gizmo · 6-DOF Depth Point-Cloud Warp · Viral Camera Rigs"),
                    ("3. OPENJUTSU SWAP", "Interactive Filmstrip Trim Modal · Single-Image Character Sheet Swap · Synced Studio Audio"),
                    ("4. DUAL SDK ENGINE", "OPEN_WEIGHTS_LOCAL (comfy-api-proxy :8189) + CLOUD_HYBRID (Comfy Cloud & Router)"),
                ]
                cy = 345
                for c_title, c_body in cards:
                    draw.rounded_rectangle([260, cy, 1660, cy + 112], radius=14, fill=(30, 28, 40, 245), outline=(234, 255, 69, 200), width=2)
                    draw.text((295, cy + 20), c_title, font=f_card_h, fill=(234, 255, 69, 255))
                    draw.text((295, cy + 60), c_body, font=f_card_b, fill=(225, 225, 238, 255))
                    cy += 132

                pil_frame.paste(Image.fromarray(sphere_gizmo).resize((210, 210)), (40, 360))

        _draw_top_and_bottom_hud(pil_frame, cur_ch, t)
        proc.stdin.write(np.asarray(pil_frame, dtype=np.uint8).tobytes())

    proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    proc.wait(timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"Demo video FFmpeg error: {stderr_bytes.decode('utf-8', errors='replace')}")

    print(f"Successfully rendered demo video: {out_mp4} ({out_mp4.stat().st_size:,} bytes)")
    return out_mp4


async def async_main() -> None:
    wav_path = await generate_narration_track()
    render_demo_video(wav_path)


if __name__ == "__main__":
    asyncio.run(async_main())
