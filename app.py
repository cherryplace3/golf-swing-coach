import streamlit as st
import tempfile
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(page_title="Pocket Swing Coach", layout="wide")

st.title("⛳ Pocket Swing Coach")
st.markdown("Upload your golf swing video for instant coaching feedback")

mp_pose = mp.solutions.pose


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees (internal use only)."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


@dataclass
class SwingSignals:
    knee_flex_too_straight: bool
    spine_too_tall: bool
    spine_too_slouched: bool
    tempo_label: str  # "Rushed" | "Balanced" | "Smooth"
    head_moves_a_lot: bool
    shoulder_turn_good: bool
    balance_solid: bool


def _extract_pose_series(video_path: str, max_frames: int = 160) -> Tuple[List[Dict[int, np.ndarray]], float]:
    """
    Extract frames from the uploaded video and run pose detection.
    Returns (frames_landmarks, fps). Frames may be sampled for performance.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # Fallback sampling.
        step = 3
    else:
        step = max(1, total // max_frames)

    frames_landmarks: List[Dict[int, np.ndarray]] = []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        idx = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue

            kept += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            lm_map: Dict[int, np.ndarray] = {}
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    # Normalized coordinates (x, y); keep z for internal stability checks.
                    lm_map[i] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            frames_landmarks.append(lm_map)
            idx += 1

            if kept >= max_frames:
                break

    cap.release()
    return frames_landmarks, fps


def _compute_signals(frames_landmarks: List[Dict[int, np.ndarray]], fps: float) -> Optional[SwingSignals]:
    if not frames_landmarks:
        return None

    # Choose "setup" as the earliest frame with enough landmarks.
    setup_idx = next((i for i, m in enumerate(frames_landmarks) if len(m) > 0), None)
    if setup_idx is None:
        return None

    # Landmarks: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    NOSE = 0
    # (Wrists exist, but tempo requirement uses shoulders.)

    def get(m: Dict[int, np.ndarray], i: int) -> Optional[np.ndarray]:
        v = m.get(i)
        if v is None:
            return None
        return v

    setup = frames_landmarks[setup_idx]
    lhip, rhip = get(setup, L_HIP), get(setup, R_HIP)
    lknee, rknee = get(setup, L_KNEE), get(setup, R_KNEE)
    lank, rank = get(setup, L_ANKLE), get(setup, R_ANKLE)
    lsho, rsho = get(setup, L_SHOULDER), get(setup, R_SHOULDER)
    nose0 = get(setup, NOSE)

    # --- Setup posture signals (internal numeric computation, qualitative output only) ---
    knee_angles: List[float] = []
    for hip, knee, ankle in [(lhip, lknee, lank), (rhip, rknee, rank)]:
        if hip is not None and knee is not None and ankle is not None:
            knee_angles.append(_angle_degrees(hip[:2], knee[:2], ankle[:2]))

    # Straight knees => larger angle near "locked out".
    knee_flex_too_straight = bool(knee_angles and (sum(knee_angles) / len(knee_angles)) > 168.0)

    # Spine "tall vs slouched" proxy using shoulder-to-hip distance (normalized).
    spine_too_tall = False
    spine_too_slouched = False
    if lsho is not None and rsho is not None and lhip is not None and rhip is not None and lank is not None and rank is not None:
        shoulder_mid = (lsho + rsho) / 2.0
        hip_mid = (lhip + rhip) / 2.0
        ankle_mid = (lank + rank) / 2.0

        # In image coordinates, y increases downward.
        shoulder_to_hip = max(1e-6, float(hip_mid[1] - shoulder_mid[1]))
        hip_to_ankle = max(1e-6, float(ankle_mid[1] - hip_mid[1]))
        spine_ratio = shoulder_to_hip / hip_to_ankle  # internal only

        spine_too_slouched = spine_ratio < 0.42
        spine_too_tall = spine_ratio > 0.62 and knee_flex_too_straight

    # --- Tempo signals ---
    # Track shoulder position changes across frames.
    shoulder_mids: List[Tuple[int, np.ndarray]] = []
    for i, m in enumerate(frames_landmarks):
        ls, rs = get(m, L_SHOULDER), get(m, R_SHOULDER)
        if ls is None or rs is None:
            continue
        shoulder_mids.append((i, (ls + rs) / 2.0))

    tempo_label = "Balanced"
    if len(shoulder_mids) >= 10:
        # Normalize motion by shoulder width from the setup frame to reduce scale effects.
        sw = 0.2
        if lsho is not None and rsho is not None:
            sw = _dist(lsho[:2], rsho[:2]) or sw

        steps: List[float] = []
        for (_, p0), (_, p1) in zip(shoulder_mids, shoulder_mids[1:]):
            steps.append(_dist(p0[:2], p1[:2]) / (sw + 1e-9))

        if steps:
            med = float(np.median(steps))
            variability = float(np.std(steps) / (med + 1e-9))

            # Qualitative classification only (no numbers shown).
            if med > 0.12 or variability > 1.2:
                tempo_label = "Rushed"
            elif med < 0.07 and variability < 0.9:
                tempo_label = "Smooth"
            else:
                tempo_label = "Balanced"

    # --- Head stability ---
    head_moves_a_lot = False
    if nose0 is not None:
        nose_positions = [get(m, NOSE) for m in frames_landmarks]
        nose_positions = [n for n in nose_positions if n is not None]
        if len(nose_positions) >= 6:
            # Normalize motion by shoulder width to be view-scale invariant.
            sw = None
            if lsho is not None and rsho is not None:
                sw = _dist(lsho[:2], rsho[:2])
            sw = sw or 0.2
            disps = [_dist(n[:2], nose0[:2]) / (sw + 1e-9) for n in nose_positions]
            head_moves_a_lot = (max(disps) > 0.55)

    # --- Shoulder turn quality proxy ---
    shoulder_turn_good = False
    # Compare shoulder line tilt at setup vs top-ish frame.
    # If top-ish frame exists with shoulders, look for meaningful rotation cue.
    top_candidate = None
    # pick frame in the later half as a "loaded" reference.
    top_candidate = int(len(frames_landmarks) * 0.6)
    mtop = frames_landmarks[min(max(0, top_candidate), len(frames_landmarks) - 1)]
    lsho_t, rsho_t = get(mtop, L_SHOULDER), get(mtop, R_SHOULDER)
    if lsho is not None and rsho is not None and lsho_t is not None and rsho_t is not None:
        # Use change in shoulder line slope magnitude as a loose proxy for rotation.
        def slope(a: np.ndarray, b: np.ndarray) -> float:
            return float((b[1] - a[1]) / ((b[0] - a[0]) + 1e-6))

        s0 = abs(slope(lsho, rsho))
        st = abs(slope(lsho_t, rsho_t))
        shoulder_turn_good = (st > s0 + 0.15)

    # --- Balance proxy ---
    # If hips stay centered relative to ankles across the motion, balance is likely solid.
    balance_solid = False
    hip_mids: List[np.ndarray] = []
    ankle_mids: List[np.ndarray] = []
    for m in frames_landmarks:
        lh, rh = get(m, L_HIP), get(m, R_HIP)
        la, ra = get(m, L_ANKLE), get(m, R_ANKLE)
        if lh is None or rh is None or la is None or ra is None:
            continue
        hip_mids.append((lh + rh) / 2.0)
        ankle_mids.append((la + ra) / 2.0)
    if len(hip_mids) >= 8 and len(ankle_mids) >= 8:
        sw = 0.2
        if lsho is not None and rsho is not None:
            sw = _dist(lsho[:2], rsho[:2]) or sw
        offsets = [abs(float(h[0] - a[0])) / (sw + 1e-9) for h, a in zip(hip_mids, ankle_mids)]
        balance_solid = (np.median(offsets) < 0.35 and max(offsets) < 0.85)

    return SwingSignals(
        knee_flex_too_straight=knee_flex_too_straight,
        spine_too_tall=spine_too_tall,
        spine_too_slouched=spine_too_slouched,
        tempo_label=tempo_label,
        head_moves_a_lot=head_moves_a_lot,
        shoulder_turn_good=shoulder_turn_good,
        balance_solid=balance_solid,
    )


def _build_report(signals: Optional[SwingSignals]) -> Dict[str, object]:
    """
    Generate strictly qualitative coaching feedback.
    Returns dict with: overview (2 sentences), tempo_label, and text blocks.
    """
    if signals is None:
        return {
            "overview": (
                "I couldn’t reliably track your body landmarks through the swing, so the feedback below is based on general down-the-line checkpoints. "
                "Try filming in good light with your full body in frame, then re-run the analysis for more personalized cues."
            ),
            "tempo_label": "Balanced",
            "setup_bullets": [
                "Start in a more athletic posture by softening your knees so you look spring-loaded rather than tall and stiff.",
                "Feel your hips fold back as if you’re sitting onto a tall stool, and keep your chest gently over the balls of your feet.",
            ],
            "tempo_bullets": [
                "Let your chest and shoulders start the swing together so the first move looks calm instead of snatched back.",
                "Use a simple count of “one-two-three” going back, then let the change of direction happen without a sudden lunge.",
            ],
            "motion_bullets": [
                "Keep your head quiet by holding your nose roughly over the same patch of grass while your shoulders rotate underneath you.",
                "Let your finish look tall and balanced, with your weight clearly on your lead side and your shirt buttons facing the target.",
            ],
            "drill": "Practice slow half-swings in front of a mirror and pause at address to check soft knees, hips back, and a steady head.",
            "positive": "You’re doing the right thing by filming your swing, because clear checkpoints and simple cues lead to fast improvement.",
        }

    # ---- Overview (exactly 2 sentences) ----
    sentence_1 = {
        "Rushed": "Your swing looks athletic, but the movement of your upper body changes quickly, which can make timing harder to repeat.",
        "Balanced": "Your swing rhythm looks fairly balanced, giving you a solid base for consistent contact.",
        "Smooth": "Your swing rhythm looks smooth and unhurried, which is a great recipe for repeatable contact.",
    }.get(signals.tempo_label, "Your swing rhythm looks fairly balanced, giving you a solid base for consistent contact.")

    if signals.knee_flex_too_straight or signals.spine_too_tall or signals.spine_too_slouched:
        sentence_2 = "The biggest opportunity is dialing in a more athletic setup so your turn looks freer and your strike becomes more predictable."
    elif signals.head_moves_a_lot:
        sentence_2 = "The biggest opportunity is keeping your head steadier so your rotation stays centered through impact."
    else:
        sentence_2 = "The biggest opportunity is committing to one simple feel so your best swings show up more often."

    overview = f"{sentence_1} {sentence_2}"

    # ---- Setup (2-3 full-sentence bullets) ----
    setup_bullets: List[str] = []
    if signals.knee_flex_too_straight:
        setup_bullets.append("Your knees look a bit too straight at address, which can make your lower body feel locked and less athletic.")
        setup_bullets.append("Try softening your knees so your legs look springy, like you could jump straight up without re-adjusting.")
    else:
        setup_bullets.append("Your lower body looks reasonably athletic at address, which helps you stay grounded as you turn.")
        setup_bullets.append("Keep a gentle softness in the knees so you look ready to move rather than perched and rigid.")

    if signals.spine_too_tall:
        setup_bullets.append("Add a little more athletic tilt by sending your hips back, and picture sitting onto a tall bar stool so your chest tips forward naturally.")
    elif signals.spine_too_slouched:
        setup_bullets.append("Stand a touch taller through your chest so your shirt buttons point slightly forward rather than collapsed down toward the ball.")
    else:
        setup_bullets.append("Maintain a tall chest while your hips sit back slightly, so your arms can hang naturally under your shoulders.")
    setup_bullets = setup_bullets[:3]

    # ---- Tempo (2-3 full-sentence bullets) ----
    tempo_bullets: List[str] = []
    if signals.tempo_label == "Rushed":
        tempo_bullets.append("Your shoulders change direction quickly, which can make the swing feel ‘grabby’ and affect consistency.")
        tempo_bullets.append("Try letting your first move back feel slow and wide, then count “one-two-three” to the top so the transition looks calm.")
        tempo_bullets.append("A good visual is that your chest finishes turning before you feel the club start down, instead of everything firing at once.")
    elif signals.tempo_label == "Smooth":
        tempo_bullets.append("Your shoulders move with a smooth pace, which makes it easier to find the same positions repeatedly.")
        tempo_bullets.append("Keep that same unhurried start and let the downswing begin as a change of direction, not a sudden hit from the top.")
        tempo_bullets.append("Picture your shoulders turning like a big wheel, with no jerky stops, all the way through to a balanced finish.")
    else:
        tempo_bullets.append("Your shoulder movement looks fairly balanced, which is a strong starting point for repeatable contact.")
        tempo_bullets.append("For even better rhythm, let the takeaway feel slightly slower than you think, then keep turning until you feel fully loaded.")
        tempo_bullets.append("A helpful image is that your shoulders and arms arrive at the top together, then your lower body leads while your head stays quiet.")
    tempo_bullets = tempo_bullets[:3]

    # ---- Motion cues (2-3 full-sentence bullets) ----
    motion_bullets: List[str] = []
    if signals.shoulder_turn_good:
        motion_bullets.append("Your shoulder turn looks good through the backswing, and it helps you create a bigger arc without forcing the arms.")
    else:
        motion_bullets.append("Focus on letting your shoulders turn more around your spine so the backswing looks like a coil rather than an arm lift.")

    if signals.head_moves_a_lot:
        motion_bullets.append("Your head moves around during the swing, so picture your nose staying over the same patch of grass while your shoulders rotate underneath you.")
        motion_bullets.append("A helpful feel is to keep your lead ear ‘stacked’ over your lead shoulder as you rotate through impact, instead of drifting toward the ball.")
    else:
        motion_bullets.append("Keep your head steady by holding your nose roughly in place while your chest and shoulders rotate underneath it through impact.")
        motion_bullets.append("Let your finish look tall and balanced, with your weight clearly on your lead side and your shirt buttons facing the target.")
    motion_bullets = motion_bullets[:3]

    # ---- One drill (single full sentence) ----
    if signals.knee_flex_too_straight or signals.spine_too_tall or signals.spine_too_slouched:
        drill = "Practice setup-only reps in front of a mirror, then make slow half-swings while keeping the ‘sit on a bar stool’ feel and a tall chest."
    elif signals.tempo_label == "Rushed":
        drill = "Make five slow rehearsals counting “one-two-three” to the top, then hit a half-swing trying to match that exact same shoulder rhythm."
    elif signals.head_moves_a_lot:
        drill = "Place a headcover just outside your lead ear and make slow half-swings without letting your head drift into it."
    else:
        drill = "Hit half-swings while holding your finish for a full breath so your body learns what balanced contact feels like."

    # ---- One positive (single full sentence) ----
    if signals.balance_solid:
        positive = "Your balance looks solid through the motion, which is an excellent foundation to build speed and consistency."
    elif signals.shoulder_turn_good:
        positive = "Your shoulder turn is a real strength, and that rotation gives you a great base for a repeatable strike."
    else:
        positive = "Your swing has a workable structure, and small setup and rhythm tweaks can make it look more repeatable quickly."

    return {
        "overview": overview,
        "tempo_label": signals.tempo_label,
        "setup_bullets": setup_bullets,
        "tempo_bullets": tempo_bullets,
        "motion_bullets": motion_bullets,
        "drill": drill,
        "positive": positive,
    }


uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'mov', 'avi'],
    help="Upload a video of your golf swing from the down-the-line view"
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("📹 Your Swing")
        st.video(video_path)
    
    if st.button("🔍 Analyze Swing", type="primary"):
        with st.spinner("Analyzing your swing..."):
            frames_landmarks, fps = _extract_pose_series(video_path=video_path)
            signals = _compute_signals(frames_landmarks=frames_landmarks, fps=fps)
            report = _build_report(signals)

            st.markdown("### 🧠 Swing Overview")
            st.write(report["overview"])

            st.markdown("### ✅ Coaching Notes")
            st.markdown("### Setup")
            for sentence in report["setup_bullets"]:
                st.markdown(f"- {sentence}")

            st.markdown(f"### Tempo: {report['tempo_label']}")
            for sentence in report["tempo_bullets"]:
                st.markdown(f"- {sentence}")

            st.markdown("### Motion Cues")
            for sentence in report["motion_bullets"]:
                st.markdown(f"- {sentence}")

            st.markdown("### Recommended Drill")
            st.markdown(f"- {report['drill']}")

            st.markdown("### You're doing well")
            st.markdown(f"- {report['positive']}")
        st.success("✅ Analysis complete!")
    # Cleanup temp file best-effort (Streamlit can rerun; keep only if still needed).
    try:
        os.unlink(video_path)
    except OSError:
        pass
else:
    st.info("👆 Upload a video to get started")
