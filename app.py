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
    hip_hinge_too_upright: bool
    tempo_label: str  # "Rushed" | "Balanced" | "Smooth"
    head_moves_a_lot: bool
    shoulder_turn_good: bool
    balance_solid: bool


def _extract_pose_series(video_path: str, max_frames: int = 140) -> Tuple[List[Dict[int, np.ndarray]], float]:
    """
    Extract a sparse time series of pose landmarks from the video.
    Returns (frames_landmarks, fps).
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
    L_WRIST, R_WRIST = 15, 16

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

    # Hip hinge proxy: shoulder-to-hip vertical alignment.
    hip_hinge_too_upright = False
    if lsho is not None and rsho is not None and lhip is not None and rhip is not None:
        shoulder_mid = (lsho + rsho) / 2.0
        hip_mid = (lhip + rhip) / 2.0
        # If shoulders stack too directly over hips, posture is likely upright.
        hip_hinge_too_upright = abs(float(shoulder_mid[0] - hip_mid[0])) < 0.03

    # --- Tempo signals ---
    # Use wrist vertical movement to estimate top-of-backswing and return-to-impact.
    ys: List[Optional[float]] = []
    for m in frames_landmarks:
        lw, rw = get(m, L_WRIST), get(m, R_WRIST)
        if lw is None and rw is None:
            ys.append(None)
            continue
        # pick the wrist that exists (or average if both)
        if lw is not None and rw is not None:
            y = float((lw[1] + rw[1]) / 2.0)
        else:
            y = float((lw or rw)[1])
        ys.append(y)

    valid = [(i, y) for i, y in enumerate(ys) if y is not None]
    tempo_label = "Balanced"
    if len(valid) >= 12 and fps > 0:
        # Setup reference y is the first valid point.
        i0, y0 = valid[0]
        # Top of backswing: minimum y (hands highest in image coordinates).
        itop, _ = min(valid, key=lambda t: t[1])
        # Impact-ish: first time after top that y returns near address band.
        band = 0.05
        iafter = [i for i, y in valid if i > itop and abs(y - y0) < band]
        iimpact = iafter[0] if iafter else valid[-1][0]

        backswing = max(1, itop - i0)
        downswing = max(1, iimpact - itop)
        ratio = backswing / downswing

        # Qualitative classification only (no numbers shown).
        if ratio < 2.0:
            tempo_label = "Rushed"
        elif ratio > 3.3:
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
    # pick frame at/near itop if we computed it; else last third.
    if "itop" in locals():
        top_candidate = itop
    else:
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
        hip_hinge_too_upright=hip_hinge_too_upright,
        tempo_label=tempo_label,
        head_moves_a_lot=head_moves_a_lot,
        shoulder_turn_good=shoulder_turn_good,
        balance_solid=balance_solid,
    )


def _build_report(signals: Optional[SwingSignals]) -> Dict[str, object]:
    """
    Generate strictly qualitative coaching feedback.
    Returns dict with: overview (2 sentences), tempo_label, sections.
    """
    if signals is None:
        return {
            "overview": (
                "I couldn’t reliably track your body landmarks through the swing, so the feedback below is based on general down-the-line checkpoints. "
                "Try filming in good light with your full body in frame, then re-run the analysis for more personalized cues."
            ),
            "tempo_label": "Balanced",
            "sections": {
                "Setup": [
                    "Start in a more athletic posture by softening your knees so you look spring-loaded rather than tall and stiff.",
                    "Feel your hips fold back as if you’re sitting onto a tall stool, and keep your chest gently over the balls of your feet.",
                ],
                "Tempo": [
                    "Let the club and arms start together so the takeaway looks unhurried instead of snatched back.",
                    "Use a simple count of “one-two-three” to the top, then let the downswing start as a natural change of direction.",
                ],
                "Motion Cues": [
                    "Keep your head quiet by holding your nose roughly over the same patch of grass while your shoulders rotate underneath you.",
                    "Make your finish look balanced and tall, with your belt buckle facing the target and your weight clearly on the lead side.",
                ],
                "Drill": [
                    "Do slow half-swings in front of a mirror and pause at address to check soft knees, hips back, and a steady head.",
                    "Repeat that same setup feel before every rep so your body learns one consistent starting picture.",
                ],
                "Positive": [
                    "You’re building the right habit by filming and checking key positions, which is the fastest way to improve.",
                    "Stick with one simple cue at a time and your swing will start to look more repeatable very quickly.",
                ],
            },
        }

    # ---- Overview (2 sentences) ----
    overview_bits: List[str] = []
    if signals.tempo_label == "Rushed":
        overview_bits.append("Your swing looks athletic, but the transition feels quick, which can make contact less consistent.")
    elif signals.tempo_label == "Smooth":
        overview_bits.append("Your swing tempo looks unhurried and composed, which is a great recipe for consistent strikes.")
    else:
        overview_bits.append("Your swing tempo looks fairly balanced, giving you a good base to build consistent contact.")

    if signals.knee_flex_too_straight or signals.hip_hinge_too_upright:
        overview_bits.append("The biggest opportunity is improving your setup posture so you can rotate more freely through the ball.")
    elif signals.head_moves_a_lot:
        overview_bits.append("The biggest opportunity is keeping your head steadier so your rotation stays centered through impact.")
    else:
        overview_bits.append("The biggest opportunity is sharpening one simple feel so your best swings show up more often.")

    overview = " ".join(overview_bits[:2])

    # ---- Setup (2-3 full sentences) ----
    setup: List[str] = []
    if signals.knee_flex_too_straight:
        setup.append("Your knees look a bit too straight at address, which can make your lower body feel locked and less athletic.")
        setup.append("Try softening your knees so your legs look springy, like you could jump straight up without re-adjusting.")
    else:
        setup.append("Your lower body looks reasonably athletic at address, which helps you stay grounded as you turn.")
        setup.append("Keep a gentle softness in the knees so you look ready to move rather than perched and rigid.")

    if signals.hip_hinge_too_upright:
        setup.append("Add a touch more hip hinge by sending your hips back, and let your chest tip forward so your arms can hang naturally.")
    else:
        setup.append("Maintain your hip hinge and let your arms hang under your shoulders so the club starts from a relaxed, repeatable spot.")
    setup = setup[:3]

    # ---- Tempo (2-3 full sentences, label in header) ----
    tempo: List[str] = []
    if signals.tempo_label == "Rushed":
        tempo.append("Your backswing appears quick, which can cause your timing to feel different from swing to swing.")
        tempo.append("Try counting “one-two-three” during the takeaway so the club stays in sync with your body and the top feels organized.")
    elif signals.tempo_label == "Smooth":
        tempo.append("Your backswing has a smooth pace, which makes it easier to find the same positions repeatedly.")
        tempo.append("Keep that same unhurried takeaway and let the change of direction feel like a gentle ‘turn-and-go’ rather than a hard hit from the top.")
    else:
        tempo.append("Your tempo looks fairly balanced, which gives you a good platform for consistent contact.")
        tempo.append("For even better rhythm, let the first few feet of the takeaway feel slow and wide, then keep turning until you’re fully loaded.")
    # Optional third sentence that stays specific and visual:
    tempo.append("A good visual is that your hands and chest arrive at the top together, then your lower body leads the downswing while your head stays calm.")
    tempo = tempo[:3]

    # ---- Motion Cues (2-3 full sentences) ----
    motion: List[str] = []
    if signals.shoulder_turn_good:
        motion.append("Your shoulder turn looks good through the backswing, and it helps you create a bigger arc without forcing the arms.")
    else:
        motion.append("Focus on letting your shoulders turn more around your spine so the backswing looks like a coil rather than an arm lift.")

    if signals.head_moves_a_lot:
        motion.append("Your head moves around during the swing, so picture your nose staying over the same patch of grass while your shoulders rotate underneath you.")
        motion.append("A helpful feel is to keep your lead ear ‘stacked’ over your lead shoulder as you rotate through impact, instead of drifting toward the ball.")
    else:
        motion.append("Keep your head steady by holding your nose roughly in place while your chest and shoulders rotate underneath it through impact.")
        motion.append("Let your finish look tall and balanced, with your weight clearly on your lead side and your belt buckle facing the target.")
    motion = motion[:3]

    # ---- Drill (2-3 full sentences) ----
    drill: List[str] = []
    if signals.knee_flex_too_straight or signals.hip_hinge_too_upright:
        drill.append("Practice setup-only reps in front of a mirror, then make slow half-swings while keeping that same ‘hips back, soft knees’ picture.")
        drill.append("Pause at the top for a moment, then swing through and finish balanced so you learn a calm transition without rushing.")
    elif signals.head_moves_a_lot:
        drill.append("Place an alignment stick or a tall object just outside your lead ear and make slow swings without letting your head bump into it.")
        drill.append("Start with half-swings, then build to fuller swings while keeping your nose quiet and your rotation centered.")
    else:
        drill.append("Do slow half-swings and hold your finish for a full breath so your body learns what balanced contact feels like.")
        drill.append("Repeat the same rhythm every rep, and stop the set as soon as you feel yourself speeding up.")
    drill = drill[:3]

    # ---- Positive (2-3 full sentences) ----
    positive: List[str] = []
    if signals.balance_solid:
        positive.append("Your balance throughout the swing is solid, which is a great foundation to build on.")
        positive.append("Keep finishing with your chest up and weight clearly on your lead side, because that stable finish usually tracks with better contact.")
    else:
        positive.append("You already have a workable motion, and a few small checkpoints will make it look more repeatable quickly.")
        positive.append("If you can hold your finish without stepping or wobbling, you’ll know your swing is staying more centered.")
    positive.append("Stay patient and focus on one cue at a time, because the best swings come from simple pictures you can repeat.")
    positive = positive[:3]

    return {
        "overview": overview,
        "tempo_label": signals.tempo_label,
        "sections": {
            "Setup": setup,
            "Tempo": tempo,
            "Motion Cues": motion,
            "Drill": drill,
            "Positive": positive,
        },
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
            sections = report["sections"]

            st.markdown("#### Setup:")
            for sentence in sections["Setup"]:
                st.markdown(f"- {sentence}")

            st.markdown(f"#### Tempo: {report['tempo_label']}")
            for sentence in sections["Tempo"]:
                st.markdown(f"- {sentence}")

            st.markdown("#### Motion Cues:")
            for sentence in sections["Motion Cues"]:
                st.markdown(f"- {sentence}")

            st.markdown("#### Drill:")
            for sentence in sections["Drill"]:
                st.markdown(f"- {sentence}")

            st.markdown("#### Positive:")
            for sentence in sections["Positive"]:
                st.markdown(f"- {sentence}")
        st.success("✅ Analysis complete!")
    # Cleanup temp file best-effort (Streamlit can rerun; keep only if still needed).
    try:
        os.unlink(video_path)
    except OSError:
        pass
else:
    st.info("👆 Upload a video to get started")
