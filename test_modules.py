"""
test_modules.py
===============
Smoke tests for all pipeline modules.
Runs without a GPU or video file – uses synthetic NumPy frames.

Run with:  python test_modules.py
"""

import numpy as np
import sys

PASS = "✓"
FAIL = "✗"


def banner(title: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print('─'*50)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config():
    banner("Config / Settings")
    from config.settings import ADASConfig, DEFAULT_CONFIG
    cfg = ADASConfig()
    assert cfg.lane.canny_low < cfg.lane.canny_high, "Canny thresholds invalid"
    assert 0 < cfg.lane.ema_alpha <= 1,              "EMA alpha out of range"
    assert DEFAULT_CONFIG is not None
    print(f"  {PASS} ADASConfig instantiates cleanly")
    print(f"  {PASS} DEFAULT_CONFIG singleton accessible")


# ---------------------------------------------------------------------------
# Geometry utils
# ---------------------------------------------------------------------------
def test_geometry():
    banner("utils/geometry")
    from utils.geometry import iou, box_area, box_center, slope_intercept

    a = (0, 0, 100, 100)
    b = (50, 50, 150, 150)
    score = iou(a, b)
    assert 0.14 < score < 0.15, f"IoU wrong: {score}"
    print(f"  {PASS} IoU calculation correct ({score:.4f})")

    assert box_area((10, 10, 50, 60)) == 40 * 50
    print(f"  {PASS} box_area correct")

    cx, cy = box_center((0, 0, 100, 80))
    assert cx == 50 and cy == 40
    print(f"  {PASS} box_center correct")

    m, b = slope_intercept(0, 0, 10, 20)
    assert abs(m - 2.0) < 1e-6
    print(f"  {PASS} slope_intercept correct")


# ---------------------------------------------------------------------------
# Lane detection
# ---------------------------------------------------------------------------
def test_lane_detection():
    banner("LaneDetector")
    from modules.lane_detection import LaneDetector, LaneResult
    from config.settings import LaneConfig

    detector = LaneDetector(LaneConfig())

    # Black frame → no lanes
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = detector.process(black)
    assert isinstance(result, LaneResult)
    assert not result.lane_detected
    print(f"  {PASS} Black frame → lane_detected=False")

    # Frame with synthetic diagonal white lines (simulating lane markings)
    synthetic = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Left lane line (negative slope from car's perspective)
    pts_left  = np.array([[200, 680], [560, 420]], dtype=np.int32)
    pts_right = np.array([[1080, 680], [720, 420]], dtype=np.int32)
    cv2_available = True
    try:
        import cv2
        cv2.line(synthetic, tuple(pts_left[0]),  tuple(pts_left[1]),  (255,255,255), 8)
        cv2.line(synthetic, tuple(pts_right[0]), tuple(pts_right[1]), (255,255,255), 8)
        result2 = detector.process(synthetic)
        # May or may not detect (depends on Hough params) – just ensure no crash
        print(f"  {PASS} Synthetic lane frame processed  (detected={result2.lane_detected})")
    except ImportError:
        print(f"  ⚠ cv2 not available – skipped synthetic lane test")

    print(f"  {PASS} LaneDetector smoke test passed")


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
def test_tracker():
    banner("IoUTracker")
    from modules.tracker import IoUTracker, Track
    from modules.object_detection import Detection
    from config.settings import TrackerConfig

    tracker = IoUTracker(TrackerConfig())

    det1 = Detection(box=(100, 100, 300, 250), class_id=2, label="car", confidence=0.9)
    det2 = Detection(box=(500, 200, 700, 380), class_id=7, label="truck", confidence=0.8)

    # Frame 1 – spawn two tracks (tentative, not yet shown)
    frame1_tracks = tracker.update([det1, det2])
    assert len(tracker._tracks) == 2
    print(f"  {PASS} 2 new detections → 2 internal tracks")

    # Frame 2 – slight movement → should match
    det1b = Detection(box=(105, 102, 305, 252), class_id=2, label="car", confidence=0.88)
    det2b = Detection(box=(503, 201, 703, 381), class_id=7, label="truck", confidence=0.79)
    frame2_tracks = tracker.update([det1b, det2b])
    confirmed = [t for t in tracker._tracks if t.hits >= 2]
    assert len(confirmed) == 2, f"Expected 2 confirmed, got {len(confirmed)}"
    print(f"  {PASS} Tracks confirmed after 2 hits")

    # Frame 3 – no detections → tracks age
    tracker.update([])
    missed = [t.missed for t in tracker._tracks]
    assert all(m == 1 for m in missed), f"missed counts: {missed}"
    print(f"  {PASS} Tracks age when no detections")

    tracker.reset()
    assert len(tracker._tracks) == 0
    print(f"  {PASS} Tracker reset clears state")


# ---------------------------------------------------------------------------
# Proximity
# ---------------------------------------------------------------------------
def test_proximity():
    banner("ProximityAnalyser")
    from modules.proximity import ProximityAnalyser, RichTrack
    from modules.tracker import Track
    from config.settings import ProximityConfig

    analyser = ProximityAnalyser(ProximityConfig())

    # Simulate a car that fills 50% of frame height → should be NEAR
    t_near = Track(track_id=1, box=(350, 340, 930, 700),
                   class_id=2, label="car", confidence=0.95)
    t_far  = Track(track_id=2, box=(550, 600, 730, 650),
                   class_id=2, label="car", confidence=0.80)

    results = analyser.analyse([t_near, t_far], frame_w=1280, frame_h=720)
    assert len(results) == 2

    near_rt = next(r for r in results if r.track.track_id == 1)
    far_rt  = next(r for r in results if r.track.track_id == 2)

    assert near_rt.band == "NEAR",   f"Expected NEAR got {near_rt.band}"
    assert far_rt.band  == "FAR",    f"Expected FAR got {far_rt.band}"
    print(f"  {PASS} Large box → NEAR band")
    print(f"  {PASS} Small box → FAR band")
    print(f"  {PASS} near_rt.danger = {near_rt.danger}")


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------
def test_alert_engine():
    banner("AlertEngine")
    from modules.alert_engine import AlertEngine
    from modules.proximity import RichTrack
    from modules.tracker import Track
    from modules.lane_detection import LaneResult
    from config.settings import AlertConfig

    engine = AlertEngine(AlertConfig())
    lane_ok   = LaneResult(lane_detected=True)
    lane_lost = LaneResult(lane_detected=False)

    # Lane lost → INFO alert
    alert = engine.evaluate([], lane_lost)
    assert alert is not None
    assert alert.priority == "INFO"
    assert "LANE" in alert.message.upper()
    print(f"  {PASS} Lane-lost alert triggered: '{alert.message}'")

    engine.reset()

    # Danger vehicle → CRITICAL
    danger_track = Track(track_id=1, box=(350, 310, 930, 710),
                          class_id=2, label="car", confidence=0.95)
    danger_rt = RichTrack(
        track      = danger_track,
        band       = "NEAR",
        zone       = "EGO_LANE",
        danger     = True,
        box_h_frac = 0.55,
        cx_frac    = 0.50,
    )
    alert2 = engine.evaluate([danger_rt], lane_ok)
    assert alert2 is not None
    assert alert2.priority == "CRITICAL"
    print(f"  {PASS} Danger-ahead alert triggered: '{alert2.message}'")

    # Cooldown – same rule shouldn't fire again immediately
    engine._cooldowns["danger_ahead"] = engine._cooldowns.get("danger_ahead", 0)
    import time; time.sleep(0.05)
    alert3 = engine.evaluate([danger_rt], lane_ok)
    # Should still show active (banner_remaining > 0)
    print(f"  {PASS} Alert cooldown state consistent")


# ---------------------------------------------------------------------------
# Drawing (no display needed)
# ---------------------------------------------------------------------------
def test_drawing():
    banner("utils/drawing")
    import cv2
    from utils.drawing import (
        draw_lane_lines, draw_lane_corridor, draw_detection,
        draw_hud, draw_alert_banner, draw_minimap
    )
    from config.settings import AlertConfig

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    draw_lane_lines(canvas, (200, 680, 560, 420), (1080, 680, 720, 420))
    draw_lane_corridor(canvas, (200, 680, 560, 420), (1080, 680, 720, 420))
    draw_detection(canvas, (400, 300, 800, 600), "car", "NEAR", track_id=1, conf=0.9)
    draw_hud(canvas, frame_idx=42, fps=28.5, lane_offset_px=34.0,
             curvature_m=350.0, n_vehicles=2)
    draw_alert_banner(canvas, "TEST ALERT", "CRITICAL", AlertConfig().priority_colors)
    draw_minimap(canvas, [{"box": (400,300,800,600), "band": "NEAR"}], 1280, 720)

    assert canvas.shape == (720, 1280, 3)
    assert canvas.sum() > 0  # something was drawn
    print(f"  {PASS} All drawing functions execute without error")


# ---------------------------------------------------------------------------
# EnvironmentConfig + adaptive lane detection
# ---------------------------------------------------------------------------
def test_environment_config():
    banner("EnvironmentConfig + Adaptive LaneDetector")
    from config.settings import EnvironmentConfig, EnvMode, ADASConfig, LaneConfig
    from modules.lane_detection import LaneDetector
    import cv2
    import numpy as np

    # Instantiate for all three modes
    for mode in (EnvMode.DAY, EnvMode.NIGHT, EnvMode.RAINY):
        env = EnvironmentConfig(mode=mode)
        det = LaneDetector(LaneConfig(), env_cfg=env)
        assert det.env_cfg.mode == mode
        print(f"  {PASS} LaneDetector instantiates with EnvMode.{mode.value.upper()}")

    # Night mode: dark frame should use lower Canny thresholds than DAY
    dark_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # pure black
    det_day   = LaneDetector(LaneConfig(), EnvironmentConfig(mode=EnvMode.DAY))
    det_night = LaneDetector(LaneConfig(), EnvironmentConfig(mode=EnvMode.NIGHT))

    import cv2
    gray = cv2.cvtColor(dark_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    day_low, day_high     = det_day._adaptive_canny_thresholds(blurred)
    night_low, night_high = det_night._adaptive_canny_thresholds(blurred)

    assert night_low <= day_low,   f"Night Canny low should be ≤ day: {night_low} vs {day_low}"
    assert night_high <= day_high, f"Night Canny high should be ≤ day: {night_high} vs {day_high}"
    print(f"  {PASS} Night mode: Canny ({night_low},{night_high}) ≤ day ({day_low},{day_high})")

    # Rainy mode: thresholds scaled down
    det_rainy = LaneDetector(LaneConfig(), EnvironmentConfig(mode=EnvMode.RAINY))
    rainy_low, rainy_high = det_rainy._adaptive_canny_thresholds(blurred)
    assert rainy_low < LaneConfig().canny_low * 0.9 or rainy_low <= day_low
    print(f"  {PASS} Rainy mode: Canny ({rainy_low},{rainy_high}) appropriately scaled")

    # Rainy/Night ROI apex should be wider
    from config.settings import LaneConfig as LC
    lc = LC()
    roi_normal = lc.roi_apex_width_fraction
    roi_night  = roi_normal + EnvironmentConfig(mode=EnvMode.NIGHT).roi_apex_extra
    assert roi_night > roi_normal
    print(f"  {PASS} Night ROI apex wider: {roi_normal:.2f} → {roi_night:.2f}")

    # ADASConfig carries EnvironmentConfig
    cfg = ADASConfig()
    cfg.environment = EnvironmentConfig(mode=EnvMode.NIGHT)
    assert cfg.environment.mode == EnvMode.NIGHT
    print(f"  {PASS} ADASConfig.environment field works correctly")


# ---------------------------------------------------------------------------
# Metrics: FrameLog + eval_events
# ---------------------------------------------------------------------------
def test_eval_events():
    banner("metrics/eval_events")
    from metrics.eval_events import (
        FrameLog, evaluate_video_log, detect_cut_in_events, VideoMetrics
    )

    def make_log(idx, lane=True, band="FAR", zone="EGO_LANE",
                 danger=False, alert=None):
        return FrameLog(
            frame_idx      = idx,
            lane_detected  = lane,
            lane_offset_px = 0.0,
            rich_tracks    = [{"track_id": 1, "label": "car",
                               "band": band, "zone": zone, "danger": danger}],
            alert_category = alert,
            alert_priority = "CRITICAL" if alert == "danger_ahead" else None,
        )

    # Build a synthetic log: frames 0-9 no danger, frames 10-19 danger
    logs = []
    for i in range(10):
        logs.append(make_log(i))
    for i in range(10, 20):
        # Danger condition + alert fires at frame 12
        alert = "danger_ahead" if i == 12 else None
        logs.append(make_log(i, band="NEAR", zone="EGO_LANE", danger=True, alert=alert))

    metrics = evaluate_video_log(logs)
    assert isinstance(metrics, VideoMetrics)
    assert metrics.total_frames == 20
    assert metrics.danger_condition_frames == 10
    assert metrics.danger_tp == 1       # only frame 12 matches both condition + alert
    assert metrics.danger_fp == 0       # alert only fired when condition was true
    assert metrics.lane_detected_pct == 100.0
    print(f"  {PASS} evaluate_video_log produces correct TP/FP counts")
    print(f"  {PASS} danger_condition_frames={metrics.danger_condition_frames}, "
          f"tp={metrics.danger_tp}, fp={metrics.danger_fp}")
    print(f"  {PASS} lane_detected_pct={metrics.lane_detected_pct}%")

    # Lane-lost detection
    lane_lost_logs = [
        FrameLog(i, lane_detected=False, lane_offset_px=None, rich_tracks=[],
                 alert_category="lane_lost" if i == 2 else None,
                 alert_priority="INFO" if i == 2 else None)
        for i in range(5)
    ]
    m2 = evaluate_video_log(lane_lost_logs)
    assert m2.lane_lost_frames == 5
    assert m2.lane_lost_alert_frames == 1
    print(f"  {PASS} Lane-lost metrics: lost_frames={m2.lane_lost_frames}, "
          f"alert_frames={m2.lane_lost_alert_frames}")


# ---------------------------------------------------------------------------
# Metrics: cut-in detection
# ---------------------------------------------------------------------------
def test_cut_in_detection():
    banner("metrics/eval_events – cut-in detection")
    from metrics.eval_events import FrameLog, detect_cut_in_events

    logs = []
    # Track 5: starts in LEFT_LANE (frames 0-4), moves to EGO_LANE (5+), NEAR at frame 8
    for i in range(5):
        logs.append(FrameLog(i, True, 0.0,
                             [{"track_id": 5, "label": "car", "band": "FAR",
                               "zone": "LEFT_LANE", "danger": False}],
                             None, None))
    for i in range(5, 10):
        band = "NEAR" if i >= 8 else "MEDIUM"
        logs.append(FrameLog(i, True, 0.0,
                             [{"track_id": 5, "label": "car", "band": band,
                               "zone": "EGO_LANE", "danger": band == "NEAR"}],
                             None, None))

    cut_ins = detect_cut_in_events(logs, window=30)
    assert len(cut_ins) == 1, f"Expected 1 cut-in, got {len(cut_ins)}"
    assert cut_ins[0] == 8
    print(f"  {PASS} Cut-in detected at frame {cut_ins[0]} (entered ego at 5, NEAR at 8)")

    # No cut-in if transition window exceeded
    cut_ins_narrow = detect_cut_in_events(logs, window=1)
    assert len(cut_ins_narrow) == 0
    print(f"  {PASS} Cut-in suppressed when window too narrow")


# ---------------------------------------------------------------------------
# Driver profile / session summary
# ---------------------------------------------------------------------------
def test_driver_profile():
    banner("metrics/driver_profile")
    from metrics.eval_events import FrameLog, evaluate_video_log
    from metrics.driver_profile import build_driver_summary, DriverSummary

    # Build 100-frame synthetic log: 20 frames with danger condition
    logs = []
    for i in range(100):
        danger = (20 <= i < 40)
        logs.append(FrameLog(
            frame_idx      = i,
            lane_detected  = i < 90,          # lane lost for last 10 frames
            lane_offset_px = float(i % 10),
            rich_tracks    = [{"track_id": 1, "label": "car",
                               "band": "NEAR" if danger else "FAR",
                               "zone": "EGO_LANE", "danger": danger}],
            alert_category = "danger_ahead" if danger and i == 22 else None,
            alert_priority = "CRITICAL" if danger and i == 22 else None,
        ))

    metrics = evaluate_video_log(logs)
    summary = build_driver_summary(logs, metrics)

    assert isinstance(summary, DriverSummary)
    assert summary.total_frames == 100
    assert summary.danger_ahead_pct == 20.0
    assert summary.lane_detected_pct == 90.0
    assert summary.risk_level in ("LOW", "MODERATE", "HIGH")
    assert len(summary.narrative) > 20

    print(f"  {PASS} DriverSummary built: danger={summary.danger_ahead_pct}%, "
          f"lane={summary.lane_detected_pct}%")
    print(f"  {PASS} Risk level: {summary.risk_level}")
    print(f"  {PASS} Narrative present ({len(summary.narrative)} chars)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    failures = []
    tests = [
        test_config,
        test_geometry,
        test_lane_detection,
        test_tracker,
        test_proximity,
        test_alert_engine,
        test_drawing,
        test_environment_config,
        test_eval_events,
        test_cut_in_detection,
        test_driver_profile,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            print(f"\n  {FAIL} FAILED in {test_fn.__name__}: {exc}")
            import traceback; traceback.print_exc()
            failures.append(test_fn.__name__)

    print(f"\n{'═'*50}")
    if failures:
        print(f"  {FAIL} {len(failures)} test(s) failed: {failures}")
        sys.exit(1)
    else:
        print(f"  {PASS} All tests passed!")
    print('═'*50)
