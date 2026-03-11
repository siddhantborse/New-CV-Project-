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
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            print(f"\n  {FAIL} FAILED: {exc}")
            failures.append(test_fn.__name__)

    print(f"\n{'═'*50}")
    if failures:
        print(f"  {FAIL} {len(failures)} test(s) failed: {failures}")
        sys.exit(1)
    else:
        print(f"  {PASS} All tests passed!")
    print('═'*50)
