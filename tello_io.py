"""
Drone and video I/O shared by every tool: open the Tello, and turn what it
streams into the BGR frames OpenCV expects.

Two traps in djitellopy 2.5 that this module exists to close once:

  RGB, not BGR. BackgroundFrameRead builds frames with
      np.array(frame.to_image()) - a PIL image, so channel order is RGB.
      Every OpenCV call in this repo assumes BGR. The recorded flights show
      it: faces came out blue. Detection mostly survives (it is greyscale),
      but colour display, saved calibration frames and recordings do not.

  No None before the first frame. The reader starts out holding a black
      300x400 placeholder, so `if frame is None` never fires and the first
      seconds of "video" are a blank image of the wrong size.
"""

import time

import cv2
import numpy as np

# djitellopy >= 2.5 decodes through PIL -> RGB. Set False for older
# versions that used to_ndarray(format="bgr24").
DRONE_FRAMES_ARE_RGB = True

PLACEHOLDER_SHAPE = (300, 400, 3)


def open_drone(stream=True):
    """Connected Tello, video on. Nothing here takes off."""
    from djitellopy import Tello
    drone = Tello()
    drone.connect()
    if stream:
        drone.streamon()
    return drone


def is_placeholder(frame):
    return (frame is None or frame.shape == PLACEHOLDER_SHAPE
            and not frame.any())


def to_bgr(frame):
    """Drone frame -> BGR, or None for the startup placeholder."""
    if is_placeholder(frame):
        return None
    if DRONE_FRAMES_ARE_RGB and frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


class FrameGrabber:
    """Latest frame from a drone's reader, as BGR, with a flag for whether
    it is new since the last call. A control loop running slower than the
    video must not feed the same frame to a filter twice, and one running
    faster must not stall on a frame that has not arrived."""

    def __init__(self, drone):
        self.reader = drone.get_frame_read()
        self._last_raw = None
        self._last = None
        self.n_new = 0

    def grab(self):
        raw = self.reader.frame
        if raw is self._last_raw:
            return self._last, False
        self._last_raw = raw
        self._last = to_bgr(raw)
        if self._last is None:
            return None, False
        self.n_new += 1
        return self._last, True

    def peek(self):
        """The newest frame as BGR without marking it as seen - for a
        display or recorder running alongside the loop that consumes them."""
        return to_bgr(self.reader.frame)


def tello_frames(drone, stop=None, poll_s=0.005):
    """Yield each new BGR frame from a connected, streaming drone."""
    g = FrameGrabber(drone)
    while stop is None or not stop.is_set():
        f, new = g.grab()
        if new:
            yield f
        else:
            time.sleep(poll_s)


def file_frames(path, loop=False, stop=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    try:
        while stop is None or not stop.is_set():
            ok, f = cap.read()
            if not ok:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            yield f
    finally:
        cap.release()


def black_frame(size=(960, 720)):
    return np.zeros((size[1], size[0], 3), np.uint8)


_RING = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def put_text(img, s, org, scale, colour, thick=1, line_type=cv2.LINE_8,
             outline=(0, 0, 0)):
    """Text with a dark outline, readable over any picture.

    The usual trick - the text in black at thickness 4, then in colour at
    thickness 1 - only lines up while thickness leaves the glyphs' width
    alone. OpenCV 5 draws thicker text wider (a 30-character line grows
    ~7 px), so the black copy drifted off the coloured one and every
    overlay read as a doubled smear. The outline here is the same thin
    text, drawn one pixel off in each direction."""
    x, y = org
    for dx, dy in _RING:
        cv2.putText(img, s, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, outline,
                    thick, line_type)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, line_type)
