"""
Interactive Tello command line - send any SDK command by hand.

    python tello_cli.py                         # interactive shell
    python tello_cli.py --exec "takeoff; up 50; land"
    python tello_cli.py --keys                  # manual flight override
    python tello_cli.py --keys --pose           # + live board pose readout

Anything you type that is not a dot-command is sent verbatim to the drone,
so undocumented commands work too (downvision 1, EXT ..., and so on).

Dot-commands are local, never sent:
    .help  .state  .watch  .keys  .quit

WHY THE LOCK: djitellopy has no send lock and pops responses off a shared
deque, so a keepalive thread and your typed command will steal each other's
replies. Every send here goes through self.lock.
"""

import argparse
import threading
import time

try:
    import readline  # noqa: F401  (history + editing on Linux/macOS)
except ImportError:
    try:
        import pyreadline3  # noqa: F401  (pip install pyreadline3 on Windows)
    except ImportError:
        pass

COMMON = [
    "command", "takeoff", "land", "emergency", "streamon", "streamoff",
    "up", "down", "left", "right", "forward", "back", "cw", "ccw", "flip",
    "go", "curve", "rc", "speed", "stop", "keepalive", "motoron", "motoroff",
    "throwfly", "downvision", "reboot",
    "speed?", "battery?", "time?", "wifi?", "sdk?", "sn?",
    "height?", "temp?", "attitude?", "baro?", "tof?", "acceleration?",
]


class TelloShell:
    def __init__(self, keepalive=True, timeout=7):
        from tello_io import open_drone
        self.t = open_drone(stream=False)
        self.lock = threading.Lock()
        self.timeout = timeout
        self._ka_mode = "keepalive"
        self._stop = threading.Event()
        self.airborne = False
        if keepalive:
            self.ka = threading.Thread(target=self._keepalive_loop, daemon=True)
            self.ka.start()

    # -- sending ------------------------------------------------------
    def send(self, cmd, timeout=None):
        """Raw passthrough. Returns the drone's reply as a string."""
        cmd = cmd.strip()
        if not cmd:
            return ""
        with self.lock:
            if cmd.startswith("rc "):
                # rc is fire-and-forget; the drone sends no reply, so waiting
                # for one would just burn the timeout.
                self.t.send_command_without_return(cmd)
                return "(no reply expected)"
            to = timeout or (20 if cmd == "takeoff" else self.timeout)
            r = self.t.send_command_with_return(cmd, timeout=to)
        if cmd == "takeoff" and "ok" in str(r).lower():
            self.airborne = True
        if cmd in ("land", "emergency"):
            self.airborne = False
        return r

    def _keepalive_loop(self):
        """The drone auto-lands after 15 s of silence. Ping every 10."""
        while not self._stop.wait(10.0):
            try:
                if self._ka_mode == "keepalive":
                    r = self.send("keepalive")
                    if "ok" not in str(r).lower():
                        # SDK 2.0 only; older firmware rejects it. A read
                        # command resets the same watchdog and is harmless.
                        self._ka_mode = "battery?"
                        print("\n(keepalive unsupported, using battery? instead)")
                else:
                    self.send("battery?")
            except Exception as e:
                print(f"\n(keepalive failed: {e})")

    # -- info ---------------------------------------------------------
    def state_line(self):
        s = self.t.get_current_state()
        if not s:
            return "(no state packets yet)"
        keys = ["bat", "h", "tof", "pitch", "roll", "yaw",
                "vgx", "vgy", "vgz", "templ", "temph", "time"]
        return "  ".join(f"{k}={s[k]}" for k in keys if k in s)

    def close(self, land=True):
        self._stop.set()
        try:
            if land and self.airborne:
                print("landing...")
                self.send("rc 0 0 0 0")
                self.send("land")
        except Exception:
            pass
        try:
            self.t.end()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Manual flight override
# ----------------------------------------------------------------------

KEYS_HELP = """
  w/s  forward/back      a/d  left/right
  r/f  up/down           q/e  yaw left/right
  SPACE  stop (hover)    t  takeoff    l  land
  [ / ]  speed -/+       ESC  land and quit

Velocities are STICKY: a key sets it until you change it or press SPACE.
Key-repeat is unreliable across platforms, so held keys are not detected.
"""


def keys_mode(sh, speed=30, pose=False):
    import cv2
    import numpy as np

    from tello_io import FrameGrabber, put_text

    est = tracker = None
    if pose:
        from board_config import load_board
        from tello_pose import BoardTracker, PoseEstimator, load_calibration
        K, dist, real = load_calibration()
        est = PoseEstimator(K, dist, load_board())
        tracker = BoardTracker(est)
        if not real:
            print("(untrusted intrinsics - pose scale is approximate)")

    sh.send("streamon")
    grab = FrameGrabber(sh.t)
    print(KEYS_HELP)
    a = b = c = d = 0

    try:
        while True:
            frame, _ = grab.grab()          # BGR; None until the stream starts
            if frame is None:
                sh.send(f"rc {a} {b} {c} {d}")
                if cv2.waitKey(30) & 0xFF == 27:
                    break
                continue
            view = frame.copy()
            if est is not None:
                corners, ids, _ = est.detect(frame)
                imu = (sh.t.get_current_state() or {}).get("yaw")
                p = tracker.update(detection=(corners, ids), imu_yaw=imu)
                view = est.draw(view, p, corners, ids)
            hud = f"rc a{a:+4d} b{b:+4d} c{c:+4d} d{d:+4d}  speed {speed}"
            h = view.shape[0]
            put_text(view, hud, (12, h - 18), 0.6, (255, 255, 0))
            cv2.imshow("tello manual - ESC lands", view)

            k = cv2.waitKey(30) & 0xFF
            if k == 27:                       # ESC
                break
            elif k == ord(" "):
                a = b = c = d = 0
            elif k == ord("w"):
                b = speed
            elif k == ord("s"):
                b = -speed
            elif k == ord("a"):
                a = -speed
            elif k == ord("d"):
                a = speed
            elif k == ord("r"):
                c = speed
            elif k == ord("f"):
                c = -speed
            elif k == ord("q"):
                d = -speed
            elif k == ord("e"):
                d = speed
            elif k == ord("["):
                speed = max(10, speed - 10)
            elif k == ord("]"):
                speed = min(100, speed + 10)
            elif k == ord("t"):
                print(sh.send("takeoff"))
            elif k == ord("l"):
                a = b = c = d = 0
                sh.send("rc 0 0 0 0")
                print(sh.send("land"))

            sh.send(f"rc {a} {b} {c} {d}")
    finally:
        try:
            sh.send("rc 0 0 0 0")
        except Exception:
            pass
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# Shell
# ----------------------------------------------------------------------

def watch(sh, seconds=None):
    print("(ctrl-c to stop)")
    t0 = time.time()
    try:
        while seconds is None or time.time() - t0 < seconds:
            print("\r" + sh.state_line()[:150].ljust(150), end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print()


def interactive(sh, keys_speed=30, pose=False):
    print("type a Tello command, or .help    (ctrl-d / .quit to exit)")
    print(f"battery {sh.t.get_battery()}%")
    while True:
        try:
            line = input("tello> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("."):
            cmd = line[1:].split()[0].lower() if len(line) > 1 else ""
            if cmd in ("quit", "q", "exit"):
                break
            elif cmd == "help":
                print(__doc__)
                print("known commands:", " ".join(COMMON))
            elif cmd == "state":
                print(sh.state_line())
            elif cmd == "watch":
                watch(sh)
            elif cmd == "keys":
                keys_mode(sh, keys_speed, pose)
            else:
                print(f"unknown dot-command: {line}")
            continue
        try:
            print(sh.send(line))
        except Exception as e:
            print(f"error: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="send Tello SDK commands by hand")
    ap.add_argument("--exec", dest="script",
                    help="semicolon-separated commands, then exit")
    ap.add_argument("--keys", action="store_true",
                    help="manual flight with live video")
    ap.add_argument("--pose", action="store_true",
                    help="overlay board pose in --keys mode")
    ap.add_argument("--speed", type=int, default=30,
                    help="rc magnitude for --keys (default 30)")
    ap.add_argument("--no-keepalive", action="store_true",
                    help="do not ping; drone auto-lands after 15 s idle")
    ap.add_argument("--no-land-on-exit", action="store_true",
                    help="leave it flying when the CLI exits (risky)")
    args = ap.parse_args()

    sh = TelloShell(keepalive=not args.no_keepalive)
    try:
        if args.script:
            for c in args.script.split(";"):
                c = c.strip()
                if c:
                    print(f"{c:20s} -> {sh.send(c)}")
        elif args.keys:
            keys_mode(sh, args.speed, args.pose)
        else:
            interactive(sh, args.speed, args.pose)
    finally:
        sh.close(land=not args.no_land_on_exit)


if __name__ == "__main__":
    main()
