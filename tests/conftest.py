import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

# The printed 2x2 board, whatever board.json says about your rig.
os.environ["TELLO_BOARD"] = "default"
