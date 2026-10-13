"""Frozen entry point: VoxShift.exe (windowed desktop app)."""
import sys

from voxshift.gui import main

main(sys.argv[1:])
