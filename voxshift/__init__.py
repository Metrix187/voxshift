"""VoxShift — local real-time voice changer."""
import os

# numpy's OpenBLAS reserves a work buffer per CPU thread at import: ~1.4 GB of
# commit on a 24-thread laptop that the audio code never uses (it is single-
# threaded and FFT-bound). Two threads keep matrix maths quick at ~160 MB.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

__version__ = "0.2.2"
