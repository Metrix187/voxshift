import threading
import time

import numpy as np

from voxshift.ai.onnx_backend import OnnxSeedVCModel


class FakeSession:
    """Records how many threads are inside run() at once."""

    def __init__(self):
        self.inside = self.peak = 0
        self.guard = threading.Lock()

    def run(self, outputs, feeds):
        with self.guard:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
        time.sleep(0.01)
        with self.guard:
            self.inside -= 1
        return [feeds["x"]]


def test_runs_are_serialised_across_threads():
    # several processors share one model (e.g. a voice reloading while the old one still
    # converts); DirectML must never see two threads in Run at once
    sess = FakeSession()
    m = OnnxSeedVCModel.__new__(OnnxSeedVCModel)
    m.sess = {"dit_step": sess}
    m._run_lock = threading.Lock()

    def work():
        for _ in range(5):
            m._run("dit_step", x=np.zeros(4))

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sess.peak == 1
