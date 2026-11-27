from dex_frax.resources import GIB, reclaim_idle


class Slot:
    def __init__(self, pid):
        self.pid = pid
        self.telemetry = {'cache_state': 'jax_hot'}


def test_drained_batch_reclaims_idle_worker_while_preserving_active_workers():
    running = Slot(1)
    idle = Slot(2)
    usage = {running: 3 * GIB, idle: 2 * GIB}
    snapshot = {'available': 20 * GIB, 'cgroup_headroom': 20 * GIB}

    assert reclaim_idle([running, idle], {running}, usage, snapshot, 6 * GIB) is None
    assert reclaim_idle(
        [running, idle], {running}, usage, snapshot, 6 * GIB, queue_empty=True
    ) is idle
    assert reclaim_idle(
        [running], {running}, usage, snapshot, 6 * GIB, queue_empty=True
    ) is None
