from voxshift.presets import SMOOTHING, load_spec, smoothing_overrides


def test_off_keeps_voice_settings():
    spec = {"block_time": 0.2, "extra_time_right": 0.02}
    assert smoothing_overrides(spec, "off") == {"block_time": 0.2}


def test_levels_only_add_delay():
    spec = load_spec("ai-realtime")
    prev_block, prev_right = spec["block_time"], spec["extra_time_right"]
    for level in list(SMOOTHING)[1:]:
        o = smoothing_overrides(spec, level)
        assert o["block_time"] > prev_block
        assert o["extra_time_right"] >= prev_right
        prev_block, prev_right = o["block_time"], o["extra_time_right"]


def test_unknown_level_falls_back_to_off():
    spec = {"block_time": 0.5}
    assert smoothing_overrides(spec, "bogus") == {"block_time": 0.5}
