from voxshift.audio.cable import find_cables


def dev(name, i, o):
    return {"name": name, "max_input_channels": i, "max_output_channels": o}


def test_detects_pairs_and_prefers_vbcable():
    devices = [
        dev("Speakers (Realtek(R) Audio)", 0, 2),
        dev("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", 0, 8),
        dev("VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)", 8, 0),
        dev("CABLE Input (VB-Audio Virtual Cable)", 0, 8),
        dev("CABLE Output (VB-Audio Virtual Cable)", 8, 0),
    ]
    cables = find_cables(devices=devices)
    assert [c.label for c in cables] == ["VB-CABLE", "VoiceMeeter"]
    assert cables[0].playback == "CABLE Input (VB-Audio Virtual Cable)"
    assert cables[0].capture == "CABLE Output (VB-Audio Virtual Cable)"


def test_half_a_cable_is_not_a_cable():
    # a capture endpoint without its playback side cannot carry our audio
    assert find_cables(devices=[dev("CABLE Output (VB-Audio Virtual Cable)", 8, 0)]) == []


def test_cable_a_is_not_mistaken_for_cable():
    devices = [dev("CABLE-A Input (VB-Audio Cable A)", 0, 8), dev("CABLE-A Output (VB-Audio Cable A)", 8, 0)]
    assert [c.label for c in find_cables(devices=devices)] == ["VB-CABLE A"]
