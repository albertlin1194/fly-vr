import pytest
from audio.stimuli import SinStim, AudioStimPlaylist
from two_photon.two_photon_control import TwoPhotonController


@pytest.fixture
def stim1():
    return SinStim(frequency=230, amplitude=2.0, phase=0.0, sample_rate=40000,
                   duration=200, intensity=1.0, pre_silence=0, post_silence=0,
                   attenuator=None)

@pytest.fixture
def stim2():
    return SinStim(frequency=330, amplitude=2.0, phase=0.0, sample_rate=40000,
                   duration=200, intensity=1.0, pre_silence=0, post_silence=0,
                   attenuator=None)

@pytest.fixture
def stim3():
    return SinStim(frequency=430, amplitude=2.0, phase=0.0, sample_rate=40000,
                   duration=200, intensity=1.0, pre_silence=0, post_silence=0,
                   attenuator=None)


def test_two_photon_control(stim1, stim2, stim3):
    two_photon_control = TwoPhotonController(start_channel_name="line0/port0",
                                             stop_channel_name="line0/por1",
                                             next_file_channel_name="line0/port2",
                                             num_samples=50)

    stims = [stim1, stim2, stim3]
    stimList = AudioStimPlaylist(stims, shuffle_playback=False)
    for stim in stimList.stims:
        stim.next_event_callbacks = two_photon_control.make_next_signal_callback()

    playGen = stimList.data_generator()
    play2P = two_photon_control.data_generator()

    playGen.next()
    data = play2P.next().data
    assert( (two_photon_control.start_signal.data == data).all() )

    data = play2P.next().data
    assert ((two_photon_control.zero_signal.data == data).all())

    playGen.next()
    data = play2P.next().data
    assert ((two_photon_control.next_signal.data == data).all())

    playGen.next()
    data = play2P.next().data
    assert ((two_photon_control.next_signal.data == data).all())

