import re
import abc
import uuid
import os.path
import logging

import h5py
import numpy as np
import scipy
from scipy import io
from scipy import signal

from flyvr.audio.signal_producer import SignalProducer, SampleChunk, MixedSignal
from flyvr.common import Randomizer


class AudioStim(SignalProducer, metaclass=abc.ABCMeta):
    """
    The AudioStim class is base class meant to encapsulate common functionality and implementation details found in
    any audio stimulus. Mainly, this includes the generation and storage of audio signal data, basic information about
    the stimulus, inclusion of pre and post silence signals to the main signal, etc. If you wish to add new audio
    stimulus functionality you should create a new class that inherits from this class and implements its abstract
    methods.
    """

    NAME = None

    def __init__(self, sample_rate, duration, intensity=1.0, pre_silence=0, post_silence=0, attenuator=None,
                 frequency=None, max_value=10.0, min_value=-10.0, next_event_callbacks=None, identifier=None):
        """
        Create an audio stimulus object that encapsulates the generation of the underlying audio
        data.
        :param int sample_rate: The sample rate in Hz of the underlying audio data.
        :param int duration: The duration of the sound in milliseconds
        :param int pre_silence: The duration (in milliseconds) of silence to add to the start of the signal.
        :param int post_silence: The duration (in milliseconds) of silence to add to the end of the signal.
        :param list next_event_callbacks: A list of control functions to call whenever the generator produced by this
        class yields a value.
        """

        # Attach event next callbacks to this object, since it is a signal producer
        super(AudioStim, self).__init__(next_event_callbacks)

        self.__sample_rate = sample_rate
        self.__duration = duration
        self.__pre_silence = pre_silence
        self.__post_silence = post_silence
        self.__attenuator = attenuator
        self.__frequency = frequency
        self.__intensity = intensity
        self.__max_value = max_value
        self.__min_value = min_value
        self.__identifier = identifier or ('%s-%s' % (self.__class__.__name__, uuid.uuid4().hex))

        # How many samples have been generated by calls to data_generator() iterators
        self.num_samples_generated = 0

        # Setup a dictionary for the parameters of the stimulus. We will send this as part
        # of an event message to the next_event_callbacks
        self.event_message = {"name": type(self).__name__,
                              "sample_rate": sample_rate, "duration": duration, "pre_silence": pre_silence,
                              "post_silence": post_silence, "frequency": frequency, "intensity": intensity,
                              "max_clamp_value": max_value, "min_clamp_value": min_value}
        if attenuator is None:
            self.event_message["attenuation"] = None
        else:
            self.event_message["attenuation"] = np.column_stack((attenuator.frequencies, attenuator.factors))

        # Initialize data to null
        self.__data = []

    def _gen_silence(self, silence_duration):
        """
        Little helper function to generate silence data.

        :param int silence_duration: Amount of silence to generate in milliseconds.
        :return: The silence signal.
        :rtype: numpy.ndarray
        """
        return np.zeros(int(np.ceil((silence_duration / 1000.0) * self.sample_rate)))

    def _add_silence(self, data):
        """
        A helper function to add pre and post silence to a generated signal.

        :param numpy.ndarray data: The data to add silence to.
        :return: The data with silence added to its start and end.
        :rtype: numpy.ndarray
        """
        return np.concatenate([self._gen_silence(self.pre_silence), data, self._gen_silence(self.post_silence)])

    @abc.abstractmethod
    def _generate_data(self):
        """
        Generate any internal data necessary for the stimulus, called when parameters
        are changed only. This is so we don't have to keep generating the data with every
        call to get_data. This method should be overloaded in any sub-class to generate the
        actual audio data.

        :return: The data representing the stimulus.
        :rtype: numpy.ndarray
        """

    @property
    def identifier(self):
        return self.__identifier

    @property
    def data(self):
        """
        Get the voltage signal data associated with this stimulus.

        :return: A 1D numpy.ndarray of data that can be passed directly to the DAQ.
        :rtype: numpy.ndarray
        """
        return self.__data

    @data.setter
    def data(self, data):
        """
        Set the data for the audio stimulus directly. This function will add any required silence
        as a side-effect. Sub-classes of AudioStim should use this setter.

        :param numpy.ndarray data: The raw audio signal data representing this stimulus.
        """

        # If the user provided an attenuator, attenuate the signal
        if self.__attenuator is not None:
            data = self.__attenuator.attenuate(data, self.__frequency)

        data = self._add_silence(data)

        # Multiply the signal by the intensity factor
        data = data * self.__intensity

        self.__data = data

        # Perform limit check on data, make sure we are not exceeding
        if data.max() > self.__max_value:
            raise ValueError("Audio stimulus value exceeded max level (%r, %s vs %s)" % (self,
                                                                                         data.max(),
                                                                                         self.__max_value))

        if data.min() < self.__min_value:
            raise ValueError("Audio stimulus value lower than min level (%r, %s vs %s)" % (self,
                                                                                           data.min(),
                                                                                           self.__min_value))

    def describe(self):
        return {'name': self.NAME,
                'sample_rate': self.__sample_rate,
                'duration': self.__duration,
                'intensity': self.__intensity,
                'pre_silence': self.__pre_silence,
                'post_silence': self.__post_silence,
                'attenuator': self.__attenuator,
                'frequency': self.__frequency,
                'max_value': self.__max_value,
                'min_value': self.__min_value}

    def data_generator(self):
        """
        Return a generator that yields the data member when next is called on it. Simply provides another interface to
        the same data stored in the data member.

        :return: A generator that yields an array containing the sample data.
        """
        while True:
            self.num_samples_generated = self.num_samples_generated + self.data.shape[0]
            # self.trigger_next_callback(message_data=self.event_message, num_samples=self.data.shape[0])

            yield SampleChunk(data=self.data, producer_id=self.producer_id)

    @property
    def sample_rate(self):
        """
        Get the sample rate of the audio stimulus in Hz.

        :return: The sample rate of the audio stimulus in Hz.
        :rtype: int
        """
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        """
        Set the sample rate of the audio stimulus in Hz.

        :param int sample_rate: The sample rate of the audio stimulus in Hz.
        """
        self.__sample_rate = sample_rate
        self.data = self._generate_data()

    @property
    def duration(self):
        """
        Get the duration of the audio signal in milliseconds.

        :return: The duration of the audio signal in milliseconds.
        :rtype: int
        """
        return self.__duration

    @duration.setter
    def duration(self, duration):
        """
        Set the duration of the audio signal in milliseconds.

        :param int duration: The duration of the audio signal in milliseconds.
        """
        self.__duration = duration
        self.data = self._generate_data()

    @property
    def intensity(self):
        """
        Get the intensity of the audio signal. This is a scalar multiplicative factor of the signal.

        :return: A scalar multiplicative factor of the signal.
        :rtype: double
        """
        return self.__intensity

    @intensity.setter
    def intensity(self, intensity):
        """
         Set the intensity of the audio signal. This is a scalar multiplicative factor of the signal.

        :param double intensity: A scalar multiplicative factor of the signal.
        """
        self.__intensity = intensity
        self.data = self._generate_data()

    @property
    def pre_silence(self):
        """
        Get the amount (in milliseconds) of pre-silence added to the audio signal.

        :return: Get the amount (in milliseconds) of pre-silence added to the audio signal.
        :rtype: int
        """
        return self.__pre_silence

    @pre_silence.setter
    def pre_silence(self, pre_silence):
        """
        Set the amount (in milliseconds) of pre-silence added to the audio signal.

        :param int pre_silence: The amount (in milliseconds) of pre-silence added to the audio signal.
        """
        self.__pre_silence = pre_silence
        self.data = self._generate_data()

    @property
    def post_silence(self):
        """
        Set the amount (in milliseconds) of post-silence added to the audio signal.

        :return: The amount (in milliseconds) of post-silence added to the audio signal.
        """
        return self.__post_silence

    @post_silence.setter
    def post_silence(self, post_silence):
        """
        Set the amount (in milliseconds) of post-silence added to the audio signal.

        :param int post_silence: The amount (in milliseconds) of post-silence added to the audio signal.
        """
        self.__post_silence = post_silence
        self.data = self._generate_data()

    @property
    def attenuator(self):
        """
        Get the attenuator object used to attenuate the sin signal.

        :return: The attenuator object used to attenuate the sin signal.
        :rtype: audio.attenuation.Attenuator
        """
        return self.__attenuator

    @attenuator.setter
    def attenuator(self, attenuator):
        """
        Set the attenuator object used to attenuate the sin signal.

        :param audio.stimuli.Attenuator attenuator: The attenuator object used to attenuate the sin signal.
        """
        self.__attenuator = attenuator
        self.data = self._generate_data()

    @property
    def frequency(self):
        """
        Get the frequency of the sin signal in Hz.

        :return: The frequency of the sin signal in Hz.
        :rtype: float
        """
        return self.__frequency

    @frequency.setter
    def frequency(self, frequency):
        """
        Set the frequency of the sin signal in Hz.

        :param float frequency: The frequency of the sin signal in Hz.
        """
        self.__frequency = frequency
        self.data = self._generate_data()


class SinStim(AudioStim):
    """
       The SinStim class provides a simple interface for generating sinusoidal audio stimulus data
       appropriate for feeding directly as voltage signals to a DAQ for playback. It allows parameterization
       of the sinusoid as well as attenuation by a custom attenuation object.
    """

    NAME = 'sin'

    def __init__(self, frequency, amplitude, phase, sample_rate, duration, intensity=1.0, pre_silence=0,
                 post_silence=0, attenuator=None, next_event_callbacks=None, identifier=None):
        # Initiatialize the base class members
        super(SinStim, self).__init__(sample_rate=sample_rate, duration=duration, intensity=intensity,
                                      pre_silence=pre_silence, post_silence=post_silence, attenuator=attenuator,
                                      frequency=frequency, next_event_callbacks=next_event_callbacks,
                                      identifier=identifier)

        self.__amplitude = amplitude
        self.__phase = phase

        # Add class specific parameters to the event message that is sent to functions in next_event_callbacks from the
        # base class generator.
        self.event_message["amplitude"] = amplitude
        self.event_message["phase"] = phase

        self.data = self._generate_data()
        self.dtype = self.data.dtype

    def describe(self):
        desc = super(SinStim, self).describe()
        desc['amplitude'] = self.__amplitude
        desc['phase'] = self.__phase
        return desc

    @property
    def amplitude(self):
        """
        Get the amplitude of the sin signal.

        :return: The amplitude of the sin signal.
        :rtype: float
        """
        return self.__amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        """
        Set the amplitude of the sin signal.

        :param float amplitude: Set the amplitude of the sin signal.
        """
        self.__amplitude = amplitude
        self.data = self._generate_data()

    @property
    def phase(self):
        """
        Get the phase of the sin, in radians.

        :return: The phase of the sin, in radians.
        :rtype: float
        """
        return self.__phase

    @phase.setter
    def phase(self, phase):
        """
        Set the phase of the sin, in radians.

        :param float phase: The phase of the sin, in radians.
        """
        self.__phase = phase
        self.data = self._generate_data()

    def _generate_data(self):
        """
        Generate the sin sample data according to the parameters. Also attenuatte the signal if an attenuator
        is provided.

        :return: The sin signal data, ready to be passed to the DAQ as voltage signals.
        :rtype: numpy.ndarray
        """
        # noinspection PyPep8Naming
        T = np.linspace(0.0, float(self.duration) / 1000.0, int((float(self.sample_rate) / 1000.0) * self.duration))

        # Generate the samples of the sin wave with specified amplitude, frequency, and phase.
        data = self.amplitude * np.sin(2 * np.pi * self.frequency * T + self.phase)

        return data


class SquareWaveStim(AudioStim):
    """
       The SquareWaveStim class provides a simple interface for generating square wave audio stimulus data
       appropriate for feeding directly as samples for sound card playback. It allows parameterization
       of the square wave as well as attenuation scaling.
    """

    NAME = 'square'

    def __init__(self, frequency, duty_cycle, amplitude, sample_rate, duration, intensity=1.0, pre_silence=0,
                 post_silence=0, attenuator=None, next_event_callbacks=None, identifier=None):
        # Initiatialize the base class members
        super(SquareWaveStim, self).__init__(sample_rate=sample_rate, duration=duration, intensity=intensity,
                                             pre_silence=pre_silence, post_silence=post_silence, attenuator=attenuator,
                                             frequency=frequency, next_event_callbacks=next_event_callbacks,
                                             identifier=identifier)

        self.__duty_cycle = duty_cycle
        self.__amplitude = amplitude

        # Add class specific parameters to the event message that is sent to functions in next_event_callbacks from the
        # base class generator.
        self.event_message["duty_cycle"] = duty_cycle
        self.event_message["amplitude"] = amplitude

        self.data = self._generate_data()
        self.dtype = self.data.dtype

    def describe(self):
        desc = super(SquareWaveStim, self).describe()
        desc['duty_cycle'] = self.__duty_cycle
        desc['amplitude'] = self.__amplitude
        return desc

    @property
    def amplitude(self):
        """
        Get the amplitude of the sin signal.

        :return: The amplitude of the sin signal.
        :rtype: float
        """
        return self.__amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        """
        Set the amplitude of the sin signal.

        :param float amplitude: Set the amplitude of the sin signal.
        """
        self.__amplitude = amplitude
        self.data = self._generate_data()

    @property
    def duty_cycle(self):
        """
        Get the duty cycle of the signal, between 0 and 1.

        :return: The duty cycle of the signal, between 0 and 1.
        :rtype: float
        """
        return self.__duty_cycle

    @duty_cycle.setter
    def phase(self, duty_cycle):
        """
        Set the duty cyclce of the signal.

        :param float duty_cycle: The duty cycle, between 0 and 1.
        """
        self.__duty_cycle = duty_cycle
        self.data = self._generate_data()

    def _generate_data(self):
        """
        Generate the sin sample data according to the parameters. Also attenuatte the signal if an attenuator
        is provided.

        :return: The sin signal data, ready to be passed to the DAQ as voltage signals.
        :rtype: numpy.ndarray
        """
        # noinspection PyPep8Naming
        T = np.linspace(0.0, float(self.duration) / 1000.0, int((float(self.sample_rate) / 1000.0) * self.duration))
        # Generate the samples of the sin wave with specified amplitude, frequency, and phase.
        return self.amplitude * signal.square(T * 2 * np.pi * self.frequency, duty=self.duty_cycle)


class ConstantStim(AudioStim):
    """A constant value signal, however supporting also pre_ and post_silence and attenuator """

    NAME = 'constant'

    def __init__(self, sample_rate, duration, amplitude=1.0, pre_silence=0, post_silence=0, attenuator=None,
                 intensity=1.0, next_event_callbacks=None, identifier=None):
        super(ConstantStim, self).__init__(sample_rate=sample_rate, duration=duration, intensity=intensity,
                                           pre_silence=pre_silence, post_silence=post_silence, attenuator=attenuator,
                                           frequency=None, next_event_callbacks=next_event_callbacks,
                                           identifier=identifier)
        self._amplitude = amplitude
        self.data = self._generate_data()
        self.dtype = self.data.dtype

    def _generate_data(self):
        return np.full((int(self.duration / 1000. * self.sample_rate),), self._amplitude)

    def describe(self):
        desc = super(ConstantStim, self).describe()
        desc['amplitude'] = self._amplitude
        desc.pop('frequency')
        return desc


class PulseStim(AudioStim):

    NAME = 'pulse'

    def __init__(self, sample_rate, duration_a, amplitude_a, duration_b, amplitude_b,
                 pre_silence=0, post_silence=0, intensity=1.0, attenuator=None,
                 next_event_callbacks=None, identifier=None):
        super(PulseStim, self).__init__(sample_rate=sample_rate, duration=duration_a + duration_b, intensity=intensity,
                                        pre_silence=pre_silence, post_silence=post_silence, attenuator=attenuator,
                                        frequency=None, next_event_callbacks=next_event_callbacks,
                                        identifier=identifier)

        self._amplitudes = amplitude_a, amplitude_b
        self._durations = duration_a, duration_b

        self.data = self._generate_data()
        self.dtype = self.data.dtype

    def _generate_data(self):
        return np.concatenate([
            np.full((int(self._durations[0] / 1000. * self.sample_rate),), self._amplitudes[0]),
            np.full((int(self._durations[1] / 1000. * self.sample_rate),), self._amplitudes[1])])

    def describe(self):
        desc = super(PulseStim, self).describe()
        desc.pop('frequency')
        desc['amplitude_a'] = self._amplitudes[0]
        desc['amplitude_b'] = self._amplitudes[0]
        desc['duration_a'] = self._durations[0]
        desc['duration_b'] = self._durations[0]
        return desc


class MATFileStim(AudioStim):
    """A class to encapsulate stimulus data that has been pre-generated and stored as MATLAB MAT files. The lab has a
    significant number of pre-generated audio stimulus patterns stored as MAT files. This class allows
    loading of these data files and playing them through the DAQ."""

    NAME = 'matfile'

    MATFILE_EXTENSION = '.mat'

    def __init__(self, filename, frequency, sample_rate, intensity=1.0, pre_silence=0, post_silence=0, attenuator=None,
                 next_event_callbacks=None, identifier=None):

        # Initiatialize the base class members
        super(MATFileStim, self).__init__(sample_rate=sample_rate, duration=None, intensity=intensity,
                                          pre_silence=pre_silence, post_silence=post_silence, attenuator=attenuator,
                                          frequency=frequency, next_event_callbacks=next_event_callbacks,
                                          identifier=identifier)

        # strip .mat for internal use
        if filename.endswith('.mat'):
            self.__filename = filename[:-4]
        else:
            self.__filename = filename

        # Add class specific parameters to the event message that is sent to functions in next_event_callbacks from the
        # base class generator.
        self.event_message["stim_filename"] = self.__filename

        self.data = self._generate_data()
        self.dtype = self.data.dtype

    def describe(self):
        desc = super(MATFileStim, self).describe()
        desc.pop('duration')
        desc['filename'] = self.__filename
        return desc

    @property
    def filename(self):
        """
        Get the filename that stored the audio data.

        :return: The filename that stored the audio data.
        :rtype: str
        """
        return self.__filename

    @filename.setter
    def filename(self, filename):
        """
        Set the filename and load the data.

        :param str filename: The name of the file that stores the audio stimulus data.
        """
        self.__filename = filename
        self.data = self._generate_data()

    def _generate_data(self):
        """
        Load the sample data from the file with path stored in __filename.

        :return: The audio stimulus data, ready to be passed to the DAQ as voltage signals.
        :rtype: numpy.ndarray
        """
        try:
            data = scipy.io.loadmat(self.__filename, variable_names=['stim'], squeeze_me=True, appendmat=True)
            data = data['stim']

            return data
        except NotImplementedError:
            # This exception indicates that this is an HDF5 file and not an old type MAT file
            h5_file = h5py.File(self.__filename + '.mat', 'r')
            data = np.squeeze(h5_file['stim'])

            return data


def _legacy_factory(chan_name, rate, silencePre, silencePost, intensity, freq, basedirs=None, attenuator=None):
    basedirs = basedirs or [os.getcwd()]

    # note: I think the legacy playlist never really supported sin or square waves because there was
    #  never any ability to specify things like duration independent of sample-rate

    if chan_name == "optooff" or chan_name.strip() == "":
        chan = ConstantStim(amplitude=0.0,
                            duration=30000,
                            sample_rate=int(1e4))
    elif chan_name == "optoon":
        chan = ConstantStim(amplitude=5.0,
                            duration=30000,
                            sample_rate=int(1e4))
    elif chan_name == "square":
        chan = SquareWaveStim(frequency=freq, duty_cycle=0.75,
                              amplitude=intensity, sample_rate=int(1e4),
                              duration=int(rate * 1000.0), intensity=1.0,
                              pre_silence=silencePre,
                              post_silence=silencePost)
    elif chan_name == "sin":
        chan = SinStim(frequency=freq,
                       amplitude=intensity,
                       phase=0.0,
                       sample_rate=44100,
                       duration=int(rate * 1000.0),  # can't be specified
                       pre_silence=silencePre,
                       post_silence=silencePost)
    else:
        if freq == -1:
            atten = None
        else:
            atten = attenuator

        filename = None
        for bd in basedirs:
            # legacy did not include .mat extension
            _filename = os.path.join(bd, chan_name + MATFileStim.MATFILE_EXTENSION)
            if os.path.exists(_filename):
                filename = _filename[:-4]
                break

        if filename is None:
            raise IOError("could not find '%s' in any directory: %r" % (chan_name, basedirs))

        chan = MATFileStim(filename=filename,
                           frequency=freq,
                           sample_rate=int(rate),
                           intensity=intensity,
                           pre_silence=int(silencePre),
                           post_silence=int(silencePost),
                           attenuator=atten)

    return chan


def stimulus_factory(**conf):
    basedirs = conf.pop('basedirs', [os.getcwd()])

    try:
        # format was
        # stimFileName	rate	trial	silencePre	silencePost	delayPost	intensity	freq
        # but trial was always 1 and delayPost always 0
        return _legacy_factory(conf['stimFileName'], conf['rate'], conf['silencePre'],
                               conf['silencePost'], conf['intensity'], conf['freq'],
                               basedirs=basedirs)
    except KeyError:
        name = conf.pop('name')
        if name == 'sin':
            return SinStim(frequency=conf['frequency'],
                           amplitude=conf['amplitude'],
                           phase=conf.get('phase', 0.0),
                           sample_rate=conf.get('sample_rate', 44100),
                           duration=conf['duration'],
                           pre_silence=conf.get('pre_silence', 0),
                           post_silence=conf.get('post_silence', 0),
                           attenuator=conf.get('attenuator'),
                           identifier=conf.get('identifier'))
        elif name == 'matfile':
            filename = None
            if os.path.isabs(conf['filename']):
                filename = conf['filename']
            else:
                # take the first file that exists
                # but be lenient with or without .mat extension
                for bd in basedirs:
                    _filename = os.path.join(bd, conf['filename'] + MATFileStim.MATFILE_EXTENSION)
                    if os.path.exists(_filename):
                        # convention is we dont include the mat file extension
                        filename = _filename[:-4]
                        break

                    _filename = os.path.join(bd, conf['filename'])
                    if os.path.exists(_filename):
                        filename = _filename
                        break

            if filename is None:
                raise IOError("could not find '%s' in any directory: %r" % (conf['filename'],
                                                                            basedirs))

            return MATFileStim(filename=filename,
                               frequency=conf['frequency'],
                               sample_rate=conf.get('sample_rate', 44100),
                               intensity=conf['intensity'],
                               pre_silence=conf.get('pre_silence', 0),
                               post_silence=conf.get('post_silence', 0),
                               attenuator=conf.get('attenuator'),
                               identifier=conf.get('identifier'))

        elif name == 'constant':
            return ConstantStim(sample_rate=conf.get('sample_rate', 44100),
                                amplitude=conf['amplitude'],
                                duration=conf['duration'],
                                pre_silence=conf.get('pre_silence', 0),
                                post_silence=conf.get('post_silence', 0),
                                attenuator=conf.get('attenuator'),
                                identifier=conf.get('identifier'))
        elif name == "square":
            return SquareWaveStim(frequency=conf['frequency'],
                                  amplitude=conf['amplitude'],
                                  duty_cycle=conf.get('duty_cycle', 0.5),
                                  sample_rate=conf.get('sample_rate', 44100),
                                  duration=conf['duration'],
                                  pre_silence=conf.get('pre_silence', 0),
                                  post_silence=conf.get('post_silence', 0),
                                  attenuator=conf.get('attenuator'),
                                  identifier=conf.get('identifier'))
        elif name == "pulse":
            return PulseStim(amplitude_a=conf['amplitude_a'],
                             amplitude_b=conf['amplitude_b'],
                             duration_a=conf['duration_a'],
                             duration_b=conf['duration_b'],
                             sample_rate=conf.get('sample_rate', 44100),
                             pre_silence=conf.get('pre_silence', 0),
                             post_silence=conf.get('post_silence', 0),
                             attenuator=conf.get('attenuator'),
                             identifier=conf.get('identifier'))

        return NotImplementedError


def legacy_factory(lines, basedirs, attenuator=None):
    def _parse_list(_s):
        _list = re.match(r"""\[([\d\s.+-]+)\]""", _s)
        if _list:
            return list(float(v.strip()) for v in _list.groups()[0].split())
        else:
            return [float(_s)]

    basedirs = basedirs or [os.getcwd()]

    stims = []
    for idx, _line in enumerate(lines):
        line = _line.rstrip()
        try:
            # noinspection PyPep8Naming
            stimFileName, rate, trial, silencePre, silencePost, delayPost, intensity, freq = \
                map(str.strip, line.split('\t'))
        except ValueError:
            raise ValueError("incorrect formatting: '%r', split: %r, (ntabs: %d)" % (line,
                                                                                     line.split('\t'),
                                                                                     line.count('\t')))

        frequencies = _parse_list(freq)
        intensities = _parse_list(intensity)

        chans = []
        for chan_idx, chan_name in enumerate(stimFileName.split(';')):

            chan = _legacy_factory(chan_name=chan_name.lower(),
                                   rate=int(rate),
                                   silencePre=int(silencePre),
                                   silencePost=int(silencePost),
                                   intensity=intensities[chan_idx],  # float
                                   freq=frequencies[chan_idx],  # float
                                   basedirs=basedirs,
                                   attenuator=attenuator)
            chans.append(chan)

        # Combine these stimuli into one analog signal with a channel for each.
        if len(chans) > 1:
            mixed_stim = MixedSignal(chans)
        else:
            mixed_stim = chans[0]

        # Append to playlist
        stims.append(mixed_stim)

    return stims


class AudioStimPlaylist(SignalProducer):
    """A simple class that provides a generator for a sequence of AudioStim objects."""

    def __init__(self, stims, random=None, paused=False):
        super(AudioStimPlaylist, self).__init__()
        self._log = logging.getLogger('flyvr.audio.AudioStimPlaylist')

        self._stims = stims

        for s in self._stims:
            self._log.debug('playlist item: %s (%r)' % (s.identifier, s))

        if random is None:
            random = Randomizer(*[s.identifier for s in stims])
        self._random = random

        self._log.info('playlist paused: %s order: %r' % (paused, self._random))

        self.paused = paused

    def __iter__(self):
        for s in self._stims:
            yield s

    def initialize(self, flyvr_shared_state, backend):
        super().initialize(flyvr_shared_state, backend)
        self._log = logging.getLogger('flyvr.audio.AudioStimPlaylist(%s)' % backend)

    def describe(self):
        return [{s.identifier: s.describe()} for s in self._stims]

    @classmethod
    def from_legacy_filename(cls, filename, random=None, attenuator=None, paused=False):
        with open(filename, 'rt') as f:
            return cls(legacy_factory(f.readlines()[1:], basedirs=[os.path.dirname(filename)], attenuator=attenuator),
                       random=random, paused=paused)

    @classmethod
    def fromitems(cls, items, random=None, paused=False, attenuator=None, basedirs=None):
        stims = []
        for item_def in items:
            id_, defn = item_def.popitem()
            defn['identifier'] = id_
            stims.append(stimulus_factory(**defn,
                                          basedirs=basedirs or []))
        return cls(stims, random=random, paused=paused)

    @classmethod
    def from_playlist_definition(cls, stim_playlist, basedirs, paused_fallback, default_repeat, attenuator=None):
        stims = []
        stim_ids = []
        option_item_defn = {}

        for item_def in stim_playlist:
            id_, defn = item_def.popitem()

            if id_ == Randomizer.IN_PLAYLIST_IDENTIFIER:
                option_item_defn = {id_: defn}
                continue

            stims.append({id_: defn})
            stim_ids.append(id_)

        random = Randomizer.new_from_playlist_option_item(option_item_defn, *stim_ids, repeat=default_repeat)
        paused = option_item_defn.pop('paused', None)

        return cls.fromitems(items=stims,
                             random=random,
                             paused=paused if paused is not None else paused_fallback,
                             basedirs=basedirs,
                             attenuator=attenuator)

    def play_item(self, identifier):
        # it's actually debatable if it's best do it this way or explicitly reset a global+sticky next_id
        for stim in self._stims:
            if stim.identifier == identifier:
                return stim.data_generator()
        raise ValueError('%s not found' % identifier)

    def play_pause(self, pause):
        self.paused = pause

    def data_generator(self):
        """
        Return a generator that yields each AudioStim in the playlist in succession. If shuffle_playback is set to true
        then we will get a non-repeating randomized sequence of all stimuli, then they will be shuffled, and the process
        repeated.
        :return: A generator that yields an array containing the sample data.
        """
        data_gens = {s.identifier: s.data_generator() for s in self._stims}
        playlist_iter = self._random.iter_items()

        while True:

            if self.paused:
                yield None
            else:
                try:
                    next_id = next(playlist_iter)
                except StopIteration:
                    # playlist finished
                    next_id = None

                if next_id is None:
                    yield None
                else:
                    self._log.info('playing item: %s' % next_id)
                    if self.flyvr_shared_state is not None:
                        self.flyvr_shared_state.signal_new_playlist_item(next_id, self.backend)

                    sample_chunk_obj = next(data_gens[next_id])
                    yield sample_chunk_obj

    def _to_array(self, fix_repeat_forver=False):
        if self._random.repeat_forever:
            if fix_repeat_forver:
                self._log.warning('previous playlist repeated forever - permuting this instance to return data. this'
                                  'instance should not be used for anything else in this process')
                before = repr(self._random)
                # noinspection PyProtectedMember
                self._random = self._random._copy_thyself(repeat=1)
                self._log.info('copied randomizer %s -> %r' % (before, self._random))
            else:
                raise ValueError('playlist repeats forever - so data array is infinite')

        arrs = []
        for chunk in self.data_generator():
            if chunk is None:
                break
            arrs.append(chunk.data)

        return np.concatenate(arrs)


