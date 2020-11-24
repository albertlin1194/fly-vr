# -*- coding: utf-8 -*-
import threading
import logging
import time

import PyDAQmx as daq
# noinspection PyUnresolvedReferences
from PyDAQmx.DAQmxFunctions import (DAQmxCreateTask, DAQmxCreateAOVoltageChan,
                                    DAQmxCfgSampClkTiming, DAQmxStartTask,
                                    DAQmxWriteAnalogScalarF64, DAQmxWaitForNextSampleClock, DAQmxStopTask,
                                    DAQmxClearTask)
# noinspection PyUnresolvedReferences
from PyDAQmx.DAQmxConstants import (DAQmx_Val_RSE, DAQmx_Val_Volts, DAQmx_Val_Rising, DAQmx_Val_HWTimedSinglePoint,
                                    DAQmx_Val_Acquired_Into_Buffer, DAQmx_Val_ContSamps,
                                    DAQmx_Val_Transferred_From_Buffer,
                                    DAQmx_Val_DoNotAllowRegen, DAQmx_Val_AllowRegen, DAQmx_Val_GroupByChannel,
                                    DAQmx_Val_Auto, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByScanNumber,
                                    DAQmx_Val_Diff,
                                    DAQmx_Val_ChanPerLine)

import numpy as np
from ctypes import byref, c_ulong

from flyvr.audio.signal_producer import chunker
from flyvr.audio.stimuli import AudioStim, AudioStimPlaylist
from flyvr.audio.util import get_paylist_object
from flyvr.common import BACKEND_DAQ
from flyvr.common.concurrent_task import ConcurrentTask
from flyvr.common.plot_task import plot_task_daq
from flyvr.common.build_arg_parser import setup_logging
from flyvr.common.ipc import PlaylistReciever

DAQ_SAMPLE_RATE = 10000
DAQ_NUM_OUTPUT_SAMPLES = 5000
DAQ_NUM_OUTPUT_SAMPLES_PER_EVENT = 250
DAQ_NUM_INPUT_SAMPLES = 10000
DAQ_NUM_INPUT_SAMPLES_PER_EVENT = 10000


# noinspection PyPep8Naming
class IOTask(daq.Task):
    """
    IOTask encapsulates the an input-output task that communicates with the NIDAQ. It works with a list of input or
    output channel names.
    """

    def __init__(self, dev_name="Dev1", cha_name=("ai0",), cha_type="input", limits=10.0, rate=DAQ_SAMPLE_RATE,
                 num_samples_per_chan=DAQ_SAMPLE_RATE, num_samples_per_event=None, digital=False, has_callback=True,
                 shared_state=None, done_callback=None, use_RSE=True):
        # check inputs
        daq.Task.__init__(self)

        self._log = logging.getLogger('flyvr.daq.IOTask')

        _digital = 'digital' if digital else 'analog'
        self._log.info(f'DAQ:{dev_name}: {_digital}{cha_type}/{cha_name} (limits: {limits}, '
                       f'SR: {rate}, nSamp/ch: {num_samples_per_chan}, '
                       f'nSamp/event: {num_samples_per_event}, RSE: {use_RSE})')

        self.dev_name = dev_name

        if not isinstance(cha_name, list):
            cha_name = [cha_name]

        self.flyvr_shared_state = shared_state

        # Is this a digital task
        self.digital = digital

        # A function to call on task completion
        self.done_callback = done_callback

        # A task to send signals to everytime we write a chunk of samples. We will send the current sample number and
        # the current FicTrac frame number
        self.logger = shared_state.logger

        # These are just some dummy values for pass by reference C functions that the NI DAQ api has.
        self.read = daq.int32()
        self.read_float64 = daq.float64()

        self.limits = limits
        self.cha_type = cha_type
        self.cha_name = ['%s/%s' % (dev_name, ch) for ch in cha_name]  # append device name
        self.cha_string = ", ".join(self.cha_name)
        self.num_samples_per_chan = num_samples_per_chan
        self.num_samples_per_event = num_samples_per_event  # self.num_samples_per_chan*self.num_channels

        if self.num_samples_per_event is None:
            self.num_samples_per_event = num_samples_per_chan

        clock_source = None  # use internal clock
        self.callback = None
        self.data_gen = None  # called at start of control
        self._data_recorders = None  # called at end of control

        if self.cha_type is "input":
            if not self.digital:
                if use_RSE:
                    self.CreateAIVoltageChan(self.cha_string, "", DAQmx_Val_RSE, -limits, limits, DAQmx_Val_Volts, None)
                else:
                    self.CreateAIVoltageChan(self.cha_string, "", DAQmx_Val_Diff, -limits, limits, DAQmx_Val_Volts,
                                             None)
            else:
                self.CreateDIChan(self.cha_string, "", DAQmx_Val_ChanPerLine)

            # Get the number of channels from the task
            nChans = c_ulong()
            self.GetTaskNumChans(nChans)
            self.num_channels = nChans.value

            if has_callback:
                self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.num_samples_per_event, 0)
                self.CfgInputBuffer(self.num_samples_per_chan * self.num_channels * 4)

        elif self.cha_type is "output":
            if not self.digital:
                self.CreateAOVoltageChan(self.cha_string, "", -limits, limits, DAQmx_Val_Volts, None)
            else:
                self.CreateDOChan(self.cha_string, "", DAQmx_Val_ChanPerLine)

            # Get the number of channels from the task
            nChans = c_ulong()
            self.GetTaskNumChans(nChans)
            self.num_channels = nChans.value

        # We need to create a dataset for log messages.
        if cha_type == "output" and not digital:
            self.logger.create("/fictrac/daq_synchronization_info", shape=[1024, 2], maxshape=[None, 2],
                               dtype=np.int64,
                               chunks=(1024, 2))
        elif cha_type == "input" and not digital:
            self.samples_dset_name = "/input/samples"
            self.samples_time_dset_name = "/input/systemtime"
            self.logger.create(self.samples_dset_name, shape=[512, self.num_channels],
                               maxshape=[None, self.num_channels],
                               chunks=(512, self.num_channels),
                               dtype=np.float64, scaleoffset=8)
            self.logger.create(self.samples_time_dset_name, shape=[1024, 1], chunks=(1024, 1),
                               maxshape=[None, 1], dtype=np.float64)
        elif cha_type == "input" and digital:
            self.samples_dset_name = "/input/digital/samples"
            self.samples_time_dset_name = "/input/digital/systemtime"
            self.logger.create(self.samples_dset_name, shape=[2048, self.num_channels],
                               maxshape=[None, self.num_channels],
                               chunks=(2048, self.num_channels),
                               dtype=np.uint8)
            self.logger.create(self.samples_time_dset_name, shape=[1024, 1], chunks=(1024, 1),
                               maxshape=[None, 1], dtype=np.float64)

        if not digital:
            self._data = np.zeros((self.num_samples_per_chan, self.num_channels),
                                  dtype=np.float64)  # init empty data array
        else:
            self._data = np.zeros((self.num_samples_per_chan, self.num_channels), dtype=np.uint8)

        # Since this data did not come from a sample chunk object, set it to None
        self._sample_chunk = None

        self.CfgSampClkTiming(clock_source, rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.num_samples_per_chan)
        self.AutoRegisterDoneEvent(0)

        if has_callback:
            self._data_lock = threading.Lock()
            self._newdata_event = threading.Event()
            if self.cha_type is "output":

                cbf = rate / float(self.num_samples_per_event)
                self._log.info('buffer size: %d (buffer callback called every %.3fs, at %.1fHz)' % (
                    self.num_samples_per_event, 1./cbf, cbf))

                self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Transferred_From_Buffer, self.num_samples_per_event, 0)
                # ensures continuous output and avoids collision of old and new data in buffer
                # self.SetAODataXferReqCond(self.cha_name[0], DAQmx_Val_OnBrdMemEmpty)
                self.SetWriteRegenMode(DAQmx_Val_DoNotAllowRegen)
                self.CfgOutputBuffer(self.num_samples_per_chan * self.num_channels * 2)

                self.EveryNCallback()  # fill buffer on init
        else:
            self.SetWriteRegenMode(DAQmx_Val_AllowRegen)
            self.CfgOutputBuffer(self.num_samples_per_chan * self.num_channels * 2)

    def stop(self):
        if self.data_gen is not None:
            self._data = self.data_gen.close()

        if self.data_recorders is not None:
            for data_rec in self.data_recorders:
                data_rec.finish()
                data_rec.close()

    def set_data_generator(self, data_generator):
        """
        Set the data generator for the audio stimulus directly.

        :param data_generator: A generator function of audio data.
        """
        with self._data_lock:
            chunked_gen = chunker(data_generator, chunk_size=self.num_samples_per_chan)
            self.data_gen = chunked_gen

    @property
    def data_recorders(self):
        return self._data_recorders

    @data_recorders.setter
    def data_recorders(self, value):

        if value is None:
            self._data_recorders = None

        # We need to store the data recorders as a list internally, because we will iterate over them later
        elif not isinstance(value, list):
            self._data_recorders = [value]

        else:
            self._data_recorders = value

    def send(self, data):
        if self.cha_type == "input":
            raise ValueError("Cannot send on an input channel, it must be an output channel.")
        if self.digital:
            self.WriteDigitalLines(data.shape[0], False, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel, data, None,
                                   None)
        else:
            self.WriteAnalogF64(data.shape[0], 0, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel, data,
                                daq.byref(self.read), None)

    def EveryNCallback(self):
        with self._data_lock:
            systemtime = time.clock()

            # get data from data generator
            if self.data_gen is not None:
                self._sample_chunk = next(self.data_gen)
                self._data = self._sample_chunk.data

            if self.cha_type is "input":
                if not self.digital:
                    self.ReadAnalogF64(DAQmx_Val_Auto, 1.0, DAQmx_Val_GroupByScanNumber,
                                       self._data, self.num_samples_per_chan * self.num_channels, daq.byref(self.read),
                                       None)
                else:
                    numBytesPerSamp = daq.int32()
                    self.ReadDigitalLines(self.num_samples_per_chan, 1.0, DAQmx_Val_GroupByScanNumber,
                                          self._data, self.num_samples_per_chan * self.num_channels,
                                          byref(self.read), byref(numBytesPerSamp), None)

            elif self.cha_type is "output":

                # Log output syncrhonization info only if the logger is valid and the task is not digital.
                if self.logger is not None and not self.digital:
                    self.logger.log("/fictrac/daq_synchronization_info",
                                    np.array([self.flyvr_shared_state.FICTRAC_FRAME_NUM,
                                              self.flyvr_shared_state.DAQ_OUTPUT_NUM_SAMPLES_WRITTEN]))

                if not self.digital:
                    self.WriteAnalogF64(self._data.shape[0], 0, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByScanNumber,
                                        self._data, daq.byref(self.read), None)

                    # Keep track of how many samples we have written out in a global variable
                    if self.flyvr_shared_state is not None:
                        self.flyvr_shared_state.DAQ_OUTPUT_NUM_SAMPLES_WRITTEN = \
                            self.flyvr_shared_state.DAQ_OUTPUT_NUM_SAMPLES_WRITTEN + self._data.shape[0]
                else:
                    self.WriteDigitalLines(self._data.shape[0], False, DAQmx_Val_WaitInfinitely,
                                           DAQmx_Val_GroupByScanNumber, self._data, None, None)

            # Send the data to a control if requested.
            if self.data_recorders is not None:
                for data_rec in self.data_recorders:
                    if self._data is not None:
                        data_rec.send((self._data, systemtime))

            # Send the data to our logging process
            if self.logger is not None and self.cha_type == "input":
                self.logger.log(self.samples_dset_name, self._data)
                self.logger.log(self.samples_time_dset_name, np.array([systemtime]))

            self._newdata_event.set()

        return 0  # The function should return an integer

    # noinspection PyUnusedLocal
    def DoneCallback(self, status):

        if self.done_callback is not None:
            self.done_callback(self)

        return 0  # The function should return an integer


def setup_playback_callbacks(stim, logger, flyvr_shared_state):
    """
    This function setups a control function for each stimulus in the playlist to be called when a set of data is
    generated. This control will send a log message to a logging process indicating the amount of samples generated and
    the stimulus that generated them.

    :param stim: The stimulus playlist to setup callbacks on.
    :param logger: The DatasetLogger object to send log signals to.
    :param flyvr_shared_state: The shared state variable that contains options to the program.
    :return: None
    """

    # noinspection PyUnusedLocal
    def make_log_stim_playback(_logger, _state):
        def _callback(chunk):
            print('callback', chunk)
            # _logger.log("/output/history",
            #             np.array([event_message.metadata['stim_playlist_idx'], event_message.num_samples]))

        return _callback

    stim.initialize(flyvr_shared_state, BACKEND_DAQ)

    # Make the control function
    callbacks = make_log_stim_playback(logger, flyvr_shared_state)

    # Setup the control.
    if isinstance(stim, AudioStim):
        stim.add_next_event_callback(callbacks)
    elif isinstance(stim, AudioStimPlaylist):
        for s in stim:
            s.add_next_event_callback(callbacks)

    # Lets setup the logging dataset that these log events will be sent to
    logger.create("/output/history",
                  shape=[2048, 2], maxshape=[None, 2],
                  chunks=(2048, 2), dtype=np.int32)


# noinspection PyPep8Naming
def io_task_loop(message_pipe, flyvr_shared_state, options):

    log = logging.getLogger('flyvr.daq')
    log.info('starting DAQ process')

    analog_in_channels = tuple(sorted(options.analog_in_channels))
    analog_out_channels = tuple(sorted(options.analog_out_channels))

    if len(analog_out_channels) > 1:
        raise NotImplementedError('only a single DAQ output channel is supported')

    if len(analog_in_channels) < 1:
        raise NotImplementedError('at least 1 DAQ analog channel must be read')

    daq_stim, _ = get_paylist_object(options, playlist_type='daq',
                                     paused_fallback=False,
                                     default_repeat=1,  # repeat=1 is more sensible for DAQ?
                                     attenuator=None)

    if daq_stim is not None:
        if daq_stim.num_channels != 1:
            raise NotImplementedError('only a single DAQ output channel is supported '
                                      '(yet the playlist has 2 channels of data)')

    is_analog_out = (daq_stim is not None) and len(analog_out_channels) == 1

    # noinspection PyBroadException
    try:

        running = True
        taskAO = None
        taskAI = None

        while running:
            log.info("initializing DAQ Tasks")

            taskAO = None
            if is_analog_out:
                # Get the input and output channels from the options
                output_chans = ["ao" + str(s) for s in analog_out_channels]
                taskAO = IOTask(cha_name=output_chans, cha_type="output",
                                num_samples_per_chan=DAQ_NUM_OUTPUT_SAMPLES,
                                num_samples_per_event=DAQ_NUM_OUTPUT_SAMPLES_PER_EVENT,
                                shared_state=flyvr_shared_state)

            input_chans = ["ai" + str(s) for s in analog_in_channels]
            input_chan_names = [options.analog_in_channels[s] for s in analog_in_channels]
            taskAI = IOTask(cha_name=input_chans, cha_type="input",
                            num_samples_per_chan=DAQ_NUM_INPUT_SAMPLES,
                            num_samples_per_event=DAQ_NUM_INPUT_SAMPLES_PER_EVENT,
                            shared_state=flyvr_shared_state, use_RSE=options.use_RSE)

            disp_task = ConcurrentTask(task=plot_task_daq, comms="pipe",
                                       taskinitargs=[input_chan_names, taskAI.num_samples_per_chan, 5])

            # Setup the display task to receive messages from recording task.
            taskAI.data_recorders = [disp_task]
            # start disp early so the user sees something
            disp_task.start()

            # Setup callbacks that will generate log messages to the logging process. These will signal what is playing
            # and when.
            if is_analog_out:
                setup_playback_callbacks(daq_stim, flyvr_shared_state.logger, flyvr_shared_state)

            if taskAO is not None:
                # Setup the stimulus playlist as the data generator
                taskAO.set_data_generator(daq_stim.data_generator())

                # Connect AO start to AI start
                taskAO.CfgDigEdgeStartTrig("ai/StartTrigger", DAQmx_Val_Rising)

            _ = flyvr_shared_state.signal_ready(BACKEND_DAQ)

            if not flyvr_shared_state.wait_for_start():
                log.info('did not receive start signal')
                running = False
                continue

            log.info("starting DAQ tasks")
            if taskAO is not None:
                # Arm the AO task
                # It won't start until the start trigger signal arrives from the AI task
                taskAO.StartTask()

            # Start the AI task
            # This generates the AI start trigger signal and triggers the AO task
            taskAI.StartTask()

            while running:
                # fixme: should replace this with a queue like the other backends and plumb
                #  in the IPC messages
                # fixme: just pop from a queue here
                if message_pipe.poll(0.1):
                    # noinspection PyBroadException
                    try:
                        msg = message_pipe.recv()

                        # If we have received a stimulus object, feed this object to output task for playback
                        if isinstance(msg, AudioStim) | isinstance(msg, AudioStimPlaylist):

                            # Setup callbacks that will generate log messages to the logging process.
                            # these will signal what is playing and when.
                            setup_playback_callbacks(msg, flyvr_shared_state.logger, flyvr_shared_state)

                            if taskAO is not None:
                                # Setup the stimulus playlist as the data generator
                                taskAO.set_data_generator(msg.data_generator())

                        if isinstance(msg, str) and msg == "STOP":
                            break
                    except Exception:
                        pass

                if flyvr_shared_state.is_stopped():
                    running = False

        log.info('stopped')

        if taskAO is not None:
            taskAO.StopTask()
            taskAO.stop()

        taskAI.StopTask()
        taskAI.stop()

        if taskAO is not None:
            taskAO.ClearTask()

        if taskAI is not None:
            taskAI.ClearTask()

    except Exception:
        flyvr_shared_state.runtime_error(1)


def _ipc_main(q):
    pr = PlaylistReciever()
    log = logging.getLogger('flyvr.daq.ipc_main')

    log.debug('starting')

    while True:
        elem = pr.get_next_element()
        if elem:
            # noinspection PyBroadException
            try:
                if 'daq_item' in elem:
                    q.put(elem['daq_item']['identifier'])
                elif 'daq_action' in elem:
                    q.put(elem['daq_action'])
                else:
                    log.debug("ignoring message: %r" % elem)
            except Exception:
                log.error('could not parse playlist item', exc_info=True)


def run_io(options):
    from flyvr.common import SharedState
    from flyvr.common.logger import DatasetLogServerThreaded

    setup_logging(options)

    class _MockPipe:
        # noinspection PyUnusedLocal,PyMethodMayBeStatic
        def poll(self, *args):
            return False

    with DatasetLogServerThreaded() as log_server:
        logger = log_server.start_logging_server(options.record_file.replace('.h5', '.daq.h5'))
        state = SharedState(options=options, logger=logger, where=BACKEND_DAQ)
        io_task_loop(_MockPipe(), state, options)


def main_io():
    from flyvr.common.build_arg_parser import build_argparser, parse_options, setup_logging
    from flyvr.audio.util import plot_playlist

    parser = build_argparser()
    parser.add_argument('--plot', action='store_true', help='plot the stimulus playlist')
    options = parse_options(parser.parse_args(), parser)

    if options.plot:
        setup_logging(options)

        if not options.playlist.get('daq'):
            return parser.error('Config file contains no daq playlist')

        plot_playlist(options, 'daq')

        return parser.exit(0)

    run_io(options)
