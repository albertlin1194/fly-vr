import numpy as np
import os


import h5py
import time

from audio.signal_producer import SignalProducer, SignalNextEventData
from common.concurrent_task import ConcurrentTask

"""
    The logger module implements a thread\process safe interface for logging datasets to a storage backend 
    (HDF5 file currently). It implements this via a multi process client server model of multiple producers (log event
    generators) and a single consumer (log event writer). The main class of interest for users is the DatasetLogServer
    class which maintaints the log server and the DatasetLogger class which is a pickable class that can be sent to 
    other processes for sending log events.
"""

class DatasetLogger:
    """
    DatasetLogger is basically a proxy class for sending log events to the DatasetLogServer. This logger class should
    only be instantiated by DatasetLogServer and returned from a call to start_logging_server(). It provides an API
    for creating and writing datasets. It can be called from multiple processes and threads in a safe manner.
    """

    def __init__(self, sender_queue):
        """
        Create the DatasetLogger for a specific message queue..

        :param sender_queue: The queue to send messages on.
        """
        self._sender_queue = sender_queue

    def create(self, *args, **kwargs):
        """
        Create an HDF5 dataset for logging. The arguments for this function is identical to h5py create_dataset command.

        :param name: Name of dataset to create. May be an absolute or relative path. Provide None to create an anonymous dataset, to be linked into the file later.
        :param shape: Shape of new dataset (Tuple).
        :param dtype: Data type for new dataset
        :param data: Initialize dataset to this (NumPy array).
        :param chunks: Chunk shape, or True to enable auto-chunking.
        :param maxshape: Dataset will be resizable up to this shape (Tuple). Automatically enables chunking. Use None for the axes you want to be unlimited.
        :param compression: Compression strategy. See Filter pipeline.
        :param compression_opts: Parameters for compression filter.
        :param scaleoffset: See Scale-Offset filter.
        :param shuffle: Enable shuffle filter (T/F). See Shuffle filter.
        :param fletcher32: Enable Fletcher32 checksum (T/F). See Fletcher32 filter.
        :param fillvalue: This value will be used when reading uninitialized parts of the dataset.
        :param track_times: Enable dataset creation timestamps (T/F).

        :return: None
        """
        create_event = DatasetCreateEvent(args=args, kwargs=kwargs)
        self._sender_queue.put(create_event)

    def log(self, dataset_name, obj, append=True):
        """
        Write data to a dataset. Supports appending data.


        :param dataset_name: The name of the dataset to modify.
        :param obj: An object to write to the dataset. Currently, this should either be a numpy array or a dictionary
        that contains numpy arrays, strings, or lists, for its values.
        :param append: Should this data be appended based on the last write to this dataset. Only valid for numpy arrays.
        :return:
        """
        log_event = DatasetWriteEvent(dataset_name=dataset_name, obj=obj, append=append)
        self._sender_queue.put(log_event)


class DatasetLogServer:
    """
    The DatasetLogServer implements the backend of a thread\process safe interface for logging datasets to a storage
    backend (HDF5 file currently). It implements this via running a separate logging process with a message queue that
    receives logging events from other processes.
    """

    def __init__(self):
        """
        Create the logging server. Does not start the logging process.
        """
        self._log_task = ConcurrentTask(task=self._thread_main, comms="queue", taskinitargs=[])

        # For each dataset, we will keep track of the current write position. This will allow us to append to it if
        # nescessary. We will store the write positions as integers in a dictionary of dataset_names
        self.dataset_write_pos = {}

    def start_logging_server(self, filename):
        """
        Start the logging server process. After this method is called log messages (datasets) can be sent to the
        logging server from other processes via the DatasetLogger object that is returned from this method.

        :param filename: The filename to write logged datasets to.
        :return: A DatasetLogger object that provides methods for sending log messages to the server for processing.
        """
        self.log_file_name = filename
        self._log_task.start()

        return DatasetLogger(self._log_task.sender)

    def stop_logging_server(self):
        """
        Stop the logging server gracefully. Flush and close the files and make sure all events have been processed.
        :return:
        """
        self._log_task.finish()
        self._log_task.close()
        self.log_file_name = None

    def _thread_main(self, frame_queue):
        """
        The main of the logging process.

        :param frame_queue: The message queue from which we will receive log messages.
        :return: None
        """

        # Setup the storage backend
        self._initialize_storage()

        # Run a message processing loop
        run = True
        while run:

            # Get the message
            msg = frame_queue.get()

            # If we get a None msg, its a shutdown signal
            if msg is None:
                run = False
            elif isinstance(msg, DatasetLogEvent):
                msg.process(self)
            else:
                raise ValueError("Bad message sent to logging thread.")

        # Close out the storage
        self._finalize_storage()


    def _initialize_storage(self):
        """
        Setup the storage backend.

        :return:
        """
        self.file = h5py.File(self.log_file_name, "w")

        # Reset all the write positions for any datasets
        self.dataset_write_pos = {}

    def _finalize_storage(self):
        """
        Close out the storage backend.

        :return:
        """

        # Flush and close the log file.
        self.file.flush()
        self.file.close()

    def wait_till_close(self):
        while self._log_task.process.is_alive():
            time.sleep(0.1)


class DatasetLogEvent:
    """
    DatasetLogEvent is the base class representing dataset logging events. It is not meant to be instatiated
    directly but to serve as a base class for different dataset logging events to inherit from. It provides a common
    interface for the DatasetLogServer to invoke processing.
    """

    def __init__(self, dataset_name):
        """
        Create a dataset log event for a specific dataset

        :param dataset_name: The str name of the dataset for which this event pertains.
        """
        self.dataset_name = dataset_name

    def process(self, server):
        """
        Process this event on the server. This method is not implemented for the base class.

        :param server: The DatasetLogServer object that is receiving this event.
        :return: None
        """
        raise NotImplemented("process is not implemented for base class DatasetLogEvent")


class DatasetCreateEvent(DatasetLogEvent):
    """
    DatasetCreateEvent implements the creation of datasets on the logging servers storage.
    """

    def __init__(self, args, kwargs):
        """
        Create a DatasetCreateEvent with arguments that are passed directly to the storage backed, HDF5 currently.

        :param args: List of arguments to pass to the dataset create command.
        :param kwargs: List of keyword arguments to pass to the dataset create command.
        """

        # We can extract the dataset name from kwargs or args
        try:
            dataset_name = kwargs['name']
        except KeyError:
            dataset_name = args[0]

        # Rather then reimplement all of h5py arguments for dataset create, we just take args and kwargs
        self.args = args
        self.kwargs = kwargs
        super(DatasetCreateEvent,self).__init__(dataset_name)

    def process(self, server):
        """
        Process the event by creating the dataset on the storage backend. args and kwargs are passed directly to the
        create dataset command.

        :param server: The DatasetLogServer to create the dataset on. Should provide an open file to write to.
        :return: None
        """
        server.file.require_dataset(*self.args, **self.kwargs)

class DatasetWriteEvent(DatasetLogEvent):
    """
    The DatasetWriteEvent implements writing to datasets stored on the DatasetLogServer.
    """

    def __init__(self, dataset_name, obj, append=True):
        """
        Create a DatasetWriteEvent that can be sent to the logging server.

        :param dataset_name: The name of the dataset to modify.
        :param obj: An object to write to the dataset. Currently, this should either be a numpy array or a dictionary
        that contains numpy arrays, strings, or lists, for its values.
        :param append: Should this data be appended based on the last write to this dataset. Only valid for numpy arrays.
        """

        if isinstance(obj, np.ndarray):
            self.obj = np.atleast_2d(obj)
        else:
            self.obj = obj

        self.append = append

        super(DatasetWriteEvent,self).__init__(dataset_name)

    def process(self, server):
        """
        Process this event on the logging server.

        :param server: The DataLogServer object that this event was received on. Should have and open file.
        :return: None
        """

        file_handle = server.file

        # Now, if this is a dictionary, we can try to simple write it recusively to this dataset
        if isinstance(self.obj, dict):
            recursively_save_dict_contents_to_group(file_handle, self.dataset_name, self.obj)

        # If we have a numpy array, then we need to write this as a dataset
        elif isinstance(self.obj, np.ndarray):

            # Get a handle to the dataset, we assume it has been created
            dset = file_handle[self.dataset_name]

            # If we are not appending to our dataset, just ovewrite
            if not self.append:

                # Make sure the size of the dataset is identical to the size of the array
                if not np.array_equal(dset.shape, self.obj.shape):
                    raise ValueError("Array cannot be logged to datset name {} because it has incompatible shape!")
                else:
                    dset[:] = self.obj

            else:

                # If we are appending, get the current write position for this dataset. If it doesnt exist, we haven't
                # written yet so lets set it to 0
                try:
                    write_pos = server.dataset_write_pos[self.dataset_name]
                except KeyError:
                    server.dataset_write_pos[self.dataset_name] = 0
                    write_pos = 0

                newsize = write_pos + self.obj.shape[0]

                dset.resize(newsize, axis=0)
                dset[write_pos:,:] = self.obj

                server.dataset_write_pos[self.dataset_name] = dset.shape[0]

        file_handle.flush()

def test_worker(msg_queue):
    while True:
        print ("Test\n")

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Saves dictionary to an HDF5 files, calls itself recursively if items in
    dictionary are not np.ndarray, np.int64, np.float64, str, bytes. Objects
    must be iterable.
    """
    for key, item in list(dic.items()):
        if item is None:
            h5file[path + key] = ""
        elif isinstance(item, bool):
            h5file[path + key] = int(item)
        elif isinstance(item, list):
            items_encoded = []
            for it in item:
                if isinstance(it, str):
                    items_encoded.append(it.encode('utf8'))
                else:
                    items_encoded.append(it)

            h5file[path + key] = np.asarray(items_encoded)
        elif isinstance(item, (str)):
            h5file[path + key] = item.encode('utf8')
        elif isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, float)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))


def make_event_metadata_dtype(metadata):
    type_desc = []
    for field_name in metadata:
        value = metadata[field_name]
        type_desc.append( (field_name, type(value)) )

def log_audio_task_main(frame_queue, state, sizeincrement=100, out_hist_size_increment=1000):

    # Get the output log file name from the options
    filename = state.options.record_file

    # Get the number of channels
    num_in_channels = len(state.options.analog_in_channels)
    num_out_channels = len(state.options.analog_out_channels)

    # For each output channel, keep track of number of samples generated.
    num_samples_out = [0 for i in range(num_out_channels)]

    # Open the HDF5 file for writing
    f = h5py.File(filename, "w")

    input_grp = f.create_group("input")
    dset_samples = input_grp.create_dataset("samples", shape=[sizeincrement, num_in_channels],
                                    maxshape=[None, num_in_channels],
                                    chunks=(sizeincrement, num_in_channels),
                                    dtype=np.float64, scaleoffset=8)
    dset_systemtime = input_grp.create_dataset("systemtime", shape=[sizeincrement, 1],
                                       maxshape=[None, 1], dtype=np.float64)

    # Lets add options meta-data to samples dataset as attributes
    options_grp = f.create_group('options')
    recursively_save_dict_contents_to_group(options_grp, '/options/', state.options.__dict__)

    # The output group will contain event data about samples written to the DAQ.
    output_grp = f.create_group('output')

    # The output group will contain event data about samples written to the DAQ.
    fictrac_grp = f.create_group('fictrac')

    # Add a group for each output channel
    channel_grps = [output_grp.create_group("ao" + str(channel)) for channel in state.options.analog_out_channels]

    # Add a producer group, this will hold unique instances of signal producers
    prod_grps = [channel_grp.create_group("producers") for channel_grp in channel_grps]

    # Add a history dataset that records occurrences of events
    hist_dsets = [channel_grps[i].create_dataset("history", shape=[out_hist_size_increment, 2],
                                               maxshape=[None, 2], dtype=np.int64)
                  for i in range(num_out_channels)]

    # Keep track of the current index for each channels events
    hist_indices = [0 for i in range(num_out_channels)]

    # Create a dataset for fictrac history
    fictrac_size_increment = 10000
    fictrac_curr_idx = 0
    fictrac_dset = fictrac_grp.create_dataset("frames_samples", shape=[fictrac_size_increment, 2],
                               maxshape=[None, 2], dtype=np.int64)

    # A dictionary that stores data generation events we have received. We just want to keep track of unique
    # events.
    data_generation_events = {}
    data_event_max_index = 0

    framecount = 0
    RUN = True
    playlist = None
    while RUN:

        # Get the message
        msg = frame_queue.get()

        # If we get a None msg, its a shutdown signal
        if msg is None:
            RUN = False
        # If it is tuple with a numpy array and a float then it is a frame and system time message from aquisition
        elif isinstance(msg, tuple) and len(msg) == 2 and isinstance(msg[0], np.ndarray) and isinstance(msg[1], float):
            frame_systemtime = msg
            #sys.stdout.write("\r   {:1.1f} seconds: saving {} ({})".format(
                #frame_systemtime[1], frame_systemtime[0].shape, framecount))
            dset_samples.resize(dset_samples.shape[0] + frame_systemtime[0].shape[0], axis=0)
            dset_samples[-frame_systemtime[0].shape[0]:, :] = frame_systemtime[0]
            dset_systemtime[framecount, :] = frame_systemtime[1]
            framecount += 1

            # Resize the system time dataset if needed.
            if framecount % sizeincrement == sizeincrement - 1:
                f.flush()
                dset_systemtime.resize(dset_systemtime.shape[0] + sizeincrement, axis=0)
        elif isinstance(msg, SignalNextEventData):
            # Ok, check if this is a signal producer event. This means a signal generator's next method was called.

            # Get the channel that this output is occurring on
            channel = msg.channel

            # Get the history dataset for this channel
            dset = hist_dsets[channel]

            # Resize the channels events history dataset if needed
            if hist_indices[channel] % out_hist_size_increment == out_hist_size_increment - 1:
                f.flush()
                dset.resize(dset.shape[0] + out_hist_size_increment, axis=0)

            # Record the event in the table by adding its index and start sample number
            dset[hist_indices[channel], :] = [msg.producer_id, num_samples_out[channel]]
            hist_indices[channel] += 1

            num_samples_out[channel] += msg.num_samples
        elif isinstance(msg, tuple) and len(msg) == 2:

            fictrac_dset[fictrac_curr_idx, :] = [msg[0], msg[1]]
            fictrac_curr_idx += 1

            # Resize the dataset if needed.
            if fictrac_curr_idx % fictrac_size_increment == fictrac_size_increment - 1:
                f.flush()
                fictrac_dset.resize(fictrac_dset.shape[0] + fictrac_size_increment, axis=0)

        else:
            raise ValueError("Bad message sent to logging thread.")

    # Shrink the data sets if we didn't fill them up
    for i in range(num_out_channels):
        hist_dsets[i].resize(hist_indices[i], axis=0)

    fictrac_dset.resize(fictrac_curr_idx, axis=0)

    f.flush()
    f.close()
#    print("   closed file \"{0}\".".format(filename))




# These are some test dataset we will write to HDF5 to check things
test1_dataset = np.zeros((1600,3))
test1_dataset[:,0] = np.arange(0,1600)
test1_dataset[:,1] = np.arange(0,1600)*2
test1_dataset[:,2] = np.arange(0,1600)*3

# A worker thread main that writes the above dataset to the logger in chunks
def log_event_worker(msg_queue, dataset_name, logger):

    logger.create(dataset_name, shape=[512, 3],
                                maxshape=[None, 3],
                                chunks=(512, 3),
                                dtype=np.float64, scaleoffset=8)

    # Write the test data in chunk_size chunks
    chunk_size = 8
    for i in range(200):
        data_chunk = test1_dataset[i*chunk_size:(i*chunk_size+chunk_size), :]
        logger.log(dataset_name, data_chunk)

test2_dataset = {"data1": "This is a test", "data2": np.ones(shape=(3,2))}

def log_event_worker2(msg_queue, dataset_name, logger):
        logger.log(dataset_name, test2_dataset)

def main():

    # If we have a test file already, delete it.
    try:
        os.remove('test.h5')
    except OSError:
        pass

    # Start a HDF5 logging server
    server = DatasetLogServer()
    logger = server.start_logging_server("test.h5")

    # Start two processes that will be send log messages simultaneouslly
    task1 = ConcurrentTask(task=log_event_worker, taskinitargs=["test1", logger])
    task2 = ConcurrentTask(task=log_event_worker2, taskinitargs=["/deeper/test2/", logger])
    task1.start()
    task2.start()

    # Wait until they are done
    while task1.process.is_alive() and task2.process.is_alive():
        pass

    # Stop the logging server
    server.stop_logging_server()

    # Wait till it is done
    while server._log_task.process.is_alive():
        pass

    # Make sure the HDF5 file has been created.
    #assert(os.path.isfile('test.h5'))

    # Now lets load the HDF5 file we just wrote and make sure it contains the correct stuff
    f = h5py.File('test.h5', 'r')

    # Check if the first dataset exists
    #assert('test1' in f)

    # Check if it is equal to the dataset we have stored in memory
    #assert(np.array_equal(f['test1'], test1_dataset))

    # Check if the second dataset exists and is equal
    #assert('/deeper/test2/data1' in f)
    #assert(f['/deeper/test2/data1'] == test2_dataset['data1'])
    #assert('/deeper/test2/data2' in f and np.array_equal(f['/deeper/test2/data2'], test2_dataset['data2']))

    f.close()

    #os.remove('test.h5')

if __name__ == "__main__":
    main()