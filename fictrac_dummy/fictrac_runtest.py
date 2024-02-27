from flyvr.fictrac.fictrac_driver import FicTracV2Driver as FicTracDriver
from flyvr.common.build_arg_parser import parse_arguments
from flyvr.common.concurrent_task import ConcurrentTask


import numpy as np
import pytest
import os
import shutil
import re

from distutils.dir_util import copy_tree

if __name__ == "__main__":

    opts, parser = parse_arguments("--config demo_experiment_and_playlist.yml", return_parser=True)

    driver = FicTracDriver('config.txt', 'output.txt', pgr_enable=False)

    fictrac_task = ConcurrentTask(task=driver.run, comms=None, taskinitargs=[opts])
    fictrac_task.start()

    while fictrac_task.is_alive():
        continue

    # Open the output file and make sure it contains 299 processes frames like it supposed to.
    with open('output.txt') as f:
        output_txt = f.read()
        assert "Frame 299" in output_txt

    # Get the timestamp fictrac is using for the output files.
    ts = re.findall("fictrac-([0-9_]+).log", output_txt)[0]

    # Make sure we got the data in the log.
    import h5py as h5
    with h5.File(opts.record_file) as f:
        logged_data = f['/fictrac/output'][:]

    # Load the output data from the dat file, this is what fictrac writes.
    dat_file_data = np.loadtxt(f'sample-{ts}.dat', delimiter=',')

    # They better match!
    assert np.allclose(logged_data, dat_file_data[int(logged_data[0, 0]):, 0:23])
