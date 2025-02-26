import argparse
import sys
import pickle
import h5py
import numpy as np

from utils import MyFormatter


def get_args():
    description = """
    Pipe data to h5 file

    accepts a dictionary of the form e.g.:
        {
        'file': 'test.h5',
        'mode': 'r+',
        'array': np.arange(10),
        'entry_1/scalar': 10,
        'sub_array': {'slice': ((10, 20), (0, 15)), 'data': np.ones((10, 15))}
        }

    useful when multiple processes must write to a single file.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=MyFormatter
    )
    parser.add_argument(
        '-i',
        '--input',
        type=argparse.FileType('rb'),
        default=sys.stdin.buffer,
        help="Python pickle output file. \
        The result is written as a dictionary with the key 'object'"
    )
    args = parser.parse_args()
    return args


def write_to_h5(
    f,
    key,
    value,
    s=()
):
    """
    write value to open h5 file

    try to overwrite existing data as this saves space
    """
    if (
        isinstance(value, np.ndarray) and
        key in f and
        value.dtype == f[key].dtype
    ):
        try:
            f[key][s] = value
            return
        except TypeError:
            pass

    if key in f:
        del f[key]

    f[key] = value
    return


def check_key_value(k, v, mode):
    """
    k must be a string (but let h5py throw that error)
    value can be str, scalar, numpy array
    or
    value can be dict with keys 'slice' and 'data'
    but only if mode is 'r+' or 'a'
    """
    if isinstance(v, dict):
        if not ('slice' in v and 'data' in v):
            err = f"dict must contain 'slice' and \
                    'data' {list(v.keys())=} {k=}"
            raise ValueError(err)

        if not (mode == 'r+' or mode == 'a'):
            err = f'{mode=} incompatible with dict data {k=}'
            raise ValueError(err)

        s = v['slice']
        data = v['data']
    else:
        s = ()
        data = v
    return data, s


def write_package(msg):
    assert (isinstance(msg, dict))
    file = msg.pop('file')

    if 'mode' in msg:
        mode = msg.pop('mode')
    else:
        mode = 'a'

    # open file
    with h5py.File(file, mode=mode) as f:
        for k, v in msg.items():
            data, s = check_key_value(k, v, mode)
            write_to_h5(f, k, data, s)


if __name__ == '__main__':
    args = get_args()

    # loop over input and write
    while True:
        try:
            package = pickle.load(args.input)

            # if data is a dictionary then recursively
            # iterate over key, value pairs
            write_package(package)

        except EOFError:
            break
