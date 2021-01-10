import subprocess
from contextlib import contextmanager
import torch
import io
import pickle
import numpy as np

import pdb

HADOOP_BIN = 'hadoop'


@contextmanager
def hopen(hdfs_path, mode="r"):
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} fs -text {}".format(HADOOP_BIN, hdfs_path), shell=True, stdout=subprocess.PIPE)
        yield pipe.stdout
        pipe.stdout.close()
        pipe.wait()
        return
    if mode == "wa":
        pipe = subprocess.Popen(
            "{} fs -appendToFile - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} fs -put -f - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def create_hdfs_loader(load_func):
    def load(filepath, **kwargs):
        assert filepath.startswith("hdfs://")
        with hopen(filepath, "rb") as reader:
            accessor = io.BytesIO(reader.read())
            content = load_func(accessor, **kwargs)
            del accessor
            return content

    return load


torch_load = create_hdfs_loader(torch.load)
pickle_load = create_hdfs_loader(pickle.load)
numpy_load = create_hdfs_loader(np.load)


def torch_save(obj, filepath, **kwargs):
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)


def pickle_save(obj, filepath, **kwargs):
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            pickle.dump(obj, writer, **kwargs)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f, **kwargs)


def numpy_save(obj, filepath):
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            np.save(writer, obj)
    else:
        np.save(filepath, obj)


if __name__ == '__main__':
    filepath = 'hdfs:///home/byte_arnold_hl_vc/user/chenjiacheng/data/coco/original_updown/id_mapping.pkl'
    obj = pickle_load(filepath)
    pdb.set_trace()
    filepath = 'hdfs:///home/byte_arnold_hl_vc/user/chenjiacheng/data/coco/original_updown/test_ims.npy'
    data = numpy_load(filepath)
    pdb.set_trace()
