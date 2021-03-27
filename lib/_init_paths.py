import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


root_dir = osp.abspath(osp.dirname(osp.join(__file__, '..')))

# Add lib to PYTHONPATH
lib_path = osp.join(root_dir, 'lib')
datasets_path = osp.join(root_dir, 'datasets')
add_path(lib_path)
add_path(datasets_path)
