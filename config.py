from __future__ import division
from __future__ import print_function

import os
import errno
from itertools import chain, starmap
import yaml
import json
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C
__C.cfg_file = ""
__C.gpu_id = "-1"
__C.num_workers = 0
__C.random_seed = None
__C.logdir = ""
__C.logtb = False
__C.logcomet = False
__C.run_name = ""
__C.experiment_name = ""
__C.comet_project_name = ""
# __C.cuda = False
__C.debug = False
__C.eval = False
__C.data_dir = ""

__C.train = edict(
    resume="",
    batch_size=12,
    epochs=5,
    snapshot_interval=1,
    it_log_interval=1000,
    val_batch_size=32,
    val_beta=10000000,
    lrs=edict(stn=1e-4, beta=1e-3, general=1e-4,),
    betas=[0.5, 0.999],
    lambdas=edict(recon=1, presence=1, boundary=1e-5, beta=1,),
    p_min_thresh=5e-3,
)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("{} is not a valid config key".format(k))
            # print("{} is not a valid config key".format(k))
            # continue

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif isinstance(b[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif b[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(
                    ("Type mismatch ({} vs. {}) " "for config key: {}").format(
                        type(b[k]), type(v), k
                    )
                )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    if filename:
        with open(filename, "r") as f:
            _, ext = os.path.splitext(filename)
            if ext == ".yml" or ext == ".yaml":
                file_cfg = edict(yaml.safe_load(f))
            elif ext == ".json":
                file_cfg = edict(json.load(f))

        _merge_a_into_b(file_cfg, __C)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def flatten_json_iterative_solution(dictionary):
    """Flatten a nested json file"""

    def unpack(parent_key, parent_value):
        """Unpack one level of nesting in json file"""
        # Unpack one level only!!!

        if isinstance(parent_value, dict):
            for key, value in parent_value.items():
                temp1 = parent_key + "." + key
                yield temp1, value
        elif isinstance(parent_value, list):
            i = 0
            for value in parent_value:
                temp2 = parent_key + "." + str(i)
                i += 1
                yield temp2, value
        else:
            yield parent_key, parent_value

    # Keep iterating until the termination condition is satisfied
    while True:
        # Keep unpacking the json file until all values are atomic elements (not dictionary or list)
        dictionary = dict(chain.from_iterable(starmap(unpack, dictionary.items())))
        # Terminate condition: not any value in the json file is dictionary or list
        if not any(
            isinstance(value, dict) for value in dictionary.values()
        ) and not any(isinstance(value, list) for value in dictionary.values()):
            break

    return dictionary


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))
