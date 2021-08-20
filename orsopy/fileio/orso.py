"""
Implementation of the top level class for the ORSO header.
"""

# author: Andrew R. McCluskey (arm61)

import re
import yaml
from typing import List, Union, TextIO
from dataclasses import dataclass
from .base import (Header, Column, Creator, _possibly_open_file,
                   _read_header_data, _nested_update, _dict_diff)
from .data_source import DataSource
from .reduction import Reduction

import numpy as np

ORSO_designate = ("# ORSO reflectivity data file | 0.1 standard | "
                  "YAML encoding | https://www.reflectometry.org/")


@dataclass(repr=False)
class Orso(Header):
    """
    The Orso object collects the necessary metadata.
    """
    creator: Creator
    data_source: DataSource
    reduction: Reduction
    data_set: Union[str, int]
    columns: List[Column]

    __repr__ = Header._staggered_repr

    def column_header(self):
        out = '# '
        for ci in self.columns:
            if ci.unit is None:
                out += '%-22s ' % ci.name
            else:
                out += '%-22s ' % (f"{ci.name} ({ci.unit})")
            if ci is self.columns[0]:
                out = out[:-4]  # strip two characters from first column to align
        return out[:-1]

    def from_difference(self, other_dict):
        # recreate info from difference dictionary
        output = self.to_dict()
        output = _nested_update(output, other_dict)
        return Orso(**output)

    def to_difference(self, other: 'Orso'):
        # return a dictionary of differences to other object
        my_dict = self.to_dict()
        other_dict = other.to_dict()
        out_dict = _dict_diff(my_dict, other_dict)
        return out_dict


@dataclass
class OrsoDataset:
    info: Orso
    data: np.ndarray

    def __post_init__(self):
        if self.data.shape[1] != len(self.info.columns):
            raise ValueError("Data has to have the same number of columns as header")

    def header(self):
        out = self.info.to_yaml()
        out += self.info.column_header()
        return out

    def diff_header(self, other: 'OrsoDataset'):
        # return a header string that only contains changes to other OrsoDataset
        # ensure that data_set is the first entry
        out_dict = {"data_set": None}

        _diff = self.info.to_difference(other.info)
        out_dict.update(_diff)
        out_dict["data_set"] = other.info.data_set

        out = yaml.dump(out_dict, sort_keys=False)
        out += self.info.column_header()
        return out

    def save(self, fname):
        return save_orso([self], fname)

    def __eq__(self, other: 'OrsoDataset'):
        return self.info == other.info and (self.data == other.data).all()


class ORSOIOError(IOError):
    pass


def save_orso(datasets: List[OrsoDataset], fname: Union[TextIO, str]):
    with _possibly_open_file(fname, 'w') as f:
        header = f"{ORSO_designate}\n"
        ds1 = datasets[0]
        header += ds1.header()
        np.savetxt(f, ds1.data, header=header, fmt='%-22.16e')

        for dsi in datasets[1:]:
            hi = ds1.diff_header(dsi)
            np.savetxt(f, dsi.data, header=hi, fmt='%-22.16e')


def load_orso(fname: Union[TextIO, str]):
    dct_list, datas = _read_header_data(fname)

    ods = []
    for dct, data in zip(dct_list, datas):
        o = Orso(**dct)
        od = OrsoDataset(o, data)
        ods.append(od)
    return ods
