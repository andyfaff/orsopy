"""
Implementation of the top level class for the ORSO header.
"""

# author: Andrew R. McCluskey (arm61)

import re
import yaml
from typing import List, Union, TextIO
from dataclasses import dataclass
from .base import Header, Column, Creator, _possibly_open_file
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
        self._update_dict(output, other_dict)
        return Orso(**output)

    @staticmethod
    def _update_dict(old, new):
        for key, value in new.items():
            if key in old and type(value) is dict:
                Orso._update_dict(old[key], value)
            else:
                old[key] = value

    def to_difference(self, other: 'Orso'):
        # return a dictionary of differences to other object
        my_dict = self.to_dict()
        other_dict = other.to_dict()
        out_dict = self._dict_diff(my_dict, other_dict)
        return out_dict

    @staticmethod
    def _dict_diff(old, new):
        # recursive find differences between two dictionaries
        out = {}
        for key, value in new.items():
            if key in old:
                if type(value) is dict:
                    diff = Orso._dict_diff(old[key], value)
                    if diff == {}:
                        continue
                    else:
                        out[key] = diff
                elif old[key] == value:
                    continue
                else:
                    out[key] = value
            else:
                out[key] = value
        return out


@dataclass
class OrsoDataset:
    info: Orso
    data: np.ndarray

    def __post_init__(self):
        if self.data.shape[0] != len(self.info.columns):
            raise ValueError("Data has to have the same number of columns as header")

    def header(self):
        out = self.info.to_yaml()
        out += self.info.column_header()
        return out

    def diff_header(self, other: 'OrsoDataset'):
        # return a header string that only contains changes to other OrsoDataset
        out_dict = self.info.to_difference(other.info)
        if 'data_set' in out_dict:
            del out_dict['data_set']
        out = f'data_set: {other.info.data_set}\n'
        if out_dict != {}:
            out += yaml.dump(out_dict, sort_keys=False)
        out += self.info.column_header()
        return out

    def save(self, fname):
        return save([self], fname)


class ORSOIOError(IOError):
    pass


def save(datasets: List[OrsoDataset], fname: Union[TextIO, str]):
    with _possibly_open_file(fname, 'w') as f:
        header = f"{ORSO_designate}\n"
        ds1 = datasets[0]
        header += ds1.header()
        np.savetxt(f, ds1.data.T, header=header, fmt='%-22.16e')

        for dsi in datasets[1:]:
            hi = ds1.diff_header(dsi)
            np.savetxt(f, dsi.data.T, header=hi, fmt='%-22.16e')


def read_data(text_data):
    # read the data from the text, faster then numpy loadtxt with StringIO
    data = [li.split() for li in text_data.strip().splitlines() if not li.startswith('#')]
    return np.array(data, dtype=float).T


def load(fname: Union[TextIO, str]):
    with _possibly_open_file(fname, 'r') as fh:
        # check if this is the right file type
        l1 = fh.readline()
        if not l1.lower().startswith('# # orso'):
            raise ORSOIOError('Not an ORSO reflectivity text file, wrong header')
        text = fh.read()
    ftype_info = list(map(str.strip, l1.split('|')))
    # find end of comment block (first line not starting with #
    data_start = re.search(r'\n[^#]', text).start()
    header = text[:data_start - 1]
    header = header[2:].replace('\n# ', '\n')  # remove header comment to make the text valid yaml
    header_encoding = ftype_info[2].lower().split()[0]
    if header_encoding == 'yaml':
        main_header_dict = yaml.load(header, Loader=yaml.FullLoader)
        ds_string = '\n# data_set:'
    elif header_encoding == 'json':
        raise NotImplementedError('JSON will come in future')
    else:
        raise ORSOIOError('Unknown header encoding "%s"' % header_encoding)
    main_info = Orso(**main_header_dict)

    # implement possibility of more then one data block:
    if ds_string in text:
        split_indices = [match.start() + data_start for
                         match in re.finditer(ds_string, text[data_start:])] + [-1]
        output = [OrsoDataset(main_info,
                              read_data(text[data_start:split_indices[0]]))]
        for i, si in enumerate(split_indices[:-1]):
            sub_header_length = re.search(r'\n[^#]', text[si:]).start()
            data = read_data(text[si + sub_header_length:split_indices[i + 1]])
            sub_header_text = text[si + 2:si + sub_header_length].rsplit('\n', 1)[0].replace('\n# ', '\n').strip()
            if header_encoding == 'yaml':
                sub_header_data = yaml.load(sub_header_text, Loader=yaml.FullLoader)
            elif header_encoding == 'json':
                raise NotImplementedError('JSON will come in future')
            # create a merged dictionary
            sub_info = main_info.from_difference(sub_header_data)
            output.append(OrsoDataset(sub_info, data))
        return output
    else:
        data = read_data(text[data_start:])
        return [OrsoDataset(main_info, data)]
