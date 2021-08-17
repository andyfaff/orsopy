"""
Implementation of the base classes for the ORSO header.
"""

# author: Andrew R. McCluskey (arm61)

import os.path
from collections.abc import Mapping
from typing import Optional, Union, List, get_args, get_origin, Literal
from inspect import isclass
from dataclasses import field, dataclass, fields
import datetime
import pathlib
import warnings
import json
import yaml
from contextlib import contextmanager
import re

import numpy as np


def _noop(self, *args, **kw):
    pass


yaml.emitter.Emitter.process_tag = _noop


def __datetime_representer(dumper, data):
    """
    Ensures that datetime objects are represented correctly."""
    value = data.isoformat("T")
    return dumper.represent_scalar("tag:yaml.org,2002:timestamp", value)


yaml.add_representer(datetime.datetime, __datetime_representer)


class BaseHeader:
    """
    The super class for all of the items in the orso module.
    """

    _orso_optionals: List[str] = []

    def __post_init__(self):
        """Make sure Header types are correct."""
        for fld in fields(self):
            attr = getattr(self, fld.name, None)
            type_attr = type(attr)
            if attr is None or type_attr is fld.type:
                continue
            else:
                updt = self._resolve_type(fld.type, attr)
                if updt is not None:
                    # convert to dataclass instance
                    setattr(self, fld.name, updt)
                else:
                    raise ValueError(f"No suitable conversion found for {fld.type} with value {attr}")
        if hasattr(self, 'unit'):
            self._check_unit(self.unit)

    @staticmethod
    def _resolve_type(hint, item):
        if isclass(hint):
            # simple type that we can work with, no Union or List/Dict
            if isinstance(item, hint):
                return item
            if issubclass(hint, Header):
                # convert to dataclass instance
                try:
                    return hint(**item)
                except (ValueError, TypeError):
                    return None
            else:
                # convert to type
                try:
                    return hint(item)
                except (ValueError, TypeError):
                    return None
        else:
            # the hint is a combined type (Union/List etc.)
            hbase = get_origin(hint)
            if hbase is list:
                t0 = get_args(hint)[0]
                if type(item) is list:
                    return [Header._resolve_type(t0, i) for i in item]
                else:
                    return [Header._resolve_type(t0, item)]
            elif hbase in [Union, Optional]:
                for subt in get_args(hint):
                    res = Header._resolve_type(subt, item)
                    if res is not None:
                        return res
            elif hbase is Literal:
                if item in get_args(hint):
                    return item
        return None

    @classmethod
    def empty(cls):
        """
        Create an empty instance of this item containing
        all non-option attributes as None
        """
        attr_items = {}
        for fld in fields(cls):
            if type(None) in get_args(fld.type):
                # skip optional arguments
                continue
            elif isclass(fld.type) and issubclass(fld.type, Header):
                attr_items[fld.name] = fld.type.empty()
            elif get_origin(fld.type) is Union and issubclass(get_args(fld.type)[0], Header):
                attr_items[fld.name] = get_args(fld.type)[0].empty()
            elif get_origin(fld.type) is list and isclass(get_args(fld.type)[0])\
                    and issubclass(get_args(fld.type)[0], Header):
                attr_items[fld.name] = [get_args(fld.type)[0].empty()]
            else:
                attr_items[fld.name] = None
        return cls(**attr_items)

    def to_dict(self):
        """
        Produces a clean dictionary of the Header object, removing
        any optional attributes with the value `None`.

        :return: Cleaned dictionary
        :rtype: dict
        """
        out_dict = {}
        for i, value in self.__dict__.items():
            if i.startswith("_") or (
                value is None and i in self._orso_optionals
            ):
                continue

            if hasattr(value, "_orso_optionals"):
                out_dict[i] = value.to_dict()
            elif isinstance(value, list):
                cleaned_list = []
                for j in value:
                    if hasattr(j, "_orso_optionals"):
                        cleaned_list.append(j.to_dict())
                    else:
                        cleaned_list.append(j)
                out_dict[i] = cleaned_list
            elif i == "data_set" and value == 0:
                continue
            else:
                out_dict[i] = value
        return out_dict

    def to_yaml(self):
        """
        Return the yaml string for the Header item

        :return: Yaml string
        :rtype: str
        """
        return yaml.dump(self.to_dict(), sort_keys=False)

    @staticmethod
    def _check_unit(unit):
        """
        Check if the unit is valid, in future this could include
        recommendations.

        :param unit: Value to check if it is a value unit
        :type unit: str
        :raises: ValueError is the unit is not ASCII text
        """
        if unit is not None:
            if not unit.isascii():
                raise ValueError("The unit must be in ASCII text.")

    def __repr__(self):
        # representation that does not show empty arguments
        out = f'{self.__class__.__name__}('
        for fi in fields(self):
            if fi.name in self._orso_optionals and getattr(self, fi.name) is None:
                # ignore empty optional arguments
                continue
            out += f'{fi.name}={getattr(self, fi.name).__repr__()}, '
        out = out[:-2]+')'
        return out

    def _staggered_repr(self):
        """
        Generate a string representation distributed over multiple lines
        to improve readability.

        To use in a subclass, the __repr__ method has to be replaced with this one.
        """
        slen = len(self.__class__.__name__)
        out = f'{self.__class__.__name__}(\n'
        for fi in fields(self):
            if fi.name in self._orso_optionals and getattr(self, fi.name) is None:
                # ignore empty optional arguments
                continue
            nlen = len(fi.name)
            ftxt = getattr(self, fi.name).__repr__()
            ftxt = ftxt.replace('\n', '\n' + ' ' * (slen + nlen + 2))
            out += ' ' * (slen + 1) + f'{fi.name}={ftxt},\n'
        out += ' ' * (slen + 1) + ')'
        return out

@dataclass(repr=False)
class Comment(BaseHeader):
    """A comment."""

    comment: str


class HeaderMeta(type):
    """
    Metaclass for Header.
    Creates a dataclass with an additional comment attribute.
    """

    def __new__(cls, name, bases, attrs, **kwargs):
        if '__annotations__' in attrs:
            # only applies to dataclass children of Header
            # add optional comment attribute, needs to come last
            attrs['__annotations__']['comment']=Optional[Comment]
            attrs['comment']=field(default=None)

            # create the _orso_optional attribute
            attrs['_orso_optionals']=[]
            for fname, ftype in attrs['__annotations__'].items():
                if type(None) in get_args(ftype):
                    attrs['_orso_optionals'].append(fname)
            for base in bases:
                if hasattr(base, '_orso_optionals'):
                    attrs['_orso_optionals']+=getattr(base, '_orso_optionals')
        return type.__new__(cls, name, bases, attrs, **kwargs)


class Header(BaseHeader, metaclass=HeaderMeta):
    """
    Class containing any header information that may include a comment.
    """


@dataclass(repr=False)
class Value(Header):
    """A value or list of values with an optional unit."""

    magnitude: Union[float, List[float]]
    unit: Optional[str] = field(
        default=None, metadata={"description": "SI unit string"}
    )


@dataclass(repr=False)
class ValueRange(Header):
    """A range or list of ranges with mins, maxs, and an optional unit."""

    min: Union[float, List[float]]
    max: Union[float, List[float]]
    unit: Optional[str] = field(
        default=None, metadata={"description": "SI unit string"}
    )


@dataclass(repr=False)
class ValueVector(Header):
    """A vector or list of vectors with an optional unit.

    For vectors relating to the sample, such as polarisation,
    the follow is defined:

    * x is defined as parallel to the radiation beam, positive going\
        with the beam direction

    * y is defined from the other two based on the right hand rule

    * z is defined as normal to the sample surface, positive direction\
        in scattering direction
    """

    x: Union[float, List[float]]
    y: Union[float, List[float]]
    z: Union[float, List[float]]
    unit: Optional[str] = field(
        default=None, metadata={"description": "SI unit string"}
    )


@dataclass(repr=False)
class Person(Header):
    """Information about a person, including name, affilation(s), and email."""

    name: str
    affiliation: Union[str, List[str]]
    contact: Optional[str] = field(
        default=None, metadata={"description": "Contact (email) address"}
    )


@dataclass(repr=False)
class Creator(Header):
    name: str
    affiliation: Union[str, List[str]]
    time: datetime.datetime
    computer: str
    contact: Optional[str] = field(
        default=None, metadata={"description": "Contact (email) address"}
    )


@dataclass(repr=False)
class Column(Header):
    """Information about a data column"""

    name: str
    unit: Optional[str] = field(
        default=None, metadata={"description": "SI unit string"}
    )
    dimension: Optional[str] = field(
        default=None, metadata={"dimension": "A description of the column"}
    )


@dataclass(repr=False)
class File(Header):
    """A file with a last modified timestamp."""

    file: str
    created: Optional[datetime.datetime] = field(
        default=None,
        metadata={
            "description": "Last modified timestamp if not given and available"
        },
    )

    def __post_init__(self):
        Header.__post_init__(self)
        fname = pathlib.Path(self.file)
        if not fname.exists():
            warnings.warn(f"The file {self.file} cannot be found.")
        else:
            if self.created is None:
                self.created = datetime.datetime.fromtimestamp(
                    fname.stat().st_mtime
                )


def _read_header_data(file):
    """
    Reads the header and data contained within an ORSO file, parsing it into
    json dictionaries and numerical arrays.

    Parameters
    ----------
    file: str or file-like

    Returns
    -------
    dct_list, data_sets: list, list
        `dct_list` is a list of json dicts containing the parsed yaml header.
        This has to be processed further.
        `data_sets` is a Python list containing numpy arrays holding the
        reflectometry data in the file. It's contained in a list because each
        of the datasets may have a different number of columns.
    """

    with _possibly_open_file(file, "r") as fi:
        header = []

        # variables to keep track of where the numerical datasets are
        ds_lines = []
        in_ds = False
        start_line = -1
        end_line = -1
        first_dataset = True

        for i, line in enumerate(fi.readlines()):
            if not line.startswith("#"):
                if not in_ds:
                    # you're in the first line of a dataset
                    in_ds = True
                    start_line = i
                continue

            # at this point all lines begin with #
            if in_ds:
                # you've reached the first line after a comment line
                end_line = i - 1
                ds_lines.append((start_line, end_line))
                start_line = end_line = -1
                in_ds = False

            if line.startswith("# data_set") and first_dataset:
                header.append(line[1:])
                first_dataset = False
            elif line.startswith("# data_set") and not first_dataset:
                # append '---' to signify the start of a new yaml document
                # Subsequent datasets get parsed into a separate dictionary,
                # which can be used to synthesise new datasets from the first.
                header.append(" ---\n")
                header.append(line[1:])
            else:
                header.append(line[1:])

        yml = "".join(header)

        # first line of an ORSO file should have the magic string
        pattern = re.compile(
            r"^(# ORSO reflectivity data file \| ([0-9]+\.?[0-9]*|\.[0-9]+)"
            r" standard \| YAML encoding \| https://www\.reflectometry\.org/)$"
        )

        if not pattern.match(header[0].lstrip(" ")):
            raise ValueError(
                "First line does not appear to match that of an ORSO file"
            )

        dcts = yaml.safe_load_all(yml)

        # synthesise json dicts for each dataset from the first dataset, and
        # updates to the yaml.
        first_dct = next(dcts)
        dct_list = [_nested_update(first_dct.copy(), dct) for dct in dcts]
        dct_list.insert(0, first_dct)

        # now load the numerical data
        # finished reading the file, have to append the last dataset
        ds_lines.append((start_line, i))

        data = []
        for ds_line in ds_lines:
            fi.seek(0, 0)
            start, end = ds_line
            arr = np.loadtxt(fi, skiprows=start, max_rows=end - start + 1)
            data.append(arr)

        return dct_list, data


def _validate_header_data(dct_list: List[dict]):
    """
    Checks whether a json dictionary corresponds to a valid ORSO header.

    Obtain these dct_list by loading from _read_header_data first.

    Parameters
    ----------
    dct_list : List[dict]
        dicts corresponding to parsed yaml headers from the ORT file.
    """
    import jsonschema

    pth = os.path.dirname(__file__)
    schema_pth = os.path.join(pth, "schema", "refl_header.schema.json")
    with open(schema_pth, "r") as f:
        schema = json.load(f)

    # d contains datetime.datetime objects, which would fail the
    # jsonschema validation, so force those to be strings.
    modified_dct_list = [
        json.loads(json.dumps(dct, default=str)) for dct in dct_list
    ]
    for dct in modified_dct_list:
        jsonschema.validate(dct, schema)


@contextmanager
def _possibly_open_file(f, mode="wb"):
    """
    Context manager for files.
    Parameters
    ----------
    f : file-like or str
        If `f` is a file, then yield the file. If `f` is a str then open the
        file and yield the newly opened file.
        On leaving this context manager the file is closed, if it was opened
        by this context manager (i.e. `f` was a string).
    mode : str, optional
        mode is an optional string that specifies the mode in which the file
        is opened.
    Yields
    ------
    g : file-like
        On leaving the context manager the file is closed, if it was opened by
        this context manager.
    """
    close_file = False
    if (hasattr(f, "read") and hasattr(f, "write")) or f is None:
        g = f
    else:
        g = open(f, mode)
        close_file = True
    yield g
    if close_file:
        g.close()


def _nested_update(d, u):
    # nested dictionary update.
    for k, v in u.items():
        if isinstance(d, Mapping):
            if isinstance(v, Mapping):
                r = _nested_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}
    return d
