"""
Implementation of the top level class for the ORSO header.
"""

# author: Andrew R. McCluskey (arm61)

from typing import List, Union
from dataclasses import dataclass
from .base import Header, Column, Person, Creator
from .data_source import (DataSource, Experiment, Sample, InstrumentSettings,
                          Measurement)
from .reduction import Reduction, Software
import datetime


@dataclass
class Orso(Header):
    """
    The Orso object collects the necessary metadata.
    """
    creator: Creator
    data_source: DataSource
    reduction: Reduction
    columns: List[Column]
    data_set: Union[str, int]
    _orso_optionals = []


def make_empty():
    """
    Create a complete Orso class object, with :code:`None` in all values
    which should be filled with objects from :code:`orsopy.fileio.base`
    or some Python base object (:code:`int`, :code:`list`, etc.).

    :return: Uninformative object
    :rtype: orsopy.fileio.orso.Orso
    """
    _person = Person(None, None, None)
    _creator = Creator(None, None, None, None, None)
    _experiment = Experiment(None, None, None, None)
    _instrument_settings = InstrumentSettings(None, None)
    _measurement = Measurement(_instrument_settings, None)
    _data_source = DataSource(
        _person, _experiment, Sample(None), _measurement
    )

    _reduction = Reduction(Software(None, None, None), None, None, None)
    return Orso(_creator, _data_source, _reduction, None, None)
