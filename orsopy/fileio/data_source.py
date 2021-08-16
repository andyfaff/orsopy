"""
Implementation of the data_source for the ORSO header.
"""

# author: Andrew R. McCluskey (arm61)

import enum
from typing import Optional, Dict, List, Union
from dataclasses import field, dataclass
import datetime
try:
    from typing import Literal
except ImportError:
    # not available until 3.8
    from typing_extensions import Literal

from .base import File, Header, ValueRange, Value, ValueVector, Person


@dataclass
class Experiment(Header):
    """A definition of the experiment performed."""
    title: str
    instrument: str
    date: datetime.datetime
    probe: Union[Literal["neutrons", "x-rays"]]
    facility: Optional[str] = field(default=None)
    ID: Optional[str] = field(default=None)
    doi: Optional[str] = field(default=None)
    _orso_optionals = ['facility', 'ID', 'doi']

@dataclass
class Sample(Header):
    """A description of the sample measured."""
    name: str
    type: Optional[str] = field(default=None)
    composition: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)
    environment: Optional[str] = field(default=None)
    sample_parameters: Optional[Dict] = field(
        default=None,
        metadata={
            'description':
            'Using keys for parameters and Value* objects for values.'
        })
    _orso_optionals = [
        'type', 'composition', 'description', 'environment',
        'sample_parameters'
    ]


# Enum does not work with yaml, if we really want this it has to be handled as special case.
#
# class Polarization(str, enum.Enum):
#     """
#     The first symbol indicates the magnetisation direction of the incident
#     beam. An optional second symbol indicates the direction of the scattered
#     beam, if a spin analyser is present.
#     """
#
#     unpolarized = "unpolarized"
#     p = "+"
#     m = "-"
#     mm = "--"
#     mp = "-+"
#     pm = "+-"
#     pp = "++"


@dataclass
class InstrumentSettings(Header):
    """Settings associated with the instrumentation."""
    incident_angle: Union[Value, ValueRange]
    wavelength: Union[Value, ValueRange]
    polarization: Optional[Union[Literal['unpolarized', '+', '-', '--', '-+', '+-','++'],
                                 ValueVector]] = field(
        default='unpolarized',
        metadata={
            'description':
            'Polarization described as p / m / pp / pm / mp / mm / vector'
        })
    configuration: Optional[str] = field(
        default=None,
        metadata={
            'description': 'half / full polarized | liquid_surface | etc'
        })
    _orso_optionals = ['configuration']

    __repr__=Header._staggered_repr

@dataclass
class Measurement(Header):
    """The measurement elements for the header."""
    instrument_settings: InstrumentSettings
    data_files: List[Union[File, str]]
    references: Optional[List[Union[File, str]]] = field(default=None)
    scheme: Optional[Union[
        Literal[
            "angle- and energy-dispersive",
            "angle-dispersive",
            "energy-dispersive",
        ]
    ]] = None
    _orso_optionals = ['references', 'scheme']

    __repr__=Header._staggered_repr


@dataclass
class DataSource(Header):
    """The data_source object definition."""
    owner: Person
    experiment: Experiment
    sample: Sample
    measurement: Measurement
    _orso_optionals = []

    __repr__=Header._staggered_repr
