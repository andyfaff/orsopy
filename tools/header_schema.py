"""
Generates the schema for an ORSO file.

Author: Brian Maranville (NIST)
"""
import datetime
import enum
from typing import Optional, Union, List, Literal, Dict, Any, Tuple
from dataclasses import field

from pydantic import Field

GENERATE_SCHEMA = True


def d(t):
    return field(metadata={"description": t})


if GENERATE_SCHEMA:
    from pydantic.dataclasses import dataclass as _dataclass

    class Config:
        @staticmethod
        def schema_extra(schema: Dict[str, Any]) -> None:
            for prop in schema.get("properties", {}).values():
                prop.pop("title", None)

    dataclass = _dataclass(config=Config)
else:
    from dataclasses import dataclass


@dataclass
class Person:
    """Information about a person, including name, affilation(s), and email."""

    name: str
    affiliation: Union[str, List[str]]
    contact: Optional[str] = field(
        default=None, metadata={"description": "Contact (email) address"}
    )


@dataclass
class Creator(Person):
    time: datetime.datetime = field(
        default=None,
        metadata={"description": "timestamp string, formatted as ISO 8601 datetime"}
    )
    computer: str = ""


@dataclass
class Sample:
    name: str


@dataclass
class Experiment:
    instrument: str
    probe: Union[Literal["neutrons", "x-rays"]]
    facility: Optional[str] = None
    ID: Optional[str] = None
    date: Optional[datetime.datetime] = field(
        metadata={
            "description": "timestamp string, formatted as ISO 8601 datetime"
        },
        default=None,
    )
    title: Optional[str] = None


class Polarization(str, enum.Enum):
    """The first symbol indicates the magnetisation direction of the incident beam.
    An optional second symbol indicates the direction of the scattered beam, if a spin analyser is present."""

    unpolarized = "unpolarized"
    p = "+"
    m = "-"
    mm = "--"
    mp = "-+"
    pm = "+-"
    pp = "++"


@dataclass
class data_file:
    file: str
    created: datetime.datetime


@dataclass
class Value:
    magnitude: Union[float, List[float]]
    unit: str = field(metadata={"description": "SI unit string"})


@dataclass
class ValueRange:
    min: float
    max: float
    unit: str = field(metadata={"description": "SI unit string"})
    steps: Optional[int] = None


@dataclass
class instrument_settings:
    incident_angle: Union[Value, ValueRange]
    wavelength: Union[Value, ValueRange]
    polarization: Optional[Union[Polarization]] = Polarization.unpolarized


@dataclass
class Measurement:
    scheme: Union[
        Literal[
            "angle- and energy-dispersive",
            "angle-dispersive",
            "energy-dispersive",
        ]
    ]
    instrument_settings: instrument_settings
    data_files: List[Union[data_file, str]]


@dataclass
class Software:
    """Description of the reduction software."""
    name: str
    version: Optional[str] = None
    platform: Optional[str] = None


@dataclass
class Reduction:
    software: Union[Software, str]
    call: Optional[str] = ""


@dataclass
class DataSource:
    owner: Person
    experiment: Experiment
    sample: Sample
    measurement: Measurement


@dataclass
class column:
    """
    Information on a data column.
    """

    name: str = d("The name of the column")
    dimension: Optional[str] = ""
    unit: Optional[Literal["1/angstrom", "1/nm"]] = field(
        default=None, metadata={"description": "SI unit string"}
    )
    # description: Optional[str] = field(
    #    default=None, metadata={"description": "A description of the column"})


@dataclass
class qz_column(column):
    name: Literal["Qz"]


@dataclass
class R_column(column):
    name: Literal["R"]


@dataclass
class sR_column(column):
    name: Literal["sR"]


@dataclass
class sQz_column(column):
    name: Literal["sQz"]


@dataclass
class ORSOHeader:
    creator: Creator
    data_source: DataSource
    columns: Union[
        Tuple[qz_column, R_column],
        Tuple[qz_column, R_column, sR_column],
        Tuple[qz_column, R_column, sR_column, sQz_column],
    ]
    reduction: Optional[Reduction] = None
    data_set: Union[str, int] = None


if GENERATE_SCHEMA:
    schema = ORSOHeader.__pydantic_model__.schema()
    print(schema)
    import json

    open("refl_header.schema.json", "wt").write(json.dumps(schema, indent=2))

    import yaml

    open("refl_header.schema.yaml", "w").write(yaml.dump(schema))
