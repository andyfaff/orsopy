"""
Implementation of the data_source for the ORSO header.
"""
from dataclasses import field, dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import yaml

from .base import ComplexValue, File, Header, Person, Value, ValueRange, ValueVector
from .model_language import SampleModel

# typing stuff introduced in python 3.8
try:
    from typing import Literal
except ImportError:
    from .typing_backport import Literal


@dataclass
class Experiment(Header):
    """
    A definition of the experiment performed.

    :param title: Proposal or project title.
    :param instrument: Reflectometer identifier.
    :param start_date: Start date for the experiment.
    :param probe: Radiation probe, either :code:`'neutron'` or
        :code:`'x-ray'`.
    :param facility: Facility where the experiment was performed.
    :param proposalID: Identifier for experiment at a facility.
    :param doi: Digital object identifier for the experiment, possibly
        provided by the facility.
    """

    title: str
    instrument: str
    start_date: datetime
    probe: Literal["neutron", "x-ray"]
    facility: Optional[str] = None
    proposalID: Optional[str] = None
    doi: Optional[str] = None

    comment: Optional[str] = None

    def __init__(
            self,
            title,
            instrument,
            start_date,
            probe,
            facility=None,
            proposalID=None,
            doi=None,
            *,
            comment=None,
            **kwds
    ):
        super(Experiment, self).__init__()
        self.title = title
        self.instrument = instrument
        self.start_date = start_date
        self.probe = probe
        self.facility = facility
        self.proposalID = proposalID
        self.doi = doi
        for k, v in kwds.items():
            setattr(self, k, v)
        self.comment = comment
        self.__post_init__()


@dataclass
class Sample(Header):
    """
    A description of the sample measured.

    :param name: An identified for the individual sample or the subject and
        state being measured.
    :param category: Simple sample description, front (beam side) / back,
        each side should be one of :code:`'solid/liquid'`,
        :code:`'liquid/solid'`, :code:`'gas/liquid'`,
        :code:`'liquid/liquid'`, :code:`'solid/gas'`, :code:`'gas/solid'`.
    :param composition: Notes on the nominal composition of the sample e.g.
        :code:`Si | SiO2 (20 angstrom) | Fe (200 angstrom) |
        air (beam side)`.
    :param description: Further details of the sample, e.g. size.
    :param size: Sample size in x, y, z direction, where z is parallel to the surface normal
        and x is along the beam direction (important for footprint correction).
    :param environment: Name of the sample environment device(s).
    :param sample_parameters: Dictionary of sample parameters.
    """

    name: str
    category: Optional[str] = None
    composition: Optional[str] = None
    description: Optional[str] = None
    size: Optional[ValueVector] = None
    environment: Optional[List[str]] = None
    sample_parameters: Optional[Dict[str, Union[Value, ValueRange, ValueVector, ComplexValue]]] = field(
        default=None, metadata={"description": "Using keys for parameters and Value* objects for values."}
    )
    model: Optional[SampleModel] = None

    comment: Optional[str] = None

    def __init__(
            self,
            name,
            category=None,
            composition=None,
            description=None,
            size=None, environment=None,
            sample_parameters=None,
            model=None,
            *,
            comment=None,
            **kwds
    ):
        super(Sample, self).__init__()
        self.name = name
        self.category = category
        self.composition = composition
        self.description = description
        self.size = size
        self.environment = environment
        self.sample_parameters = sample_parameters
        self.model = model
        for k, v in kwds.items():
            setattr(self, k, v)
        self.comment = comment
        self.__post_init__()


class Polarization(str, Enum):
    """
    Polarization of the beam used for the reflectivity.

    Neutrons:
    The first symbol indicates the magnetisation direction of the incident
    beam, the second symbol indicates the direction of the scattered
    beam. If either polarization or analysis are not employed the
    symbol is replaced by "o".

    X-rays:
    Uses the conventional names pi, sigma, left and right. In experiments
    with polarization analysis the incident and outgoing polarizations
    are separated with an underscore "_".
    """

    unpolarized = "unpolarized"
    # half polarized states
    po = "po"
    mo = "mo"
    op = "op"
    om = "om"
    # full polarization analysis
    mm = "mm"
    mp = "mp"
    pm = "pm"
    pp = "pp"
    # x-ray polarizations
    pi = "pi"  # in scattering plane
    sigma = "sigma"  # perpendicular to scattering plane
    left = "left"  # circular left
    right = "right"  # circular right
    pi_pi = "pi_pi"
    sigma_sigma = "sigma_sigma"
    pi_sigma = "pi_sigma"
    sigma_pi = "sigma_pi"

    def yaml_representer(self, dumper: yaml.Dumper):
        output = self.value
        return dumper.represent_str(output)


@dataclass
class InstrumentSettings(Header):
    """
    Settings associated with the instrumentation.

    :param incident_angle: Angle (range) of incidence.
    :param wavelength: Neutron/x-ray wavelength (range).
    :param polarization: Radiation polarization as one of
        :code:`'unpolarized'`, :code:`'p'`, :code:`'m'`, :code:`'pp'`,
        :code:`'pm'`, :code:`'mp'`, :code:`'mm'`, or a
        :py:class:`orsopy.fileio.base.ValueVector`.
    :param configuration: Description of the instreument configuration (full
        polarized/liquid surface/etc).
    """

    incident_angle: Union[Value, ValueRange]
    wavelength: Union[Value, ValueRange]
    polarization: Optional[Union[Polarization, ValueVector]] = field(
        default="unpolarized",
        metadata={
            "description": "Polarization described as unpolarized/ po/ mo / op / om / pp / pm / mp / mm / vector"
        },
    )
    configuration: Optional[str] = field(
        default=None, metadata={"description": "half / full polarized | liquid_surface | etc"}
    )

    __repr__ = Header._staggered_repr

    comment: Optional[str] = None

    def __init__(self, incident_angle, wavelength, polarization=None, configuration=None, *, comment=None, **kwds):
        super(InstrumentSettings, self).__init__()
        self.incident_angle = incident_angle
        self.wavelength = wavelength
        self.polarization = polarization
        self.configuration = configuration
        for k, v in kwds.items():
            setattr(self, k, v)
        self.comment = comment
        self.__post_init__()


@dataclass
class Measurement(Header):
    """
    The measurement elements for the header.

    :param instrument_settings: Instrumentation details.
    :param data_files: Raw data files produced in the measurement.
    :param references: Raw reference files used in the reduction.
    :param scheme: Measurement scheme (one of :code:`'angle-dispersive'`,
        :code:`'energy-dispersive'`/:code:`'angle- and energy-dispersive'`).
    """

    instrument_settings: InstrumentSettings
    data_files: List[Union[File, str]]
    additional_files: Optional[List[Union[File, str]]] = None
    scheme: Optional[Literal["angle- and energy-dispersive", "angle-dispersive", "energy-dispersive"]] = None

    __repr__ = Header._staggered_repr

    comment: Optional[str] = None

    def __init__(self, instrument_settings, data_files, additional_files=None, scheme=None, *, comment=None, **kwds):
        super(Measurement, self).__init__()
        self.instrument_settings = instrument_settings
        self.data_files = data_files
        self.additional_files = additional_files
        self.scheme = scheme
        for k, v in kwds.items():
            setattr(self, k, v)
        self.comment = comment
        self.__post_init__()


@dataclass
class DataSource(Header):
    """
    The data_source object definition.

    :param owner: This refers to the actual owner of the data set, i.e. the
        main proposer or the person doing the measurement on a lab
        reflectometer.
    :param experiment: Details of the experimental.
    :param sample: Sample information.
    :param measurement: Measurement specifics.
    """

    owner: Person
    experiment: Experiment
    sample: Sample
    measurement: Measurement
    _orso_optionals = []

    __repr__ = Header._staggered_repr

    comment: Optional[str] = None

    def __init__(self, owner, experiment, sample, measurement, *, comment=None, **kwds):
        super(DataSource, self).__init__()
        self.owner = owner
        self.experiment = experiment
        self.sample = sample
        self.measurement = measurement
        for k, v in kwds.items():
            setattr(self, k, v)
        self.comment = comment
        self.__post_init__()
