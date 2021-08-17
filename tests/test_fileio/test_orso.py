"""
Tests for fileio module
"""

# author: Andrew R. McCluskey (arm61)

import unittest
import pathlib
from datetime import datetime
import yaml
from orsopy.fileio.orso import Orso, make_empty
from orsopy.fileio.data_source import (DataSource, Experiment, Sample,
                                       Measurement, InstrumentSettings)
from orsopy.fileio.reduction import Reduction, Software
from orsopy.fileio.base import Person, ValueRange, Value, File, Column, Creator
from orsopy.fileio.base import _validate_header_data


class TestOrso(unittest.TestCase):
    """
    Testing the Orso class.
    """
    def test_creation(self):
        """
        Creation of Orso object.
        """
        c = Creator(
            'A Person', 'Some Uni', "wally@wallyland.com", datetime.now(), ""
        )
        e = Experiment(
            'Experiment 1', 'ESTIA', datetime(2021, 7, 7, 16, 31, 10),
            'neutrons'
        )
        s = Sample('The sample')
        inst = InstrumentSettings(
            Value(4.0, 'deg'), ValueRange(2., 12., 'angstrom')
        )
        df = [File('README.rst', None)]
        m = Measurement(inst, df, scheme="angle-dispersive")
        p = Person('A Person', 'Some Uni')
        ds = DataSource(p, e, s, m)

        soft = Software('orsopy', '0.0.1', 'macOS-10.15')
        p2 = Person('Andrew McCluskey', 'European Spallation Source')
        redn = Reduction(
            soft, datetime(2021, 7, 14, 10, 10, 10),
            p2, ['footprint', 'background']
        )

        cols = [Column("Qz"), Column("R")]
        value = Orso(c, ds, redn, cols, 0)

        assert value.creator.name == "A Person"
        assert value.creator.contact == "wally@wallyland.com"
        ds = value.data_source
        dsm = ds.measurement
        assert ds.owner.name == 'A Person'
        assert dsm.data_files[0].file == 'README.rst'
        assert dsm.instrument_settings.incident_angle.magnitude == 4.0
        assert dsm.instrument_settings.wavelength.min == 2.0
        assert dsm.instrument_settings.wavelength.max == 12.0
        assert value.reduction.software.name == 'orsopy'
        assert value.reduction.software.version == "0.0.1"
        assert value.reduction.time == datetime(2021, 7, 14, 10, 10, 10)
        assert value.columns[0].name == 'Qz'
        assert value.columns[1].name == 'R'
        assert value.data_set == 0

        h = value.to_yaml()
        h = "\n".join(
            ["# ORSO reflectivity data file | 0.1 standard | YAML encoding"
             " | https://www.reflectometry.org/",
             h]
        )
        g = yaml.safe_load_all(h)
        _validate_header_data([next(g)])

    def test_creation_data_set1(self):
        """
        Creation of Orso object with a non-zero data_set.
        """
        c = Creator(
            'A Person', 'Some Uni', "wally@wallyland.com", datetime.now(), ""
        )
        e = Experiment(
            'Experiment 1', 'ESTIA', datetime(2021, 7, 7, 16, 31, 10),
            'neutrons'
        )
        s = Sample('The sample')
        inst = InstrumentSettings(
            Value(4.0, 'deg'), ValueRange(2., 12., 'angstrom')
        )
        df = [File('README.rst', None)]
        m = Measurement(inst, df, scheme="angle-dispersive")
        p = Person('A Person', 'Some Uni')
        ds = DataSource(p, e, s, m)

        soft = Software('orsopy', '0.0.1', 'macOS-10.15')
        p2 = Person('Andrew McCluskey', 'European Spallation Source')
        redn = Reduction(
            soft, datetime(2021, 7, 14, 10, 10, 10), p2,
            ['footprint', 'background']
        )

        cols = [Column("Qz"), Column("R")]
        value = Orso(c, ds, redn, cols, 1)

        dsm = value.data_source.measurement
        assert value.data_source.owner.name == 'A Person'
        assert dsm.data_files[0].file == 'README.rst'
        assert value.reduction.software.name == 'orsopy'
        assert value.columns[0].name == 'Qz'
        assert value.data_set == 1


class TestFunctions(unittest.TestCase):
    """
    Tests for functionality in the Orso module.
    """
    def test_make_empty(self):
        """
        Creation of the empty Orso object.
        """
        empty = make_empty()
        assert issubclass(empty.__class__, Orso)
        ds = empty.data_source
        assert ds.owner.name is None
        assert ds.experiment.title is None
        assert ds.experiment.instrument is None
        assert ds.experiment.date is None
        assert ds.experiment.probe is None
        assert ds.sample.name is None
        assert ds.measurement.instrument_settings.incident_angle is None
        assert ds.measurement.instrument_settings.wavelength is None
        assert ds.measurement.data_files is None
        assert empty.reduction.software.name is None
        assert empty.reduction.software.version is None
        assert empty.reduction.software.platform is None
        assert empty.reduction.time is None
        assert empty.reduction.creator is None
        assert empty.reduction.corrections is None
        assert empty.columns is None
        assert empty.data_set is None

    def test_empty_to_yaml(self):
        """
        Checking yaml string form empty Orso object.
        """
        empty = make_empty()
        assert empty.to_yaml() == (
            'creator:\n  name: null\n  affiliation: null\n  time: null\n'
            '  computer: null\ndata_source:\n  owner:\n    name: null\n'
            '    affiliation: null\n  experiment:\n    title: null\n'
            '    instrument: null\n    date: null\n    probe: null\n'
            '  sample:\n    name: null\n  measurement:\n'
            '    instrument_settings:\n      incident_angle: null\n'
            '      wavelength: null\n      polarization: unpolarized\n'
            '    data_files: null\nreduction:\n  software:\n    name: null\n'
            '  time: null\n  creator: null\n  corrections: null\n'
            'columns: null\ndata_set: null\n'
        )
