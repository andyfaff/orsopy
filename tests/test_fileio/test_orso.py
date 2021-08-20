"""
Tests for fileio module
"""

# author: Andrew R. McCluskey (arm61)

import unittest
import pathlib
from datetime import datetime
import yaml
from orsopy.fileio.orso import Orso
from orsopy.fileio.data_source import (DataSource, Experiment, Sample,
                                       Measurement, InstrumentSettings)
from orsopy.fileio.reduction import Reduction, Software
from orsopy.fileio.base import Person, ValueRange, Value, File, Column, Creator
from orsopy.fileio.base import _validate_header_data
from orsopy import fileio as fileio
import numpy as np


class TestOrso(unittest.TestCase):
    """
    Testing the Orso class.
    """
    def test_creation(self):
        """
        Creation of Orso object.
        """
        c = Creator(
            'A Person', 'Some Uni', datetime.now(), "",
            contact="wally@wallyland.com"
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
        value = Orso(c, ds, redn, 0, cols)

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
        assert value.data_set == '0'

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
            'A Person', 'Some Uni', datetime.now(), "",
            contact="wally@wallyland.com"
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
        value = Orso(c, ds, redn, 1, cols)

        dsm = value.data_source.measurement
        assert value.data_source.owner.name == 'A Person'
        assert dsm.data_files[0].file == 'README.rst'
        assert value.reduction.software.name == 'orsopy'
        assert value.columns[0].name == 'Qz'
        assert value.data_set == '1'

    def test_write_read(self):
        # test write and read of multiple datasets
        info = fileio.Orso.empty()
        info2 = fileio.Orso.empty()
        data = np.zeros((100, 3))
        data[:] = np.arange(100.0)[:, None]

        info.columns = [
            fileio.Column("q", "1/A"),
            fileio.Column("R", "1"),
            fileio.Column("sR", "1"),
        ]
        info2.columns = info.columns
        info.data_source.measurement.instrument_settings.polarization = "+"
        info2.data_source.measurement.instrument_settings.polarization = "-"
        info.data_set = "up polarization"
        info2.data_set = "down polarization"
        info2.data_source.sample.comment = "this is a comment"

        ds = fileio.OrsoDataset(info, data)
        ds2 = fileio.OrsoDataset(info2, data)

        info3 = fileio.Orso(
            creator=fileio.Creator(
                name="Artur Glavic",
                affiliation="Paul Scherrer Institut",
                time=datetime.now(),
                computer="localhost",
            ),
            data_source=fileio.DataSource(
                sample=fileio.Sample(
                    name="My Sample",
                    type="solid",
                    description="Something descriptive",
                ),
                experiment=fileio.Experiment(
                    title="Main experiment",
                    instrument="Reflectometer",
                    date=datetime.now(),
                    probe="x-rays",
                ),
                owner=fileio.Person("someone", "important"),
                measurement=fileio.Measurement(
                    instrument_settings=fileio.InstrumentSettings(
                        incident_angle=fileio.Value(13.4, "deg"),
                        wavelength=fileio.Value(5.34, "A"),
                    ),
                    data_files=["abc", "def", "ghi"],
                    references=["more", "files"],
                    scheme="angle-dispersive",
                ),
            ),
            reduction=fileio.Reduction(software="awesome orso"),
            data_set="Filled header",
            columns=info.columns,
        )
        ds3 = fileio.OrsoDataset(info3, data)

        fileio.save_orso([ds, ds2, ds3], "test.ort")

        ls1, ls2, ls3 = fileio.load_orso("test.ort")
        assert ls1 == ds
        assert ls2 == ds2
        assert ls3 == ds3


class TestFunctions(unittest.TestCase):
    """
    Tests for functionality in the Orso module.
    """
    def test_make_empty(self):
        """
        Creation of the empty Orso object.
        """
        empty = Orso.empty()
        assert issubclass(empty.__class__, Orso)
        ds = empty.data_source
        assert ds.owner.name is None
        assert ds.experiment.title is None
        assert ds.experiment.instrument is None
        assert ds.experiment.date is None
        assert ds.experiment.probe is None
        assert ds.sample.name is None
        assert ds.measurement.instrument_settings.incident_angle.magnitude is None
        assert ds.measurement.instrument_settings.wavelength.magnitude is None
        assert ds.measurement.data_files is None
        assert empty.reduction.software.name is None
        assert empty.reduction.software.version is None
        assert empty.reduction.software.platform is None
        assert empty.reduction.time is None
        assert empty.reduction.creator is None
        assert ds.owner.affiliation is None
        assert ds.sample.name is None
        assert empty.reduction.corrections is None
        assert empty.reduction.creator is None
        assert empty.columns == [Column.empty()]
        assert empty.data_set is None

    def test_empty_to_yaml(self):
        """
        Checking yaml string form empty Orso object.

        TODO: Fix once correct format is known.
        """
        empty = Orso.empty()
        assert empty.to_yaml() == (
            'creator:\n  name: null\n  affiliation: null\n  time: null\n'
            '  computer: null\ndata_source:\n  owner:\n    name: null\n'
            '    affiliation: null\n  experiment:\n    title: null\n'
            '    instrument: null\n    date: null\n    probe: null\n'
            '  sample:\n    name: null\n  measurement:\n'
            '    instrument_settings:\n      incident_angle:\n        magnitude: null\n'
            '      wavelength:\n        magnitude: null\n      polarization: unpolarized\n'
            '    data_files: null\nreduction:\n  software:\n    name: null\n'
            'data_set: null\ncolumns:\n- name: null\n'

        )
