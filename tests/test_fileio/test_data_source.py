"""
Tests for fileio.data_source module
"""

# author: Andrew R. McCluskey (arm61)

import unittest
import pathlib
from datetime import datetime
from orsopy.fileio import data_source, base


class TestExperiment(unittest.TestCase):
    """
    Testing the Experiment class.
    """
    def test_creation(self):
        """
        Creation with minimal set.
        """
        value = data_source.Experiment('My First Experiment',
                                       'A Lab Instrument',
                                       datetime(1992, 7, 14, 10, 10,
                                                10), 'x-rays')
        assert value.title == "My First Experiment"
        assert value.instrument == 'A Lab Instrument'
        assert value.date == datetime(1992, 7, 14, 10, 10, 10)
        assert value.probe == 'x-rays'
        assert value.facility is None
        assert value.ID is None
        assert value.doi is None

    def test_to_yaml(self):
        """
        Transformation to yaml with minimal set.
        """
        value = data_source.Experiment('My First Experiment',
                                       'A Lab Instrument',
                                       datetime(1992, 7, 14, 10, 10,
                                                10), 'x-rays')
        assert value.to_yaml() == 'title: My First Experiment\n'\
            + 'instrument: A Lab Instrument\ndate: 1992-07-14T'\
            + '10:10:10\nprobe: x-rays\n'

    def test_creation_optionals(self):
        """
        Creation with optionals.
        """
        value = data_source.Experiment('My First Neutron Experiment',
                                       'TAS8',
                                       datetime(1992, 7, 14, 10, 10, 10),
                                       'neutrons',
                                       facility='Risoe',
                                       ID='abc123',
                                       doi='10.0000/abc1234')
        assert value.title == "My First Neutron Experiment"
        assert value.instrument == 'TAS8'
        assert value.date == datetime(1992, 7, 14, 10, 10, 10)
        assert value.probe == 'neutrons'
        assert value.facility == 'Risoe'
        assert value.ID == 'abc123'
        assert value.doi == '10.0000/abc1234'

    def test_to_yaml_optionals(self):
        """
        Transformation to yaml with optionals.
        """
        value = data_source.Experiment('My First Neutron Experiment',
                                       'TAS8',
                                       datetime(1992, 7, 14, 10, 10, 10),
                                       'neutrons',
                                       facility='Risoe',
                                       ID='abc123',
                                       doi='10.0000/abc1234')
        assert value.to_yaml() == 'title: My First Neutron Experiment\n'\
            + 'instrument: TAS8\ndate: 1992-07-14T'\
            + '10:10:10\nprobe: neutrons\nfacility: Risoe\nID: '\
            + 'abc123\ndoi: 10.0000/abc1234\n'


class TestSample(unittest.TestCase):
    """
    Testing for the Sample class.
    """
    def test_creation(self):
        """
        Creation with a minimal set.
        """
        value = data_source.Sample('A Perfect Sample')
        assert value.name == 'A Perfect Sample'
        assert value.type is None
        assert value.composition is None
        assert value.description is None
        assert value.environment is None

    def test_to_yaml(self):
        """
        Transformation to yaml with a minimal set.
        """
        value = data_source.Sample('A Perfect Sample')
        assert value.to_yaml() == 'name: A Perfect Sample\n'

    def test_creation_optionals(self):
        """
        Creation with a optionals.
        """
        value = data_source.Sample(
            'A Perfect Sample',
            type='solid/gas',
            composition='Si | SiO2(20 A) | Fe(200 A) | air(beam side)',
            description='The sample is without flaws',
            environment='Temperature cell')
        assert value.name == 'A Perfect Sample'
        assert value.type == 'solid/gas'
        assert value.composition == 'Si | SiO2(20 A) | '\
            + 'Fe(200 A) | air(beam side)'
        assert value.description == 'The sample is without flaws'
        assert value.environment == 'Temperature cell'

    def test_to_yaml_optionals(self):
        """
        Transformation to yaml with optionals.
        """
        value = data_source.Sample(
            'A Perfect Sample',
            type='solid/gas',
            composition='Si | SiO2(20 A) | Fe(200 A) | air(beam side)',
            description='The sample is without flaws',
            environment='Temperature cell')
        assert value.to_yaml() == 'name: A Perfect Sample\ntype: '\
            + 'solid/gas\ncomposition: Si | SiO2(20 A) | Fe(200 A) | air'\
            + '(beam side)\ndescription: The sample is without flaws\n'\
            + 'environment: Temperature cell\n'


class TestDataSource(unittest.TestCase):
    """
    Tests for the DataSource class.
    """
    def test_creation(self):
        """
        Creation with only default.
        """
        inst = data_source.InstrumentSettings(
            base.Value(0.25, unit='deg'),
            base.ValueRange(2, 20, unit="angstrom")
        )
        df = [
            base.File("1.nx.hdf", datetime.now()),
            base.File("2.nx.hdf", datetime.now())
        ]
        m = data_source.Measurement(inst, df, scheme="angle-dispersive")

        value = data_source.DataSource(
            base.Person('A Person', 'Some Uni'),
            data_source.Experiment('My First Experiment', 'A Lab Instrument',
                                   datetime(1992, 7, 14, 10, 10, 10), 'x-rays'),
            data_source.Sample('A Perfect Sample'),
            m)
        assert value.owner.name == 'A Person'
        assert value.owner.affiliation == 'Some Uni'
        assert value.experiment.title == 'My First Experiment'
        assert value.experiment.instrument == 'A Lab Instrument'
        assert value.experiment.date == datetime(1992, 7, 14, 10, 10, 10)
        assert value.experiment.probe == 'x-rays'
        assert value.sample.name == 'A Perfect Sample'


class TestInstrumentSettings(unittest.TestCase):
    """
    Tests for the InstrumentSettings class.
    """
    def test_creation(self):
        """
        Creation with minimal settings.
        """
        value = data_source.InstrumentSettings(
            base.Value(4., 'deg'),
            base.ValueRange(2., 12., 'angstrom'),
        )
        assert value.incident_angle.magnitude == 4.
        assert value.incident_angle.unit == 'deg'
        assert value.wavelength.min == 2.
        assert value.wavelength.max == 12.
        assert value.wavelength.unit == 'angstrom'
        assert value.polarization == 'unpolarized'
        assert value.configuration is None

    def test_to_yaml(self):
        """
        Transformation to yaml with minimal set.
        """
        value = data_source.InstrumentSettings(
            base.Value(4., 'deg'),
            base.ValueRange(2., 12., 'angstrom'),
        )
        assert value.to_yaml() == 'incident_angle:\n  magnitude: '\
            + '4.0\n  unit: deg\nwavelength:\n  min: 2.0\n  '\
            + 'max: 12.0\n  unit: angstrom\npolarization: unpolarized\n'

    def test_creation_config_and_polarization(self):
        """
        Creation with optional items.
        """
        value = data_source.InstrumentSettings(base.Value(4., 'deg'),
                                               base.ValueRange(
                                                   2., 12., 'angstrom'),
                                               polarization='+',
                                               configuration='liquid surface')
        assert value.incident_angle.magnitude == 4.
        assert value.incident_angle.unit == 'deg'
        assert value.wavelength.min == 2.
        assert value.wavelength.max == 12.
        assert value.wavelength.unit == 'angstrom'
        assert value.polarization == '+'
        assert value.configuration == 'liquid surface'

    def test_to_yaml_config_and_polarization(self):
        """
        Transformation to yaml with optional items.
        """
        value = data_source.InstrumentSettings(base.Value(4., 'deg'),
                                               base.ValueRange(
                                                   2., 12., 'angstrom'),
                                               polarization='+',
                                               configuration='liquid surface')
        assert value.to_yaml() == 'incident_angle:\n  magnitude: '\
            + '4.0\n  unit: deg\nwavelength:\n  min: 2.0\n  '\
            + 'max: 12.0\n  unit: angstrom\npolarization: +\n'\
            + 'configuration: liquid surface\n'


class TestMeasurement(unittest.TestCase):
    """
    Tests for the Measurement class.
    """
    def test_creation(self):
        """
        Creation with minimal set.
        """
        value = data_source.Measurement(
            data_source.InstrumentSettings(
                base.Value(4., 'deg'),
                base.ValueRange(2., 12., 'angstrom'),
            ), [
                base.File(str(pathlib.Path().resolve().joinpath("README.rst")),
                          None)
            ])
        assert value.instrument_settings.incident_angle.magnitude == 4.0
        assert value.instrument_settings.incident_angle.unit == 'deg'
        assert value.instrument_settings.wavelength.min == 2.0
        assert value.instrument_settings.wavelength.max == 12.0
        assert value.instrument_settings.wavelength.unit == 'angstrom'
        assert value.data_files[0].file == str(
            pathlib.Path().resolve().joinpath("README.rst"))
        assert value.data_files[0].created == datetime.fromtimestamp(
            pathlib.Path('README.rst').stat().st_mtime)

    def test_to_yaml(self):
        """
        Transform to yaml with minimal set.
        """
        value = data_source.Measurement(
            data_source.InstrumentSettings(
                base.Value(4., 'deg'),
                base.ValueRange(2., 12., 'angstrom'),
            ), [
                base.File(str(pathlib.Path().resolve().joinpath("README.rst")),
                          None)
            ])
        fname = pathlib.Path('README.rst')
        assert value.to_yaml() == 'instrument_settings:\n  incident_angle:'\
            + '\n    magnitude: 4.0\n    unit: deg\n  wavelength:\n    min: '\
            + '2.0\n    max: 12.0\n    unit: angstrom\n  polarization: '\
            + 'unpolarized\ndata_files:\n- file: '\
            + f'{str(fname.absolute())}\n  created: '\
            + f'{datetime.fromtimestamp(fname.stat().st_mtime).isoformat()}\n'

    def test_creation_optionals(self):
        """
        Creation with optionals.
        """
        value = data_source.Measurement(
            data_source.InstrumentSettings(
                base.Value(4., 'deg'),
                base.ValueRange(2., 12., 'angstrom'),
            ), [
                base.File(str(pathlib.Path().resolve().joinpath("README.rst")),
                          None)
            ], [
                base.File(
                    str(pathlib.Path().resolve().joinpath("AUTHORS.rst")),
                    None)
            ])
        assert value.instrument_settings.incident_angle.magnitude == 4.0
        assert value.instrument_settings.incident_angle.unit == 'deg'
        assert value.instrument_settings.wavelength.min == 2.0
        assert value.instrument_settings.wavelength.max == 12.0
        assert value.instrument_settings.wavelength.unit == 'angstrom'
        assert value.data_files[0].file == str(
            pathlib.Path().resolve().joinpath("README.rst"))
        assert value.data_files[0].created == datetime.fromtimestamp(
            pathlib.Path('README.rst').stat().st_mtime)
        assert value.references[0].file == str(
            pathlib.Path().resolve().joinpath("AUTHORS.rst"))
        assert value.references[
            0].created == datetime.fromtimestamp(
                pathlib.Path('AUTHORS.rst').stat().st_mtime)

    def test_to_yaml_optionals(self):
        """
        Transform to yaml with optionals.
        """
        value = data_source.Measurement(
            data_source.InstrumentSettings(
                base.Value(4., 'deg'),
                base.ValueRange(2., 12., 'angstrom'),
            ), [
                base.File(str(pathlib.Path().resolve().joinpath("README.rst")),
                          None)
            ], [
                base.File(
                    str(pathlib.Path().resolve().joinpath("AUTHORS.rst")),
                    None)
            ], 'energy-dispersive')
        fname = pathlib.Path('README.rst')
        gname = pathlib.Path('AUTHORS.rst')
        assert value.to_yaml() == 'instrument_settings:\n  incident_angle:'\
            + '\n    magnitude: 4.0\n    unit: deg\n  wavelength:\n    min: '\
            + '2.0\n    max: 12.0\n    unit: angstrom\n  polarization: '\
            + 'unpolarized\ndata_files:\n- file: '\
            + f'{str(fname.absolute())}\n  created: '\
            + f'{datetime.fromtimestamp(fname.stat().st_mtime).isoformat()}\n'\
            + 'references:\n- file: '\
            + f'{str(gname.absolute())}\n  created: '\
            + f'{datetime.fromtimestamp(gname.stat().st_mtime).isoformat()}\n'\
            + 'scheme: energy-dispersive\n'
