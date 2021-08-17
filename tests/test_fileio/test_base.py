"""
Tests for fileio.base module
"""

# author: Andrew R. McCluskey (arm61)
# pylint: disable=R0201

from datetime import datetime
import unittest
import warnings
import pathlib
from numpy.testing import assert_equal
from orsopy.fileio import base


class TestValue(unittest.TestCase):
    """
    Testing the Value class.
    """
    def test_single_value(self):
        """
        Creation of an object with a magnitude and unit.
        """
        value = base.Value(1., 'm')
        assert value.magnitude == 1.
        assert value.unit == 'm'

    def test_list(self):
        """
        Creation of an object with a a list of values and a unit.
        """
        value = base.Value([1, 2, 3], 'm')
        assert_equal(value.magnitude, [1, 2, 3])
        assert value.unit == 'm'

    def test_bad_unit(self):
        """
        Rejection of non-ASCII units.
        """
        with self.assertRaises(ValueError):
            _ = base.Value(1., 'Å')

    def test_to_yaml(self):
        """
        Transform to yaml.
        """
        value = base.Value(1., 'm')
        assert value.to_yaml() == 'magnitude: 1.0\nunit: m\n'

    def test_no_magnitude_to_yaml(self):
        """
        Transform to yaml with a non-optional ORSO item.
        """
        value = base.Value(None)
        assert value.to_yaml() == 'magnitude: null\n'


class TestValueVector(unittest.TestCase):
    """
    Testing the ValueVector class
    """
    def test_single_value(self):
        """
        Creation of an object with three dimensions and unit.
        """
        value = base.ValueVector(1, 2, 3, 'm')
        assert value.x == 1
        assert value.y == 2
        assert value.z == 3
        assert value.unit == 'm'

    def test_list(self):
        """
        Creation of an object with three dimensions of lists and unit.
        """
        value = base.ValueVector([1, 2], [2, 3], [3, 4], 'm')
        assert_equal(value.x, [1, 2])
        assert_equal(value.y, [2, 3])
        assert_equal(value.z, [3, 4])
        assert value.unit == 'm'

    def test_bad_unit(self):
        """
        Rejection of non-ASCII unit.
        """
        with self.assertRaises(ValueError):
            _ = base.ValueVector(1, 2, 3, 'Å')

    def test_to_yaml(self):
        """
        Transform to yaml.
        """
        value = base.ValueVector(1., 2., 3., 'm')
        assert value.to_yaml() == 'x: 1.0\ny: 2.0\nz: 3.0\nunit: m\n'

    def test_two_dimensional_to_yaml(self):
        """
        Transform to yaml with only two dimensions.
        """
        value = base.ValueVector(1., 2., None, 'm')
        assert value.to_yaml() == 'x: 1.0\ny: 2.0\nz: null\nunit: m\n'


class TestValueRange(unittest.TestCase):
    """
    Testing the ValueRange class
    """
    def test_single_value(self):
        """
        Creation of an object with a max, min and unit.
        """
        value = base.ValueRange(1., 2., 'm')
        assert value.min == 1
        assert value.max == 2
        assert value.unit == 'm'

    def test_list(self):
        """
        Creation of an object of a list of max and list of min and a unit.
        """
        value = base.ValueRange([1, 2, 3], [2, 3, 4], 'm')
        assert_equal(value.min, [1, 2, 3])
        assert_equal(value.max, [2, 3, 4])
        assert value.unit == 'm'

    def test_bad_unit(self):
        """
        Rejection of non-ASCII unit.
        """
        with self.assertRaises(ValueError):
            _ = base.ValueRange(1., 2., 'Å')

    def test_to_yaml(self):
        """
        Transform to yaml.
        """
        value = base.ValueRange(1., 2., 'm')
        assert value.to_yaml() == 'min: 1.0\nmax: 2.0\nunit: m\n'

    def test_no_upper_to_yaml(self):
        """
        Transform to yaml with no max.
        """
        value = base.ValueRange(1., None)
        assert value.to_yaml() == 'min: 1.0\nmax: null\n'

    def test_no_lower_to_yaml(self):
        """
        Transform to yaml with no min.
        """
        value = base.ValueRange(None, 1.)
        assert value.to_yaml() == 'min: null\nmax: 1.0\n'


class TestPerson(unittest.TestCase):
    """
    Testing the Person class
    """
    def test_creation(self):
        """
        Creation with no email.
        """
        value = base.Person('Joe A. User', 'Ivy League University')
        assert value.name == 'Joe A. User'
        assert value.affiliation == 'Ivy League University'

    def test_creation_with_contact(self):
        """
        Creation with an email.
        """
        value = base.Person('Joe A. User', 'Ivy League University',
                            'jauser@ivy.edu')
        assert value.name == 'Joe A. User'
        assert value.affiliation == 'Ivy League University'
        assert value.contact == 'jauser@ivy.edu'

    def test_to_yaml(self):
        """
        Transform to yaml with no email.
        """
        value = base.Person('Joe A. User', 'Ivy League University')
        assert value.to_yaml(
        ) == 'name: Joe A. User\naffiliation: Ivy League University\n'

    def test_no_affiliation_to_yaml(self):
        """
        Transform to yaml without affiliation.
        """
        value = base.Person('Joe A. User', None)
        assert value.to_yaml() == 'name: Joe A. User\naffiliation: null\n'

    def test_no_name_to_yaml(self):
        """
        Transform to yaml without name.
        """
        value = base.Person(None, 'A University')
        assert value.to_yaml() == 'name: null\naffiliation: A University\n'

    def test_email_to_yaml(self):
        """
        Transform to yaml with an email.
        """
        value = base.Person('Joe A. User', 'Ivy League University',
                            'jauser@ivy.edu')
        assert value.to_yaml(
        ) == 'name: Joe A. User\naffiliation: Ivy League University'\
            + '\ncontact: jauser@ivy.edu\n'


class TestColumn(unittest.TestCase):
    """
    Testing the Column class
    """
    def test_creation(self):
        """
        Creation of a column.
        """
        value = base.Column('q', '1/angstrom', 'qz vector')
        assert value.dimension == 'qz vector'
        assert value.name == 'q'
        assert value.unit == '1/angstrom'

    def test_bad_unit(self):
        """
        Rejection of non-ASCII unit.
        """
        with self.assertRaises(ValueError):
            _ = base.Column('q', 'Å', 'qz vector')

    def test_to_yaml(self):
        """
        Transformation to yaml.
        """
        value = base.Column('q', '1/angstrom', 'qz vector')
        assert value.to_yaml(
        ) == 'name: q\nunit: 1/angstrom\ndimension: qz vector\n'

    def test_no_description_to_yaml(self):
        """
        Transformation to yaml.
        """
        value = base.Column('q', '1/angstrom')
        assert value.to_yaml() == 'name: q\nunit: 1/angstrom\n'


class TestFile(unittest.TestCase):
    """
    Testing the File class.
    """
    def test_creation_for_nonexistent_file(self):
        """
        Creation of a file that does not exist.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = base.File('not_a_file.txt',
                              datetime(2021, 7, 12, 14, 4, 20))
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert 'The file not_a_file.txt cannot be found.' == str(
                w[0].message)
        assert value.file == 'not_a_file.txt'
        assert value.created == datetime(2021, 7, 12, 14, 4, 20)

    def test_to_yaml_for_nonexistent_file(self):
        """
        Transformation to yaml of a file that does not exist.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = base.File('not_a_file.txt',
                              datetime(2021, 7, 12, 14, 4, 20))
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert 'The file not_a_file.txt cannot be found.' == str(
                w[0].message)
        assert value.to_yaml() == 'file: not_a_file.txt\ncreated: '\
            + '2021-07-12T14:04:20\n'

    def test_creation_for_existing_file(self):
        """
        Creation for a file that does exist with a given modified date.
        """
        fname = pathlib.Path('README.rst')
        value = base.File(str(fname.absolute()),
                          datetime.fromtimestamp(fname.stat().st_mtime))
        assert value.file == str(
            pathlib.Path().resolve().joinpath('README.rst'))
        assert value.created == datetime.fromtimestamp(fname.stat().st_mtime)

    def test_to_yaml_for_existing_file(self):
        """
        Transformation to yaml a file that does exist with a given modified
        date.
        """
        fname = pathlib.Path('README.rst')
        value = base.File(str(fname.absolute()),
                          datetime.fromtimestamp(fname.stat().st_mtime))
        assert value.to_yaml(
        ) == f'file: {str(pathlib.Path().resolve().joinpath("README.rst"))}\n'\
            + 'created: '\
            + f'{datetime.fromtimestamp(fname.stat().st_mtime).isoformat()}\n'

    def test_creation_for_existing_file_no_mod_time(self):
        """
            Transformation to yaml a file that does exist without a given
            modified date.
            """
        fname = pathlib.Path('AUTHORS.rst')
        value = base.File(str(fname.absolute()), None)
        assert value.file == str(
            pathlib.Path().resolve().joinpath("AUTHORS.rst"))
        assert value.created == datetime.fromtimestamp(fname.stat().st_mtime)

    def test_to_yaml_for_existing_file_no_mod_time(self):
        """
            Transformation to yaml a file that does exist without a given
            modified date.
            """
        fname = pathlib.Path('AUTHORS.rst')
        value = base.File(str(fname.absolute()), None)
        assert value.to_yaml(
        ) == 'file: '\
            + f'{str(pathlib.Path().resolve().joinpath("AUTHORS.rst"))}\n'\
            + 'created: '\
            + f'{datetime.fromtimestamp(fname.stat().st_mtime).isoformat()}\n'
