import json
import os.path
import jsonschema
import yaml
from orsopy import fileio
from orsopy.fileio.base import _read_header_data, _validate_header_data
import numpy as np


class TestSchema:
    def test_example_ort(self):
        pth = os.path.dirname(fileio.__file__)
        schema_pth = os.path.join(pth, "schema", "refl_header.schema.json")
        with open(schema_pth, "r") as f:
            schema = json.load_orso(f)

        dct_list, data = _read_header_data(
            os.path.join("tests", "test_example.ort"),
            validate=True
        )
        assert data[0].shape == (2, 4)

        # d contains datetime.datetime objects, which would fail the
        # jsonschema validation, so force those to be strings.
        modified_dct_list = [
            json.loads(json.dumps(dct, default=str)) for dct in dct_list
        ]
        for dct in modified_dct_list:
            jsonschema.validate(dct, schema)

        _validate_header_data(dct_list)

        # try a 2 dataset file
        dct_list, data = _read_header_data(
            os.path.join("tests", "test_example2.ort"),
            validate=True
        )
        assert len(dct_list) == 2
        assert dct_list[1]["data_set"] == "spin_down"
        assert data[1].shape == (4, 4)
        np.testing.assert_allclose(data[1][2:], data[0])
