import json
import os.path
import jsonschema
import yaml
from orsopy import fileio
from orsopy.fileio.base import _read_header_data, _validate_header_data


class TestSchema:
    def test_example_ort(self):
        pth = os.path.dirname(fileio.__file__)
        schema_pth = os.path.join(pth, "schema", "refl_header.schema.json")
        with open(schema_pth, "r") as f:
            schema = json.load(f)

        dct_list, data = _read_header_data(
            os.path.join("tests", "test_example.ort")
        )

        # d contains datetime.datetime objects, which would fail the
        # jsonschema validation, so force those to be strings.
        modified_dct_list = [
            json.loads(json.dumps(dct, default=str)) for dct in dct_list
        ]
        for dct in modified_dct_list:
            jsonschema.validate(dct, schema)

        _validate_header_data(dct_list)
