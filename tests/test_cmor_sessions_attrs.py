# tests/test_cmor_session_attrs.py
"""test cmor session attributes"""
from cmip7_prep import cmor_writer as cw


def test_session_sets_tracking_prefix_and_normalizes_license_product(fake_cmor):
    """test tracking_prefix and license strings"""
    fake, tables_path = fake_cmor
    print("Running test_session_sets_tracking_prefix_and_normalizes_license_product")
    # No tracking_id provided; product wrong to test normalization
    sess = cw.CmorSession(
        tables_path=tables_path,
        dataset_attrs={
            "institution_id": "NCAR",
            "product": "output",  # will be normalized to model-output
        },
    )
    with sess:
        pass

    # setup/dataset_json called
    assert fake.inpath == str(tables_path)
    assert isinstance(fake.dataset_json_path, str)

    # tracking_prefix set; tracking_id cleared (so CMOR can generate)
    assert fake.attrs.get("tracking_prefix") == "hdl:21.14100/"
    assert fake.attrs.get("tracking_id") == ""

    # product normalized
    assert fake.attrs.get("product") == "model-output"

    # license is a long paragraph; just sanity check start and URL
    lic = fake.attrs.get("license", "")
    assert lic.startswith("CMIP6 model data produced by NCAR")
    assert "https://creativecommons.org/licenses/by/4.0/" in lic
