# -*- coding: utf-8 -*-
# Copyright : INSEE, 2026

import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from pynsee.melodi import get_catalog, get_dataset, get_idbank, get_range
from pynsee.melodi.data import _parse_dataset_observations, _parse_metadata


def _mock_response(data):
    r = MagicMock()
    r.json.return_value = data
    return r


_METADATA_PAYLOAD = {
    "title": {"fr": "Jeu de données test", "en": "Test dataset"},
    "identifier": "DS_TEST",
    "publisher": {
        "id": "INSEE",
        "label": [
            {"lang": "fr", "content": "Institut national de la statistique"},
            {"lang": "en", "content": "National Institute of Statistics"},
        ],
    },
    "paging": {"count": 0, "isLast": True},
    "observations": [],
}

_OBS_DYNAMIC_MEASURE = {
    "observations": [
        {
            "dimensions": {"TIME_PERIOD": "2024-01", "FREQ": "M"},
            "attributes": {"OBS_STATUS": "A"},
            "measures": {"OBS_VALUE_INDICE_DE_PRIX": {"value": 105.3}},
        },
        {
            "dimensions": {"TIME_PERIOD": "2024-02", "FREQ": "M"},
            "attributes": {"OBS_STATUS": "A"},
            "measures": {"OBS_VALUE_INDICE_DE_PRIX": {"value": None}},
        },
    ]
}

_OBS_NO_MEASURES = {
    "observations": [
        {
            "dimensions": {"TIME_PERIOD": "2024"},
            "attributes": {"OBS_STATUS": "A"},
        }
    ]
}


class TestFunction(TestCase):

    # --- _parse_metadata ---

    def test_parse_metadata_all_languages(self):
        meta = _parse_metadata(_mock_response(_METADATA_PAYLOAD), language="all")
        test = isinstance(meta, dict)
        test = test & ("title_fr" in meta) & ("title_en" in meta)
        test = test & ("publisher_fr" in meta) & ("publisher_en" in meta)
        test = test & (meta["identifier"] == "DS_TEST")
        self.assertTrue(test)

    def test_parse_metadata_single_language(self):
        meta = _parse_metadata(_mock_response(_METADATA_PAYLOAD), language="fr")
        test = ("title_fr" in meta) & ("title_en" not in meta)
        test = test & ("publisher_fr" in meta) & ("publisher_en" not in meta)
        self.assertTrue(test)

    # --- _parse_dataset_observations ---

    def test_parse_observations_dynamic_measure_key(self):
        df = _parse_dataset_observations(_mock_response(_OBS_DYNAMIC_MEASURE))
        test = isinstance(df, pd.DataFrame)
        test = test & ("OBS_VALUE_INDICE_DE_PRIX" in df.columns)
        test = test & ("OBS_VALUE_NIVEAU" not in df.columns)
        test = test & (df["OBS_VALUE_INDICE_DE_PRIX"].iloc[0] == 105.3)
        test = test & pd.isna(df["OBS_VALUE_INDICE_DE_PRIX"].iloc[1])
        self.assertTrue(test)

    def test_parse_observations_no_measures_column(self):
        df = _parse_dataset_observations(_mock_response(_OBS_NO_MEASURES))
        test = isinstance(df, pd.DataFrame)
        test = test & ("TIME_PERIOD" in df.columns)
        self.assertTrue(test)

    def test_parse_observations_dimensions_flattened(self):
        df = _parse_dataset_observations(_mock_response(_OBS_DYNAMIC_MEASURE))
        test = isinstance(df, pd.DataFrame)
        test = test & ("TIME_PERIOD" in df.columns) & ("FREQ" in df.columns)
        test = test & ("dimensions" not in df.columns)
        test = test & ("attributes" not in df.columns)
        self.assertTrue(test)

    # --- get_catalog ---

    def test_get_catalog(self):
        df = get_catalog()
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_get_catalog_language_fr(self):
        test = True
        df = get_catalog(language="fr")
        test = test & isinstance(df, pd.DataFrame)
        test = test & ("dataset_identifier" in df.columns)
        test = test & (len(df) > 0)
        self.assertTrue(test)

    # --- get_range ---

    def test_get_range_sparse(self):
        test = True
        df = get_range("DS_TICM_PRATIQUES", language="fr")
        test = test & isinstance(df, pd.DataFrame)
        test = test & ("concept_code" in df.columns)
        test = test & ("concept_fr" in df.columns)
        test = test & (len(df) > 0)
        self.assertTrue(test)

    def test_get_range_with_values(self):
        test = True
        df = get_range(
            "DS_TICM_PRATIQUES",
            language="fr",
            include_values=True,
        )
        test = test & isinstance(df, pd.DataFrame)
        test = test & ("code" in df.columns)
        test = test & ("value_fr" in df.columns)
        test = test & (len(df) > 0)
        self.assertTrue(test)

    # --- get_dataset ---

    def test_get_dataset_1(self):
        df = get_dataset(
            "DS_TICM_PRATIQUES",
            language="fr",
            time_period=2025,
            sex="F",
            age="Y45T59",
        )
        test = isinstance(df, pd.DataFrame)
        test = test & ("OBS_VALUE_NIVEAU" in df.columns)
        test = test & (len(df) > 0)
        self.assertTrue(test)

    def test_get_dataset_dynamic_measures(self):
        # DS_IPC_PRINC expose OBS_VALUE_INDICE_DE_PRIX, pas OBS_VALUE_NIVEAU
        df = get_dataset(
            "DS_IPC_PRINC",
            language="fr",
            freq="M",
            time_period=2024,
        )
        test = isinstance(df, pd.DataFrame)
        test = test & ("OBS_VALUE_INDICE_DE_PRIX" in df.columns)
        test = test & df["OBS_VALUE_INDICE_DE_PRIX"].notna().any()
        self.assertTrue(test)

    # --- get_idbank ---

    def test_get_idbank_1(self):
        df = get_idbank("010770930")
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_get_idbank_2(self):
        df = get_idbank("010598544+010770930")
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_get_idbank_language_fr(self):
        df = get_idbank("010770930", language="fr")
        test = isinstance(df, pd.DataFrame)
        test = test & any(c.endswith("_fr") for c in df.columns)
        self.assertTrue(test)

    def test_get_idbank_dead_bdm_ids(self):
        df = get_idbank("001565530+001565531")
        test = isinstance(df, pd.DataFrame)
        test = test & df.empty
        self.assertTrue(test)


if __name__ == "__main__":
    unittest.main()