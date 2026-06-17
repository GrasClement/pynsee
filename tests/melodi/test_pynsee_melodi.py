# -*- coding: utf-8 -*-
# Copyright : INSEE, 2024

import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from pynsee.melodi import get_catalog, get_dataset, get_idbank, get_range
from pynsee.melodi.data import _parse_dataset_observations, _parse_metadata

# ---------------------------------------------------------------------------
# Mocks: the only mocks in the pynsee tests.
# Every other module tests internal helpers against the real API.
# Exception here: the bugs fixed in melodi/data/py live inside parsing helpers
# called inside get_dataset. Testing them through the real API would
# require downloading DS_IPC_PRINC on every test run.
# ---------------------------------------------------------------------------


def _mock_response(data):
    r = MagicMock()
    r.json.return_value = data
    return r


# Bug fix: measure key was hardcoded as OBS_VALUE_NIVEAU. DS_IPC_PRINC uses
# OBS_VALUE_INDICE_DE_PRIX: old code returned None for every price.
# Second obs has value=None to verify None -> NaN (not a raw dict).
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

# Bug fix: observations without a "measures" key crashed on obs.drop("measures").
# Guard added: if "measures" in obs.columns.
_OBS_NO_MEASURES = {
    "observations": [
        {
            "dimensions": {"TIME_PERIOD": "2024"},
            "attributes": {"OBS_STATUS": "A"},
        }
    ]
}

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


# ---------------------------------------------------------------------------
# DS_TICM_PRATIQUES = annual internet usage survey. Chosen because it is
# small (~350 obs), stable structure, downloads in one page.
# ---------------------------------------------------------------------------

# Exact set of 10 dimensions: fails visibly if INSEE restructures the dataset.
_TICM_CONCEPT_CODES = {
    "OBS_STATUS",
    "PCS_ESE",
    "SEX",
    "TICM_MEASURE",
    "EMPSTA",
    "FREQ",
    "TIME_PERIOD",
    "EDUC",
    "AGE",
    "MEASURE",
}

# Exact set of 16 survey measures: fails visibly if INSEE adds/removes one.
_TICM_MEASURES = {
    "ACHA_12MOIS",
    "ACHA_3MOIS",
    "VENTE_3MOIS",
    "COMPTE_BANCAIRE",
    "WEB_12MOIS_X_EGOV",
    "EGOV_12MOIS",
    "QUOTIDIEN",
    "EMAIL",
    "RESEAUX_SOC",
    "INFO_PRODUITS",
    "NO_EGOV",
    "WEB_3MOIS",
    "TEL_PAR_INT",
    "NO_WEB_12MOIS",
    "LECTURE",
    "PROFIL",
}


class TestFunction(TestCase):

    def test_parse_metadata_all_languages(self):
        # {"lang": "fr", "content": "..."} must be flattened to {"publisher_fr": ...}
        meta = _parse_metadata(
            _mock_response(_METADATA_PAYLOAD), language="all"
        )
        test = isinstance(meta, dict)
        test = test & ("title_fr" in meta) & ("title_en" in meta)
        test = test & ("publisher_fr" in meta) & ("publisher_en" in meta)
        test = test & (meta["identifier"] == "DS_TEST")
        self.assertTrue(test)

    def test_parse_metadata_single_language(self):
        # language="fr" must exclude _en columns
        meta = _parse_metadata(
            _mock_response(_METADATA_PAYLOAD), language="fr"
        )
        test = ("title_fr" in meta) & ("title_en" not in meta)
        test = test & ("publisher_fr" in meta) & ("publisher_en" not in meta)
        self.assertTrue(test)

    def test_parse_observations_dynamic_measure_key(self):
        # Regression: measure key must be extracted dynamically, not hardcoded.
        df = _parse_dataset_observations(_mock_response(_OBS_DYNAMIC_MEASURE))
        test = isinstance(df, pd.DataFrame)
        test = test & ("OBS_VALUE_INDICE_DE_PRIX" in df.columns)
        test = test & ("OBS_VALUE_NIVEAU" not in df.columns)
        test = test & (df["OBS_VALUE_INDICE_DE_PRIX"].iloc[0] == 105.3)
        test = test & pd.isna(df["OBS_VALUE_INDICE_DE_PRIX"].iloc[1])
        self.assertTrue(test)

    def test_parse_observations_no_measures_column(self):
        # Regression: observations without "measures" must not crash.
        df = _parse_dataset_observations(_mock_response(_OBS_NO_MEASURES))
        test = isinstance(df, pd.DataFrame)
        test = test & ("TIME_PERIOD" in df.columns)
        self.assertTrue(test)

    def test_parse_observations_dimensions_flattened(self):
        # "dimensions" and "attributes" must be unpacked into top-level columns.
        df = _parse_dataset_observations(_mock_response(_OBS_DYNAMIC_MEASURE))
        test = isinstance(df, pd.DataFrame)
        test = test & ("TIME_PERIOD" in df.columns) & ("FREQ" in df.columns)
        test = test & ("dimensions" not in df.columns)
        test = test & ("attributes" not in df.columns)
        self.assertTrue(test)

    def test_get_catalog(self):
        df = get_catalog()
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_get_catalog_known_datasets_present(self):
        # dataset_identifier is not unique (one dataset can have multiple products).
        # Assert presence of stable, long-running datasets rather than len > 0.
        test = True
        df = get_catalog(language="fr")
        known = {"DS_IPC_PRINC", "DS_TICM_PRATIQUES", "DS_ICA"}
        test = test & isinstance(df, pd.DataFrame)
        test = test & known.issubset(set(df["dataset_identifier"]))
        self.assertTrue(test)

    def test_get_range_sparse(self):
        # Assert exact dimension set, not just column presence.
        test = True
        df = get_range("DS_TICM_PRATIQUES", language="fr")
        test = test & isinstance(df, pd.DataFrame)
        test = test & (set(df["concept_code"]) == _TICM_CONCEPT_CODES)
        self.assertTrue(test)

    def test_get_range_with_values(self):
        # Assert exact measure set for TICM_MEASURE.
        test = True
        df = get_range("DS_TICM_PRATIQUES", language="fr", include_values=True)
        test = test & isinstance(df, pd.DataFrame)
        measures = set(
            df[df["concept_code"] == "TICM_MEASURE"]["code"].dropna()
        )
        test = test & (measures == _TICM_MEASURES)
        self.assertTrue(test)

    def test_get_dataset_1(self):
        # With sex=F, age=Y45T59, time_period=2025: exactly one row per measure
        # (all other dimensions have a single "total" modality _T).
        # Also verify the API actually applied the filters.
        df = get_dataset(
            "DS_TICM_PRATIQUES",
            language="fr",
            time_period=2025,
            sex="F",
            age="Y45T59",
        )
        test = isinstance(df, pd.DataFrame)
        test = test & (len(df) == len(_TICM_MEASURES))
        test = test & (set(df["TICM_MEASURE"]) == _TICM_MEASURES)
        test = test & (df["SEX"] == "F").all()
        test = test & (df["AGE"] == "Y45T59").all()
        self.assertTrue(test)

    def test_get_dataset_dynamic_measures(self):
        # End-to-end regression for the dynamic measure key fix.
        # freq="M", time_period=2024: keeps it to ~1 page (full year, published).
        df = get_dataset(
            "DS_IPC_PRINC", language="fr", freq="M", time_period=2024
        )
        test = isinstance(df, pd.DataFrame)
        test = test & ("OBS_VALUE_INDICE_DE_PRIX" in df.columns)
        test = test & df["OBS_VALUE_INDICE_DE_PRIX"].notna().any()
        self.assertTrue(test)

    def test_get_idbank_1(self):
        df = get_idbank("010770930")
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_get_idbank_series_history_stable(self):
        # 010770930 = DS_ICA monthly series, sector 46.19A (non-food purchasing
        # centres), base 2021. Open series: do NOT assert len (grows monthly).
        # Start date 1999-01 is a historical fact and will not change.
        test = True
        df = get_idbank("010770930", language="fr")
        test = test & isinstance(df, pd.DataFrame)
        test = test & (df["TIME_PERIOD"].min() == "1999-01")
        test = test & (df["idBank"] == "010770930").all()
        self.assertTrue(test)

    def test_get_idbank_multi(self):
        # nunique == 2 confirms both series were returned, not one duplicated.
        df = get_idbank("010598544+010770930")
        test = isinstance(df, pd.DataFrame)
        test = test & (df["idBank"].nunique() == 2)
        self.assertTrue(test)

    def test_get_idbank_language_fr(self):
        # Regression: "key2 in {'all', language}" was backwards: _fr columns
        # were absent when language="fr". Fixed to "language in {'all', key2}".
        df = get_idbank("010770930", language="fr")
        test = isinstance(df, pd.DataFrame)
        test = test & any(c.endswith("_fr") for c in df.columns)
        self.assertTrue(test)

    def test_get_idbank_dead_bdm_ids(self):
        # BDM series with no MELODI dataset yet -> HTTP 200 + empty list -> empty df.
        df = get_idbank("001565530+001565531")
        test = isinstance(df, pd.DataFrame)
        test = test & df.empty
        self.assertTrue(test)


if __name__ == "__main__":
    unittest.main()
