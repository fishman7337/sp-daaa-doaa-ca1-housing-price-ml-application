"""Preprocessing helpers aligned with the feature-engineering pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

STRUCTURED_BASE_COLS: tuple[str, ...] = (
    "bed",
    "bath",
    "house_size",
    "acre_lot",
    "city",
    "state",
    "status",
)

CITY_PATTERN = re.compile(r"^[a-z][a-z .'-]+$")
CITY_BAD_TOKENS = {
    "rd",
    "road",
    "ave",
    "avenue",
    "st",
    "street",
    "dr",
    "drive",
    "ln",
    "lane",
    "trl",
    "trail",
    "hwy",
    "highway",
    "pkwy",
    "parkway",
    "unit",
    "county",
    "route",
    "ct",
    "court",
    "blvd",
    "terrace",
    "ter",
    "sq",
    "park",
    "center",
    "centre",
    "ctr",
    "circle",
    "cir",
    "pl",
    "place",
    "way",
    "heights",
}
STATE_NAME_TO_ABBR: dict[str, str] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "guam": "GU",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "puerto rico": "PR",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virgin islands": "VI",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

STATE_ABBR_TO_NAME: dict[str, str] = {
    abbr.lower(): name for name, abbr in STATE_NAME_TO_ABBR.items()
}


def _clean_city_name(name: str | None) -> str | None:
    """Normalize and filter noisy city labels down to plausible US cities."""
    if not isinstance(name, str):
        return None
    cleaned = name.strip().lower()
    if not cleaned or len(cleaned) < 2 or len(cleaned) > 60:
        return None
    if any(ch.isdigit() for ch in cleaned):
        return None
    if not CITY_PATTERN.fullmatch(cleaned):
        return None
    tokens = cleaned.split()
    if len(tokens) > 4:
        return None
    trailing = tokens[1:] if len(tokens) > 1 else []
    if any(tok in CITY_BAD_TOKENS for tok in trailing):
        return None
    return cleaned


def _clean_state_name(name: str | None) -> str | None:
    """Normalize state names/abbreviations to canonical lowercase names."""
    if not isinstance(name, str):
        return None
    cleaned = name.strip().lower().replace(".", "")
    if not cleaned:
        return None
    if cleaned in STATE_NAME_TO_ABBR:
        return cleaned
    if cleaned in STATE_ABBR_TO_NAME:
        return STATE_ABBR_TO_NAME[cleaned]
    return None


def _build_city_state_lookup(valid_states: set[str]) -> dict[str, set[str]]:
    """Build a lookup of city -> possible states using pgeocode (offline dataset).

    Only unambiguous mappings are later used; this merely captures candidates.
    """
    try:
        import pgeocode
    except ImportError:
        return {}

    nomi = pgeocode.Nominatim("US")
    df_geo = nomi._data[["place_name", "state_name", "state_code"]].dropna()

    lookup: dict[str, set[str]] = {}
    for _, row in df_geo.iterrows():
        states = {
            st
            for st in (
                _clean_state_name(row.get("state_name")),
                _clean_state_name(row.get("state_code")),
            )
            if st
        }
        if valid_states:
            states = {st for st in states if st in valid_states}
        if not states:
            continue

        place = str(row.get("place_name") or "")
        for part in place.split(","):
            city_clean = _clean_city_name(part)
            if not city_clean:
                continue
            lookup.setdefault(city_clean, set()).update(states)
    return lookup


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    """Element-wise safe division used by the feature engineering pipeline."""
    d_safe = d.replace(0, np.nan)
    return n / d_safe


def _safe_int(val: object) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def load_listing_counts(
    path: Path = Path("models/city_state_counts.json"),
    city_csv: Path = Path("models/city_listing_summary.csv"),
    state_csv: Path = Path("models/state_listing_summary.csv"),
) -> dict[str, dict[str, int]]:
    """Load city/state listing counts, cleaning noise and building a state->cities map.

    Preference is given to the CSV summaries shipped alongside the project. If
    they are absent, the legacy JSON fallback is used.
    """
    city_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    state_to_cities: dict[str, dict[str, int]] = {}

    # Load state counts first so we can validate state names when mapping cities.
    if state_csv.exists():
        df_state = pd.read_csv(state_csv)
        state_col = next((c for c in df_state.columns if "state" in c.lower()), None)
        cnt_col = next((c for c in df_state.columns if "count" in c.lower()), None)
        if state_col and cnt_col:
            for _, row in df_state.iterrows():
                state_clean = _clean_state_name(row.get(state_col))
                if not state_clean:
                    continue
                state_counts[state_clean] = _safe_int(row.get(cnt_col))

    # Load city-level counts, optionally with explicit state information.
    if city_csv.exists():
        df_city = pd.read_csv(city_csv)
        city_col = next((c for c in df_city.columns if "city" in c.lower()), None)
        cnt_col = next((c for c in df_city.columns if "count" in c.lower()), None)
        state_col = next((c for c in df_city.columns if "state" in c.lower()), None)

        if city_col and cnt_col:
            for _, row in df_city.iterrows():
                cleaned_city = _clean_city_name(row.get(city_col))
                if not cleaned_city:
                    continue
                count_val = _safe_int(row.get(cnt_col))
                city_counts[cleaned_city] = city_counts.get(cleaned_city, 0) + count_val

                if state_col:
                    state_clean = _clean_state_name(row.get(state_col))
                    if state_clean:
                        state_to_cities.setdefault(state_clean, {})[cleaned_city] = count_val

    # If city CSV lacked state column, derive a best-effort mapping using pgeocode.
    if city_counts and not state_to_cities:
        lookup = _build_city_state_lookup(set(state_counts.keys()))
        for city, count in city_counts.items():
            states = lookup.get(city, set())
            if len(states) == 1:
                st = next(iter(states))
                state_to_cities.setdefault(st, {})[city] = count
            elif len(states) > 1 and state_counts:
                # Break ties by picking the state with the larger listing volume.
                st = max(states, key=lambda s: state_counts.get(s, 0))
                state_to_cities.setdefault(st, {})[city] = count

    # Drop cities that cannot be mapped to a US state to avoid leaking addresses or non-US places.
    if state_to_cities:
        valid_cities: set[str] = set()
        for cities in state_to_cities.values():
            valid_cities.update(cities.keys())
        city_counts = {city: count for city, count in city_counts.items() if city in valid_cities}

    if state_to_cities and not state_counts:
        state_counts = {state: sum(vals.values()) for state, vals in state_to_cities.items()}

    # Fallback to legacy JSON if CSVs were missing or incomplete.
    if path.exists():
        raw = json.loads(path.read_text())
        for name, count in (raw.get("city_listing_count") or {}).items():
            cleaned = _clean_city_name(name)
            if cleaned and cleaned not in city_counts:
                city_counts[cleaned] = _safe_int(count)

        for name, count in (raw.get("state_listing_count") or {}).items():
            state_clean = _clean_state_name(name)
            if state_clean and state_clean not in state_counts:
                state_counts[state_clean] = _safe_int(count)

        if not state_to_cities and raw.get("state_to_cities"):
            # Legacy cache may already include a mapping.
            for state_name, cities in raw.get("state_to_cities", {}).items():
                state_clean = _clean_state_name(state_name)
                if not state_clean:
                    continue
                state_to_cities[state_clean] = {
                    _clean_city_name(city): _safe_int(cnt)
                    for city, cnt in (cities or {}).items()
                    if _clean_city_name(city)
                }

    return {
        "city_listing_count": city_counts,
        "state_listing_count": state_counts,
        "state_to_cities": state_to_cities,
    }


def engineer_minimal_from_payload(
    payload: dict[str, object],
    counts: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """Build a single-row DataFrame limited to the deployed feature set.

    Expected order:
    acre_lot_log1p, bath_log1p, city_listing_count, house_size,
    state_listing_count, status_is_ready_to_build, status_is_sold, total_rooms.
    """

    def _float(val: object) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    bed = _float(payload.get("bed") or payload.get("bedrooms"))
    bath = _float(payload.get("bath") or payload.get("bathrooms"))
    house_size = _float(payload.get("house_size") or payload.get("living_area"))

    acre = _float(payload.get("acre_lot"))
    if acre == 0.0:
        lot_sqft = _float(payload.get("lot_size"))
        acre = lot_sqft / 43_560 if lot_sqft else 0.0

    total_rooms = bed + bath

    city_raw = payload.get("city") or "unknown"
    state_raw = payload.get("state") or "unknown"
    status_raw = payload.get("status") or "unknown"

    city_clean = _clean_city_name(str(city_raw)) or "unknown"
    state_clean = _clean_state_name(str(state_raw)) or str(state_raw).strip().lower()
    status_clean = str(status_raw).strip().lower()

    counts = counts or {"city_listing_count": {}, "state_listing_count": {}}
    city_listing_count = counts.get("city_listing_count", {}).get(city_clean, 0)
    state_listing_count = counts.get("state_listing_count", {}).get(state_clean, 0)

    status_is_ready_to_build = int(status_clean == "ready_to_build")
    status_is_sold = int(status_clean == "sold")
    # For sale (or anything else) leaves both flags at 0, matching one-hot expectation.

    df = pd.DataFrame(
        [
            {
                "acre_lot_log1p": np.log1p(max(acre, 0)),
                "bath_log1p": np.log1p(max(bath, 0)),
                "city_listing_count": city_listing_count,
                "house_size": house_size,
                "state_listing_count": state_listing_count,
                "status_is_ready_to_build": status_is_ready_to_build,
                "status_is_sold": status_is_sold,
                "total_rooms": total_rooms,
            }
        ]
    )

    # Ensure column order exactly matches deployed model expectations.
    desired_cols = [
        "acre_lot_log1p",
        "bath_log1p",
        "city_listing_count",
        "house_size",
        "state_listing_count",
        "status_is_ready_to_build",
        "status_is_sold",
        "total_rooms",
    ]
    return df[desired_cols]


__all__ = [
    "STRUCTURED_BASE_COLS",
    "engineer_minimal_from_payload",
    "load_listing_counts",
    "STATE_NAME_TO_ABBR",
    "STATE_ABBR_TO_NAME",
]
