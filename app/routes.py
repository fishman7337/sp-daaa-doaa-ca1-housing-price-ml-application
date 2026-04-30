"""Flask routes for auth, prediction, and API endpoints."""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

from flask import (
    Blueprint,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import requests
import re
import tempfile

from .extensions import db
from .forms import LoginForm, PredictionForm, SignupForm
from .models import ChatMessage, Prediction, User
from .services.model_service import ModelService
from .services.preprocessing import (
    STATE_NAME_TO_ABBR,
    STATE_ABBR_TO_NAME,
    _clean_city_name,
    _clean_state_name,
)

main_bp = Blueprint("main", __name__)
TREND_CACHE: Dict[str, List[Dict[str, object]]] = {}


def _resolve_time_series_path() -> Optional[Path]:
    """Resolve the pre-aggregated time series CSV shipped in data/."""

    configured = current_app.config.get("TREND_TIMESERIES_PATH")
    if configured:
        p = Path(configured)
        if p.exists():
            return p
    project_root = Path(current_app.root_path).parent
    candidate_paths = [
        project_root / "data" / "usa_real_estate_price_time_series.csv",
        Path("data/usa_real_estate_price_time_series.csv"),
    ]
    return next((p for p in candidate_paths if p.exists()), None)


def _resolve_histogram_path() -> Optional[Path]:
    """Resolve the pre-aggregated histogram CSV shipped in data/."""

    configured = current_app.config.get("TREND_HISTOGRAM_PATH")
    if configured:
        p = Path(configured)
        if p.exists():
            return p
    project_root = Path(current_app.root_path).parent
    candidate_paths = [
        project_root / "data" / "usa_real_estate_price_histogram.csv",
        Path("data/usa_real_estate_price_histogram.csv"),
    ]
    return next((p for p in candidate_paths if p.exists()), None)


def _extract_drive_id(url: str) -> Optional[str]:
    """Pull the Google Drive file id from a typical share URL."""

    if "drive.google.com" in url and "/d/" in url:
        try:
            return url.split("/d/")[1].split("/")[0]
        except Exception:
            return None
    if "id=" in url:
        try:
            return url.split("id=")[1].split("&")[0]
        except Exception:
            return None
    return None


def _normalize_drive_url(url: str) -> str:
    """Convert a Google Drive share link to a direct download URL."""

    file_id = _extract_drive_id(url)
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def _download_csv_bytes(remote_url: str) -> Optional[bytes]:
    """Download CSV content; handle Google Drive virus scan confirmation for large files."""

    file_id = _extract_drive_id(remote_url)
    session = requests.Session()
    dl_url = _normalize_drive_url(remote_url)

    def _looks_like_html(content: bytes) -> bool:
        prefix = content.lstrip()[:50].lower()
        return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")

    try:
        resp = session.get(dl_url, timeout=90)
        resp.raise_for_status()
        content = resp.content
        if file_id and _looks_like_html(content):
            # Try to extract confirm token
            match = re.search(rb'confirm=([0-9A-Za-z_-]+)', content)
            token = match.group(1).decode() if match else "t"
            confirm_url = "https://drive.usercontent.google.com/download"
            resp = session.get(confirm_url, params={"id": file_id, "export": "download", "confirm": token}, timeout=180)
            resp.raise_for_status()
            content = resp.content
        if _looks_like_html(content) or len(content) < 1024:
            # Fallback to gdown for large Drive files
            if file_id:
                try:
                    import gdown

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        tmp_path = tmp.name
                    gdown.download(id=file_id, output=tmp_path, quiet=True, use_cookies=False)
                    with open(tmp_path, "rb") as fh:
                        content = fh.read()
                    Path(tmp_path).unlink(missing_ok=True)
                    if _looks_like_html(content) or len(content) < 1024:
                        return None
                    return content
                except Exception:
                    return None
            return None
        return content
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Failed to download CSV: %s", exc)
        return None


def _resolve_trend_data_path() -> Optional[Path]:
    """
    Resolve the trend/distribution CSV:
    1) Use TREND_DATA_PATH if it exists.
    2) Otherwise download from TREND_DATA_URL (supports Google Drive) to instance/cache.
    3) Fall back to local project paths.
    """

    configured = current_app.config.get("TREND_DATA_PATH")
    if configured:
        p = Path(configured)
        if p.exists():
            return p

    remote_url = current_app.config.get("TREND_DATA_URL") or os.environ.get("TREND_DATA_URL")
    if remote_url:
        cache_dir = Path(current_app.instance_path) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / "usa_real_estate_clean.csv"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        content = _download_csv_bytes(remote_url)
        if content:
            dest.write_bytes(content)
            return dest
        else:
            current_app.logger.warning("Failed to download trend data from %s", remote_url)
            if dest.exists():
                dest.unlink(missing_ok=True)

    project_root = Path(current_app.root_path).parent
    candidate_paths = [
        project_root / "data" / "processed" / "usa_real_estate_clean.csv",
        Path("data/processed/usa_real_estate_clean.csv"),
    ]
    return next((p for p in candidate_paths if p.exists()), None)


def _load_price_dataframe(data_path: Path) -> pd.DataFrame:
    """
    Load the price CSV with robust column normalization.

    Accepts variations like upper-case headers or alternative names
    (e.g., sold_price, sold_date). Returns an empty DataFrame if required
    columns are missing.
    """

    try:
        df = pd.read_csv(data_path)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Failed to read price CSV: %s", exc)
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]
    col_map = {}

    # price column
    if "price" in df.columns:
        col_map["price"] = "price"
    elif "sold_price" in df.columns:
        col_map["price"] = "sold_price"

    # date column
    if "prev_sold_date" in df.columns:
        col_map["prev_sold_date"] = "prev_sold_date"
    elif "sold_date" in df.columns:
        col_map["prev_sold_date"] = "sold_date"
    elif "date" in df.columns:
        col_map["prev_sold_date"] = "date"

    # city/state
    if "city" in df.columns:
        col_map["city"] = "city"
    if "state" in df.columns:
        col_map["state"] = "state"

    required = {"price", "prev_sold_date"}
    if not required.issubset(col_map):
        return pd.DataFrame()

    df = df.rename(columns={v: k for k, v in col_map.items()})
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_model_service() -> ModelService:
    if "model_service" not in g:
        g.model_service = ModelService(
            model_dir=Path(current_app.config["MODEL_DIR"]),
            upload_dir=Path(current_app.config["UPLOAD_FOLDER"]),
        )
    # Refresh listing counts if the cached service lacks the state->city map.
    if not getattr(g.model_service, "listing_counts", {}).get("state_to_cities"):
        from .services.preprocessing import load_listing_counts  # local import to avoid cycles

        g.model_service.listing_counts = load_listing_counts()
    return g.model_service


def _valid_city_names(listing_counts: Dict[str, Dict[str, object]]) -> list[str]:
    """Return a sorted list of cities mapped to a known US state."""

    state_to_cities = listing_counts.get("state_to_cities") or {}
    if state_to_cities:
        city_pool = {
            city for cities in state_to_cities.values() for city in (cities or {}).keys()
        }
    else:
        city_pool = set((listing_counts.get("city_listing_count") or {}).keys())
    return sorted(city_pool)


def _load_price_trend(state: str | None = None, city: str | None = None) -> list[dict]:
    """Load (and cache) monthly mean/median price trend for optional state/city filters."""

    cache_key = f"{state or ''}|{city or ''}"
    if cache_key in TREND_CACHE:
        return TREND_CACHE[cache_key]

    ts_path = _resolve_time_series_path()
    if not ts_path or not ts_path.exists():
        TREND_CACHE[cache_key] = []
        return []

    try:
        df = pd.read_csv(ts_path)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Failed to read time series CSV: %s", exc)
        TREND_CACHE[cache_key] = []
        return []

    df.columns = [c.strip().lower() for c in df.columns]
    # Map expected columns
    if "price_mean" in df.columns:
        df = df.rename(columns={"price_mean": "mean"})
    if "price_median" in df.columns:
        df = df.rename(columns={"price_median": "median"})
    if "n_obs" in df.columns:
        df = df.rename(columns={"n_obs": "count"})

    if "date" not in df.columns or not {"mean", "median"}.issubset(df.columns):
        TREND_CACHE[cache_key] = []
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if state and "state" in df.columns:
        df = df[df["state"].str.lower() == state]
    if city and "city" in df.columns:
        df = df[df["city"].str.lower() == city]
    df = df[df["date"].notna()]
    df = df.sort_values("date")
    if df.empty:
        TREND_CACHE[cache_key] = []
        return []

    trend: list[dict] = []
    for _, row in df.iterrows():
        try:
            trend.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "mean": float(row["mean"]),
                    "median": float(row["median"]),
                    "count": int(row["count"]) if "count" in df.columns else 0,
                }
            )
        except Exception:
            continue

    TREND_CACHE[cache_key] = trend
    return trend


def _load_price_distribution() -> dict:
    """Return histogram edges/counts for recent prices (last 5 years), cached."""

    cache_key = "price_distribution"
    if cache_key in TREND_CACHE:
        return TREND_CACHE[cache_key]

    hist_path = _resolve_histogram_path()
    if not hist_path or not hist_path.exists():
        TREND_CACHE[cache_key] = {"edges": [], "counts": []}
        return TREND_CACHE[cache_key]

    try:
        hdf = pd.read_csv(hist_path)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Failed to read histogram CSV: %s", exc)
        TREND_CACHE[cache_key] = {"edges": [], "counts": []}
        return TREND_CACHE[cache_key]

    hdf.columns = [c.strip().lower() for c in hdf.columns]
    if "price" not in hdf.columns:
        TREND_CACHE[cache_key] = {"edges": [], "counts": []}
        return TREND_CACHE[cache_key]

    prices = pd.to_numeric(hdf["price"], errors="coerce").dropna().values
    prices = prices[prices > 0]
    if prices.size == 0:
        TREND_CACHE[cache_key] = {"edges": [], "counts": []}
        return TREND_CACHE[cache_key]

    counts, edges = np.histogram(prices, bins=20)
    result = {"edges": edges.tolist(), "counts": counts.tolist()}
    TREND_CACHE[cache_key] = result
    return result


def _save_images(files: list[FileStorage]) -> list[str]:
    """Persist uploaded images and return relative paths."""

    saved: list[str] = []
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files or []:
        if not file:
            continue
        filename = secure_filename(file.filename or "")
        if not filename:
            continue
        token = secrets.token_hex(8)
        filename = f"{token}_{filename}"
        filepath = upload_dir / filename
        file.save(filepath)
        saved.append(f"uploads/{filename}")
    return saved


def _collect_structured_payload(form: PredictionForm) -> Optional[Dict[str, object]]:
    """Extract structured fields if any were provided."""

    fields = {
        "city": form.city.data,
        "state": form.state.data,
        "status": form.status.data,
        "bedrooms": form.bedrooms.data,
        "bathrooms": form.bathrooms.data,
        "living_area": form.living_area.data,
        "lot_size": form.lot_size.data,
    }
    if any(v not in (None, "", 0) for v in fields.values()):
        return fields
    return None


def _record_prediction(
    preds: Dict[str, float],
    final_price: float,
    structured_payload: Optional[Dict[str, object]],
    description: Optional[str],
    image_rel_path: Optional[str],
) -> Prediction:
    """Persist a prediction record for the current user."""

    prediction = Prediction(
        user_id=current_user.id,
        structured_payload=structured_payload,
        description=(description or "").strip(),
        image_path=image_rel_path,
        tabular_price=preds.get("tabular"),
        nlp_price=preds.get("nlp"),
        cnn_price=preds.get("cnn"),
        final_price=final_price,
    )
    db.session.add(prediction)
    db.session.commit()
    return prediction


def _build_location_choices(listing_counts: Dict[str, Dict[str, int]]):
    """Return city/state select choices from cleaned listing counts."""

    valid_cities = _valid_city_names(listing_counts)
    state_counts = listing_counts.get("state_listing_count") or {}
    state_to_cities = listing_counts.get("state_to_cities") or {}
    state_keys = sorted(set(state_counts.keys()) | set(state_to_cities.keys()))

    city_choices = [("", "Select city")] + [(c, c.title()) for c in valid_cities]
    state_choices = [
        ("", "Select state")
    ] + [
        (s, f"{STATE_NAME_TO_ABBR.get(s, s.upper())} - {s.title()}")
        for s in state_keys
    ]
    return city_choices, state_choices


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------
@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash("Username already exists.", "danger")
            return render_template("signup.html", form=form)
        if User.query.filter_by(email=form.email.data).first():
            flash("Email already registered.", "danger")
            return render_template("signup.html", form=form)

        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Account created. Please log in.", "success")
        return redirect(url_for("main.login"))
    return render_template("signup.html", form=form)


@main_bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            flash("Welcome back!", "success")
            return redirect(url_for("main.dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("login.html", form=form)


@main_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("main.index"))


# ---------------------------------------------------------------------------
# Prediction dashboard
# ---------------------------------------------------------------------------
@main_bp.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    form = PredictionForm()
    service = _get_model_service()
    listing_counts = getattr(service, "listing_counts", {}) or {}
    city_choices, state_choices = _build_location_choices(listing_counts)
    # Pre-populate city/state choices from filtered US cities
    form.city.choices = city_choices
    form.state.choices = state_choices
    city_options = _valid_city_names(listing_counts)

    history = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )
    history_payload = [item.to_dict() for item in history]

    if form.validate_on_submit():
        image_rel_paths = _save_images(form.image.data) if form.image.data else []
        image_abs_paths = [
            Path(current_app.config["UPLOAD_FOLDER"]) / Path(rel_path).name
            for rel_path in image_rel_paths
        ]

        structured_payload = _collect_structured_payload(form)
        description = form.description.data or ""

        try:
            preds, final_price = service.predict(
                structured_payload=structured_payload,
                description=description,
                image_paths=image_abs_paths,
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"Prediction failed: {exc}", "danger")
            return render_template(
                "dashboard.html",
                form=form,
                history=history,
                history_payload=history_payload,
                city_options=city_options,
            )

        _record_prediction(
            preds=preds,
            final_price=final_price,
            structured_payload=structured_payload,
            description=description,
            image_rel_path=image_rel_paths[0] if image_rel_paths else None,
        )
        flash("Prediction completed!", "success")
        return redirect(url_for("main.dashboard"))

    return render_template(
        "dashboard.html",
        form=form,
        history=history,
        history_payload=history_payload,
        city_options=city_options,
        state_cities_map={
            state: sorted((cities or {}).keys())
            for state, cities in (listing_counts.get("state_to_cities") or {}).items()
        },
    )


# ---------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------
@main_bp.route("/api/history", methods=["GET"])
@login_required
def api_history():
    history = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(50)
        .all()
    )
    return jsonify([item.to_dict() for item in history])


@main_bp.route("/api/history/<int:history_id>", methods=["DELETE"])
@login_required
def api_delete_history(history_id: int):
    """Delete a single prediction owned by the current user."""

    record = Prediction.query.filter_by(id=history_id, user_id=current_user.id).first()
    if not record:
        return jsonify({"error": "Not found"}), 404
    db.session.delete(record)
    db.session.commit()
    return jsonify({"status": "deleted", "id": history_id})


@main_bp.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    form = PredictionForm()
    service = _get_model_service()
    listing_counts = getattr(service, "listing_counts", {}) or {}
    form.city.choices, form.state.choices = _build_location_choices(listing_counts)

    if not form.validate_on_submit():
        return jsonify({"error": "Invalid input", "messages": form.errors}), 400

    image_rel_paths = _save_images(form.image.data) if form.image.data else []
    image_abs_paths = [
        Path(current_app.config["UPLOAD_FOLDER"]) / Path(p).name for p in image_rel_paths
    ]
    structured_payload = _collect_structured_payload(form)
    description = form.description.data or ""

    try:
        preds, final_price = _get_model_service().predict(
            structured_payload=structured_payload,
            description=description,
            image_paths=image_abs_paths,
        )
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Prediction failed: %s", exc)
        return jsonify({"error": "Prediction failed"}), 500
    record = _record_prediction(
        preds=preds,
        final_price=final_price,
        structured_payload=structured_payload,
        description=description,
        image_rel_path=image_rel_paths[0] if image_rel_paths else None,
    )
    safe_preds = {k: (float(v) if v is not None else None) for k, v in preds.items()}
    return jsonify({"predictions": safe_preds, "final_price": float(final_price), "history_id": record.id})


@main_bp.route("/api/cities")
@login_required
def api_cities():
    """Return cities for the current user filtered by state if available."""

    state_param = (request.args.get("state") or "").strip().lower()
    service = _get_model_service()
    listing_counts = getattr(service, "listing_counts", {}) or {}
    state_to_cities = listing_counts.get("state_to_cities") or {}
    valid_city_pool = set(_valid_city_names(listing_counts))

    cities: set[str] = set()
    if state_param:
        normalized_states = {state_param}
        if state_param in STATE_ABBR_TO_NAME:
            normalized_states.add(STATE_ABBR_TO_NAME[state_param])
        for state_name in normalized_states:
            cities.update(state_to_cities.get(state_name, {}).keys())
        cities = {c for c in cities if c in valid_city_pool}
    if not cities and state_param:
        # Legacy fallback: infer from prior predictions if mapping is missing.
        records = (
            Prediction.query.filter(Prediction.structured_payload.isnot(None))
            .order_by(Prediction.created_at.desc())
            .limit(1000)
            .all()
        )
        for rec in records:
            payload = rec.structured_payload or {}
            st = str(payload.get("state") or "").strip().lower()
            city = str(payload.get("city") or "").strip()
            if st == state_param and city:
                cities.add(city)

    # If still empty, fall back to all known cities so the user isn't blocked
    if not cities:
        cities = valid_city_pool
    return jsonify(sorted(cities))


@main_bp.route("/api/price-trend", methods=["GET"])
@login_required
def api_price_trend():
    """Return monthly mean/median price trend, optionally filtered by state/city."""

    state_param = (request.args.get("state") or "").strip().lower()
    city_param = (request.args.get("city") or "").strip().lower()
    try:
        state_clean = _clean_state_name(state_param) if state_param else None
        city_clean = _clean_city_name(city_param) if city_param else None
        trend = _load_price_trend(state_clean, city_clean)
        return jsonify(trend)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Trend API failed: %s", exc)
        return jsonify([])


@main_bp.route("/api/price-distribution", methods=["GET"])
@login_required
def api_price_distribution():
    """Return histogram counts/edges for recent sale prices."""

    try:
        data = _load_price_distribution()
        return jsonify(data)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Distribution API failed: %s", exc)
        return jsonify({"edges": [], "counts": []})


@main_bp.route("/api/chat/history", methods=["GET"])
@login_required
def api_chat_history():
    """Return chat history for the logged-in user."""

    messages = (
        ChatMessage.query.filter_by(user_id=current_user.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return jsonify([m.to_dict() for m in messages])


@main_bp.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    """Chat endpoint. Tied to user accounts; stores/retrieves per-user history."""

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    lower = message.lower()
    housing_terms = [
        "house", "housing", "home", "property", "real estate", "listing", "mls", "zillow",
        "price", "worth", "valuation", "appraisal", "comps", "rent", "lease", "mortgage",
        "down payment", "interest rate", "hoa", "tax", "neighborhood", "zip", "bedroom", "bath",
    ]
    if not any(term in lower for term in housing_terms):
        return jsonify({"reply": "I'm focused on housing questions (pricing, comps, listings, mortgages). Ask me about a property or the market."})

    # Pull prior history from DB for context (limit to last 20)
    history_msgs = (
        ChatMessage.query.filter_by(user_id=current_user.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    context = [{"role": m.role, "content": m.content} for m in history_msgs][-20:]

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI  # type: ignore

            messages = [{"role": "system", "content": "You are a helpful housing assistant. Only answer housing-related questions (valuations, comps, listings, mortgages, renovations, neighborhoods) and politely refuse unrelated topics. Keep responses concise and useful."}]
            for item in context:
                role = item.get("role")
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                if role == "bot":
                    role = "assistant"
                if role not in ("user", "assistant", "system"):
                    continue
                messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": message})

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=current_app.config.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                max_tokens=180,
            )
            reply = completion.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            current_app.logger.exception("OpenAI chat failed: %s", exc)
            reply = None
    else:
        reply = None

    lower = message.lower()

    def respond_fallback() -> str:
        if any(k in lower for k in ["price", "worth", "valuation", "value"]):
            return "To estimate a home's price, combine recent comps, local medians, and your inputs. Try filling the form so the ensemble can blend tabular, text, and images."
        if any(k in lower for k in ["sell", "listing", "list"]):
            return "Strong listings highlight bedrooms/bathrooms, lot size, recent upgrades, and bright images. Pricing near recent neighborhood medians often drives faster interest."
        if any(k in lower for k in ["buy", "offer", "bid"]):
            return "Compare recent sales, days on market, and price per sqft. If the model's predicted price is below ask, consider negotiating with contingencies."
        if any(k in lower for k in ["mortgage", "loan", "rate"]):
            return "Watch current rates and plan for taxes/insurance/HOA. A 1% rate change can shift affordability meaningfully."
        if any(k in lower for k in ["image", "photo", "cnn"]):
            return "You can upload multiple images; the CNN averages them. Clear, well-lit exterior and main room shots tend to be most informative."
        return "Ask me about housing prices, comps, images, or listing strategy. For the most accurate valuation, submit the prediction form with full tabular info and any images."

    final_reply = reply or respond_fallback()

    # Persist user message and bot reply
    try:
        db.session.add(ChatMessage(user_id=current_user.id, role="user", content=message))
        db.session.add(ChatMessage(user_id=current_user.id, role="assistant", content=final_reply))
        db.session.commit()
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Failed to save chat messages: %s", exc)
        db.session.rollback()

    return jsonify({"reply": final_reply})
