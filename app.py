"""
app.py
------
Flask entry point for the Digital Twin Classroom Scheduler.
Defines all routes and passes data to Jinja2 templates.
"""

import csv
import os
import re
import traceback
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import google.generativeai as genai
    has_genai = True
except ImportError:
    has_genai = False

import analyzer

if load_dotenv:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "classroom-scheduler-dev-secret")
SCHEDULE_COLUMNS = pd.read_csv(analyzer.SCHEDULE_CSV, nrows=0).columns.tolist()
ROOMS_COLUMNS = pd.read_csv(analyzer.ROOMS_CSV, nrows=0).columns.tolist()
DATA_DIR = os.path.join(analyzer.BASE_DIR, "data")
ACTIVE_SCHEDULE_POINTER = os.path.join(DATA_DIR, ".active_schedule")
BOOKING_LOG_CSV = os.path.join(DATA_DIR, "booking_changes.csv")
DAY_ALIAS_TO_CANONICAL = {
    "sun": "Sun",
    "sunday": "Sun",
    "mon": "Mon",
    "monday": "Mon",
    "tue": "Tue",
    "tues": "Tue",
    "tuesday": "Tue",
    "wed": "Wed",
    "wednesday": "Wed",
    "thu": "Thu",
    "thur": "Thu",
    "thurs": "Thu",
    "thursday": "Thu",
    "fri": "Fri",
    "friday": "Fri",
    "sat": "Sat",
    "saturday": "Sat",
}
DAY_SORT_ORDER = {"Sun": 0, "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6}
IMSIU_BOOKING_DAYS = {"Sun", "Mon", "Tue", "Wed", "Thu"}


def _get_gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _get_schedule_datasets():
    active_dataset = _get_active_schedule_dataset()
    datasets = []
    try:
        csv_files = sorted(
            file_name for file_name in os.listdir(DATA_DIR)
            if file_name.lower().endswith(".csv") and file_name.startswith("schedule")
        )
    except FileNotFoundError:
        return []

    for schedule_file in csv_files:
        datasets.append(
            {
                "id": schedule_file,
                "schedule_file": schedule_file,
                "is_active": schedule_file == active_dataset,
                "label": schedule_file,
            }
        )
    return datasets


def _normalize_schedule_dataset_name(name: str) -> Optional[str]:
    candidate = os.path.basename(str(name or "").strip())
    if not candidate:
        return None
    if not candidate.lower().endswith(".csv"):
        return None
    if not candidate.startswith("schedule"):
        return None
    full_path = os.path.join(DATA_DIR, candidate)
    if not os.path.isfile(full_path):
        return None
    return candidate


def _get_active_schedule_dataset() -> str:
    default_name = "schedule.csv"
    if not os.path.exists(ACTIVE_SCHEDULE_POINTER):
        return default_name
    try:
        with open(ACTIVE_SCHEDULE_POINTER, "r", encoding="utf-8") as handle:
            stored = handle.read().strip()
    except OSError:
        return default_name

    normalized = _normalize_schedule_dataset_name(stored)
    return normalized or default_name


def _set_active_schedule_dataset(dataset_name: str) -> None:
    normalized = _normalize_schedule_dataset_name(dataset_name)
    target = normalized or "schedule.csv"
    with open(ACTIVE_SCHEDULE_POINTER, "w", encoding="utf-8") as handle:
        handle.write(target)


@app.context_processor
def inject_import_schedule_datasets():
    return {
        "import_schedule_datasets": _get_schedule_datasets(),
    }


def get_data():
    rooms = analyzer.load_rooms()
    schedule = analyzer.load_schedule()
    twin_state = analyzer.build_twin_state(rooms, schedule)
    return rooms, schedule, twin_state


def _load_schedule_from_dataset(dataset_name: str) -> Optional[pd.DataFrame]:
    normalized = _normalize_schedule_dataset_name(dataset_name)
    if not normalized:
        return None
    dataset_path = os.path.join(DATA_DIR, normalized)
    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        return None

    required_columns = {"Room_ID", "CRN", "Day", "Time", "Enrolled", "Capacity"}
    if not required_columns.issubset(set(df.columns)):
        return None

    df["Room_ID"] = df["Room_ID"].astype(str)
    df["CRN"] = df["CRN"].astype(str)
    df["Day"] = df["Day"].astype(str).str.strip()
    df["Time"] = df["Time"].astype(str).str.strip()
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df["Enrolled"] = pd.to_numeric(df["Enrolled"], errors="coerce")
    df = df.dropna(subset=["Capacity", "Enrolled"]).copy()
    df["Capacity"] = df["Capacity"].astype(int)
    df["Enrolled"] = df["Enrolled"].astype(int)
    df["Status"] = df.apply(lambda row: _calc_status_label(int(row["Enrolled"]), int(row["Capacity"])), axis=1)
    df["Utilization_Pct"] = ((df["Enrolled"] / df["Capacity"]) * 100).round(1)
    return df


def _parse_time_token(token: str) -> Optional[str]:
    value = token.strip().lower().replace(".", "")
    value = re.sub(r"\s+", " ", value)
    formats = ("%H:%M", "%H", "%I:%M %p", "%I %p", "%I%p", "%I:%M%p")
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.strftime("%H:%M")
        except ValueError:
            continue
    return None


def _extract_query_time(message: str) -> Optional[str]:
    range_match = re.search(
        r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*[-–]\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
        message,
        re.IGNORECASE,
    )
    if range_match:
        start = _parse_time_token(range_match.group(1))
        end = _parse_time_token(range_match.group(2))
        if start and end:
            return f"{start}-{end}"

    time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", message, re.IGNORECASE)
    if time_match:
        return _parse_time_token(time_match.group(1))
    return None


def _match_time_slots(schedule: pd.DataFrame, query_time: str) -> list[str]:
    times = schedule["Time"].astype(str).str.strip().dropna().unique().tolist()
    times = sorted(times)
    if "-" in query_time:
        return [query_time] if query_time in times else []

    query_dt = datetime.strptime(query_time, "%H:%M")
    matched = []
    for slot in times:
        parts = slot.split("-")
        if len(parts) != 2:
            continue
        start_str = parts[0].strip()
        end_str = parts[1].strip()
        try:
            start_dt = datetime.strptime(start_str, "%H:%M")
            end_dt = datetime.strptime(end_str, "%H:%M")
        except ValueError:
            continue

        if query_dt == start_dt or (start_dt <= query_dt < end_dt):
            matched.append(slot)
    return matched


def _extract_room_id(message: str, rooms: pd.DataFrame) -> Optional[str]:
    room_ids = set(rooms["Room_ID"].astype(str).tolist())
    explicit_match = re.search(r"\broom\s*#?\s*([a-zA-Z0-9_-]+)\b", message, re.IGNORECASE)
    if explicit_match:
        candidate = explicit_match.group(1)
        if candidate in room_ids:
            return candidate

    numeric_matches = re.findall(r"\b(\d{3,6})\b", message)
    for candidate in numeric_matches:
        if candidate in room_ids:
            return candidate
    return None


def _build_saved_schedule_name(original_filename):
    base_name = secure_filename(os.path.splitext(original_filename)[0]) or "uploaded"
    candidate = f"schedule_{base_name}.csv"
    if candidate == "schedule.csv":
        candidate = "schedule_uploaded.csv"

    counter = 1
    while os.path.exists(os.path.join(DATA_DIR, candidate)):
        candidate = f"schedule_{base_name}_{counter}.csv"
        counter += 1
    return candidate


def _normalize_day_value(value: str) -> Optional[str]:
    raw = str(value).strip()
    if not raw:
        return None
    return DAY_ALIAS_TO_CANONICAL.get(raw.lower(), raw)


def _normalize_time_value(value: str) -> Optional[str]:
    raw = str(value).strip()
    if not raw:
        return None

    if "-" in raw or "–" in raw:
        parts = re.split(r"\s*[-–]\s*", raw)
        if len(parts) != 2:
            return None
        start = _parse_time_token(parts[0])
        end = _parse_time_token(parts[1])
        if not start or not end:
            return None
        return f"{start}-{end}"

    parsed = _parse_time_token(raw)
    return parsed


def _calc_status_label(enrolled: int, capacity: int) -> str:
    if enrolled > capacity:
        return "Overcrowded"
    if enrolled < 0.4 * capacity:
        return "Underutilized"
    return "Normal"


def _sort_times(values: list[str]) -> list[str]:
    def key_fn(value: str):
        first = str(value).split("-", 1)[0].strip()
        parsed = _parse_time_token(first)
        if parsed:
            dt = datetime.strptime(parsed, "%H:%M")
            return (0, dt.hour * 60 + dt.minute, value)
        return (1, 0, value)

    return sorted(values, key=key_fn)


def _validate_and_autofix_schedule_df(schedule_df: pd.DataFrame, rooms_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    expected_columns = list(SCHEDULE_COLUMNS)
    expected_set = set(expected_columns)
    source_columns = schedule_df.columns.tolist()
    source_set = set(source_columns)

    missing_columns = [column for column in expected_columns if column not in source_set]
    extra_columns = [column for column in source_columns if column not in expected_set]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}.")
    if extra_columns:
        warnings.append(f"Ignored extra columns: {', '.join(extra_columns)}.")

    if errors:
        return schedule_df, errors, warnings

    fixed = schedule_df.copy()
    fixed = fixed[expected_columns]

    for column in ["CRN", "Course_Code", "Course_Name", "Instructor", "Day", "Time", "Room_ID", "Room_Type"]:
        fixed[column] = fixed[column].astype(str).str.strip()

    fixed["Day"] = fixed["Day"].map(_normalize_day_value)
    invalid_day_rows = fixed[fixed["Day"].isna() | ~fixed["Day"].isin(DAY_SORT_ORDER.keys())]
    if not invalid_day_rows.empty:
        bad_examples = invalid_day_rows["Day"].astype(str).head(5).tolist()
        errors.append(f"Invalid day values detected (examples: {', '.join(bad_examples)}).")

    fixed["Time"] = fixed["Time"].map(_normalize_time_value)
    invalid_time_rows = fixed[fixed["Time"].isna()]
    if not invalid_time_rows.empty:
        errors.append("Invalid time format detected. Use values like 10:00-11:30 or 14:00-15:30.")

    fixed["Capacity"] = pd.to_numeric(fixed["Capacity"], errors="coerce")
    fixed["Enrolled"] = pd.to_numeric(fixed["Enrolled"], errors="coerce")
    invalid_numeric = fixed[fixed["Capacity"].isna() | fixed["Enrolled"].isna()]
    if not invalid_numeric.empty:
        errors.append("Capacity and Enrolled must be numeric for all rows.")

    if errors:
        return fixed, errors, warnings

    fixed["Capacity"] = fixed["Capacity"].astype(int)
    fixed["Enrolled"] = fixed["Enrolled"].astype(int)

    if (fixed["Capacity"] <= 0).any():
        errors.append("Capacity must be greater than 0 for all rows.")
    if (fixed["Enrolled"] < 0).any():
        errors.append("Enrolled cannot be negative.")

    duplicate_crns = fixed[fixed["CRN"].duplicated(keep=False)]["CRN"].drop_duplicates().tolist()
    if duplicate_crns:
        errors.append(f"Duplicate CRN values found: {', '.join(duplicate_crns[:8])}.")

    rooms_lookup = rooms_df.copy()
    rooms_lookup["Room_ID"] = rooms_lookup["Room_ID"].astype(str)
    room_map = rooms_lookup.set_index("Room_ID").to_dict(orient="index")

    unknown_rooms = sorted(set(fixed[~fixed["Room_ID"].isin(room_map.keys())]["Room_ID"].tolist()))
    if unknown_rooms:
        errors.append(f"Unknown Room_ID values (not in rooms.csv): {', '.join(unknown_rooms[:8])}.")

    if errors:
        return fixed, errors, warnings

    capacity_fixed_count = 0
    room_type_fixed_count = 0
    for idx, row in fixed.iterrows():
        room_id = row["Room_ID"]
        room_info = room_map.get(room_id)
        if not room_info:
            continue
        canonical_capacity = int(room_info["Capacity"])
        canonical_type = str(room_info["Type"]).strip()

        if int(row["Capacity"]) != canonical_capacity:
            fixed.at[idx, "Capacity"] = canonical_capacity
            capacity_fixed_count += 1
        if str(row["Room_Type"]).strip() != canonical_type:
            fixed.at[idx, "Room_Type"] = canonical_type
            room_type_fixed_count += 1

    if capacity_fixed_count:
        warnings.append(f"Auto-fixed Capacity in {capacity_fixed_count} row(s) to match rooms.csv.")
    if room_type_fixed_count:
        warnings.append(f"Auto-fixed Room_Type in {room_type_fixed_count} row(s) to match rooms.csv.")

    high_enrollment_rows = fixed[fixed["Enrolled"] > fixed["Capacity"] * 2]
    if not high_enrollment_rows.empty:
        warnings.append("Some sections have Enrolled > 200% of room capacity; review those rows.")

    fixed["Status"] = fixed.apply(lambda row: _calc_status_label(int(row["Enrolled"]), int(row["Capacity"])), axis=1)
    return fixed[expected_columns], errors, warnings


def _build_recommendation_preview_metrics(schedule_df: pd.DataFrame) -> dict:
    conflicts_count = len(analyzer.detect_room_conflicts(schedule_df))
    overcrowded_count = int((schedule_df["Status"] == "Overcrowded").sum())
    underutilized_count = int((schedule_df["Status"] == "Underutilized").sum())
    avg_util = round(float(schedule_df["Utilization_Pct"].mean()), 1) if not schedule_df.empty else 0.0
    return {
        "conflicts": conflicts_count,
        "overcrowded": overcrowded_count,
        "underutilized": underutilized_count,
        "avg_utilization": avg_util,
    }


def _build_preview_schedule(schedule_df: pd.DataFrame, rooms_df: pd.DataFrame, crn: str, target_room: str) -> Optional[pd.DataFrame]:
    preview = schedule_df.copy()
    mask = preview["CRN"].astype(str) == str(crn)
    if not mask.any():
        return None

    rooms_lookup = rooms_df.copy()
    rooms_lookup["Room_ID"] = rooms_lookup["Room_ID"].astype(str)
    room_match = rooms_lookup[rooms_lookup["Room_ID"] == str(target_room)]
    if room_match.empty:
        return None

    room_info = room_match.iloc[0]
    preview.loc[mask, "Room_ID"] = str(room_info["Room_ID"])
    preview.loc[mask, "Room_Type"] = str(room_info["Type"])
    preview.loc[mask, "Capacity"] = int(room_info["Capacity"])
    preview["Status"] = preview.apply(lambda row: _calc_status_label(int(row["Enrolled"]), int(row["Capacity"])), axis=1)
    preview["Utilization_Pct"] = ((preview["Enrolled"] / preview["Capacity"]) * 100).round(1)
    return preview


@app.route("/import-schedule-csv", methods=["POST"])
def import_schedule_csv():
    source_mode = request.form.get("source_mode", "upload").strip().lower()
    schedule_file = request.files.get("schedule_csv")
    selected_data_schedule_id = request.form.get("data_schedule_id", "").strip()
    rooms_df = analyzer.load_rooms()

    if source_mode == "data":
        if not selected_data_schedule_id:
            flash("Choose a schedule dataset from the data folder.", "danger")
            return redirect(url_for("index"))

        datasets_by_id = {dataset["id"]: dataset for dataset in _get_schedule_datasets()}
        selected_dataset = datasets_by_id.get(selected_data_schedule_id)
        if selected_dataset is None:
            flash("The selected schedule dataset was not found in the data folder.", "danger")
            return redirect(url_for("index"))

        schedule_path = os.path.join(DATA_DIR, selected_dataset["schedule_file"])
        try:
            schedule_df = pd.read_csv(schedule_path)
        except Exception:
            flash("Could not read the selected schedule file from the data folder.", "danger")
            return redirect(url_for("index"))
    else:
        if schedule_file is None or not schedule_file.filename:
            flash("Please attach a schedule CSV file.", "danger")
            return redirect(url_for("index"))

        if not schedule_file.filename.lower().endswith(".csv"):
            flash("Only CSV schedule files are supported.", "danger")
            return redirect(url_for("index"))

        try:
            schedule_df = pd.read_csv(schedule_file)
        except Exception:
            flash("Could not read the uploaded schedule file. Please upload a valid CSV.", "danger")
            return redirect(url_for("index"))

    if schedule_df.empty or rooms_df.empty:
        flash("The selected schedule or current rooms data is empty.", "danger")
        return redirect(url_for("index"))

    fixed_schedule_df, validation_errors, validation_warnings = _validate_and_autofix_schedule_df(schedule_df, rooms_df)
    if validation_errors:
        flash("Schedule validation failed. Please fix the file and try again.", "danger")
        for msg in validation_errors[:8]:
            flash(msg, "danger")
        return redirect(url_for("index"))
    for msg in validation_warnings[:8]:
        flash(msg, "warning")

    schedule_df = fixed_schedule_df[SCHEDULE_COLUMNS]
    rooms_df = rooms_df[ROOMS_COLUMNS]

    try:
        schedule_df.to_csv(analyzer.SCHEDULE_CSV, index=False)
    except OSError:
        flash("Could not save the active schedule file (data/schedule.csv).", "danger")
        return redirect(url_for("index"))

    if source_mode == "upload":
        saved_schedule_name = _build_saved_schedule_name(schedule_file.filename)
        saved_schedule_path = os.path.join(DATA_DIR, saved_schedule_name)
        try:
            schedule_df.to_csv(saved_schedule_path, index=False)
            _set_active_schedule_dataset(saved_schedule_name)
        except OSError:
            flash("Schedule imported to data/schedule.csv, but saving the named copy failed.", "warning")
    else:
        selected_schedule_path = os.path.join(DATA_DIR, selected_data_schedule_id)
        try:
            schedule_df.to_csv(selected_schedule_path, index=False)
        except OSError:
            flash("Loaded dataset into data/schedule.csv, but updating the source dataset file failed.", "warning")
        _set_active_schedule_dataset(selected_data_schedule_id)

    if source_mode == "data":
        flash(
            f"Schedule {selected_data_schedule_id} loaded successfully and set as active. Rooms stayed unchanged and the site was rebuilt.",
            "success",
        )
    else:
        flash(
            f"Schedule imported successfully as {saved_schedule_name}. Rooms stayed unchanged and the site was rebuilt.",
            "success",
        )
    return redirect(url_for("index"))


@app.route("/")
@app.route("/dashboard")
def index():
    rooms, schedule, _ = get_data()
    kpis = analyzer.get_kpis(rooms, schedule)
    bar_chart = analyzer.get_room_occupancy_chart(schedule)
    pie_chart = analyzer.get_status_pie_chart(schedule)
    overview = analyzer.get_building_overview_kpis(rooms, schedule)
    problems = analyzer.get_problem_summary(rooms, schedule)
    room_heatmap = analyzer.build_room_time_heatmap(rooms, schedule)
    floor_heatmap = analyzer.build_floor_time_heatmap(rooms, schedule)

    return render_template(
        "index.html",
        kpis=kpis,
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        overview=overview,
        problems=problems,
        room_heatmap=room_heatmap,
        floor_heatmap=floor_heatmap,
    )


@app.route("/rooms")
def rooms_view():
    rooms, _, _ = get_data()
    max_capacity = int(rooms["Capacity"].max()) if not rooms.empty else 1

    floor_filter = request.args.get("floor", "all")
    type_filter = request.args.get("type", "all")

    filtered = rooms.copy()
    if floor_filter != "all":
        filtered = filtered[filtered["Floor"] == int(floor_filter)]
    if type_filter != "all":
        filtered = filtered[filtered["Type"] == type_filter]

    return render_template(
        "rooms.html",
        rooms=filtered.to_dict(orient="records"),
        floor_filter=floor_filter,
        type_filter=type_filter,
        max_capacity=max_capacity,
    )


@app.route("/schedule")
def schedule_view():
    rooms, schedule, _ = get_data()

    day_filter = request.args.get("day", "all")
    time_filter = request.args.get("time", "all")
    room_filter = request.args.get("room_id", "all")
    instructor_filter = request.args.get("instructor", "all")
    floor_filter = request.args.get("floor", "all")
    status_filter = request.args.get("status", "all")
    search_query = request.args.get("q", "").strip()

    rooms_with_floor = rooms[["Room_ID", "Floor"]].copy()
    rooms_with_floor["Room_ID"] = rooms_with_floor["Room_ID"].astype(str)
    filtered = schedule.copy().merge(rooms_with_floor, on="Room_ID", how="left")

    if day_filter != "all":
        filtered = filtered[filtered["Day"] == day_filter]
    if time_filter != "all":
        filtered = filtered[filtered["Time"] == time_filter]
    if room_filter != "all":
        filtered = filtered[filtered["Room_ID"] == room_filter]
    if instructor_filter != "all":
        filtered = filtered[filtered["Instructor"] == instructor_filter]
    if floor_filter != "all":
        try:
            floor_value = int(floor_filter)
            filtered = filtered[filtered["Floor"] == floor_value]
        except ValueError:
            pass
    if status_filter != "all":
        filtered = filtered[filtered["Status"] == status_filter]
    if search_query:
        search_series = (
            filtered["CRN"].astype(str) + " " +
            filtered["Course_Code"].astype(str) + " " +
            filtered["Course_Name"].astype(str) + " " +
            filtered["Instructor"].astype(str) + " " +
            filtered["Day"].astype(str) + " " +
            filtered["Time"].astype(str) + " " +
            filtered["Room_ID"].astype(str) + " " +
            filtered["Room_Type"].astype(str) + " " +
            filtered["Status"].astype(str) + " " +
            filtered["Floor"].astype(str)
        ).str.lower()
        filtered = filtered[search_series.str.contains(search_query.lower(), regex=False, na=False)]

    days = sorted(schedule["Day"].unique().tolist(), key=lambda d: DAY_SORT_ORDER.get(d, 99))
    times = _sort_times(schedule["Time"].unique().tolist())
    room_ids = sorted(schedule["Room_ID"].unique().tolist())
    instructors = sorted(schedule["Instructor"].unique().tolist())
    floors = sorted([int(f) for f in rooms["Floor"].dropna().unique().tolist()])
    statuses = ["Normal", "Overcrowded", "Underutilized"]

    return render_template(
        "schedule.html",
        schedule=filtered.to_dict(orient="records"),
        days=days,
        times=times,
        room_ids=room_ids,
        instructors=instructors,
        floors=floors,
        statuses=statuses,
        day_filter=day_filter,
        time_filter=time_filter,
        room_filter=room_filter,
        instructor_filter=instructor_filter,
        floor_filter=floor_filter,
        status_filter=status_filter,
        search_query=search_query,
    )


@app.route("/conflicts")
def conflicts_view():
    _, schedule, _ = get_data()
    room_conflicts = analyzer.detect_room_conflicts(schedule)
    overcrowded = analyzer.detect_overcrowded(schedule)
    underutilized = analyzer.detect_underutilized(schedule)

    return render_template(
        "conflicts.html",
        room_conflicts=room_conflicts,
        overcrowded=overcrowded,
        underutilized=underutilized,
    )


@app.route("/simulation", methods=["GET", "POST"])
def simulation_view():
    rooms, schedule, _ = get_data()

    unique_sections = schedule.drop_duplicates(subset=["CRN"]).sort_values("CRN")
    crns = unique_sections[["CRN", "Course_Name"]].to_dict(orient="records")
    room_ids = sorted(rooms["Room_ID"].unique().tolist())
    result = None
    error = None

    selected_crn = request.form.get("crn", "").strip() or request.args.get("crn", "").strip()
    selected_room = request.form.get("room_id", "").strip() or request.args.get("room_id", "").strip()

    if selected_crn and selected_room:
        try:
            result = analyzer.simulate_move(
                schedule, rooms, selected_crn, selected_room
            )
        except (IndexError, KeyError) as exc:
            error = f"Data error: {str(exc)}"

    return render_template(
        "simulation.html",
        crns=crns,
        room_ids=room_ids,
        result=result,
        error=error,
        selected_crn=selected_crn,
        selected_room=selected_room,
    )


@app.route("/recommendations")
def recommendations_view():
    rooms, schedule, twin_state = get_data()
    active_schedule_dataset = _get_active_schedule_dataset()
    overcrowded_recommendations = analyzer.generate_recommendations(schedule, rooms)
    underutilized_recommendations = analyzer.generate_underutilized_recommendations(schedule, rooms)
    conflict_recommendations = analyzer.generate_conflict_recommendations(schedule, rooms)
    smart_recommendations = []
    for rec in overcrowded_recommendations:
        if rec.get("actionable"):
            smart_recommendations.append({
                "type": "overcrowded",
                "priority": 1,
                "crn": rec.get("crn"),
                "course_name": rec.get("course_name"),
                "slot": f"{rec.get('day')} / {rec.get('time')}",
                "current_room": rec.get("current_room"),
                "suggested_room": rec.get("suggested_room"),
                "action": rec.get("recommendation_reason"),
            })
    for rec in conflict_recommendations:
        if rec.get("actionable"):
            smart_recommendations.append({
                "type": "conflict",
                "priority": 2,
                "crn": rec.get("crn"),
                "course_name": rec.get("course_name"),
                "slot": f"{rec.get('day')} / {rec.get('time')}",
                "current_room": rec.get("current_room"),
                "suggested_room": rec.get("suggested_room"),
                "action": rec.get("action"),
            })
    for rec in underutilized_recommendations:
        if rec.get("actionable"):
            smart_recommendations.append({
                "type": "underutilized",
                "priority": 3,
                "crn": rec.get("crn"),
                "course_name": rec.get("course_name"),
                "slot": f"{rec.get('day')} / {rec.get('time')}",
                "current_room": rec.get("current_room"),
                "suggested_room": rec.get("suggested_room"),
                "action": rec.get("action"),
            })
    smart_recommendations.sort(key=lambda x: (x["priority"], x["slot"], x["course_name"]))
    smart_recommendations = smart_recommendations[:8]
    impact = analyzer.calculate_optimization_impact(overcrowded_recommendations, twin_state, rooms)
    return render_template(
        "recommendations.html",
        overcrowded_recommendations=overcrowded_recommendations,
        underutilized_recommendations=underutilized_recommendations,
        conflict_recommendations=conflict_recommendations,
        smart_recommendations=smart_recommendations,
        impact=impact,
        active_schedule_dataset=active_schedule_dataset,
    )


def _get_recommendations_by_type(rec_type: str, schedule: pd.DataFrame, rooms: pd.DataFrame) -> list[dict]:
    if rec_type == "overcrowded":
        return analyzer.generate_recommendations(schedule, rooms)
    if rec_type == "underutilized":
        return analyzer.generate_underutilized_recommendations(schedule, rooms)
    if rec_type == "conflict":
        return analyzer.generate_conflict_recommendations(schedule, rooms)
    return []


@app.route("/recommendations/apply", methods=["POST"])
def apply_recommendation_change():
    rec_type = str(request.form.get("rec_type", "")).strip().lower()
    crn = str(request.form.get("crn", "")).strip()
    requested_dataset = str(request.form.get("schedule_dataset", "")).strip()
    if rec_type not in {"overcrowded", "underutilized", "conflict"} or not crn:
        flash("Invalid recommendation request.", "danger")
        return redirect(url_for("recommendations_view"))

    active_dataset = _normalize_schedule_dataset_name(requested_dataset) or _get_active_schedule_dataset()
    _set_active_schedule_dataset(active_dataset)

    rooms = analyzer.load_rooms()
    schedule = _load_schedule_from_dataset(active_dataset)
    if schedule is None:
        flash(f"Could not load active dataset ({active_dataset}) for applying the change.", "danger")
        return redirect(url_for("recommendations_view"))

    recommendations = _get_recommendations_by_type(rec_type, schedule, rooms)
    rec = next((item for item in recommendations if str(item.get("crn")) == crn), None)
    if not rec:
        flash("Recommendation not found for this CRN.", "danger")
        return redirect(url_for("recommendations_view"))

    target_room = str(rec.get("suggested_room", "")).strip()
    if not rec.get("actionable") or not target_room or target_room in {
        "No alternative available",
        "No suitable smaller room",
        "No free room",
        "-",
    }:
        reason = rec.get("recommendation_reason") or rec.get("action") or "No valid target room available."
        flash(f"Change was not applied. {reason}", "warning")
        return redirect(url_for("recommendations_view"))

    preview_schedule = _build_preview_schedule(schedule, rooms, crn=crn, target_room=target_room)
    if preview_schedule is None:
        flash("Could not apply this change due to invalid CRN or room.", "danger")
        return redirect(url_for("recommendations_view"))

    previous_room = str(schedule[schedule["CRN"].astype(str) == crn].iloc[0]["Room_ID"])
    schedule_to_save = preview_schedule[SCHEDULE_COLUMNS]
    active_path = os.path.join(DATA_DIR, active_dataset)
    try:
        schedule_to_save.to_csv(active_path, index=False)
    except OSError:
        flash(f"Could not save change to imported dataset ({active_dataset}).", "danger")
        return redirect(url_for("recommendations_view"))

    if os.path.abspath(active_path) != os.path.abspath(analyzer.SCHEDULE_CSV):
        try:
            schedule_to_save.to_csv(analyzer.SCHEDULE_CSV, index=False)
        except OSError:
            flash(
                f"Applied change to imported dataset ({active_dataset}), but failed to sync schedule.csv.",
                "warning",
            )
            return redirect(url_for("recommendations_view"))

    flash(
        f"Applied change for CRN {crn}: room {previous_room} → room {target_room} in {active_dataset}.",
        "success",
    )
    return redirect(url_for("recommendations_view"))


@app.route("/api/recommendation-preview", methods=["POST"])
def recommendation_preview_api():
    data = request.get_json() or {}
    rec_type = str(data.get("rec_type", "")).strip().lower()
    crn = str(data.get("crn", "")).strip()
    if rec_type not in {"overcrowded", "underutilized", "conflict"} or not crn:
        return jsonify({"error": "Invalid recommendation request."}), 400

    rooms, schedule, _ = get_data()
    recommendations = _get_recommendations_by_type(rec_type, schedule, rooms)

    rec = next((item for item in recommendations if str(item.get("crn")) == crn), None)
    if not rec:
        return jsonify({"error": "Recommendation not found for this CRN."}), 404

    target_room = str(rec.get("suggested_room", "")).strip()
    if not rec.get("actionable") or not target_room or target_room in {
        "No alternative available",
        "No suitable smaller room",
        "No free room",
        "-",
    }:
        return jsonify({
            "actionable": False,
            "message": "This recommendation is not actionable yet.",
            "reason": rec.get("recommendation_reason") or rec.get("action") or "No valid target room available.",
        })

    preview_schedule = _build_preview_schedule(schedule, rooms, crn=crn, target_room=target_room)
    if preview_schedule is None:
        return jsonify({"error": "Could not build preview for this recommendation."}), 400

    before = _build_recommendation_preview_metrics(schedule)
    after = _build_recommendation_preview_metrics(preview_schedule)
    section = schedule[schedule["CRN"].astype(str) == crn].iloc[0]

    return jsonify({
        "actionable": True,
        "crn": crn,
        "course_name": rec.get("course_name", section.get("Course_Name")),
        "day": rec.get("day", section.get("Day")),
        "time": rec.get("time", section.get("Time")),
        "current_room": str(section["Room_ID"]),
        "suggested_room": target_room,
        "before": before,
        "after": after,
        "delta": {
            "conflicts": after["conflicts"] - before["conflicts"],
            "overcrowded": after["overcrowded"] - before["overcrowded"],
            "underutilized": after["underutilized"] - before["underutilized"],
            "avg_utilization": round(after["avg_utilization"] - before["avg_utilization"], 1),
        },
    })


@app.route("/heatmap")
def heatmap_view():
    rooms, schedule, _ = get_data()
    room_heatmap = analyzer.build_room_time_heatmap(rooms, schedule)
    floor_heatmap = analyzer.build_floor_time_heatmap(rooms, schedule)
    return render_template(
        "heatmap.html",
        room_heatmap=room_heatmap,
        floor_heatmap=floor_heatmap,
    )


@app.route("/analytics")
def analytics_view():
    rooms, schedule, _ = get_data()
    analytics = analyzer.get_analytics_data(rooms, schedule)
    return render_template("analytics.html", analytics=analytics)


@app.route("/about-twin")
def about_twin_view():
    rooms, schedule, twin_state = get_data()
    recommendations = analyzer.generate_recommendations(schedule, rooms)
    impact = analyzer.calculate_optimization_impact(recommendations, twin_state, rooms)
    return render_template(
        "about_twin.html",
        twin_state=twin_state,
        recommendations=recommendations,
        impact=impact,
    )


@app.route("/export-csv")
def export_csv():
    """Download the currently active schedule CSV."""
    active_dataset = _get_active_schedule_dataset()
    file_path = os.path.join(DATA_DIR, active_dataset)
    if not os.path.isfile(file_path):
        file_path = analyzer.SCHEDULE_CSV
        active_dataset = os.path.basename(file_path)
    return send_from_directory(
        DATA_DIR,
        active_dataset,
        as_attachment=True,
        download_name=active_dataset,
        mimetype="text/csv",
    )


@app.route("/ai-agent")
def ai_agent_view():
    rooms, schedule, _ = get_data()
    instructors = sorted(schedule["Instructor"].astype(str).unique().tolist())
    room_ids = sorted(rooms["Room_ID"].astype(str).unique().tolist())
    time_options = _sort_times(schedule["Time"].astype(str).str.strip().unique().tolist())
    room_types = sorted(rooms["Type"].astype(str).str.strip().unique().tolist())
    days = sorted(schedule["Day"].astype(str).unique().tolist(), key=lambda d: DAY_SORT_ORDER.get(d, 99))
    return render_template(
        "ai_agent.html",
        instructors=instructors,
        room_ids=room_ids,
        time_options=time_options,
        room_types=room_types,
        days=days,
    )


BOOKING_REQUIRED_FIELDS = ("instructor", "day", "time", "num_students", "room_type")
BOOKING_FIELD_LABELS = {
    "instructor": "instructor full name as registered in the system",
    "day": "day(s), Sunday through Thursday",
    "time": "start and end time",
    "num_students": "number of students",
    "room_type": "room type (Lab or Hall)",
    "course_name": "course name (optional)",
    "course_code": "course code (optional)",
    "preferred_room": "preferred room (optional)",
}


def _default_booking_state() -> dict:
    return {
        "instructor": None,
        "day": None,
        "time": None,
        "num_students": None,
        "room_type": None,
        "preferred_room": None,
        "course_name": None,
        "course_code": None,
    }


def _sanitize_booking_state(raw) -> dict:
    state = _default_booking_state()
    if not isinstance(raw, dict):
        return state
    for key in state:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        state[key] = value
    if isinstance(state.get("num_students"), str):
        try:
            state["num_students"] = int(state["num_students"])
        except ValueError:
            state["num_students"] = None
    if state.get("day"):
        state["day"] = _normalize_booking_days(state["day"])
    if state.get("room_type"):
        state["room_type"] = _normalize_booking_room_type(str(state["room_type"]))
    return state


def _normalize_booking_days(value) -> Optional[list[str]]:
    if value is None:
        return None
    raw_values = value if isinstance(value, list) else re.split(r"\s*(?:,|/|&|\band\b)\s*", str(value))
    days: list[str] = []
    for raw in raw_values:
        canonical = DAY_ALIAS_TO_CANONICAL.get(str(raw).strip().lower(), str(raw).strip())
        if canonical in IMSIU_BOOKING_DAYS and canonical not in days:
            days.append(canonical)
    return days or None


def _extract_booking_days(message: str) -> Optional[list[str]]:
    found: list[str] = []
    for alias, canonical in DAY_ALIAS_TO_CANONICAL.items():
        if canonical not in IMSIU_BOOKING_DAYS:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", message.lower()):
            if canonical not in found:
                found.append(canonical)
    found.sort(key=lambda d: DAY_SORT_ORDER.get(d, 99))
    return found or None


def _normalize_booking_room_type(value: str) -> Optional[str]:
    lower = str(value or "").strip().lower()
    if lower in {"lab", "computer lab", "cs lab"}:
        return "Computer Lab"
    if lower in {"hall", "lecture hall", "classroom", "lecture"}:
        return "Lecture Hall"
    return value if value in {"Computer Lab", "Lecture Hall"} else None


def _booking_room_type_label(value: str) -> str:
    if value == "Computer Lab":
        return "Lab"
    if value == "Lecture Hall":
        return "Hall"
    return str(value or "")


def _format_booking_days(value) -> str:
    days = _normalize_booking_days(value)
    return ", ".join(days) if days else ""


_INSTRUCTOR_STOP_TOKENS = {
    "dr", "doctor", "prof", "professor", "al", "mr", "mrs", "ms",
}


def _instructor_name_tokens(name: str) -> list[str]:
    tokens = [t.lower() for t in re.split(r"[^A-Za-z]+", name) if t]
    return [t for t in tokens if t not in _INSTRUCTOR_STOP_TOKENS and len(t) >= 2]


def _match_instructor(message: str, instructors: list[str]) -> Optional[str]:
    if not message or not instructors:
        return None

    for name in instructors:
        if re.search(rf"\b{re.escape(name)}\b", message, re.IGNORECASE):
            return name

    dr_match = re.search(
        r"\b(?:dr\.?|doctor|prof\.?|professor)\s+([A-Za-z][A-Za-z\-'\. ]{1,60})",
        message,
        re.IGNORECASE,
    )
    candidate_phrase: Optional[str] = None
    if dr_match:
        candidate_phrase = dr_match.group(1).strip().rstrip(",.;:")
    else:
        word_count = len(message.split())
        if word_count <= 4:
            candidate_phrase = message.strip().rstrip(",.;:")

    if not candidate_phrase:
        return None

    candidate_tokens = [
        t for t in re.split(r"[^A-Za-z]+", candidate_phrase) if len(t) >= 3
    ]
    if not candidate_tokens:
        return None

    best_name: Optional[str] = None
    best_score = 0
    for name in instructors:
        name_tokens = _instructor_name_tokens(name)
        if not name_tokens:
            continue
        score = 0
        for ct in candidate_tokens:
            ct_low = ct.lower()
            for nt in name_tokens:
                if nt == ct_low:
                    score += 3
                elif nt.startswith(ct_low) or ct_low.startswith(nt):
                    score += 2
                elif ct_low in nt or nt in ct_low:
                    score += 1
        if score > best_score:
            best_score = score
            best_name = name

    return best_name if best_score >= 2 else None


def _match_room_type(message: str) -> Optional[str]:
    lower = message.lower()
    if re.search(r"\bcomputer\s*lab\b|\bcs\s*lab\b|\blab\b", lower):
        return _normalize_booking_room_type("Lab")
    if re.search(r"\blecture\s*hall\b|\bclassroom\b|\bhall\b|\blecture\b", lower):
        return _normalize_booking_room_type("Hall")
    return None


def _match_num_students(message: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,3})\s*(?:students?|learners?|attendees?|people|seats?)\b", message, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    m = re.search(r"\bfor\s+(\d{1,3})\b", message, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _match_course(message: str) -> tuple[Optional[str], Optional[str]]:
    code = None
    name = None
    code_match = re.search(r"\b([A-Z]{2,4})\s*-?\s*(\d{3,4})\b", message)
    if code_match:
        code = f"{code_match.group(1).upper()}{code_match.group(2)}"
    name_match = re.search(r"(?:course|class|subject|teach(?:ing)?)\s+(?:is\s+|called\s+|named\s+)?['\"]?([A-Za-z][A-Za-z0-9 &\-]{2,60})['\"]?", message, re.IGNORECASE)
    if name_match:
        candidate = name_match.group(1).strip().rstrip(",.;")
        lowered = candidate.lower()
        for token in ("on ", "at ", "for ", "with ", "in "):
            idx = lowered.find(f" {token.strip()} ")
            if idx > 0:
                candidate = candidate[:idx].strip()
                break
        if len(candidate) >= 3:
            name = candidate
    return code, name


def _resolve_time_slot(message: str, schedule: pd.DataFrame) -> Optional[str]:
    query_time = _extract_query_time(message)
    if not query_time:
        return None
    matched = _match_time_slots(schedule, query_time)
    if matched:
        return matched[0]
    return query_time if "-" in query_time else None


def _extract_intent_from_message(
    message: str,
    state: dict,
    rooms: pd.DataFrame,
    schedule: pd.DataFrame,
    instructors: list[str],
) -> dict:
    updated = dict(state)

    if not updated.get("instructor"):
        instructor = _match_instructor(message, instructors)
        if instructor:
            updated["instructor"] = instructor

    if not updated.get("day"):
        days = _extract_booking_days(message)
        if days:
            updated["day"] = days

    if not updated.get("time"):
        slot = _resolve_time_slot(message, schedule)
        if slot:
            updated["time"] = slot

    if not updated.get("num_students"):
        count = _match_num_students(message)
        if count:
            updated["num_students"] = count

    if not updated.get("room_type"):
        room_type = _match_room_type(message)
        if room_type:
            updated["room_type"] = room_type

    if not updated.get("preferred_room"):
        room_id = _extract_room_id(message, rooms)
        if room_id:
            updated["preferred_room"] = room_id

    if not updated.get("course_code") or not updated.get("course_name"):
        code, name = _match_course(message)
        if code and not updated.get("course_code"):
            updated["course_code"] = code
        if name and not updated.get("course_name"):
            updated["course_name"] = name

    return updated


def _missing_booking_fields(state: dict) -> list[str]:
    return [field for field in BOOKING_REQUIRED_FIELDS if not state.get(field)]


def _build_agent_prompt_for_missing(missing: list[str]) -> str:
    if not missing:
        return ""
    human = [BOOKING_FIELD_LABELS.get(f, f) for f in missing]
    if len(human) == 1:
        return f"I just need the {human[0]}."
    if len(human) == 2:
        return f"I still need the {human[0]} and the {human[1]}."
    return "I still need: " + ", ".join(human[:-1]) + f", and {human[-1]}."


def _format_booking_state(state: dict) -> str:
    lines = []
    display_order = [
        ("instructor", "Instructor"),
        ("course_code", "Course Code"),
        ("course_name", "Course Name"),
        ("day", "Day"),
        ("time", "Time"),
        ("num_students", "Students"),
        ("room_type", "Room Type"),
        ("preferred_room", "Preferred Room"),
    ]
    for key, label in display_order:
        value = state.get(key)
        if value:
            if key == "day":
                value = _format_booking_days(value)
            elif key == "room_type":
                value = _booking_room_type_label(value)
            lines.append(f"- {label}: {value}")
    return "\n".join(lines) if lines else "- (nothing captured yet)"


def _estimate_instructor_floor(instructor: str, rooms: pd.DataFrame, schedule: pd.DataFrame) -> Optional[int]:
    if not instructor:
        return None
    rooms_lookup = rooms[["Room_ID", "Floor"]].copy()
    rooms_lookup["Room_ID"] = rooms_lookup["Room_ID"].astype(str)
    instructor_rows = schedule[schedule["Instructor"].astype(str).str.lower() == instructor.lower()].copy()
    if instructor_rows.empty:
        return None
    instructor_rows["Room_ID"] = instructor_rows["Room_ID"].astype(str)
    merged = instructor_rows.merge(rooms_lookup, on="Room_ID", how="left")
    floors = pd.to_numeric(merged["Floor"], errors="coerce").dropna()
    if floors.empty:
        return None
    return int(floors.mode().iloc[0])


def _time_start_minutes(value: str) -> Optional[int]:
    start = _parse_time_token(str(value).split("-", 1)[0])
    if not start:
        return None
    parsed = datetime.strptime(start, "%H:%M")
    return parsed.hour * 60 + parsed.minute


def _format_booking_summary(state: dict, suggestion: dict) -> str:
    room_type = _booking_room_type_label(str(suggestion.get("room_type") or state.get("room_type") or ""))
    capacity = suggestion.get("capacity", "")
    def row(label: str, value) -> str:
        text = f"{label:<11}: {str(value)}"
        return f"│ {text:<31} │"

    return (
        "┌─────────────────────────────────┐\n"
        "│ BOOKING SUMMARY - IMSIU CCIS    │\n"
        "├─────────────────────────────────┤\n"
        f"{row('Instructor', state.get('instructor'))}\n"
        f"{row('Day', _format_booking_days(suggestion.get('day') or state.get('day')))}\n"
        f"{row('Time', suggestion.get('time') or state.get('time'))}\n"
        f"{row('Room Type', room_type)}\n"
        f"{row('Room ID', suggestion.get('room_id'))}\n"
        f"{row('Capacity', capacity)}\n"
        f"{row('Students', state.get('num_students'))}\n"
        "└─────────────────────────────────┘\n\n"
        "Shall I confirm this booking?"
    )


def _rank_suggestions_for_booking(
    state: dict, rooms: pd.DataFrame, schedule: pd.DataFrame, twin_state: dict, limit: int = 5
) -> list[dict]:
    requested_days = _normalize_booking_days(state.get("day")) or []
    requested_time = state.get("time") or "Any"

    def build_suggestions(preferred_time: str) -> list[dict]:
        rows = analyzer.find_available_slots(
            twin_state=twin_state,
            rooms_df=rooms,
            schedule_df=schedule,
            instructor=state.get("instructor"),
            preferred_time=preferred_time,
            preferred_room=state.get("preferred_room") or "Any",
            num_students=int(state.get("num_students") or 0),
            room_type=state.get("room_type") or "Any",
        )
        if requested_days:
            if len(requested_days) == 1:
                rows = [s for s in rows if str(s.get("day")).lower() == requested_days[0].lower()]
            else:
                grouped: dict[tuple[str, str], dict] = {}
                for s in rows:
                    if str(s.get("day")) not in requested_days:
                        continue
                    key = (str(s.get("room_id")), str(s.get("time")))
                    entry = grouped.setdefault(key, {**s, "day": [], "_slack": int(s.get("slack", 0))})
                    entry["day"].append(str(s.get("day")))
                    entry["_slack"] = min(int(entry.get("_slack", 0)), int(s.get("slack", 0)))
                rows = []
                for entry in grouped.values():
                    unique_days = sorted(set(entry["day"]), key=lambda d: DAY_SORT_ORDER.get(d, 99))
                    if unique_days == requested_days:
                        entry["day"] = unique_days
                        entry["display_day"] = _format_booking_days(unique_days)
                        entry.pop("_slack", None)
                        rows.append(entry)
        return rows

    suggestions = build_suggestions(str(requested_time))
    used_alternative_time = False
    if not suggestions and requested_time not in (None, "", "Any"):
        suggestions = build_suggestions("Any")
        used_alternative_time = True

    target_floor = _estimate_instructor_floor(str(state.get("instructor") or ""), rooms, schedule)
    requested_minutes = _time_start_minutes(str(requested_time))
    for s in suggestions:
        floor = int(s.get("floor", 0) or 0)
        suggestion_minutes = _time_start_minutes(str(s.get("time", "")))
        time_distance = abs(suggestion_minutes - requested_minutes) if requested_minutes is not None and suggestion_minutes is not None else 0
        if used_alternative_time:
            s["alternative_note"] = "Exact slot was unavailable; this is the nearest available alternative."
        s["display_day"] = s.get("display_day") or _format_booking_days(s.get("day"))
        s["room_type_label"] = _booking_room_type_label(str(s.get("room_type") or ""))
        s["_rank"] = (
            time_distance,
            abs(floor - target_floor) if target_floor is not None else floor,
            int(s.get("slack", 0)),
            int(s.get("time_busyness", 0)),
            floor,
            str(s.get("room_id", "")),
        )
    suggestions.sort(key=lambda row: row["_rank"])
    for row in suggestions:
        row.pop("_rank", None)
    return suggestions[:limit]


def _next_crn(schedule: pd.DataFrame) -> str:
    try:
        numeric = pd.to_numeric(schedule["CRN"], errors="coerce").dropna()
        if numeric.empty:
            return "90001"
        return str(int(numeric.max()) + 1)
    except Exception:
        return "90001"


def _log_booking_change(action: str, details: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ["timestamp", "action", "crn", "instructor", "day", "time", "room_id", "details"]
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "crn": details.get("crn", ""),
        "instructor": details.get("instructor", ""),
        "day": details.get("day", ""),
        "time": details.get("time", ""),
        "room_id": details.get("room_id", ""),
        "details": details.get("details", ""),
    }
    file_exists = os.path.exists(BOOKING_LOG_CSV)
    with open(BOOKING_LOG_CSV, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _append_booking_row(state: dict, suggestion: dict) -> tuple[bool, str, Optional[dict]]:
    active_dataset = _get_active_schedule_dataset()
    rooms_df = analyzer.load_rooms()
    schedule_df = _load_schedule_from_dataset(active_dataset)
    if schedule_df is None:
        return False, f"Could not load the active dataset ({active_dataset}).", None

    room_id = str(suggestion.get("room_id", "")).strip()
    requested_days = _normalize_booking_days(suggestion.get("day") or state.get("day")) or []
    time_slot = str(suggestion.get("time", "")).strip()
    if not room_id or not requested_days or not time_slot:
        return False, "The chosen slot is missing room, day, or time.", None

    instructor = (state.get("instructor") or "").strip() or "Unknown"
    for day in requested_days:
        conflict = schedule_df[
            (schedule_df["Room_ID"].astype(str) == room_id)
            & (schedule_df["Day"].astype(str).str.strip().str.lower() == day.lower())
            & (schedule_df["Time"].astype(str) == time_slot)
        ]
        blocked = conflict[
            conflict["Course_Name"].astype(str).str.contains("blocked|maintenance", case=False, na=False)
            | conflict["Status"].astype(str).str.contains("blocked|maintenance", case=False, na=False)
        ]
        if not blocked.empty:
            return False, f"Room {room_id} on {day} at {time_slot} is blocked for maintenance.", None
        if not conflict.empty:
            return False, f"Room {room_id} on {day} at {time_slot} was just taken. Please pick another slot.", None

        if instructor == "Unknown":
            continue
        instructor_conflict = schedule_df[
            (schedule_df["Instructor"].astype(str) == instructor)
            & (schedule_df["Day"].astype(str).str.strip().str.lower() == day.lower())
            & (schedule_df["Time"].astype(str) == time_slot)
        ]
        if not instructor_conflict.empty:
            return False, f"{instructor} already has a class on {day} at {time_slot}.", None

    room_row = rooms_df[rooms_df["Room_ID"].astype(str) == room_id]
    if room_row.empty:
        return False, f"Room {room_id} is not in rooms.csv.", None
    room_info = room_row.iloc[0]
    capacity = int(room_info["Capacity"])
    room_type = str(room_info["Type"]).strip()

    num_students = int(state.get("num_students") or 0)
    if num_students <= 0:
        return False, "Number of students must be greater than zero.", None
    if capacity < num_students:
        return False, f"Room {room_id} capacity ({capacity}) is below the requested {num_students} students.", None
    if room_type != state.get("room_type"):
        return False, f"Room {room_id} is a {room_type}, not a {_booking_room_type_label(state.get('room_type'))}.", None

    new_crn = _next_crn(schedule_df)
    rows = []
    for offset, day in enumerate(requested_days):
        rows.append({
            "CRN": str(int(new_crn) + offset) if new_crn.isdigit() else f"{new_crn}-{offset + 1}",
            "Course_Code": (state.get("course_code") or "ADHOC").strip() or "ADHOC",
            "Course_Name": (state.get("course_name") or "Ad-hoc Booking").strip() or "Ad-hoc Booking",
            "Instructor": instructor,
            "Day": day,
            "Time": time_slot,
            "Room_ID": room_id,
            "Room_Type": room_type,
            "Capacity": capacity,
            "Enrolled": num_students,
            "Status": _calc_status_label(num_students, capacity),
        })

    updated = pd.concat([schedule_df, pd.DataFrame(rows)], ignore_index=True)
    updated = updated[SCHEDULE_COLUMNS]

    active_path = os.path.join(DATA_DIR, active_dataset)
    try:
        updated.to_csv(active_path, index=False)
    except OSError:
        return False, f"Could not write booking to {active_dataset}.", None

    if os.path.abspath(active_path) != os.path.abspath(analyzer.SCHEDULE_CSV):
        try:
            updated.to_csv(analyzer.SCHEDULE_CSV, index=False)
        except OSError:
            return False, "Booking saved to dataset, but failed to sync active schedule.csv.", None

    for row in rows:
        _log_booking_change("confirm", {
            "crn": row["CRN"],
            "instructor": row["Instructor"],
            "day": row["Day"],
            "time": row["Time"],
            "room_id": row["Room_ID"],
            "details": f"{row['Course_Code']} {row['Course_Name']} for {row['Enrolled']} students",
        })

    return True, "Booking confirmed and added to the schedule.", rows[0] if len(rows) == 1 else {"rows": rows}


def _gemini_extract_booking_fields(
    message: str, state: dict, instructors: list[str], room_types: list[str], days: list[str], time_slots: list[str]
) -> Optional[dict]:
    if not has_genai:
        return None
    gemini_api_key = _get_gemini_api_key()
    if not gemini_api_key:
        return None
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=(
                "You extract room-booking fields for a classroom scheduler. "
                "Return ONLY a compact JSON object with these optional keys: "
                "instructor, day, time, num_students, room_type, preferred_room, course_code, course_name. "
                "Use the allowed lists for instructor/day/time/room_type whenever possible. "
                "For day, return a single allowed day or an array of allowed days. "
                "For room_type, normalize Lab to Computer Lab and Hall to Lecture Hall. "
                "Omit keys you are not confident about. Never include prose."
            ),
        )
        prompt = (
            "Current known fields (JSON):\n"
            f"{state}\n\n"
            f"Allowed instructors: {instructors}\n"
            f"Allowed days: {days}\n"
            f"Allowed time slots: {time_slots}\n"
            f"Allowed room types: {room_types}\n\n"
            f"User message: {message}\n\n"
            "Return ONLY a JSON object of extracted fields."
        )
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            return None
        import json as _json
        extracted = _json.loads(json_match.group(0))
        if not isinstance(extracted, dict):
            return None
        cleaned: dict = {}
        for key in _default_booking_state().keys():
            value = extracted.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            cleaned[key] = value
        if "num_students" in cleaned:
            try:
                cleaned["num_students"] = int(cleaned["num_students"])
            except (TypeError, ValueError):
                cleaned.pop("num_students", None)
        return cleaned
    except Exception:
        traceback.print_exc()
        return None


def _extract_crn(message: str) -> Optional[str]:
    match = re.search(r"\bCRN\s*#?\s*(\d{4,8})\b", message, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _next_session_start(day: str, time_slot: str) -> Optional[datetime]:
    day_to_weekday = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    weekday = day_to_weekday.get(day)
    if weekday is None:
        return None
    start = _parse_time_token(str(time_slot).split("-", 1)[0])
    if not start:
        return None
    start_time = datetime.strptime(start, "%H:%M").time()
    now = datetime.now()
    days_ahead = (weekday - now.weekday()) % 7
    candidate = datetime.combine(now.date(), start_time)
    if days_ahead:
        candidate = candidate + timedelta(days=days_ahead)
    elif candidate <= now:
        candidate = candidate + timedelta(days=7)
    return candidate


def _cancel_booking_by_crn(crn: str) -> tuple[bool, str]:
    active_dataset = _get_active_schedule_dataset()
    schedule_df = _load_schedule_from_dataset(active_dataset)
    if schedule_df is None:
        return False, f"Could not load the active dataset ({active_dataset})."

    mask = schedule_df["CRN"].astype(str) == str(crn)
    if not mask.any():
        return False, f"I could not find CRN {crn} in the active schedule."

    row = schedule_df[mask].iloc[0]
    session_start = _next_session_start(str(row["Day"]), str(row["Time"]))
    if session_start is not None:
        hours_until = (session_start - datetime.now()).total_seconds() / 3600
        if hours_until < 24:
            return False, "This session is less than 24 hours away, so it cannot be cancelled through the agent."

    updated = schedule_df[~mask].copy()
    active_path = os.path.join(DATA_DIR, active_dataset)
    try:
        updated.to_csv(active_path, index=False)
        if os.path.abspath(active_path) != os.path.abspath(analyzer.SCHEDULE_CSV):
            updated.to_csv(analyzer.SCHEDULE_CSV, index=False)
    except OSError:
        return False, f"Could not save the cancellation to {active_dataset}."

    _log_booking_change("cancel", {
        "crn": crn,
        "instructor": row.get("Instructor", ""),
        "day": row.get("Day", ""),
        "time": row.get("Time", ""),
        "room_id": row.get("Room_ID", ""),
        "details": f"Cancelled {row.get('Course_Code', '')} {row.get('Course_Name', '')}",
    })
    return True, f"Cancelled CRN {crn} and logged the change. For modifications, request a new booking with the updated details."


@app.route("/api/booking-agent", methods=["POST"])
def booking_agent_api():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    raw_state = data.get("state") or {}
    reset = bool(data.get("reset"))

    state = _default_booking_state() if reset else _sanitize_booking_state(raw_state)

    rooms, schedule, twin_state = get_data()
    instructors = sorted(schedule["Instructor"].astype(str).unique().tolist())
    room_types = sorted(rooms["Type"].astype(str).str.strip().unique().tolist())
    days = sorted(schedule["Day"].astype(str).unique().tolist(), key=lambda d: DAY_SORT_ORDER.get(d, 99))
    time_slots = _sort_times(schedule["Time"].astype(str).str.strip().unique().tolist())

    if reset and not message:
        return jsonify({
            "reply": (
                "Hi doctor! I can help you book a room. Tell me who is teaching, "
                "which IMSIU day (Sunday-Thursday), start/end time, how many students, "
                "and whether you need a Lab or Hall. "
                "Example: \"I'm Dr. Sara, I need a Hall on Monday 10:00-11:30 for 30 students.\""
            ),
            "state": state,
            "stage": "gathering",
            "suggestions": [],
        })

    if not message:
        return jsonify({"error": "No message provided."}), 400

    lower_message = message.lower()
    if any(word in lower_message for word in ["cancel", "remove", "delete"]):
        crn = _extract_crn(message)
        if not crn:
            return jsonify({
                "reply": "I can cancel a booking up to 24 hours before the session. Please include the CRN, for example: cancel CRN 90001.",
                "state": state,
                "stage": "cancellation",
                "suggestions": [],
            })
        ok, cancel_message = _cancel_booking_by_crn(crn)
        return jsonify({
            "reply": cancel_message,
            "state": _default_booking_state() if ok else state,
            "stage": "cancelled" if ok else "cancellation_error",
            "suggestions": [],
        }), 200 if ok else 400

    if any(word in lower_message for word in ["modify", "reschedule", "change booking", "change crn"]):
        return jsonify({
            "reply": (
                "Modifications are handled as cancel + rebook. "
                "First say \"cancel CRN <number>\"; after it is cancelled, send the new booking details."
            ),
            "state": state,
            "stage": "modification",
            "suggestions": [],
        })

    state = _extract_intent_from_message(message, state, rooms, schedule, instructors)
    gemini_fields = _gemini_extract_booking_fields(message, state, instructors, room_types, days, time_slots)
    if gemini_fields:
        for key, value in gemini_fields.items():
            if not state.get(key) and value:
                if key == "preferred_room":
                    if str(value) in set(rooms["Room_ID"].astype(str).tolist()):
                        state[key] = str(value)
                elif key == "day":
                    normalized = _normalize_booking_days(value)
                    if normalized:
                        state[key] = normalized
                elif key == "time":
                    normalized = _normalize_time_value(str(value))
                    if normalized:
                        state[key] = normalized
                elif key == "instructor":
                    match = _match_instructor(str(value), instructors) or (str(value) if str(value) in instructors else None)
                    if match:
                        state[key] = match
                elif key == "room_type":
                    normalized = _normalize_booking_room_type(str(value))
                    if normalized in room_types:
                        state[key] = normalized
                elif key == "num_students":
                    try:
                        state[key] = int(value)
                    except (TypeError, ValueError):
                        pass
                else:
                    state[key] = str(value)

    missing = _missing_booking_fields(state)
    if missing:
        reply_lines = [
            "Got it. Here is what I have so far:",
            _format_booking_state(state),
            "",
            _build_agent_prompt_for_missing(missing),
        ]
        return jsonify({
            "reply": "\n".join(reply_lines),
            "state": state,
            "stage": "gathering",
            "suggestions": [],
            "missing": missing,
        })

    suggestions = _rank_suggestions_for_booking(state, rooms, schedule, twin_state, limit=5)
    if not suggestions:
        reply = (
            "I could not find any free slot that matches these constraints:\n"
            f"{_format_booking_state(state)}\n\n"
            "There may be a room conflict, instructor conflict, maintenance block, or no room with enough capacity. "
            "Try a nearby time, a different day, or the other room type."
        )
        return jsonify({
            "reply": reply,
            "state": state,
            "stage": "no_match",
            "suggestions": [],
        })

    preferred = suggestions[0]
    reply_lines = [
        "I found the best available match and checked room/instructor conflicts:",
        "",
        _format_booking_summary(state, preferred),
    ]
    if preferred.get("alternative_note"):
        reply_lines.insert(1, preferred["alternative_note"])
    return jsonify({
        "reply": "\n".join(reply_lines),
        "state": state,
        "stage": "awaiting_confirmation",
        "pending_suggestion": preferred,
        "suggestions": suggestions,
    })


@app.route("/api/booking-agent/confirm", methods=["POST"])
def booking_agent_confirm_api():
    data = request.get_json(silent=True) or {}
    state = _sanitize_booking_state(data.get("state") or {})
    suggestion = data.get("suggestion") or {}

    if not isinstance(suggestion, dict):
        return jsonify({"error": "Invalid suggestion."}), 400

    missing = _missing_booking_fields(state)
    if missing:
        return jsonify({
            "ok": False,
            "error": "Booking state is incomplete.",
            "missing": missing,
        }), 400

    ok, message, new_row = _append_booking_row(state, suggestion)
    if not ok:
        return jsonify({"ok": False, "error": message}), 400

    if new_row and "rows" in new_row:
        rows = new_row["rows"]
        crns = ", ".join(str(row["CRN"]) for row in rows)
        days = ", ".join(str(row["Day"]) for row in rows)
        first = rows[0]
        reply = (
            f"Booking confirmed! CRNs {crns} — {first['Course_Name']} "
            f"({first['Instructor']}) in Room {first['Room_ID']} on {days} at {first['Time']}."
        )
    else:
        reply = (
            f"Booking confirmed! CRN {new_row['CRN']} — {new_row['Course_Name']} "
            f"({new_row['Instructor']}) in Room {new_row['Room_ID']} on {new_row['Day']} at {new_row['Time']}."
        )
    return jsonify({
        "ok": True,
        "reply": reply,
        "booking": new_row,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
