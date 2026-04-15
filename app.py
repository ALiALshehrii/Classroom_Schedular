"""
app.py
------
Flask entry point for the Digital Twin Classroom Scheduler.
Defines all routes and passes data to Jinja2 templates.
"""

import os
import re
import traceback
from datetime import datetime
from typing import Optional

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for, jsonify
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
CHATBOT_SYSTEM_INSTRUCTION = """
You are the Digital Twin Classroom Scheduler assistant for this website.

Rules:
1) Only answer questions related to this website and its scheduling domain.
2) You may help with anything related to conflicts, classes, overcrowded sections, what-if scenarios, students, instructors, rooms, time slots, utilization, and similar scheduling topics.
3) If a question is outside this scope, do not answer it. Reply briefly that you can only help with website-related scheduling questions and invite the user to ask about the allowed topics.
4) Prefer direct answers from provided schedule/rooms data. Do not ask users to navigate the website for answers you can compute.
5) Handle short follow-ups like "and 1008?" or "what about 2006?" by using recent conversation context when intent is clear.
6) For availability checks, give a clear Yes/No and include room/day/time in the first line.
7) For room summary requests, return currently scheduled sections in that room with CRN, course, instructor, day, time, and status.
8) When data is missing or ambiguous, ask one concise clarifying question.
"""
SCHEDULE_COLUMNS = pd.read_csv(analyzer.SCHEDULE_CSV, nrows=0).columns.tolist()
ROOMS_COLUMNS = pd.read_csv(analyzer.ROOMS_CSV, nrows=0).columns.tolist()
DATA_DIR = os.path.join(analyzer.BASE_DIR, "data")
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


def _get_gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _get_schedule_datasets():
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
                "is_active": schedule_file == "schedule.csv",
                "label": schedule_file,
            }
        )
    return datasets


def _sanitize_chat_history(raw_history) -> list[dict]:
    if not isinstance(raw_history, list):
        return []

    cleaned = []
    for item in raw_history[-20:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _latest_user_message(history: list[dict]) -> Optional[str]:
    for item in reversed(history):
        if item.get("role") == "user":
            return item.get("content", "")
    return None


def _is_short_room_followup(message: str, room_id: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.strip().lower())
    bare = re.sub(r"[^\w-]", "", normalized)
    followup_prefix = bool(re.match(r"^(and|what about|how about|about)\b", normalized))
    bare_room = bare == room_id.lower()
    short_question = len(normalized.split()) <= 5
    return (followup_prefix or bare_room) and short_question


def _expand_followup_message(message: str, history: list[dict], rooms: pd.DataFrame) -> str:
    room_id = _extract_room_id(message, rooms)
    if not room_id:
        return message
    if _extract_day(message) or _extract_query_time(message):
        return message
    if any(word in message.lower() for word in ["available", "availability", "free", "booked", "occupied"]):
        return message

    last_user = _latest_user_message(history)
    if not last_user:
        return message
    last_day = _extract_day(last_user)
    last_time = _extract_query_time(last_user)
    if last_day and last_time and _is_short_room_followup(message, room_id):
        return f"Is room {room_id} available on {last_day} at {last_time}?"
    return message


def _summarize_room_schedule(room_id: str, schedule: pd.DataFrame, limit: int = 6) -> str:
    room_rows = schedule[schedule["Room_ID"].astype(str) == room_id]
    if room_rows.empty:
        return f"There are no scheduled sections in room {room_id}."

    rows = room_rows[
        ["CRN", "Course_Name", "Instructor", "Day", "Time", "Capacity", "Enrolled", "Status"]
    ].drop_duplicates()
    rows = rows.copy()
    rows["__day_order"] = rows["Day"].astype(str).map(lambda d: DAY_SORT_ORDER.get(str(d).strip(), 99))
    rows = rows.sort_values(by=["__day_order", "Time", "CRN"]).drop(columns="__day_order")

    lines = [f"Room {room_id} has {len(rows)} scheduled section(s):"]
    for _, row in rows.head(limit).iterrows():
        lines.append(
            f"- CRN {row['CRN']}: {row['Course_Name']} ({row['Instructor']}) on {row['Day']} at {row['Time']} "
            f"[Status: {row['Status']}, Enrolled {row['Enrolled']}/{row['Capacity']}]"
        )
    if len(rows) > limit:
        lines.append(f"- ...and {len(rows) - limit} more section(s).")
    return "\n".join(lines)


@app.context_processor
def inject_import_schedule_datasets():
    api_key_configured = bool(_get_gemini_api_key())
            
    return {
        "import_schedule_datasets": _get_schedule_datasets(),
        "api_key_configured": api_key_configured
    }


def get_data():
    rooms = analyzer.load_rooms()
    schedule = analyzer.load_schedule()
    twin_state = analyzer.build_twin_state(rooms, schedule)
    return rooms, schedule, twin_state


def _extract_day(message: str) -> Optional[str]:
    normalized = f" {message.lower()} "
    for alias, canonical in DAY_ALIAS_TO_CANONICAL.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return canonical
    return None


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


def _format_conflict_rows(conflicts: list[dict], limit: int = 6) -> str:
    if not conflicts:
        return "No room conflicts found in the current schedule."

    lines = []
    for row in conflicts[:limit]:
        lines.append(
            f"- Room {row['room_id']} on {row['day']} at {row['time']} ({row['count']} sections: {row['crns']})"
        )
    if len(conflicts) > limit:
        lines.append(f"- ...and {len(conflicts) - limit} more conflict slots.")
    return "\n".join(lines)


def _answer_from_local_data(
    message: str,
    rooms: pd.DataFrame,
    schedule: pd.DataFrame,
    history: Optional[list[dict]] = None,
) -> Optional[str]:
    text = message.strip()
    history = history or []
    resolved_text = _expand_followup_message(text, history, rooms)
    lower = resolved_text.lower()
    room_conflicts = analyzer.detect_room_conflicts(schedule)

    is_conflict_query = any(word in lower for word in ["conflict", "double-book", "double book", "overlap"])
    day = _extract_day(resolved_text)
    asks_for_days = bool(re.search(r"\b(what|which)\s+days?\b", lower))

    if is_conflict_query and day and not asks_for_days:
        day_conflicts = [c for c in room_conflicts if str(c["day"]).strip().lower() == day.lower()]
        if not day_conflicts:
            return f"There are no room conflicts on {day}."
        return f"Conflicts on {day} ({len(day_conflicts)} slots):\n{_format_conflict_rows(day_conflicts)}"

    if is_conflict_query and asks_for_days:
        if not room_conflicts:
            return "There are no room conflicts in the current schedule."
        days = sorted({c["day"] for c in room_conflicts}, key=lambda d: DAY_SORT_ORDER.get(d, 99))
        return f"Conflicts happen on: {', '.join(days)}."

    if is_conflict_query:
        return f"Total room conflict slots: {len(room_conflicts)}.\n{_format_conflict_rows(room_conflicts)}"

    is_availability_query = any(word in lower for word in ["available", "availability", "free", "booked", "occupied"])
    if is_availability_query:
        room_id = _extract_room_id(resolved_text, rooms)
        if room_id:
            day = _extract_day(resolved_text)
            query_time = _extract_query_time(resolved_text)
            if not day or not query_time:
                return "I can check that. Please include room, day, and time (example: Room 1003 on Monday at 12:00)."

            matched_slots = _match_time_slots(schedule, query_time)
            if not matched_slots:
                known_times = ", ".join(sorted(schedule["Time"].astype(str).unique().tolist()))
                return f"I couldn't match that time to schedule slots. Available slots are: {known_times}."

            occupied_rows = schedule[
                (schedule["Room_ID"].astype(str) == room_id)
                & (schedule["Day"].astype(str).str.strip().str.lower() == day.lower())
                & (schedule["Time"].astype(str).isin(matched_slots))
            ]
            if occupied_rows.empty:
                return f"Yes. Room {room_id} is available on {day} at {query_time}."

            details = occupied_rows[["Course_Name", "CRN", "Time"]].drop_duplicates().to_dict(orient="records")
            formatted = "; ".join([f"{d['Course_Name']} (CRN {d['CRN']}) at {d['Time']}" for d in details[:3]])
            return f"No. Room {room_id} is not available on {day} at {query_time}. Scheduled: {formatted}."

    room_id = _extract_room_id(resolved_text, rooms)
    if room_id:
        followup_like = bool(re.search(r"\b(what about|how about|about|and)\b", text.lower())) or \
            (re.sub(r"[^\w-]", "", text.lower()) == room_id.lower())
        if followup_like and not _extract_day(resolved_text) and not _extract_query_time(resolved_text):
            return _summarize_room_schedule(room_id, schedule)

    if "total rooms" in lower or "how many rooms" in lower:
        return f"Total rooms in the current dataset: {len(rooms)}."
    if "total sections" in lower or "how many classes" in lower:
        return f"Total scheduled sections in the current dataset: {len(schedule)}."

    return None


def _build_chat_context(rooms: pd.DataFrame, schedule: pd.DataFrame) -> str:
    conflicts = analyzer.detect_room_conflicts(schedule)
    overview = analyzer.get_building_overview_kpis(rooms, schedule)
    status_series = schedule["Status"].astype(str).str.strip()
    overcrowded_count = int((status_series.str.lower() == "overcrowded").sum())
    underutilized_count = int((status_series.str.lower() == "underutilized").sum())
    normal_count = int((status_series.str.lower() == "normal").sum())
    sample_rows = schedule[
        ["CRN", "Course_Name", "Instructor", "Day", "Time", "Room_ID", "Capacity", "Enrolled", "Status"]
    ].head(80)
    return (
        f"Dataset summary:\n"
        f"- Total rooms: {len(rooms)}\n"
        f"- Total sections: {len(schedule)}\n"
        f"- Days: {', '.join(sorted(schedule['Day'].astype(str).unique().tolist()))}\n"
        f"- Time slots: {', '.join(sorted(schedule['Time'].astype(str).unique().tolist()))}\n"
        f"- Conflict slots: {len(conflicts)}\n"
        f"- Building utilization (%): {overview.get('building_utilization_pct', 0)}\n\n"
        f"Status mix:\n"
        f"- Normal: {normal_count}\n"
        f"- Overcrowded: {overcrowded_count}\n"
        f"- Underutilized: {underutilized_count}\n\n"
        f"Helpful question examples:\n"
        f"- Is room 3007 free on Tuesday at 10:00?\n"
        f"- What conflicts happen on Wednesday?\n"
        f"- Which days have room conflicts?\n"
        f"- Show all sections in room 2006.\n"
        f"- Which rooms are overcrowded on Monday?\n"
        f"- Which instructor has the most classes?\n"
        f"- How many total rooms and sections are there?\n"
        f"- What time slots are most utilized?\n\n"
        f"Conflicts detail (first 10):\n{_format_conflict_rows(conflicts, limit=10)}\n\n"
        f"Schedule rows (CSV excerpt):\n{sample_rows.to_csv(index=False)}"
    )


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
    schedule_df.to_csv(analyzer.SCHEDULE_CSV, index=False)
    if source_mode == "upload":
        saved_schedule_name = _build_saved_schedule_name(schedule_file.filename)
        schedule_df.to_csv(os.path.join(DATA_DIR, saved_schedule_name), index=False)

    if source_mode == "data":
        flash(
            f"Schedule {selected_data_schedule_id} loaded successfully. Rooms stayed unchanged and the site was rebuilt.",
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
    )


@app.route("/api/recommendation-preview", methods=["POST"])
def recommendation_preview_api():
    data = request.get_json() or {}
    rec_type = str(data.get("rec_type", "")).strip().lower()
    crn = str(data.get("crn", "")).strip()
    if rec_type not in {"overcrowded", "underutilized", "conflict"} or not crn:
        return jsonify({"error": "Invalid recommendation request."}), 400

    rooms, schedule, _ = get_data()
    if rec_type == "overcrowded":
        recommendations = analyzer.generate_recommendations(schedule, rooms)
    elif rec_type == "underutilized":
        recommendations = analyzer.generate_underutilized_recommendations(schedule, rooms)
    else:
        recommendations = analyzer.generate_conflict_recommendations(schedule, rooms)

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


@app.route("/schedule-assistant", methods=["GET", "POST"])
def schedule_assistant_view():
    rooms, schedule, twin_state = get_data()

    instructors = sorted(schedule["Instructor"].unique().tolist())
    room_ids = sorted(rooms["Room_ID"].unique().tolist())
    time_options = ["Any"] + _sort_times(schedule["Time"].astype(str).str.strip().unique().tolist())
    room_type_options = ["Any"] + sorted(rooms["Type"].astype(str).str.strip().unique().tolist())

    selected_instructor = request.form.get("instructor", "")
    selected_time = request.form.get("preferred_time", "Any")
    selected_room = request.form.get("preferred_room", "Any")
    selected_students = request.form.get("num_students", "")
    selected_room_type = request.form.get("room_type", "Any")

    suggestions = []
    if request.method == "POST":
        num_students = int(selected_students) if str(selected_students).isdigit() else 0
        suggestions = analyzer.find_available_slots(
            twin_state=twin_state,
            rooms_df=rooms,
            schedule_df=schedule,
            instructor=selected_instructor if selected_instructor else None,
            preferred_time=selected_time,
            preferred_room=selected_room,
            num_students=num_students,
            room_type=selected_room_type,
        )

    return render_template(
        "schedule_assistant.html",
        instructors=instructors,
        room_ids=room_ids,
        time_options=time_options,
        room_type_options=room_type_options,
        suggestions=suggestions,
        selected_instructor=selected_instructor,
        selected_time=selected_time,
        selected_room=selected_room,
        selected_students=selected_students,
        selected_room_type=selected_room_type,
    )


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


@app.route("/ai-agent")
def ai_agent_view():
    return render_template("ai_agent.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    message = (data or {}).get("message", "")
    history = _sanitize_chat_history((data or {}).get("history"))
    
    if not message:
        return jsonify({"error": "No message provided."}), 400

    rooms, schedule, _ = get_data()

    resolved_message = _expand_followup_message(message, history, rooms)
    local_reply = _answer_from_local_data(message, rooms, schedule, history=history)
    if local_reply:
        return jsonify({"reply": local_reply})

    if not has_genai:
        return jsonify({
            "reply": "I can answer direct schedule questions from CSV data (rooms, conflicts, availability). "
                     "Try: 'What days have conflicts?' or 'Is room 1003 available on Monday at 12:00?'"
        })

    gemini_api_key = _get_gemini_api_key()
    if not gemini_api_key:
        return jsonify({
            "reply": "I can still answer direct CSV-based questions, but generative fallback is disabled because GEMINI_API_KEY is missing."
        })

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=CHATBOT_SYSTEM_INSTRUCTION,
        )

        context = _build_chat_context(rooms, schedule)
        history_text = "\n".join(
            f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}"
            for item in history[-10:]
        )
        prompt = (
            "Answer using the provided website dataset.\n"
            "If the question asks for conflicts/availability, give a direct answer and include exact room/day/time.\n"
            "Use recent conversation context for short follow-up questions when the intent is clear.\n"
            "If data is insufficient, say exactly what is missing.\n\n"
            f"Recent conversation:\n{history_text or '(none)'}\n\n"
            f"{context}\n\n"
            f"User question: {resolved_message}"
        )
        response = model.generate_content(prompt)
        
        return jsonify({"reply": response.text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
