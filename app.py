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
    import google.generativeai as genai
    has_genai = True
except ImportError:
    has_genai = False

import analyzer

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "classroom-scheduler-dev-secret")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CHATBOT_SYSTEM_INSTRUCTION = """
You are the Digital Twin Classroom Scheduler assistant for this website.

Rules:
1) Only answer questions related to this website and its scheduling domain.
2) You may help with anything related to conflicts, classes, overcrowded sections, what-if scenarios, students, instructors, rooms, time slots, utilization, and similar scheduling topics.
3) If a question is outside this scope, do not answer it. Reply briefly that you can only help with website-related scheduling questions and invite the user to ask about the allowed topics.
4) Prefer direct answers from provided schedule/rooms data. Do not ask users to navigate the website for answers you can compute.
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


@app.context_processor
def inject_import_schedule_datasets():
    api_key_configured = has_genai and bool(GEMINI_API_KEY)
            
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


def _answer_from_local_data(message: str, rooms: pd.DataFrame, schedule: pd.DataFrame) -> Optional[str]:
    text = message.strip()
    lower = text.lower()
    room_conflicts = analyzer.detect_room_conflicts(schedule)

    is_conflict_query = any(word in lower for word in ["conflict", "double-book", "double book", "overlap"])
    day = _extract_day(text)
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
        room_id = _extract_room_id(text, rooms)
        if room_id:
            day = _extract_day(text)
            query_time = _extract_query_time(text)
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

    if "total rooms" in lower or "how many rooms" in lower:
        return f"Total rooms in the current dataset: {len(rooms)}."
    if "total sections" in lower or "how many classes" in lower:
        return f"Total scheduled sections in the current dataset: {len(schedule)}."

    return None


def _build_chat_context(rooms: pd.DataFrame, schedule: pd.DataFrame) -> str:
    conflicts = analyzer.detect_room_conflicts(schedule)
    overview = analyzer.get_building_overview_kpis(rooms, schedule)
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

    if schedule_df.columns.tolist() != SCHEDULE_COLUMNS or rooms_df.columns.tolist() != ROOMS_COLUMNS:
        flash("Import a schedule file with the same layout as the current Schedule CSV.", "danger")
        return redirect(url_for("index"))

    schedule_df = schedule_df[SCHEDULE_COLUMNS]
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
    _, schedule, _ = get_data()

    day_filter = request.args.get("day", "all")
    time_filter = request.args.get("time", "all")
    room_filter = request.args.get("room_id", "all")
    instructor_filter = request.args.get("instructor", "all")

    filtered = schedule.copy()
    if day_filter != "all":
        filtered = filtered[filtered["Day"] == day_filter]
    if time_filter != "all":
        filtered = filtered[filtered["Time"] == time_filter]
    if room_filter != "all":
        filtered = filtered[filtered["Room_ID"] == room_filter]
    if instructor_filter != "all":
        filtered = filtered[filtered["Instructor"] == instructor_filter]

    days = sorted(schedule["Day"].unique().tolist())
    times = sorted(schedule["Time"].unique().tolist())
    room_ids = sorted(schedule["Room_ID"].unique().tolist())
    instructors = sorted(schedule["Instructor"].unique().tolist())

    return render_template(
        "schedule.html",
        schedule=filtered.to_dict(orient="records"),
        days=days,
        times=times,
        room_ids=room_ids,
        instructors=instructors,
        day_filter=day_filter,
        time_filter=time_filter,
        room_filter=room_filter,
        instructor_filter=instructor_filter,
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

    selected_crn = request.form.get("crn", "")
    selected_room = request.form.get("room_id", "")

    if request.method == "POST":
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
    impact = analyzer.calculate_optimization_impact(overcrowded_recommendations, twin_state, rooms)
    return render_template(
        "recommendations.html",
        overcrowded_recommendations=overcrowded_recommendations,
        underutilized_recommendations=underutilized_recommendations,
        conflict_recommendations=conflict_recommendations,
        impact=impact,
    )


@app.route("/schedule-assistant", methods=["GET", "POST"])
def schedule_assistant_view():
    rooms, schedule, twin_state = get_data()

    instructors = sorted(schedule["Instructor"].unique().tolist())
    room_ids = sorted(rooms["Room_ID"].unique().tolist())
    time_options = ["Any"] + analyzer.TIME_SLOTS

    selected_instructor = request.form.get("instructor", "")
    selected_time = request.form.get("preferred_time", "Any")
    selected_room = request.form.get("preferred_room", "Any")
    selected_students = request.form.get("num_students", "")
    selected_room_type = request.form.get("room_type", "Lecture Hall")

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
    
    if not message:
        return jsonify({"error": "No message provided."}), 400

    rooms, schedule, _ = get_data()

    local_reply = _answer_from_local_data(message, rooms, schedule)
    if local_reply:
        return jsonify({"reply": local_reply})

    if not has_genai:
        return jsonify({
            "reply": "I can answer direct schedule questions from CSV data (rooms, conflicts, availability). "
                     "Try: 'What days have conflicts?' or 'Is room 1003 available on Monday at 12:00?'"
        })

    if not GEMINI_API_KEY:
        return jsonify({
            "reply": "I can still answer direct CSV-based questions, but generative fallback is disabled because GEMINI_API_KEY is missing."
        })

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=CHATBOT_SYSTEM_INSTRUCTION,
        )

        context = _build_chat_context(rooms, schedule)
        prompt = (
            "Answer using the provided website dataset.\n"
            "If the question asks for conflicts/availability, give a direct answer and include exact room/day/time.\n"
            "If data is insufficient, say exactly what is missing.\n\n"
            f"{context}\n\n"
            f"User question: {message}"
        )
        response = model.generate_content(prompt)
        
        return jsonify({"reply": response.text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
