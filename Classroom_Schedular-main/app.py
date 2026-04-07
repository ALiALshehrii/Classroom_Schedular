"""
app.py
------
Flask entry point for the Digital Twin Classroom Scheduler.
Defines all routes and passes data to Jinja2 templates.
"""

import os

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for, jsonify, send_file

import analyzer

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "classroom-scheduler-dev-secret")
SCHEDULE_COLUMNS = pd.read_csv(analyzer.SCHEDULE_CSV, nrows=0).columns.tolist()
ROOMS_COLUMNS = pd.read_csv(analyzer.ROOMS_CSV, nrows=0).columns.tolist()


def get_data():
    rooms = analyzer.load_rooms()
    schedule = analyzer.load_schedule()
    twin_state = analyzer.build_twin_state(rooms, schedule)
    return rooms, schedule, twin_state


@app.route("/import-schedule-csv", methods=["POST"])
def import_schedule_csv():
    schedule_file = request.files.get("schedule_csv")
    rooms_file = request.files.get("rooms_csv")
    if schedule_file is None or not schedule_file.filename or rooms_file is None or not rooms_file.filename:
        flash("Please attach both Schedule CSV and Rooms CSV files.", "danger")
        return redirect(url_for("index"))

    if not schedule_file.filename.lower().endswith(".csv") or not rooms_file.filename.lower().endswith(".csv"):
        flash("Only CSV files are supported.", "danger")
        return redirect(url_for("index"))

    try:
        schedule_df = pd.read_csv(schedule_file)
        rooms_df = pd.read_csv(rooms_file)
    except Exception:
        flash("Could not read one or both uploaded files. Please upload valid CSV files.", "danger")
        return redirect(url_for("index"))

    if schedule_df.empty or rooms_df.empty:
        flash("One or both uploaded CSV files are empty.", "danger")
        return redirect(url_for("index"))

    if schedule_df.columns.tolist() != SCHEDULE_COLUMNS or rooms_df.columns.tolist() != ROOMS_COLUMNS:
        flash("Import files with the same layout as the existing Schedule and Rooms CSV files.", "danger")
        return redirect(url_for("index"))

    schedule_df = schedule_df[SCHEDULE_COLUMNS]
    rooms_df = rooms_df[ROOMS_COLUMNS]
    schedule_df.to_csv(analyzer.SCHEDULE_CSV, index=False)
    rooms_df.to_csv(analyzer.ROOMS_CSV, index=False)
    flash("Files imported successfully. Dashboard and all pages were rebuilt from the new files.", "success")
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

    instructors = sorted(schedule["Instructor"].dropna().unique().tolist())
    room_ids = sorted(rooms["Room_ID"].dropna().unique().tolist())
    room_types = sorted(rooms["Type"].dropna().unique().tolist())
    _, dynamic_times = analyzer._get_schedule_dimensions(schedule)
    time_options = ["Any"] + dynamic_times

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
        room_types=room_types,
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


@app.route("/calendar")
def calendar_view():
    rooms, schedule, _ = get_data()
    room_ids = sorted(rooms["Room_ID"].unique().tolist())
    room_counts = schedule["Room_ID"].value_counts().to_dict()
    # Default to "" (All Rooms) instead of implicitly forcing the first room from rooms.csv
    selected_room = request.args.get("room_id", "")
    return render_template("calendar.html", room_ids=room_ids, room_counts=room_counts, selected_room=selected_room)


@app.route("/api/events")
def api_events():
    _, schedule, _ = get_data()
    room_id = request.args.get("room_id")
    
    if room_id:
        filtered = schedule[schedule["Room_ID"] == room_id]
    else:
        filtered = schedule
        
    events = []
    # Map explicit Days to specific dates in our Dummy Week (e.g. 2024-01-07 is Sunday)
    day_mapping = {
        "Sunday": ["2024-01-07"],
        "Monday": ["2024-01-08"],
        "Tuesday": ["2024-01-09"],
        "Wednesday": ["2024-01-10"],
        "Thursday": ["2024-01-11"],
    }
    
    for _, row in filtered.iterrows():
        crn = str(row["CRN"])
        course_name = str(row['Course_Name'])
        title = f"{row['Course_Code']} - {course_name}"
        time_str = str(row["Time"]).strip()
        day_pattern = str(row["Day"]).strip()
        
        # Parse Time correctly (e.g. "09:30-10:45")
        if "-" in time_str:
            start_time = time_str.split("-")[0].strip()
            end_time = time_str.split("-")[1].strip()
        else:
            start_time = time_str
            try:
                hour = int(start_time.split(':')[0])
                mins = start_time.split(':')[1]
                end_time = f"{hour + 1:02d}:{mins}"
            except:
                end_time = start_time
        
        dates = day_mapping.get(day_pattern, [])
        for d in dates:
            events.append({
                "id": crn,
                "title": f"{title} ({row['Enrolled']}/{row['Capacity']})",
                "start": f"{d}T{start_time}:00",
                "end": f"{d}T{end_time}:00",
                "extendedProps": {
                    "day_pattern": day_pattern,
                    "crn": crn,
                    "room_id": str(row["Room_ID"]),
                    "status": str(row["Status"])
                },
                "backgroundColor": "#d9534f" if row["Status"] == "Overcrowded" else ("#f0ad4e" if row["Status"] == "Underutilized" else "#5cb85c"),
                "borderColor": "#d9534f" if row["Status"] == "Overcrowded" else ("#f0ad4e" if row["Status"] == "Underutilized" else "#5cb85c")
            })
            
    return jsonify(events)


@app.route("/api/update-event", methods=["POST"])
def api_update_event():
    data = request.json
    crn = data.get("crn")
    start_str = data.get("start")
    new_room = data.get("room_id")
    
    if not crn or not start_str or not new_room:
        return jsonify({"success": False, "error": "Missing parameters"})
        
    try:
        date_part = start_str.split("T")[0]   # "2024-01-08"
        start_time = start_str.split("T")[1][:5] # "08:00"
        
        # Reconstruct the correct schedule time block string.
        # Since we just have the drop start_time (e.g. 08:30) 
        # and we know typical classes are 1hr 15m, we can estimate it, 
        # but realistically FullCalendar passes the new END time natively if resized!
        # Let's just create a standard 1hr 15m block for now.
        try:
            shour = int(start_time.split(":")[0])
            smin = int(start_time.split(":")[1])
            total_mins = smin + 75
            ehour = shour + (total_mins // 60)
            emin = total_mins % 60
            end_time = f"{ehour:02d}:{emin:02d}"
            time_part = f"{start_time}-{end_time}"
        except:
            time_part = f"{start_time}-??:??"
        
        # Reverse map date -> day pattern
        day_pattern = "Sunday"
        if date_part == "2024-01-07": day_pattern = "Sunday"
        elif date_part == "2024-01-08": day_pattern = "Monday"
        elif date_part == "2024-01-09": day_pattern = "Tuesday"
        elif date_part == "2024-01-10": day_pattern = "Wednesday"
        elif date_part == "2024-01-11": day_pattern = "Thursday"
            
        success = analyzer.update_schedule_entry(crn, day_pattern, time_part, new_room)
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/apply-resolution", methods=["POST"])
def api_apply_resolution():
    data = request.json
    crn = data.get("crn")
    day = data.get("day")
    time = data.get("time")
    new_room = data.get("suggested_room")
    
    if not crn or not day or not time or not new_room:
        return jsonify({"success": False, "error": "Missing parameters"})
        
    try:
        success = analyzer.update_schedule_entry(str(crn), str(day), str(time), str(new_room))
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/export")
def export_csv():
    """Allows administrators to download the resolved database cleanly."""
    return send_file(
        analyzer.SCHEDULE_CSV, 
        as_attachment=True, 
        download_name="Official_CCIS_Schedule.csv"
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
