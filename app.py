"""
app.py
------
Flask entry point for the Digital Twin Classroom Scheduler.
Defines all routes and passes data to Jinja2 templates.
"""

from flask import Flask, render_template, request

import analyzer

app = Flask(__name__)


def get_data():
    rooms = analyzer.load_rooms()
    schedule = analyzer.load_schedule()
    twin_state = analyzer.build_twin_state(rooms, schedule)
    return rooms, schedule, twin_state


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
    recs = analyzer.generate_recommendations(schedule, rooms)
    impact = analyzer.calculate_optimization_impact(recs, twin_state, rooms)
    return render_template(
        "recommendations.html",
        recommendations=recs,
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
