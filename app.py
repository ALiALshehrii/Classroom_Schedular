"""
app.py
------
Flask entry point for the Digital Twin Classroom Scheduler.
Defines all routes and passes data to Jinja2 templates.
"""

from flask import Flask, render_template, request
import analyzer

app = Flask(__name__)


# ─── Helper: load fresh data on every request ─────────────────────────────────

def get_data():
    rooms    = analyzer.load_rooms()
    schedule = analyzer.load_schedule()
    return rooms, schedule


# ─── Page 1: Dashboard ────────────────────────────────────────────────────────

@app.route("/")
def index():
    rooms, schedule = get_data()
    kpis        = analyzer.get_kpis(rooms, schedule)
    bar_chart   = analyzer.get_room_occupancy_chart(schedule)
    pie_chart   = analyzer.get_status_pie_chart(schedule)
    return render_template(
        "index.html",
        kpis=kpis,
        bar_chart=bar_chart,
        pie_chart=pie_chart,
    )


# ─── Page 2: Rooms View ───────────────────────────────────────────────────────

@app.route("/rooms")
def rooms_view():
    rooms, _ = get_data()

    # Filters from query params
    floor_filter = request.args.get("floor", "all")
    type_filter  = request.args.get("type",  "all")

    filtered = rooms.copy()
    if floor_filter != "all":
        filtered = filtered[filtered["Floor"] == int(floor_filter)]
    if type_filter != "all":
        filtered = filtered[filtered["Type"] == type_filter]

    max_capacity = int(rooms["Capacity"].max()) 
    return render_template(
        "rooms.html",
        rooms=filtered.to_dict(orient="records"),
        floor_filter=floor_filter,
        type_filter=type_filter,
        max_capacity=max_capacity,
    )


# ─── Page 3: Schedule View ────────────────────────────────────────────────────

@app.route("/schedule")
def schedule_view():
    _, schedule = get_data()

    day_filter      = request.args.get("day",        "all")
    time_filter     = request.args.get("time",       "all")
    room_filter     = request.args.get("room_id",    "all")
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

    # Unique values for filter dropdowns
    days        = sorted(schedule["Day"].unique().tolist())
    times       = sorted(schedule["Time"].unique().tolist())
    room_ids    = sorted(schedule["Room_ID"].unique().tolist())
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


# ─── Page 4: Conflict Detection ───────────────────────────────────────────────

@app.route("/conflicts")
def conflicts_view():
    _, schedule = get_data()
    room_conflicts = analyzer.detect_room_conflicts(schedule)
    overcrowded    = analyzer.detect_overcrowded(schedule)
    underutilized  = analyzer.detect_underutilized(schedule)

    return render_template(
        "conflicts.html",
        room_conflicts=room_conflicts,
        overcrowded=overcrowded,
        underutilized=underutilized,
    )


# ─── Page 5: What-If Simulation ───────────────────────────────────────────────

@app.route("/simulation", methods=["GET", "POST"])
def simulation_view():
    rooms, schedule = get_data()

    crns     = sorted(schedule["CRN"].unique().tolist())
    room_ids = sorted(rooms["Room_ID"].unique().tolist())
    result   = None
    error    = None

    if request.method == "POST":
        selected_crn     = request.form.get("crn")
        selected_room_id = request.form.get("room_id")

        try:
            result = analyzer.simulate_move(
                schedule, rooms, selected_crn, selected_room_id
            )
        except (IndexError, KeyError) as e:
            error = f"خطأ في البيانات: {str(e)}"

    return render_template(
        "simulation.html",
        crns=crns,
        room_ids=room_ids,
        result=result,
        error=error,
        selected_crn=request.form.get("crn", ""),
        selected_room=request.form.get("room_id", ""),
    )


# ─── Page 6: Recommendations ──────────────────────────────────────────────────

@app.route("/recommendations")
def recommendations_view():
    rooms, schedule = get_data()
    recs = analyzer.generate_recommendations(schedule, rooms)
    return render_template("recommendations.html", recommendations=recs)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
