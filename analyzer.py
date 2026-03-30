"""
analyzer.py
-----------
Core logic module for the Digital Twin Classroom Scheduler.
Handles: data loading, occupancy calculation, conflict detection,
         what-if simulation, and room recommendations.
"""

import pandas as pd
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOMS_CSV    = os.path.join(BASE_DIR, "data", "rooms.csv")
SCHEDULE_CSV = os.path.join(BASE_DIR, "data", "schedule.csv")


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_rooms() -> pd.DataFrame:
    """Load rooms data from CSV and return as DataFrame."""
    df = pd.read_csv(ROOMS_CSV)
    df["Room_ID"] = df["Room_ID"].astype(str)
    return df


def load_schedule() -> pd.DataFrame:
    """
    Load schedule data from CSV, recalculate Status field,
    and add a Utilization_Pct column.
    """
    df = pd.read_csv(SCHEDULE_CSV)
    df["Room_ID"] = df["Room_ID"].astype(str)
    df["CRN"] = df["CRN"].astype(str)

    # Recalculate status based on business rules (ignore CSV value)
    def calc_status(row):
        if row["Enrolled"] > row["Capacity"]:
            return "Overcrowded"
        elif row["Enrolled"] < 0.4 * row["Capacity"]:
            return "Underutilized"
        else:
            return "Normal"

    df["Status"] = df.apply(calc_status, axis=1)

    # Utilization percentage (rounded to 1 decimal)
    df["Utilization_Pct"] = (
        (df["Enrolled"] / df["Capacity"]) * 100
    ).round(1)

    return df


# ─── Dashboard KPIs ───────────────────────────────────────────────────────────

def get_kpis(rooms: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Return summary KPI values for the dashboard."""
    total_rooms    = len(rooms)
    total_sections = len(schedule)
    overcrowded    = int((schedule["Status"] == "Overcrowded").sum())
    underutilized  = int((schedule["Status"] == "Underutilized").sum())
    avg_occupancy  = round(schedule["Utilization_Pct"].mean(), 1)

    return {
        "total_rooms":    total_rooms,
        "total_sections": total_sections,
        "overcrowded":    overcrowded,
        "underutilized":  underutilized,
        "avg_occupancy":  avg_occupancy,
    }


# ─── Chart Data ───────────────────────────────────────────────────────────────

def get_room_occupancy_chart(schedule: pd.DataFrame) -> dict:
    """
    Build chart data: average utilization % per room.
    Returns labels (room IDs) and data (avg utilization %).
    """
    grouped = (
        schedule.groupby("Room_ID")["Utilization_Pct"]
        .mean()
        .round(1)
        .reset_index()
        .sort_values("Room_ID")
    )
    return {
        "labels": grouped["Room_ID"].tolist(),
        "data":   grouped["Utilization_Pct"].tolist(),
    }


def get_status_pie_chart(schedule: pd.DataFrame) -> dict:
    """Return count of each status for pie chart."""
    counts = schedule["Status"].value_counts()
    return {
        "Normal":       int(counts.get("Normal", 0)),
        "Overcrowded":  int(counts.get("Overcrowded", 0)),
        "Underutilized": int(counts.get("Underutilized", 0)),
    }


# ─── Conflict Detection ───────────────────────────────────────────────────────

def detect_room_conflicts(schedule: pd.DataFrame) -> list[dict]:
    """
    Detect cases where the same Room_ID is scheduled at the
    same Day + Time slot (double-booking).
    Returns a list of conflict dicts.
    """
    conflicts = []
    grouped = schedule.groupby(["Room_ID", "Day", "Time"])

    for (room_id, day, time), group in grouped:
        if len(group) > 1:
            crns = group["CRN"].tolist()
            courses = group["Course_Name"].tolist()
            conflicts.append({
                "room_id":  room_id,
                "day":      day,
                "time":     time,
                "crns":     ", ".join(crns),
                "courses":  " | ".join(courses),
                "count":    len(group),
            })
    return conflicts


def detect_overcrowded(schedule: pd.DataFrame) -> list[dict]:
    """Return all sections where Enrolled > Capacity."""
    df = schedule[schedule["Status"] == "Overcrowded"].copy()
    return df[[
        "CRN", "Course_Name", "Instructor", "Room_ID",
        "Day", "Time", "Capacity", "Enrolled", "Utilization_Pct"
    ]].to_dict(orient="records")


def detect_underutilized(schedule: pd.DataFrame) -> list[dict]:
    """Return all sections where Enrolled < 40% of Capacity."""
    df = schedule[schedule["Status"] == "Underutilized"].copy()
    return df[[
        "CRN", "Course_Name", "Instructor", "Room_ID",
        "Day", "Time", "Capacity", "Enrolled", "Utilization_Pct"
    ]].to_dict(orient="records")


# ─── What-If Simulation ───────────────────────────────────────────────────────

def simulate_move(schedule: pd.DataFrame, rooms: pd.DataFrame,
                  crn: str, new_room_id: str) -> dict:
    """
    Simulate moving a CRN section to a new room.
    Returns before/after comparison dict.
    """
    # Current section data
    section = schedule[schedule["CRN"] == crn].iloc[0]
    new_room = rooms[rooms["Room_ID"] == new_room_id].iloc[0]

    old_capacity    = int(section["Capacity"])
    new_capacity    = int(new_room["Capacity"])
    enrolled        = int(section["Enrolled"])

    old_util = round((enrolled / old_capacity) * 100, 1)
    new_util = round((enrolled / new_capacity) * 100, 1)

    def status_label(util, enr, cap):
        if enr > cap:
            return "Overcrowded"
        elif enr < 0.4 * cap:
            return "Underutilized"
        else:
            return "Normal"

    old_status = status_label(old_util, enrolled, old_capacity)
    new_status = status_label(new_util, enrolled, new_capacity)

    # Check for conflicts in the new room at same day/time
    same_slot = schedule[
        (schedule["Room_ID"] == new_room_id) &
        (schedule["Day"]     == section["Day"]) &
        (schedule["Time"]    == section["Time"]) &
        (schedule["CRN"]     != crn)
    ]
    has_conflict = len(same_slot) > 0

    return {
        "crn":          crn,
        "course_name":  section["Course_Name"],
        "instructor":   section["Instructor"],
        "day":          section["Day"],
        "time":         section["Time"],
        "enrolled":     enrolled,
        # Before
        "old_room":     section["Room_ID"],
        "old_capacity": old_capacity,
        "old_util":     old_util,
        "old_status":   old_status,
        # After
        "new_room":     new_room_id,
        "new_room_type": new_room["Type"],
        "new_capacity": new_capacity,
        "new_util":     new_util,
        "new_status":   new_status,
        "has_conflict": has_conflict,
        "conflict_with": same_slot["CRN"].tolist() if has_conflict else [],
    }


# ─── Recommendations ──────────────────────────────────────────────────────────

def generate_recommendations(schedule: pd.DataFrame,
                              rooms: pd.DataFrame) -> list[dict]:
    """
    For every overcrowded section, suggest the best alternative room:
      - Same type (Lecture Hall ↔ Lecture Hall, Lab ↔ Lab)
      - Capacity >= Enrolled
      - Not already booked at the same Day + Time
    Returns a list of recommendation dicts.
    """
    recommendations = []
    overcrowded = schedule[schedule["Status"] == "Overcrowded"]

    for _, section in overcrowded.iterrows():
        crn      = section["CRN"]
        day      = section["Day"]
        time     = section["Time"]
        enrolled = section["Enrolled"]
        rtype    = section["Room_Type"]
        curr_room = section["Room_ID"]

        # Rooms of same type with sufficient capacity
        candidates = rooms[
            (rooms["Type"]     == rtype) &
            (rooms["Capacity"] >= enrolled) &
            (rooms["Room_ID"]  != curr_room)
        ].copy()

        # Rooms already busy at same day+time
        busy_rooms = schedule[
            (schedule["Day"]  == day) &
            (schedule["Time"] == time)
        ]["Room_ID"].tolist()

        # Free candidates
        free_candidates = candidates[
            ~candidates["Room_ID"].isin(busy_rooms)
        ].sort_values("Capacity")   # prefer smallest fitting room

        if not free_candidates.empty:
            best = free_candidates.iloc[0]
            recommendations.append({
                "crn":            crn,
                "course_name":    section["Course_Name"],
                "current_room":   curr_room,
                "suggested_room": best["Room_ID"],
                "current_cap":    int(section["Capacity"]),
                "suggested_cap":  int(best["Capacity"]),
                "enrolled":       int(enrolled),
                "room_type":      rtype,
                "day":            day,
                "time":           time,
            })
        else:
            recommendations.append({
                "crn":            crn,
                "course_name":    section["Course_Name"],
                "current_room":   curr_room,
                "suggested_room": "No alternative available",
                "current_cap":    int(section["Capacity"]),
                "suggested_cap":  "-",
                "enrolled":       int(enrolled),
                "room_type":      rtype,
                "day":            day,
                "time":           time,
            })

    return recommendations
