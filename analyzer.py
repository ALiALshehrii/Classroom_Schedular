"""
analyzer.py
-----------
Core logic module for the Digital Twin Classroom Scheduler.
Handles: data loading, occupancy calculation, conflict detection,
         what-if simulation, recommendations, heatmaps, and analytics.
"""

import os
from datetime import datetime
from typing import Optional

import pandas as pd

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOMS_CSV = os.path.join(BASE_DIR, "data", "rooms.csv")
SCHEDULE_CSV = os.path.join(BASE_DIR, "data", "schedule.csv")

# ─── Twin Dimensions ─────────────────────────────────────────────────────────
DAY_OPTIONS = ["Sun-Tue-Thu", "Mon-Wed"]
TIME_SLOTS = ["08:00", "09:00", "10:00", "11:00", "13:00"]


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
    df["Day"] = df["Day"].astype(str).str.strip()
    df["Time"] = df["Time"].astype(str).str.strip()

    def calc_status(row):
        if row["Enrolled"] > row["Capacity"]:
            return "Overcrowded"
        if row["Enrolled"] < 0.4 * row["Capacity"]:
            return "Underutilized"
        return "Normal"

    df["Status"] = df.apply(calc_status, axis=1)
    df["Utilization_Pct"] = ((df["Enrolled"] / df["Capacity"]) * 100).round(1)
    return df


# ─── Twin State ───────────────────────────────────────────────────────────────

def _slot_key(room_id: str, day: str, time: str) -> str:
    return f"{room_id}|{day}|{time}"


def _get_schedule_dimensions(schedule_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return day and time axes derived from schedule data.
    Falls back to defaults when schedule is empty.
    """
    if schedule_df.empty:
        return DAY_OPTIONS, TIME_SLOTS

    days = (
        schedule_df["Day"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    times_raw = (
        schedule_df["Time"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if not days:
        days = DAY_OPTIONS
    if not times_raw:
        return days, TIME_SLOTS

    def _time_sort_key(value: str) -> tuple[int, int, str]:
        formats = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p")
        for fmt in formats:
            try:
                parsed = datetime.strptime(value, fmt)
                return (0, parsed.hour * 60 + parsed.minute, value)
            except ValueError:
                continue
        return (1, 0, value)

    times = sorted(times_raw, key=_time_sort_key)
    return days, times


def build_twin_state(rooms_df: pd.DataFrame, schedule_df: pd.DataFrame) -> dict:
    """
    Build a full in-memory mirror of every (Room_ID, Day, Time) slot.
    Each slot stores occupancy and list of assigned sections.
    """
    slots = {}
    day_options, time_slots = _get_schedule_dimensions(schedule_df)

    for _, room in rooms_df.iterrows():
        room_id = str(room["Room_ID"])
        for day in day_options:
            for time in time_slots:
                key = _slot_key(room_id, day, time)
                slots[key] = {
                    "room_id": room_id,
                    "floor": int(room["Floor"]),
                    "room_type": room["Type"],
                    "capacity": int(room["Capacity"]),
                    "day": day,
                    "time": time,
                    "sections": [],
                    "occupied": False,
                }

    for _, row in schedule_df.iterrows():
        key = _slot_key(str(row["Room_ID"]), row["Day"], row["Time"])
        if key not in slots:
            slots[key] = {
                "room_id": str(row["Room_ID"]),
                "floor": None,
                "room_type": row["Room_Type"],
                "capacity": int(row["Capacity"]),
                "day": row["Day"],
                "time": row["Time"],
                "sections": [],
                "occupied": False,
            }

        slots[key]["sections"].append({
            "CRN": str(row["CRN"]),
            "Course_Code": row["Course_Code"],
            "Course_Name": row["Course_Name"],
            "Instructor": row["Instructor"],
            "Enrolled": int(row["Enrolled"]),
            "Capacity": int(row["Capacity"]),
            "Room_ID": str(row["Room_ID"]),
            "Room_Type": row["Room_Type"],
            "Day": row["Day"],
            "Time": row["Time"],
            "Status": row["Status"],
            "Utilization_Pct": float(row["Utilization_Pct"]),
        })
        slots[key]["occupied"] = True

    total_slots = len(slots)
    occupied_slots = sum(1 for v in slots.values() if v["occupied"])
    return {
        "slots": slots,
        "total_slots": total_slots,
        "occupied_slots": occupied_slots,
        "empty_slots": total_slots - occupied_slots,
    }


# ─── Dashboard KPIs ───────────────────────────────────────────────────────────

def get_kpis(rooms: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Return summary KPI values for the dashboard."""
    total_rooms = len(rooms)
    total_sections = len(schedule)
    overcrowded = int((schedule["Status"] == "Overcrowded").sum())
    underutilized = int((schedule["Status"] == "Underutilized").sum())
    avg_occupancy = round(schedule["Utilization_Pct"].mean(), 1)

    return {
        "total_rooms": total_rooms,
        "total_sections": total_sections,
        "overcrowded": overcrowded,
        "underutilized": underutilized,
        "avg_occupancy": avg_occupancy,
    }


# ─── Chart Data ───────────────────────────────────────────────────────────────

def get_room_occupancy_chart(schedule: pd.DataFrame) -> dict:
    """Build average utilization % per room for bar chart."""
    grouped = (
        schedule.groupby("Room_ID")["Utilization_Pct"]
        .mean()
        .round(1)
        .reset_index()
        .sort_values("Room_ID")
    )
    return {
        "labels": grouped["Room_ID"].tolist(),
        "data": grouped["Utilization_Pct"].tolist(),
    }


def get_status_pie_chart(schedule: pd.DataFrame) -> dict:
    """Return count of each status for pie chart."""
    counts = schedule["Status"].value_counts()
    return {
        "Normal": int(counts.get("Normal", 0)),
        "Overcrowded": int(counts.get("Overcrowded", 0)),
        "Underutilized": int(counts.get("Underutilized", 0)),
    }


# ─── Conflict Detection ───────────────────────────────────────────────────────

def detect_room_conflicts(schedule: pd.DataFrame) -> list:
    """Detect same Room_ID at same Day + Time slot (double-booking)."""
    conflicts = []
    grouped = schedule.groupby(["Room_ID", "Day", "Time"])

    for (room_id, day, time), group in grouped:
        if len(group) > 1:
            conflicts.append({
                "room_id": room_id,
                "day": day,
                "time": time,
                "crns": ", ".join(group["CRN"].tolist()),
                "courses": " | ".join(group["Course_Name"].tolist()),
                "count": len(group),
            })
    return conflicts


def detect_overcrowded(schedule: pd.DataFrame) -> list:
    """Return sections where Enrolled > Capacity."""
    df = schedule[schedule["Status"] == "Overcrowded"].copy()
    return df[[
        "CRN", "Course_Name", "Instructor", "Room_ID",
        "Day", "Time", "Capacity", "Enrolled", "Utilization_Pct"
    ]].to_dict(orient="records")


def detect_underutilized(schedule: pd.DataFrame) -> list:
    """Return sections where Enrolled < 40% of Capacity."""
    df = schedule[schedule["Status"] == "Underutilized"].copy()
    return df[[
        "CRN", "Course_Name", "Instructor", "Room_ID",
        "Day", "Time", "Capacity", "Enrolled", "Utilization_Pct"
    ]].to_dict(orient="records")


# ─── What-If Simulation ───────────────────────────────────────────────────────

def simulate_move(schedule: pd.DataFrame, rooms: pd.DataFrame,
                  crn: str, new_room_id: str) -> dict:
    """Simulate moving a CRN section to a new room."""
    section = schedule[schedule["CRN"] == crn].iloc[0]
    new_room = rooms[rooms["Room_ID"] == new_room_id].iloc[0]

    old_capacity = int(section["Capacity"])
    new_capacity = int(new_room["Capacity"])
    enrolled = int(section["Enrolled"])

    old_util = round((enrolled / old_capacity) * 100, 1)
    new_util = round((enrolled / new_capacity) * 100, 1)

    def status_label(enr: int, cap: int) -> str:
        if enr > cap:
            return "Overcrowded"
        if enr < 0.4 * cap:
            return "Underutilized"
        return "Normal"

    old_status = status_label(enrolled, old_capacity)
    new_status = status_label(enrolled, new_capacity)

    same_slot = schedule[
        (schedule["Room_ID"] == new_room_id) &
        (schedule["Day"] == section["Day"]) &
        (schedule["Time"] == section["Time"]) &
        (schedule["CRN"] != crn)
    ]
    has_conflict = len(same_slot) > 0

    return {
        "crn": crn,
        "course_name": section["Course_Name"],
        "instructor": section["Instructor"],
        "day": section["Day"],
        "time": section["Time"],
        "enrolled": enrolled,
        "old_room": section["Room_ID"],
        "old_room_type": section["Room_Type"],
        "old_capacity": old_capacity,
        "old_util": old_util,
        "old_status": old_status,
        "new_room": new_room_id,
        "new_room_type": new_room["Type"],
        "new_capacity": new_capacity,
        "new_util": new_util,
        "new_status": new_status,
        "has_conflict": has_conflict,
        "conflict_with": same_slot["CRN"].tolist() if has_conflict else [],
    }


# ─── Recommendations (Optimization Twin) ─────────────────────────────────────

def generate_recommendations(schedule: pd.DataFrame, rooms: pd.DataFrame) -> list:
    """
    For each overcrowded section, suggest best same-type room:
      - Capacity >= Enrolled
      - Not already occupied at same Day + Time
      - Prefer smallest fitting room
    """
    recommendations = []
    overcrowded = schedule[schedule["Status"] == "Overcrowded"]

    for _, section in overcrowded.iterrows():
        day = section["Day"]
        time = section["Time"]
        enrolled = int(section["Enrolled"])
        room_type = section["Room_Type"]
        current_room = str(section["Room_ID"])
        current_cap = int(section["Capacity"])
        overflow = max(enrolled - current_cap, 0)
        current_util_pct = round((enrolled / current_cap) * 100, 1) if current_cap else 0.0

        candidates = rooms[
            (rooms["Type"] == room_type) &
            (rooms["Capacity"] >= enrolled) &
            (rooms["Room_ID"] != current_room)
        ].copy()

        busy_rooms = schedule[
            (schedule["Day"] == day) &
            (schedule["Time"] == time)
        ]["Room_ID"].tolist()

        free_candidates = candidates[
            ~candidates["Room_ID"].isin(busy_rooms)
        ].sort_values("Capacity")

        if not free_candidates.empty:
            best = free_candidates.iloc[0]
            suggested_room = str(best["Room_ID"])
            suggested_cap = int(best["Capacity"])
            suggested_util_pct = round((enrolled / suggested_cap) * 100, 1) if suggested_cap else 0.0
            recommendations.append({
                "crn": str(section["CRN"]),
                "course_name": section["Course_Name"],
                "instructor": section["Instructor"],
                "current_room": current_room,
                "suggested_room": suggested_room,
                "current_cap": current_cap,
                "suggested_cap": suggested_cap,
                "enrolled": enrolled,
                "room_type": room_type,
                "day": day,
                "time": time,
                "overflow": overflow,
                "current_util_pct": current_util_pct,
                "suggested_util_pct": suggested_util_pct,
                "seat_gain": suggested_cap - current_cap,
                "actionable": True,
                "recommendation_status": "Actionable",
                "recommendation_reason": "Alternative room available",
            })
        else:
            recommendations.append({
                "crn": str(section["CRN"]),
                "course_name": section["Course_Name"],
                "instructor": section["Instructor"],
                "current_room": current_room,
                "suggested_room": "No alternative available",
                "current_cap": current_cap,
                "suggested_cap": "-",
                "enrolled": enrolled,
                "room_type": room_type,
                "day": day,
                "time": time,
                "overflow": overflow,
                "current_util_pct": current_util_pct,
                "suggested_util_pct": "-",
                "seat_gain": "-",
                "actionable": False,
                "recommendation_status": "No Match",
                "recommendation_reason": "No free same-type room can hold enrollment at this day/time",
            })

    recommendations.sort(
        key=lambda r: (
            not r.get("actionable", False),
            r.get("day", ""),
            r.get("time", ""),
            r.get("course_name", ""),
        )
    )
    return recommendations


def generate_underutilized_recommendations(schedule: pd.DataFrame, rooms: pd.DataFrame) -> list:
    """
    For each underutilized section, suggest a smaller same-type room:
      - Capacity >= Enrolled
      - Capacity < current room capacity
      - Not occupied at same Day + Time
      - Prefer smallest fitting room
    """
    recommendations = []
    underutilized = schedule[schedule["Status"] == "Underutilized"]

    for _, section in underutilized.iterrows():
        day = section["Day"]
        time = section["Time"]
        enrolled = int(section["Enrolled"])
        room_type = section["Room_Type"]
        current_room = str(section["Room_ID"])
        current_cap = int(section["Capacity"])
        current_util_pct = round((enrolled / current_cap) * 100, 1) if current_cap else 0.0

        candidates = rooms[
            (rooms["Type"] == room_type) &
            (rooms["Capacity"] >= enrolled) &
            (rooms["Capacity"] < current_cap) &
            (rooms["Room_ID"] != current_room)
        ].copy()

        busy_rooms = schedule[
            (schedule["Day"] == day) &
            (schedule["Time"] == time)
        ]["Room_ID"].tolist()

        free_candidates = candidates[
            ~candidates["Room_ID"].isin(busy_rooms)
        ].sort_values("Capacity")

        if not free_candidates.empty:
            best = free_candidates.iloc[0]
            suggested_room = str(best["Room_ID"])
            suggested_cap = int(best["Capacity"])
            suggested_util_pct = round((enrolled / suggested_cap) * 100, 1) if suggested_cap else 0.0
            recommendations.append({
                "crn": str(section["CRN"]),
                "course_name": section["Course_Name"],
                "day": day,
                "time": time,
                "current_room": current_room,
                "current_cap": current_cap,
                "enrolled": enrolled,
                "current_util_pct": current_util_pct,
                "suggested_room": suggested_room,
                "suggested_cap": suggested_cap,
                "suggested_util_pct": suggested_util_pct,
                "actionable": True,
                "action": "Move to smaller room",
            })
        else:
            recommendations.append({
                "crn": str(section["CRN"]),
                "course_name": section["Course_Name"],
                "day": day,
                "time": time,
                "current_room": current_room,
                "current_cap": current_cap,
                "enrolled": enrolled,
                "current_util_pct": current_util_pct,
                "suggested_room": "No suitable smaller room",
                "suggested_cap": "-",
                "suggested_util_pct": "-",
                "actionable": False,
                "action": "Keep room (no better option)",
            })

    recommendations.sort(
        key=lambda r: (
            not r.get("actionable", False),
            r.get("day", ""),
            r.get("time", ""),
            r.get("course_name", ""),
        )
    )
    return recommendations


def generate_conflict_recommendations(schedule: pd.DataFrame, rooms: pd.DataFrame) -> list:
    """
    For each double-booked slot, suggest alternative rooms for extra sections.
    First section keeps current room; others try to move.
    """
    recommendations = []
    grouped = schedule.groupby(["Room_ID", "Day", "Time"])

    for (room_id, day, time), group in grouped:
        if len(group) <= 1:
            continue

        ordered = group.sort_values("CRN")
        keep_crn = str(ordered.iloc[0]["CRN"])

        for _, section in ordered.iterrows():
            crn = str(section["CRN"])
            enrolled = int(section["Enrolled"])
            room_type = section["Room_Type"]
            current_room = str(section["Room_ID"])

            if crn == keep_crn:
                recommendations.append({
                    "crn": crn,
                    "course_name": section["Course_Name"],
                    "day": day,
                    "time": time,
                    "current_room": current_room,
                    "suggested_room": current_room,
                    "actionable": False,
                    "action": "Keep in current room",
                })
                continue

            candidates = rooms[
                (rooms["Type"] == room_type) &
                (rooms["Capacity"] >= enrolled) &
                (rooms["Room_ID"] != current_room)
            ].copy()

            busy_rooms = schedule[
                (schedule["Day"] == day) &
                (schedule["Time"] == time)
            ]["Room_ID"].tolist()

            free_candidates = candidates[
                ~candidates["Room_ID"].isin(busy_rooms)
            ].sort_values("Capacity")

            if not free_candidates.empty:
                best = free_candidates.iloc[0]
                recommendations.append({
                    "crn": crn,
                    "course_name": section["Course_Name"],
                    "day": day,
                    "time": time,
                    "current_room": current_room,
                    "suggested_room": str(best["Room_ID"]),
                    "actionable": True,
                    "action": "Move to alternative room",
                })
            else:
                recommendations.append({
                    "crn": crn,
                    "course_name": section["Course_Name"],
                    "day": day,
                    "time": time,
                    "current_room": current_room,
                    "suggested_room": "No free room",
                    "actionable": False,
                    "action": "Manual reschedule needed",
                })

    recommendations.sort(
        key=lambda r: (
            r.get("day", ""),
            r.get("time", ""),
            r.get("current_room", ""),
            r.get("crn", ""),
        )
    )
    return recommendations


def calculate_optimization_impact(
    recommendations: list,
    twin_state: dict,
    rooms_df: pd.DataFrame
) -> dict:
    """
    Calculates projected improvement if all actionable recommendations are applied.
    Returns current_avg_utilization, projected_avg_utilization, sections_resolved.
    """
    all_sections = []
    for slot in twin_state["slots"].values():
        all_sections.extend(slot["sections"])

    if not all_sections:
        return {
            "current_avg_utilization": 0.0,
            "projected_avg_utilization": 0.0,
            "sections_resolved": 0,
            "applied_recommendations": 0,
        }

    room_capacity_map = rooms_df.set_index("Room_ID")["Capacity"].to_dict()
    rec_map = {
        rec["crn"]: rec
        for rec in recommendations
        if rec["suggested_room"] != "No alternative available"
    }

    current_total_enrolled = 0
    current_total_capacity = 0
    projected_total_enrolled = 0
    projected_total_capacity = 0
    sections_resolved = 0
    applied_recommendations = 0

    for section in all_sections:
        crn = section["CRN"]
        enrolled = int(section["Enrolled"])
        current_capacity = int(section["Capacity"])
        projected_capacity = current_capacity

        current_total_enrolled += enrolled
        current_total_capacity += current_capacity

        if crn in rec_map:
            target_room = rec_map[crn]["suggested_room"]
            if target_room in room_capacity_map:
                projected_capacity = int(room_capacity_map[target_room])
                applied_recommendations += 1
                if enrolled > current_capacity and enrolled <= projected_capacity:
                    sections_resolved += 1

        projected_total_enrolled += enrolled
        projected_total_capacity += projected_capacity

    current_avg = round(
        (current_total_enrolled / current_total_capacity) * 100, 1
    ) if current_total_capacity else 0.0
    projected_avg = round(
        (projected_total_enrolled / projected_total_capacity) * 100, 1
    ) if projected_total_capacity else 0.0

    return {
        "current_avg_utilization": current_avg,
        "projected_avg_utilization": projected_avg,
        "sections_resolved": sections_resolved,
        "applied_recommendations": applied_recommendations,
    }


# ─── Scheduling Assistant ─────────────────────────────────────────────────────

def _fit_badge(utilization_pct: float) -> tuple:
    """
    Fit rating for suggested room.
    Tight: very close to or above full.
    Good Fit: high but healthy usage.
    Acceptable: lower usage.
    """
    if utilization_pct >= 95:
        return "Tight", "danger"
    if utilization_pct >= 70:
        return "Good Fit", "success"
    return "Acceptable", "warning"


def find_available_slots(
    twin_state: dict,
    rooms_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    instructor: Optional[str] = None,
    preferred_time: Optional[str] = None,
    preferred_room: Optional[str] = None,
    num_students: int = 0,
    room_type: Optional[str] = None,
) -> list:
    """
    Return ranked available (day, time, room) combinations based on constraints.
    Ranking:
      1) Smallest non-negative capacity slack
      2) Lower floor first
      3) Less busy time slot first
    """
    preferred_time = None if preferred_time in (None, "", "Any") else preferred_time
    preferred_room = None if preferred_room in (None, "", "Any") else str(preferred_room)

    rooms = rooms_df.copy()
    if room_type:
        rooms = rooms[rooms["Type"] == room_type]
    if preferred_room:
        rooms = rooms[rooms["Room_ID"] == preferred_room]

    if rooms.empty:
        return []

    instructor_busy = set()
    if instructor:
        instructor_rows = schedule_df[schedule_df["Instructor"] == instructor]
        instructor_busy = set(
            zip(instructor_rows["Day"].tolist(), instructor_rows["Time"].tolist())
        )

    time_busyness = schedule_df.groupby("Time").size().to_dict()
    day_order = {d: i for i, d in enumerate(DAY_OPTIONS)}

    suggestions = []
    for _, room in rooms.iterrows():
        room_id = str(room["Room_ID"])
        floor = int(room["Floor"])
        capacity = int(room["Capacity"])
        slack = capacity - int(num_students)

        if slack < 0:
            continue

        for day in DAY_OPTIONS:
            for time in TIME_SLOTS:
                if preferred_time and time != preferred_time:
                    continue
                if (day, time) in instructor_busy:
                    continue

                key = _slot_key(room_id, day, time)
                slot = twin_state["slots"].get(key)
                if not slot or slot["occupied"]:
                    continue

                utilization = round((int(num_students) / capacity) * 100, 1) if capacity else 0.0
                badge_label, badge_level = _fit_badge(utilization)

                suggestions.append({
                    "room_id": room_id,
                    "floor": floor,
                    "room_type": room["Type"],
                    "capacity": capacity,
                    "day": day,
                    "time": time,
                    "utilization_pct": utilization,
                    "fit_label": badge_label,
                    "fit_level": badge_level,
                    "slack": slack,
                    "time_busyness": int(time_busyness.get(time, 0)),
                    "_rank": (
                        slack,
                        floor,
                        int(time_busyness.get(time, 0)),
                        day_order.get(day, 99),
                        room_id,
                    ),
                })

    suggestions.sort(key=lambda x: x["_rank"])
    for row in suggestions:
        row.pop("_rank", None)
    return suggestions


# ─── Heatmaps ─────────────────────────────────────────────────────────────────

def occupancy_to_color_light(pct: float) -> str:
    """Map occupancy percent to heatmap color for light theme."""
    if pct == 0:
        return "#f8fafc"
    if pct < 40:
        return "#dbeafe"
    if pct < 80:
        return "#93c5fd"
    if pct < 100:
        return "#fdba74"
    return "#fca5a5"


def occupancy_to_color_dark(pct: float) -> str:
    """Map occupancy percent to heatmap color for dark theme."""
    if pct == 0:
        return "#0f172a"
    if pct < 40:
        return "#0f3d5e"
    if pct < 80:
        return "#0d6e9f"
    if pct < 100:
        return "#b45309"
    return "#b91c1c"


def occupancy_to_text_color_light(pct: float) -> str:
    """Readable labels for light theme heatmap backgrounds."""
    if pct < 80:
        return "#0f172a"
    return "#111827"


def occupancy_to_text_color_dark(pct: float) -> str:
    """Readable labels for dark theme heatmap backgrounds."""
    if pct < 5:
        return "#cbd5e1"
    return "#f8fafc"


def build_room_time_heatmap(rooms_df: pd.DataFrame, schedule_df: pd.DataFrame) -> dict:
    """
    Heatmap 1: Room x Time.
    Cell value is occupancy % averaged across day patterns.
    """
    room_map = rooms_df.set_index("Room_ID").to_dict(orient="index")
    day_options, time_slots = _get_schedule_dimensions(schedule_df)
    rows = []

    rooms_sorted = [
        str(r["Room_ID"])
        for _, r in rooms_df.sort_values(["Floor", "Room_ID"]).iterrows()
    ]
    scheduled_room_ids = (
        schedule_df["Room_ID"].astype(str).drop_duplicates().tolist()
        if not schedule_df.empty else []
    )
    extra_room_ids = sorted([rid for rid in scheduled_room_ids if rid not in room_map])
    all_room_ids = rooms_sorted + extra_room_ids

    for room_id in all_room_ids:
        room_info = room_map.get(room_id)
        floor_value = int(room_info["Floor"]) if room_info is not None else None
        floor_label = f"F{floor_value}" if floor_value is not None else "Unmapped"
        room_capacity = int(room_info["Capacity"]) if room_info is not None else None
        row_cells = []
        for time in time_slots:
            day_pcts = []
            tooltip_parts = []
            for day in day_options:
                slot_df = schedule_df[
                    (schedule_df["Room_ID"] == room_id) &
                    (schedule_df["Time"] == time) &
                    (schedule_df["Day"] == day)
                ]

                if slot_df.empty:
                    day_pcts.append(0.0)
                    tooltip_parts.append(f"{day}: Empty")
                else:
                    total_enrolled = int(slot_df["Enrolled"].sum())
                    if room_capacity is not None:
                        cap = room_capacity
                    else:
                        cap = int(slot_df["Capacity"].max()) if not slot_df["Capacity"].empty else 0
                    pct = round((total_enrolled / cap) * 100, 1) if cap else 0.0
                    day_pcts.append(pct)
                    details = " | ".join(
                        f"{r['Course_Name']} ({r['Instructor']}, {int(r['Enrolled'])}/{int(r['Capacity'])})"
                        for _, r in slot_df.iterrows()
                    )
                    tooltip_parts.append(f"{day}: {details}")

            avg_pct = round(sum(day_pcts) / len(day_options), 1)
            row_cells.append({
                "time": time,
                "value": avg_pct,
                "color": occupancy_to_color_light(avg_pct),
                "text_color": occupancy_to_text_color_light(avg_pct),
                "dark_color": occupancy_to_color_dark(avg_pct),
                "dark_text_color": occupancy_to_text_color_dark(avg_pct),
                "tooltip": f"Room {room_id} at {time} | " + " || ".join(tooltip_parts),
            })

        rows.append({
            "room_id": room_id,
            "floor": floor_value,
            "floor_label": floor_label,
            "cells": row_cells,
        })

    return {"times": time_slots, "rows": rows}


def build_floor_time_heatmap(rooms_df: pd.DataFrame, schedule_df: pd.DataFrame) -> dict:
    """
    Heatmap 2: Floor x Time.
    Cell value shows average students on floor at that time.
    Color is based on occupancy % vs total floor capacity.
    """
    room_floor = rooms_df.set_index("Room_ID")["Floor"].to_dict()
    floor_capacity = rooms_df.groupby("Floor")["Capacity"].sum().to_dict()
    day_options, time_slots = _get_schedule_dimensions(schedule_df)
    rooms_by_floor: dict[object, set[str]] = {
        int(floor): set(
            rooms_df[rooms_df["Floor"] == floor]["Room_ID"].astype(str).tolist()
        )
        for floor in sorted(rooms_df["Floor"].unique().tolist())
    }
    scheduled_room_ids = (
        schedule_df["Room_ID"].astype(str).drop_duplicates().tolist()
        if not schedule_df.empty else []
    )
    unmapped_rooms = {rid for rid in scheduled_room_ids if rid not in room_floor}
    if unmapped_rooms:
        rooms_by_floor["Unmapped"] = unmapped_rooms

    schedule_capacity_by_room = (
        schedule_df.groupby("Room_ID")["Capacity"].max().to_dict()
        if not schedule_df.empty else {}
    )

    rows = []
    floor_keys = list(rooms_by_floor.keys())
    for floor in floor_keys:
        floor_cells = []
        floor_room_ids = rooms_by_floor[floor]
        floor_label = f"Floor {int(floor)}" if floor != "Unmapped" else "Floor Unmapped"
        if floor == "Unmapped":
            cap = int(sum(schedule_capacity_by_room.get(room_id, 0) for room_id in floor_room_ids))
        else:
            cap = int(floor_capacity.get(floor, 0))
        for time in time_slots:
            day_students = []
            tooltip_parts = []
            for day in day_options:
                slot_df = schedule_df[
                    (schedule_df["Room_ID"].isin(floor_room_ids)) &
                    (schedule_df["Time"] == time) &
                    (schedule_df["Day"] == day)
                ]
                total_students = int(slot_df["Enrolled"].sum())
                day_students.append(total_students)

                if slot_df.empty:
                    tooltip_parts.append(f"{day}: 0 students")
                else:
                    courses = " | ".join(
                        f"{r['Course_Name']} ({int(r['Enrolled'])})"
                        for _, r in slot_df.iterrows()
                    )
                    tooltip_parts.append(f"{day}: {total_students} students [{courses}]")

            avg_students = round(sum(day_students) / len(day_options), 1)
            occ_pct = round((avg_students / cap) * 100, 1) if cap else 0.0

            floor_cells.append({
                "time": time,
                "students": avg_students,
                "occupancy_pct": occ_pct,
                "color": occupancy_to_color_light(occ_pct),
                "text_color": occupancy_to_text_color_light(occ_pct),
                "dark_color": occupancy_to_color_dark(occ_pct),
                "dark_text_color": occupancy_to_text_color_dark(occ_pct),
                "tooltip": (
                    f"{floor_label} at {time}: {avg_students} students avg, "
                    f"{occ_pct}% of floor capacity | " + " || ".join(tooltip_parts)
                ),
            })

        rows.append({
            "floor": floor if floor == "Unmapped" else int(floor),
            "floor_label": floor_label,
            "cells": floor_cells,
        })

    return {"times": time_slots, "rows": rows}


# ─── Analytics ────────────────────────────────────────────────────────────────

def get_building_overview_kpis(rooms_df: pd.DataFrame, schedule_df: pd.DataFrame) -> dict:
    total_rooms = int(len(rooms_df))
    lecture_halls = int((rooms_df["Type"] == "Lecture Hall").sum())
    computer_labs = int((rooms_df["Type"] == "Computer Lab").sum())
    total_sections = int(len(schedule_df))

    day_options, time_slots = _get_schedule_dimensions(schedule_df)
    active_slots = int(
        schedule_df[["Room_ID", "Day", "Time"]].drop_duplicates().shape[0]
    )
    total_possible_slots = int(total_rooms * len(day_options) * len(time_slots))
    empty_slots = int(total_possible_slots - active_slots)
    overall_utilization = round(
        (schedule_df["Enrolled"].sum() / schedule_df["Capacity"].sum()) * 100, 1
    ) if schedule_df["Capacity"].sum() else 0.0

    return {
        "total_rooms": total_rooms,
        "lecture_halls": lecture_halls,
        "computer_labs": computer_labs,
        "total_sections": total_sections,
        "active_slots": active_slots,
        "empty_slots": empty_slots,
        "overall_utilization": overall_utilization,
    }


def get_problem_summary(rooms_df: pd.DataFrame, schedule_df: pd.DataFrame) -> dict:
    overcrowded_count = int((schedule_df["Status"] == "Overcrowded").sum())
    underutilized_count = int((schedule_df["Status"] == "Underutilized").sum())

    conflicts = (
        schedule_df.groupby(["Room_ID", "Day", "Time"])
        .size()
        .reset_index(name="cnt")
    )
    double_booked = int((conflicts["cnt"] >= 2).sum())

    overflow = (schedule_df["Enrolled"] - schedule_df["Capacity"]).clip(lower=0)
    room_overflow = schedule_df.assign(Overflow=overflow).groupby("Room_ID")["Overflow"].sum()
    if room_overflow.empty or room_overflow.max() <= 0:
        most_overcrowded_room = "N/A"
    else:
        most_overcrowded_room = str(room_overflow.idxmax())

    room_avg = schedule_df.groupby("Room_ID")["Utilization_Pct"].mean()
    all_rooms = rooms_df["Room_ID"].astype(str)
    room_avg = room_avg.reindex(all_rooms, fill_value=0.0)
    most_underused_room = str(room_avg.idxmin()) if not room_avg.empty else "N/A"

    enrolled_by_time = schedule_df.groupby("Time")["Enrolled"].sum()
    peak_hour = str(enrolled_by_time.idxmax()) if not enrolled_by_time.empty else "N/A"
    quietest_hour = str(enrolled_by_time.idxmin()) if not enrolled_by_time.empty else "N/A"

    return {
        "overcrowded_sections": overcrowded_count,
        "underutilized_sections": underutilized_count,
        "double_booked_slots": double_booked,
        "most_overcrowded_room": most_overcrowded_room,
        "most_underused_room": most_underused_room,
        "peak_hour": peak_hour,
        "quietest_hour": quietest_hour,
    }


def get_floor_breakdown(rooms_df: pd.DataFrame, schedule_df: pd.DataFrame) -> list:
    rows = []
    for floor in sorted(rooms_df["Floor"].unique().tolist()):
        floor_rooms = rooms_df[rooms_df["Floor"] == floor]["Room_ID"].astype(str)
        floor_schedule = schedule_df[schedule_df["Room_ID"].isin(floor_rooms)]

        rows.append({
            "floor": int(floor),
            "rooms_count": int(len(floor_rooms)),
            "sections_count": int(len(floor_schedule)),
            "avg_occupancy_pct": round(float(floor_schedule["Utilization_Pct"].mean()), 1)
            if not floor_schedule.empty else 0.0,
            "overcrowded_count": int((floor_schedule["Status"] == "Overcrowded").sum()),
            "underutilized_count": int((floor_schedule["Status"] == "Underutilized").sum()),
        })
    return rows


def get_course_analysis(schedule_df: pd.DataFrame) -> list:
    rows = []
    for code, group in schedule_df.groupby("Course_Code"):
        status_counts = group["Status"].value_counts()
        rows.append({
            "course_code": code,
            "course_name": group["Course_Name"].iloc[0],
            "sections_count": int(len(group)),
            "total_enrolled": int(group["Enrolled"].sum()),
            "avg_occupancy_pct": round(float(group["Utilization_Pct"].mean()), 1),
            "normal_count": int(status_counts.get("Normal", 0)),
            "overcrowded_count": int(status_counts.get("Overcrowded", 0)),
            "underutilized_count": int(status_counts.get("Underutilized", 0)),
        })

    rows.sort(key=lambda x: x["course_code"])
    return rows


def get_instructor_load_summary(rooms_df: pd.DataFrame, schedule_df: pd.DataFrame) -> list:
    room_floor = rooms_df.set_index("Room_ID")["Floor"].to_dict()
    rows = []

    for instructor, group in schedule_df.groupby("Instructor"):
        rooms_used = sorted(group["Room_ID"].astype(str).unique().tolist())
        floors_used = sorted({int(room_floor.get(r, 0)) for r in rooms_used})
        common_time = group["Time"].mode().iloc[0] if not group["Time"].mode().empty else "N/A"

        rows.append({
            "instructor": instructor,
            "sections_count": int(len(group)),
            "total_students": int(group["Enrolled"].sum()),
            "rooms_used": ", ".join(rooms_used),
            "floors_used": ", ".join(str(f) for f in floors_used if f != 0),
            "most_common_time": common_time,
        })

    rows.sort(key=lambda x: x["instructor"])
    return rows


def get_analytics_data(rooms_df: pd.DataFrame, schedule_df: pd.DataFrame) -> dict:
    return {
        "overview": get_building_overview_kpis(rooms_df, schedule_df),
        "problems": get_problem_summary(rooms_df, schedule_df),
        "floor_breakdown": get_floor_breakdown(rooms_df, schedule_df),
        "course_analysis": get_course_analysis(schedule_df),
        "instructor_load": get_instructor_load_summary(rooms_df, schedule_df),
    }
