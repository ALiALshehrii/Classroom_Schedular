# 🏫 Campus Digital Twin Scheduler

Web application that creates a **virtual replica of a university campus** to monitor, simulate, and optimize classroom occupancy and academic scheduling in real time.

> Built with Flask (Python)

---

## 📌 Overview

Traditional academic scheduling leads to overcrowded rooms, wasted space, and no way to test changes before applying them. This system solves that by applying **Digital Twin principles** — any change is tested in the virtual model before touching the real schedule.

### What the system does
- 📊 **Dashboard** — Live KPIs: total rooms, overcrowded sections, avg occupancy
- 🔍 **Conflict Detection** — Flags double-bookings, overcrowded & underutilized sections
- 🔀 **What-If Simulation** — Preview moving a course to a different room before committing
- 💡 **Smart Recommendations** — Auto-suggests the best available room for overcrowded sections

---

## 🗂️ Project Structure

```
project/
├── app.py                  # Flask routes & entry point
├── analyzer.py             # Core logic: data loading, KPIs, conflicts, simulation
├── requirements.txt
├── data/
│   ├── rooms.csv           # Room inventory
│   └── schedule.csv        # Course sections
├── static/
│   └── style.css           # All UI styling
└── templates/
    ├── base.html
    ├── index.html           # Dashboard
    ├── rooms.html           # Room list
    ├── schedule.html        # Full schedule
    ├── conflicts.html       # Conflict report
    ├── simulation.html      # What-If tool
    └── recommendations.html # Room suggestions
```

> ⚠️ The `data/` folder is required. `analyzer.py` looks for CSV files there specifically.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone git clone https://github.com/ALiALshehrii/Classroom_Schedular.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 🗺️ Routes

| Method | Route | Page |
|--------|-------|------|
| GET | `/` | Dashboard |
| GET | `/rooms` | Room list with filters |
| GET | `/schedule` | Full schedule with filters |
| GET | `/conflicts` | Conflict detection report |
| GET / POST | `/simulation` | What-If simulation tool |
| GET | `/recommendations` | Room swap suggestions |

---

## 📋 Data Format

### `data/rooms.csv`

| Column | Type | Description |
|--------|------|-------------|
| `Room_ID` | string | Unique room identifier (e.g. `1001`) |
| `Floor` | integer | Floor number (1, 2, or 3) |
| `Type` | string | `Lecture Hall` or `Computer Lab` |
| `Capacity` | integer | Maximum number of seats |

### `data/schedule.csv`

| Column | Description |
|--------|-------------|
| `CRN` | Unique section identifier |
| `Course_Code` | Course code (e.g. `CS101`) |
| `Course_Name` | Full course name |
| `Instructor` | Instructor name |
| `Day` | Day pattern (e.g. `Sun-Tue-Thu`) |
| `Time` | Start time (e.g. `08:00`) |
| `Room_ID` | Assigned room |
| `Room_Type` | `Lecture Hall` or `Computer Lab` |
| `Capacity` | Room capacity |
| `Enrolled` | Number of enrolled students |
| `Status` | Ignored on load — recalculated automatically |

> The `Status` column in the CSV is always **overwritten** by `analyzer.py` based on enrollment rules.

---

## 🧠 Business Logic

### Occupancy Status Rules

| Status | Condition |
|--------|-----------|
| 🔴 Overcrowded | `Enrolled > Capacity` |
| 🟡 Underutilized | `Enrolled < 40% of Capacity` |
| 🟢 Normal | Everything else |

### Recommendation Engine

For each overcrowded section, the system finds the best room by:
1. Matching the same room type (Lecture Hall ↔ Lecture Hall)
2. Capacity must be ≥ enrolled students
3. Room must be free at the same Day + Time
4. Among valid options, pick the **smallest fitting room** (minimize wasted seats)

### Conflict Detection
- **Double Booking** — Same room assigned to two sections at the same Day + Time
- **Overcrowded** — Enrolled exceeds room capacity
- **Underutilized** — Enrolled below 40% capacity (flagged for energy/space efficiency)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.x, Flask 3.0, pandas 2.1 |
| Frontend | Bootstrap 5.3, Chart.js 4.4, Jinja2 |
| Data | CSV files (rooms.csv, schedule.csv) |
| Styling | Custom CSS with CSS variables |

---

## 📦 Dependencies

```
Flask==3.0.0
pandas==2.1.4
Werkzeug==3.0.1
```

---

## 📝 Notes

- `replace_colors.py` and `update_styles.py` are one-time migration scripts — **do not run them again**
- The simulation page only **previews** a move, it does not save changes to the schedule
- This is a **prototype** — data is loaded from CSV on every request, not from a live database
