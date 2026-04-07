# Digital Twin Classroom Scheduler

A modern, high-fidelity **Digital Twin** and smart scheduling system built for the **College of Computer and Information Sciences (CCIS)**. 

This project provides an interactive digital replica of the physical university environment, allowing administrators to monitor utilization, simulate room transfers, and instantly resolve double-bookings and overcrowding constraints.

## ✨ Features

- **Live Dashboard:** Monitor building occupancy, overcrowded rooms, and underutilized labs at a summary level.
- **Interactive Weekly Calendar:** A drag-and-drop master timeline to intuitively test space configurations and time swaps.
- **Smart AI Recommendations:** Detects constraints and provides 1-Click actionable "Apply Fix" buttons that physically resolve double-booking and overcrowding within the database.
- **Cross-Room Transfers:** An elegant unified modal for cleanly transferring overlapping classes to free CCIS rooms.
- **Scheduling Assistant:** Input a target instructor or student count, and instantly receive ranked, compatible slot/room combinations.
- **One-Click Exporting:** Instantly download the conflict-free, resolved official `schedule.csv`.

## 🚀 Setup & Execution

### Prerequisites
Make sure you have Python 3.9+ installed and pip ready.

### Installation
1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the internal Flask web application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to exactly: `http://127.0.0.1:5000`

## 🗄️ Data Control
The structure mimics a real-world enterprise database.
- `data/rooms.csv`: Contains the 30 CCIS floor allocations (Labs + Lecture Halls).
- `data/schedule.csv`: The live target schedule that handles all section enrollment tracking and class structures.
