# 🏭 LogiX: Warehouse Optimization & Play

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**LogiX** is an interactive warehouse logistics simulator that compares **A\*, Dijkstra, and Q-Learning** algorithms for autonomous guided vehicle (AGV) routing and box delivery. It features a Streamlit dashboard for simulation and comparison, plus a standalone HTML5 browser game for manual play.

---

## ✨ Features

### 🚀 Algorithm Simulation
- **Single Robot Mode** — Watch one AGV navigate a warehouse, pick up boxes, and deliver them to rack slots using A\*, Dijkstra, or Q-Learning.
- **Multi-Robot Mode (5 AGVs)** — Five robots compete to deliver 20 boxes in a shared environment, with per-robot statistics and a winner podium.
- **Step-by-Step Animation** — Visual path tracing with real-time AGV fleet status updates (position, battery, delivered count).

### 📊 Algorithm Comparison Dashboard
Run all three algorithms side-by-side on the same warehouse map and compare:
- **Path Length** — Total steps taken
- **Delivery Success Rate** — Boxes delivered vs. total available
- **Computation Time** — How fast each algorithm finds a solution
- **Radar Chart** — Multi-dimensional performance profile (path efficiency, speed, delivery rate)
- **Per-Algorithm Deep Dive** — Pros, cons, and decision traces for each algorithm
- **Automatic Winner** — Recommendation based on delivery rate and path quality

### 🎮 Manual Game (HTML5)
A standalone browser-based game where you control 5 AGVs manually:
- **D-Pad Controls** — Use arrow keys or on-screen buttons
- **Multi-Robot Switching** — Switch between 5 differently colored robots
- **Scoring System** — +20 for pickup, +50 for delivery, -5 for wall collisions
- **Battery Management** — Each robot has its own battery that depletes with movement
- **Win Condition** — Deliver all 20 boxes to win!

### 🧠 AI Decision Log
Every algorithm explains its reasoning in plain language, making it educational for students and enthusiasts learning about pathfinding and reinforcement learning.

---

## 🧩 Algorithms

| Algorithm | Type | Heuristic | Optimality | Speed |
|-----------|------|-----------|------------|-------|
| **A\*** | Informed Search | Manhattan Distance | ✅ Guaranteed (with admissible heuristic) | ⚡ Fast |
| **Dijkstra** | Uniform Search | None | ✅ Guaranteed | 🐢 Slower on open terrain |
| **Q-Learning** | Reinforcement Learning | Learned via rewards | ❌ Approximate | 🕐 Requires training (2000 episodes) |

### A\* (A-Star)
- Uses `f(n) = g(n) + h(n)` where `h(n)` is Manhattan distance
- Priority queue guides search toward the goal efficiently
- Explores fewer nodes than Dijkstra in open terrain
- **Best for:** Static maps with clear goals (warehouse routing, GPS navigation, game AI)

### Dijkstra
- Explores uniformly outward in all 4 directions
- Evaluates all reachable nodes by shortest distance from start
- No heuristic needed — pure cost-based search
- **Best for:** Network routing, scenarios where all directions are equally important

### Q-Learning
- Reinforcement learning agent trained over 2000 episodes
- State encoding: `(x, y, carrying_flag, wall_up, wall_down, wall_left, wall_right)`
- Reward structure: +50 delivery, +20 pickup, -1 step penalty, -10 wall collision
- Epsilon-greedy exploration decays from 1.0 → 0.05 over training
- Experience replay with 100,000 capacity buffer
- Q-table persistence — saves/loads from CSV for continued learning
- **Best for:** Dynamic/unknown environments, adaptive systems, educational demonstrations

---

## 🗺️ Warehouse Map

The default warehouse is a **20×20 grid** with:
- `0` — Walkable floor
- `1` — Walls (obstacles)
- `2` — Rack slots (delivery destinations)
- `3` — Boxes (pickup locations)

The map is loaded from `data/warehouse.csv`. The start position is `(0, 0)` (top-left, cyan) and the goal is `(19, 19)` (bottom-right, magenta).

The HTML5 game generates a **50×50** procedural map with seeded randomness for reproducibility.

---

## 🏗️ Project Structure

```
warehouse_LogiX/
├── app.py                    # 🚀 Main Streamlit app (simulation + comparison + game)
├── comparison_dashboard.py   # 📊 Standalone comparison dashboard
├── warehouse.html            # 🎮 Standalone HTML5 manual game
├── env.py                    # 🌍 GridWorld environment & RobotState
├── algorithms.py             # 🧠 A*, Dijkstra, Q-Learning implementations
├── map_loader.py             # 📂 CSV map loader
├── renderer.py               # 🎨 Layered rendering system (background, objects, overlay)
├── data/
│   ├── warehouse.csv         # 🗺️ 20×20 warehouse map
│   └── q_table.csv           # 💾 Persisted Q-table (saved/loaded across sessions)
├── box.png                   # 📦 Box sprite
├── brick.jpeg                # 🧱 Wall texture
└── robot.png                 # 🤖 Robot sprite
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/deasy-mandasari/LogiX.git
cd LogiX

# Install dependencies
pip install streamlit numpy pandas matplotlib pillow
```

### Run the Streamlit App

```bash
streamlit run app.py
```

This launches the main dashboard with three tabs:
1. **🚀 Simulation** — Run single or multi-robot simulations
2. **📊 Algorithm Comparison Dashboard** — Compare all algorithms side-by-side
3. **🎮 Manual Game** — Play the warehouse game (Streamlit version)

### Run the Standalone Comparison Dashboard

```bash
streamlit run comparison_dashboard.py
```

### Play the HTML5 Game

Simply open `warehouse.html` in any modern web browser — no server required!

---

## 🎮 How to Play (Manual Game)

1. Click **START GAME**
2. Use **arrow keys** or the **D-Pad** to move the selected robot
3. Switch between 5 robots using the dropdown
4. **Pick up boxes** (📦) by moving onto them
5. **Deliver boxes** to orange rack slots (📍)
6. Monitor battery levels — robots recharge when idle for 2 seconds
7. Deliver all 20 boxes to win! 🏆

---

## 📊 Interpreting Results

In the comparison dashboard, metrics are displayed as:
- **Path Steps** — Lower is better (shorter path = more efficient)
- **Delivery %** — Higher is better (more boxes delivered)
- **Computation Time** — Lower is better (faster algorithm)
- **Radar Chart** — Shows the trade-off between path efficiency, speed, and delivery rate

Typically, **A\*** offers the best balance of speed and optimality for static warehouse maps, while **Q-Learning** demonstrates how reinforcement learning can adapt to unknown environments.

---

## 🧪 Extending the Project

- **Custom Maps** — Edit `data/warehouse.csv` or create new CSV files with your own layouts
- **New Algorithms** — Add new pathfinding algorithms to `algorithms.py` and register them in `app.py`
- **Larger Maps** — The system supports any N×N grid; adjust the CSV accordingly
- **Q-Learning Tuning** — Modify hyperparameters (`alpha`, `gamma`, `epsilon`) in `QAgent.__init__()`
- **Reward Shaping** — Adjust reward values in `train_q_learning_delivery()` to change agent behavior

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io)** — Interactive web dashboard
- **[NumPy](https://numpy.org)** — Numerical computing
- **[Pandas](https://pandas.dev)** — Data manipulation and analysis
- **[Matplotlib](https://matplotlib.org)** — Visualization and rendering
- **[Pillow (PIL)](https://python-pillow.org)** — Image processing and sprite generation
- **Vanilla JavaScript + Canvas** — HTML5 game rendering

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. Areas for contribution include:
- Additional pathfinding algorithms (e.g., D\*, RRT, Theta\*)
- Enhanced Q-Learning with deep neural networks (DQN)
- More complex warehouse layouts and obstacle types
- Performance optimizations for larger maps
- Multi-language support

---

## 👤 Author

**Deasy Mandasari** — [GitHub](https://github.com/deasy-mandasari)

---

*Built with ❤️ for warehouse logistics optimization and AI education.*
