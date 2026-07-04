import sys
import io
import logging
import time
import pandas as pd

_stderr_sink = io.StringIO()
_orig_stderr = sys.stderr
sys.stderr = _stderr_sink

import streamlit as st

sys.stderr = _orig_stderr

logging.disable(logging.WARNING)
for _name in list(logging.root.manager.loggerDict):
    if _name.startswith("streamlit"):
        logging.getLogger(_name).disabled = True
logging.getLogger("streamlit").disabled = True

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from map_loader import load_csv_map
from env import GridWorld
from algorithms import (
    deliver_boxes_astar,
    deliver_boxes_dijkstra,
    train_q_learning_delivery,
    run_q_delivery_policy,
)
from renderer import draw_frame, TILE, resize, compose, build_background, build_objects, build_overlay

st.set_page_config(page_title="🏭 Smart Warehouse Hub", layout="wide")
st.title("LogiX: Optimization & Play")

# ── Global keyboard interceptor ──
# Prevents arrow keys from being used by Streamlit's tab component for tab switching.
# Arrow keys are instead routed to the game's direction buttons when the game tab is active.
import streamlit.components.v1 as components
components.html(
    """
    <script>
    (function() {
        try {
            var target = window.parent || window;
            // Use capture phase to intercept before Streamlit's tab handler
            target.addEventListener('keydown', function(e) {
                var key = e.key;
                var isArrow = (key === 'ArrowUp' || key === 'ArrowDown' || 
                               key === 'ArrowLeft' || key === 'ArrowRight');
                if (isArrow) {
                    e.preventDefault();
                    e.stopPropagation();
                    e.stopImmediatePropagation();
                    // Try to click the corresponding game direction button
                    var btnMap = {
                        'ArrowUp': '⬆',
                        'ArrowDown': '⬇',
                        'ArrowLeft': '⬅',
                        'ArrowRight': '➡'
                    };
                    var btnText = btnMap[key];
                    if (btnText) {
                        var buttons = target.document.querySelectorAll('button');
                        for (var i = 0; i < buttons.length; i++) {
                            if (buttons[i].textContent && buttons[i].textContent.indexOf(btnText) !== -1) {
                                buttons[i].click();
                                break;
                            }
                        }
                    }
                    return false;
                }
            }, {capture: true});
        } catch(err) {
            console.log('KB init error:', err);
        }
    })();
    </script>
    """,
    height=0,
    width=0,
)

# ── LOAD MAP (global, shared between tabs) ────────────────────────────────────
grid = load_csv_map("data/warehouse.csv")

def fresh_env():
    return GridWorld(copy.deepcopy(grid))

# ── AGV FLEET STATUS (shared across all tabs) ────────────────────────────────
agv_expander = st.sidebar.expander("🤖 AGV Fleet Status", expanded=True)
with agv_expander:
    agv_status_placeholder = st.empty()

def render_agv_status(placeholder, robot_id, pos, status, battery, delivered, total):
    """Update the AGV status display in the sidebar."""
    placeholder.empty()
    with placeholder:
        st.write(f"**{robot_id}** → {status} | 📍 {pos}")
        st.progress(battery / 100)
        st.write(f"🔋 {battery:.0f}% — 📦 Delivered: {delivered}/{total}")

# Initial AGV state (before any tab interaction)
initial_boxes = int(np.count_nonzero(grid == 3))
render_agv_status(agv_status_placeholder, "AGV-1", (0, 0), "⏸️ IDLE", 100, 0, initial_boxes)

# ── TABS — Simulation | Comparison Dashboard | Manual Game ────────────────────
tab_sim, tab_compare, tab_game = st.tabs(["🚀 Simulation", "📊 Algorithm Comparison Dashboard", "🎮 Manual Game"])

with tab_sim:

    mode = st.sidebar.selectbox(
        "Algorithm",
        ["A*", "Dijkstra", "Q-Learning"]
    )

    def _decision_logs_for_mode(mode):
        """Return algorithm-specific decision trace logs."""
        if mode == "A*":
            return [
                "A* pathfinding: heuristic (Manhattan distance) guides search toward goal",
                "Priority queue ordered by f(n) = g(n) + h(n) — balances cost & estimate",
                "Admissible heuristic guarantees optimal path under uniform edge weights",
                "Target selection: nearest box/slot by Manhattan distance",
                "Open set expanded from start — pruning suboptimal branches early",
            ]
        elif mode == "Dijkstra":
            return [
                "Dijkstra pathfinding: no heuristic — explores uniformly outward",
                "All reachable nodes evaluated by shortest distance from start (g-cost only)",
                "Guarantees shortest path but explores more nodes than A* in open terrain",
                "Exploration radius grows uniformly in all 4 directions",
                "Target selection: nearest box/slot by shortest path cost",
            ]
        else:  # Q-Learning
            return [
                "Q-Learning: training agent with reward-based learning (2000 episodes)",
                "State encoded as (x, y, carrying_flag) — actions: {↑, ↓, ←, →}",
                "Shaped rewards: +50 delivery, +20 pickup, -1 step penalty, -5 wall collision",
                "Epsilon-greedy exploration decays from 1.0 → 0.05 over training",
                "Policy converges via Bellman updates — 3 best-of-N attempts at inference",
            ]

    # ── AI DECISION TRACE ────────────────────────────────────────────────
    with st.sidebar.expander("🧠 AI Decision Log", expanded=True):
        decision_logs = _decision_logs_for_mode(mode)
        for log in decision_logs:
            st.write("•", log)


    def animate_path(env, full_path, events, speed_ms, agv_placeholder=None):
        """
        Animate the robot moving step-by-step along full_path.
        Events are triggered when the robot reaches the event position.
        """
        # Create a fresh environment to replay
        anim_env = fresh_env()

        # Build a set of event positions for quick lookup
        event_positions = set(events.keys())

        # Track which events have been applied
        applied_events = {}
        path_so_far = []

        # Status text
        status_text = st.empty()
        progress_bar = st.progress(0)
        total_steps = len(full_path)

        # Create a placeholder for the animation frames
        anim_placeholder = st.empty()

        for step_idx, pos in enumerate(full_path):
            # Check if this position triggers an event
            if pos in event_positions and pos not in applied_events:
                kind = events[pos]
                applied_events[pos] = kind
                if kind == 'pickup':
                    anim_env.collect_box(pos)
                    status_text.info(f"📦 Step {step_idx+1}/{total_steps}: Pick up box at {pos}")
                else:
                    anim_env.fill_slot(pos)
                    status_text.success(f"✅ Step {step_idx+1}/{total_steps}: Deliver box to slot at {pos}")
            else:
                status_text.text(f"🚶 Step {step_idx+1}/{total_steps}: Move to {pos}")

            path_so_far.append(pos)

            # Draw current frame
            with anim_placeholder.container():
                draw_frame(
                    anim_env,
                    robot_pos=pos,
                    path_so_far=path_so_far,
                    events_so_far=applied_events,
                    title=f"Step {step_idx+1} / {total_steps}  |  "
                          f"Delivered: {len(anim_env.filled_slots)}/{len(anim_env.boxes)}"
                )

            # Update progress
            progress_bar.progress((step_idx + 1) / total_steps)

            # Update AGV fleet status dynamically
            if agv_placeholder is not None:
                battery = max(5, 100 - step_idx * 0.5)
                if anim_env.is_done():
                    cur_status = "✅ COMPLETED"
                elif pos in applied_events:
                    cur_status = "🎯 PICKING" if applied_events[pos] == 'pickup' else "📦 DELIVERING"
                else:
                    cur_status = "🚶 MOVING"
                render_agv_status(agv_placeholder, "AGV-1", pos, cur_status, battery,
                                  len(anim_env.filled_slots), len(anim_env.boxes))

            # Sleep for animation
            time.sleep(speed_ms / 1000.0)

        progress_bar.empty()
        status_text.success(f"🎉 Simulation complete! {len(anim_env.filled_slots)}/{len(anim_env.boxes)} boxes delivered.")
        return anim_env


    # Placeholder for the simulation output
    output_placeholder = st.empty()

    if st.button("RUN"):
        with st.spinner("Computing path..."):
            simulation_env = fresh_env()

            if mode == "A*":
                path, events = deliver_boxes_astar(simulation_env)
            elif mode == "Dijkstra":
                path, events = deliver_boxes_dijkstra(simulation_env)
            else:
                best_path, best_events = [], {}
                best_delivered = -1
                progress_bar = st.progress(0)
                for attempt in range(3):
                    trial_env = fresh_env()
                    agent, _ = train_q_learning_delivery(trial_env, episodes=2000)
                    trial_env2 = fresh_env()
                    p, e = run_q_delivery_policy(trial_env2, agent, max_steps=400)
                    delivered = len(trial_env2.filled_slots)
                    if delivered > best_delivered:
                        best_delivered = delivered
                        best_path, best_events = p, e
                    if delivered == len(trial_env2.boxes):
                        break
                    progress_bar.progress((attempt + 1) / 3)
                path, events = best_path, best_events
                progress_bar.empty()

            # Summary stats before animation
            num_boxes = len(simulation_env.boxes)
            num_delivered = len(simulation_env.filled_slots)

            st.info(
                f"📦 **Path computed**: {len(path)} steps, "
                f"{num_delivered} / {num_boxes} boxes will be delivered. "
                f"Starting animation..."
            )

        # Animate step by step
        final_env = animate_path(simulation_env, path, events, 80, agv_status_placeholder)

        # Final summary
        num_boxes = len(final_env.boxes)
        num_delivered = len(final_env.filled_slots)
        st.success(
            f"📦 **SUMMARY**: {num_delivered} / {num_boxes} boxes delivered  —  "
            f"Robot path has {len(path)} steps"
        )

        # Final static view
        draw_frame(final_env, robot_pos=path[-1] if path else final_env.start,
                   path_so_far=path, events_so_far=events,
                   title="🏁 Finish")
    else:
        # Show initial empty map
        draw_frame(fresh_env())

with tab_compare:
    st.markdown(
        r"Run **A\***, **Dijkstra**, and **Q-Learning** on the same warehouse map "
        "and compare their performance, path quality, and delivery success."
    )

    # ── RUN ALL BUTTON ──
    col1, col2, col3 = st.columns([2, 2, 1])
    with col2:
        run_btn = st.button("🚀 Run All Algorithms", type="primary", use_container_width=True)

    # ── MAIN DASHBOARD ──
    if run_btn:
        with st.spinner("Running all 3 algorithms... This may take a moment for Q-Learning."):
            base_grid = copy.deepcopy(grid)
            results = {}

            # ── Runner helpers ──
            def _run_astar(env, label):
                t0 = time.perf_counter()
                path, events = deliver_boxes_astar(env)
                t1 = time.perf_counter()
                return {
                    "Algorithm": label,
                    "Path Length (steps)": len(path),
                    "Delivered": f"{len(env.filled_slots)} / {len(env.boxes)}",
                    "Delivered %": (len(env.filled_slots) / len(env.boxes) * 100) if env.boxes else 0,
                    "Time (s)": round(t1 - t0, 3),
                    "path": path,
                    "events": events,
                    "env": env,
                }

            def _run_dijkstra(env, label):
                t0 = time.perf_counter()
                path, events = deliver_boxes_dijkstra(env)
                t1 = time.perf_counter()
                return {
                    "Algorithm": label,
                    "Path Length (steps)": len(path),
                    "Delivered": f"{len(env.filled_slots)} / {len(env.boxes)}",
                    "Delivered %": (len(env.filled_slots) / len(env.boxes) * 100) if env.boxes else 0,
                    "Time (s)": round(t1 - t0, 3),
                    "path": path,
                    "events": events,
                    "env": env,
                }

            def _run_qlearning(env, label):
                t0 = time.perf_counter()
                best_delivered = -1
                best_path, best_events = [], {}
                for attempt in range(3):
                    trial_env = fresh_env()
                    agent, _ = train_q_learning_delivery(trial_env, episodes=2000)
                    trial_env2 = fresh_env()
                    p, e = run_q_delivery_policy(trial_env2, agent, max_steps=400)
                    delivered = len(trial_env2.filled_slots)
                    if delivered > best_delivered:
                        best_delivered = delivered
                        best_path, best_events = p, e
                    if delivered == len(trial_env2.boxes):
                        break
                t1 = time.perf_counter()
                return {
                    "Algorithm": label,
                    "Path Length (steps)": len(best_path),
                    "Delivered": f"{best_delivered} / {len(env.boxes)}",
                    "Delivered %": (best_delivered / len(env.boxes) * 100) if env.boxes else 0,
                    "Time (s)": round(t1 - t0, 3),
                    "path": best_path,
                    "events": best_events,
                    "env": env,
                }

            # ── Execute ──
            results["A*"] = _run_astar(fresh_env(), "A*")
            results["Dijkstra"] = _run_dijkstra(fresh_env(), "Dijkstra")
            results["Q-Learning"] = _run_qlearning(fresh_env(), "Q-Learning")

        # ═══════════════════════════════════════════════════════════════
        #  SECTION 1: OVERVIEW METRICS
        # ═══════════════════════════════════════════════════════════════
        st.success("✅ All 3 algorithms completed! Explore the comparison below.")

        st.header("📋 Performance Metrics")

        # Build DataFrame
        df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("path", "events", "env")}
            for r in results.values()
        ])
        df = df[["Algorithm", "Path Length (steps)", "Delivered", "Delivered %", "Time (s)"]]
        df_display = df.copy()
        df_display["Delivered %"] = df_display["Delivered %"].map("{:.1f}%".format)

        # Color-coded metric cards
        colors = {"A*": "#FF6B35", "Dijkstra": "#004E89", "Q-Learning": "#1A936F"}

        metric_cols = st.columns(3)
        for idx, (algo_name, r) in enumerate(results.items()):
            with metric_cols[idx]:
                c = colors[algo_name]
                st.markdown(
                    f"""
                    <div style="
                        background: {c}15;
                        border: 2px solid {c};
                        border-radius: 12px;
                        padding: 16px;
                        text-align: center;
                        margin-bottom: 8px;
                    ">
                        <h3 style="color: {c}; margin: 0 0 8px 0;">{algo_name}</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                            <div>
                                <div style="font-size: 24px; font-weight: bold;">{r['Path Length (steps)']}</div>
                                <div style="font-size: 12px; color: #666;">Path Steps</div>
                            </div>
                            <div>
                                <div style="font-size: 24px; font-weight: bold;">{r['Delivered']}</div>
                                <div style="font-size: 12px; color: #666;">Delivered</div>
                            </div>
                            <div>
                                <div style="font-size: 24px; font-weight: bold;">{r['Delivered %']:.1f}%</div>
                                <div style="font-size: 12px; color: #666;">Success Rate</div>
                            </div>
                            <div>
                                <div style="font-size: 24px; font-weight: bold;">{r['Time (s)']:.3f}s</div>
                                <div style="font-size: 12px; color: #666;">Compute Time</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Data table
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # ═══════════════════════════════════════════════════════════════
        #  SECTION 2: SIDE-BY-SIDE PATH VISUALIZATION
        # ═══════════════════════════════════════════════════════════════
        st.header("🗺️  Final Path Comparison")
        st.markdown("Each column shows the final path taken by the robot after completing all deliveries.")

        h = base_grid.shape[0]
        w = base_grid.shape[1]

        fig, axes = plt.subplots(1, 3, figsize=(7 * 3, 7))

        for idx, (algo_name, r) in enumerate(results.items()):
            ax = axes[idx]

            # Build environment snapshot for rendering
            if algo_name == "Q-Learning":
                replay_env = fresh_env()
                for pos, kind in r["events"].items():
                    if kind == "pickup":
                        replay_env.collect_box(pos)
                    else:
                        replay_env.fill_slot(pos)
                env_snapshot = replay_env
            else:
                env_snapshot = r["env"]

            bg = build_background(env_snapshot)
            final_pos = r["path"][-1] if r["path"] else env_snapshot.start
            obj = build_objects(env_snapshot, final_pos)
            overlay = build_overlay(env_snapshot, r["path"], r["events"])
            frame = compose(bg, obj, overlay)

            ax.imshow(frame, interpolation="nearest")
            ax.set_title(
                f"{algo_name}\n{r['Path Length (steps)']} steps · {r['Delivered']}",
                fontsize=11, fontweight="bold", color=colors[algo_name],
            )
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════
        #  SECTION 3: BAR CHARTS
        # ═══════════════════════════════════════════════════════════════
        st.header("📈  Metrics Comparison Chart")

        names = [r["Algorithm"] for r in results.values()]
        path_lens = [r["Path Length (steps)"] for r in results.values()]
        times = [r["Time (s)"] for r in results.values()]
        deliveries = [r["Delivered %"] for r in results.values()]
        bar_colors = [colors[n] for n in names]

        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))

        # ── Path Length ──
        ax1 = axes2[0]
        bars1 = ax1.bar(names, path_lens, color=bar_colors, edgecolor="white", linewidth=1.2, width=0.6)
        ax1.set_ylabel("Path Steps", fontsize=12)
        ax1.set_title("Total Path Length", fontsize=13, fontweight="bold")
        for bar, val in zip(bars1, path_lens):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(path_lens) * 0.02,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax1.set_ylim(0, max(path_lens) * 1.15)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # ── Computation Time ──
        ax2 = axes2[1]
        bars2 = ax2.bar(names, times, color=bar_colors, edgecolor="white", linewidth=1.2, width=0.6)
        ax2.set_ylabel("Time (seconds)", fontsize=12)
        ax2.set_title("Computation Time", fontsize=13, fontweight="bold")
        for bar, val in zip(bars2, times):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                f"{val:.3f}s", ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax2.set_ylim(0, max(times) * 1.15)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # ── Delivery Success Rate ──
        ax3 = axes2[2]
        bars3 = ax3.bar(names, deliveries, color=bar_colors, edgecolor="white", linewidth=1.2, width=0.6)
        ax3.set_ylabel("Delivery Success (%)", fontsize=12)
        ax3.set_title("Delivery Success Rate", fontsize=13, fontweight="bold")
        for bar, val in zip(bars3, deliveries):
            ax3.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(deliveries) * 0.02,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax3.set_ylim(0, max(deliveries) * 1.15)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ═══════════════════════════════════════════════════════════════
        #  SECTION 4: RADAR / SPIDER CHART
        # ═══════════════════════════════════════════════════════════════
        st.header("🕸️  Multi-Dimensional Comparison (Radar Chart)")

        # Normalize metrics for radar chart (higher is better)
        # Path length: invert (shorter = better)
        max_path = max(path_lens)
        min_path = min(path_lens)
        path_scores = [
            100 * (1 - (p - min_path) / (max_path - min_path)) if max_path != min_path else 100
            for p in path_lens
        ]

        # Time: invert (faster = better)
        max_time = max(times)
        min_time = min(times)
        time_scores = [
            100 * (1 - (t - min_time) / (max_time - min_time)) if max_time != min_time else 100
            for t in times
        ]

        # Delivery %: as-is (higher = better)
        delivery_scores = deliveries

        categories = ["Path Efficiency", "Speed", "Delivery Rate"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # close the loop

        fig3, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for idx, algo_name in enumerate(names):
            values = [path_scores[idx], time_scores[idx], delivery_scores[idx]]
            values += values[:1]
            ax_radar.plot(angles, values, "o-", linewidth=2, label=algo_name, color=colors[algo_name])
            ax_radar.fill(angles, values, alpha=0.1, color=colors[algo_name])

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=12, fontweight="bold")
        ax_radar.set_ylim(0, 105)
        ax_radar.set_yticks([20, 40, 60, 80, 100])
        ax_radar.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9, color="gray")
        ax_radar.set_title("Algorithm Performance Profile", fontsize=14, fontweight="bold", pad=20)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # ═══════════════════════════════════════════════════════════════
        #  SECTION 5: PER-ALGORITHM DEEP DIVE
        # ═══════════════════════════════════════════════════════════════
        st.header("🔍  Per-Algorithm Deep Dive")

        decision_info = {
            "A*": {
                "icon": "⭐",
                "pros": [
                    "Optimal path guaranteed with admissible heuristic",
                    "Manhattan distance guides search efficiently toward goal",
                    "Fast computation — explores fewer nodes than Dijkstra",
                    "Priority queue orders by f(n) = g(n) + h(n)",
                ],
                "cons": [
                    "Heuristic must be admissible for optimality guarantee",
                    "Performance depends on quality of heuristic function",
                ],
            },
            "Dijkstra": {
                "icon": "🌐",
                "pros": [
                    "Optimal path guaranteed (no heuristic needed)",
                    "Explores uniformly — works well in any terrain",
                    "Simple and predictable behavior",
                ],
                "cons": [
                    "Explores more nodes than A* in open terrain",
                    "Slower on large maps with few obstacles",
                    "No directional guidance — expands in all directions equally",
                ],
            },
            "Q-Learning": {
                "icon": "🧠",
                "pros": [
                    "Learns from experience — no map knowledge required",
                    "Adapts to dynamic environments (can retrain)",
                    "Discovers strategies beyond shortest-path",
                ],
                "cons": [
                    "Requires many training episodes to converge",
                    "May not find optimal path (policy is approximate)",
                    "Significantly slower due to training overhead",
                    "Stochastic — results vary between runs",
                ],
            },
        }

        tabs = st.tabs(["⭐ A*", "🌐 Dijkstra", "🧠 Q-Learning"])

        for tab_i, (algo_name, info) in enumerate(decision_info.items()):
            with tabs[tab_i]:
                r = results[algo_name]

                col_a, col_b = st.columns([1, 1.5])

                with col_a:
                    st.markdown(
                        f"""
                        <div style="
                            background: {colors[algo_name]}10;
                            border-radius: 10px;
                            padding: 16px;
                            border-left: 4px solid {colors[algo_name]};
                        ">
                            <h4 style="margin: 0 0 12px 0; color: {colors[algo_name]};">Key Metrics</h4>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.metric("Path Steps", r["Path Length (steps)"])
                    st.metric("Boxes Delivered", r["Delivered"])
                    st.metric("Computation Time", f"{r['Time (s)']:.3f}s")
                    st.metric("Delivery Success", f"{r['Delivered %']:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_b:
                    st.markdown(f"#### ✅ Pros")
                    for pro in info["pros"]:
                        st.write("•", pro)

                    st.markdown(f"#### ⚠️ Cons")
                    for con in info["cons"]:
                        st.write("•", con)

                    # Show algorithm-specific decision trace
                    st.markdown("#### 📋 Decision Trace")
                    if algo_name == "A*":
                        traces = [
                            "A* pathfinding: heuristic (Manhattan distance) guides search toward goal",
                            "Priority queue ordered by f(n) = g(n) + h(n) — balances cost & estimate",
                            "Admissible heuristic guarantees optimal path under uniform edge weights",
                            "Target selection: nearest box/slot by Manhattan distance",
                            "Open set expanded from start — pruning suboptimal branches early",
                        ]
                    elif algo_name == "Dijkstra":
                        traces = [
                            "Dijkstra pathfinding: no heuristic — explores uniformly outward",
                            "All reachable nodes evaluated by shortest distance from start (g-cost only)",
                            "Guarantees shortest path but explores more nodes than A* in open terrain",
                            "Exploration radius grows uniformly in all 4 directions",
                            "Target selection: nearest box/slot by shortest path cost",
                        ]
                    else:
                        traces = [
                            "Q-Learning: training agent with reward-based learning (2000 episodes)",
                            "State encoded as (x, y, carrying_flag) — actions: {↑, ↓, ←, →}",
                            "Shaped rewards: +50 delivery, +20 pickup, -1 step penalty, -5 wall collision",
                            "Epsilon-greedy exploration decays from 1.0 → 0.05 over training",
                            "Policy converges via Bellman updates — 3 best-of-N attempts at inference",
                        ]
                    for t in traces:
                        st.write("•", t)

        # ═══════════════════════════════════════════════════════════════
        #  SECTION 6: WINNER / RECOMMENDATION
        # ═══════════════════════════════════════════════════════════════
        st.header("🏆  Summary & Recommendation")

        # Determine "winner" by delivery % then path length
        sorted_results = sorted(
            results.values(),
            key=lambda r: (r["Delivered %"], -r["Path Length (steps)"]),
            reverse=True,
        )
        winner = sorted_results[0]

        col_rec1, col_rec2 = st.columns([1, 2])

        with col_rec1:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {colors[winner['Algorithm']]}22, {colors[winner['Algorithm']]}44);
                    border: 3px solid {colors[winner['Algorithm']]};
                    border-radius: 16px;
                    padding: 24px;
                    text-align: center;
                ">
                    <div style="font-size: 48px;">🏆</div>
                    <h2 style="color: {colors[winner['Algorithm']]}; margin: 8px 0;">{winner['Algorithm']}</h2>
                    <div style="font-size: 14px; color: #666;">Best Overall Performer</div>
                    <div style="margin-top: 12px;">
                        <div style="font-size: 20px; font-weight: bold;">{winner['Path Length (steps)']} steps</div>
                        <div style="font-size: 14px; color: #666;">{winner['Delivered']} delivered in {winner['Time (s)']:.3f}s</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_rec2:
            st.markdown("#### When to use each algorithm:")
            recs = [
                (r"**A\*** — Best for static maps with clear goals",
                 "Use when you need the **shortest path fast**. A* is the go-to for "
                 "warehouse routing, GPS navigation, and game AI where the environment is known."),
                ("**Dijkstra** — Best for uniform exploration",
                 "Use when you need **guaranteed shortest paths** without designing a heuristic. "
                 "Good for network routing and scenarios where all directions are equally important."),
                ("**Q-Learning** — Best for dynamic / unknown environments",
                 "Use when the environment **changes over time** or you want the robot to "
                 "**learn from experience**. Ideal for adaptive systems where optimality is "
                 "less critical than adaptability."),
            ]
            for title, desc in recs:
                st.markdown(f"- {title}: {desc}")

        # ═══════════════════════════════════════════════════════════════
        #  SECTION 7: RAW DATA EXPORT
        # ═══════════════════════════════════════════════════════════════
        with st.expander("📥 Export Raw Data"):
            st.markdown("Download the comparison results as CSV for further analysis.")
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="algorithm_comparison_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("#### Raw Metrics Table")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

    else:
        # ── IDLE STATE ──
        st.info("👆 Click **'Run All Algorithms'** to start the comparison.")

        # Show a preview of the map
        st.markdown("### 🗺️  Warehouse Map Preview")
        preview_env = fresh_env()
        draw_frame(preview_env, title="Warehouse Layout (10×10)")

        st.markdown("---")
        st.markdown(
            """
            ### What you'll see:
            1. **Performance Metrics** — Side-by-side comparison table with path length, delivery rate, and computation time
            2. **Path Visualization** — See the final path each algorithm took on the warehouse map
            3. **Bar Charts** — Visual comparison of key metrics
            4. **Radar Chart** — Multi-dimensional performance profile
            5. **Deep Dive** — Per-algorithm analysis with pros, cons, and decision traces
            6. **Winner** — Automatic recommendation based on results
            """
        )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: MANUAL GAME — Keyboard-controlled robot
# ══════════════════════════════════════════════════════════════════════════════
with tab_game:

    # ── Initialise game state in session_state ──
    if "game_env" not in st.session_state:
        st.session_state.game_env = fresh_env()
    if "game_pos" not in st.session_state:
        st.session_state.game_pos = (0, 0)
    if "game_path" not in st.session_state:
        st.session_state.game_path = [(0, 0)]
    if "game_events" not in st.session_state:
        st.session_state.game_events = {}
    if "game_steps" not in st.session_state:
        st.session_state.game_steps = 0
    if "game_battery" not in st.session_state:
        st.session_state.game_battery = 100.0
    if "game_msg" not in st.session_state:
        st.session_state.game_msg = "🎮 Use arrow keys to move the robot!"
    if "game_over" not in st.session_state:
        st.session_state.game_over = False
    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "game_balloons_shown" not in st.session_state:
        st.session_state.game_balloons_shown = False
    if "game_started" not in st.session_state:
        st.session_state.game_started = False

    def update_agv_status_from_game():
        """Update the shared AGV fleet status sidebar from the game state."""
        env = st.session_state.game_env
        pos = st.session_state.game_pos
        battery = st.session_state.game_battery
        delivered = len(env.filled_slots)
        total = len(env.boxes)

        if st.session_state.game_over:
            status = "✅ COMPLETED"
        elif env.carrying_box:
            status = "📦 DELIVERING"
        else:
            status = "🚶 MOVING"

        render_agv_status(agv_status_placeholder, "AGV-1", pos, status, battery, delivered, total)

    def reset_game():
        st.session_state.game_env = fresh_env()
        st.session_state.game_pos = (0, 0)
        st.session_state.game_path = [(0, 0)]
        st.session_state.game_events = {}
        st.session_state.game_steps = 0
        st.session_state.game_battery = 100.0
        st.session_state.game_msg = "🔄 Game reset! Use arrow keys to move."
        st.session_state.game_over = False
        st.session_state.game_score = 0
        st.session_state.game_balloons_shown = False
        st.session_state.game_started = False
        update_agv_status_from_game()

    def move_robot(dx, dy):
        """Attempt to move the robot by (dx, dy). Returns True if moved."""
        env = st.session_state.game_env
        x, y = st.session_state.game_pos
        nx, ny = x + dx, y + dy
        new_pos = (nx, ny)

        # Check bounds and walls
        if not env.is_valid(new_pos):
            st.session_state.game_score = max(0, st.session_state.game_score - 5)
            st.session_state.game_msg = f"🚫 Blocked! Wall or obstacle. (-5 pts, Score: {st.session_state.game_score})"
            # Still update AGV status to reflect blocked state
            update_agv_status_from_game()
            return False

        # Move the robot — consume battery
        st.session_state.game_pos = new_pos
        st.session_state.game_path.append(new_pos)
        st.session_state.game_steps += 1
        st.session_state.game_battery = max(0.0, st.session_state.game_battery - 0.5)

        # Check for box pickup
        if new_pos in env.get_available_boxes() and not env.carrying_box:
            env.collect_box(new_pos)
            st.session_state.game_events[new_pos] = "pickup"
            st.session_state.game_score += 20
            st.session_state.game_msg = f"📦 Picked up box at {new_pos}! (+20 pts, Score: {st.session_state.game_score})"
        # Check for delivery (carrying box + standing on empty rack slot)
        elif new_pos in env.get_empty_slots() and env.carrying_box:
            env.fill_slot(new_pos)
            st.session_state.game_events[new_pos] = "delivery"
            st.session_state.game_score += 50
            st.session_state.game_msg = f"✅ Delivered box to slot at {new_pos}! (+50 pts, Score: {st.session_state.game_score})"
            if env.is_done():
                st.session_state.game_msg = f"🎉 YOU WIN! All {len(env.boxes)} boxes delivered in {st.session_state.game_steps} steps! Final Score: {st.session_state.game_score}"
                st.session_state.game_over = True
                if not st.session_state.game_balloons_shown:
                    st.balloons()
                    st.session_state.game_balloons_shown = True
        else:
            st.session_state.game_msg = f"🚶 Moved to {new_pos}"

        # Update AGV fleet status after move
        update_agv_status_from_game()
        return True

    # Update AGV fleet status to reflect current game state
    update_agv_status_from_game()

    # ── Start button (shown before game begins) ──
    if not st.session_state.game_started:
        st.caption("🎮 Press **Start** to begin the game.")

        col_game_map, col_game_controls = st.columns([3, 1])

        with col_game_map:
            # Draw initial map without active game
            env = st.session_state.game_env
            draw_frame(
                env,
                robot_pos=st.session_state.game_pos,
                path_so_far=st.session_state.game_path,
                events_so_far=st.session_state.game_events,
                title="🎮 Manual Game — Ready"
            )

        with col_game_controls:
            st.markdown("### 🎮 Game")
            st.markdown("Navigate the robot through the warehouse to pick up 📦 boxes and deliver them to empty rack slots 🗄️.")
            st.markdown("---")
            st.markdown("#### 🏁 Get Ready")
            st.markdown(
                """
                - **Arrow Keys** to move the robot
                - Pick up boxes by moving onto them
                - Deliver boxes to empty rack slots
                - Deliver all boxes to win!
                """
            )
            st.markdown("---")
            if st.button("▶️ **START GAME**", use_container_width=True, type="primary"):
                st.session_state.game_started = True
                st.session_state.game_msg = "🎮 Game started! Use arrow keys to move the robot."
                st.rerun()

        # Update AGV status
        update_agv_status_from_game()
    else:
        st.caption("🎮 **Arrow Keys** to move the robot.")

        # ── Layout: two columns ──
        col_game_map, col_game_controls = st.columns([3, 1])

        with col_game_map:
            # Draw the current game state
            env = st.session_state.game_env
            pos = st.session_state.game_pos
            path = st.session_state.game_path
            events = st.session_state.game_events

            draw_frame(
                env,
                robot_pos=pos,
                path_so_far=path,
                events_so_far=events,
                title=f"🎮 Manual Game — Step {st.session_state.game_steps}"
            )

        with col_game_controls:
            st.markdown("### 🎮 Controls")
            st.markdown("Click a direction button to move the robot.")

            # ── Direction pad buttons ──
            st.markdown("#### Direction Pad")
            # Row 1: empty, up, empty
            r1 = st.columns(3)
            with r1[0]:
                st.write("")
            with r1[1]:
                up_btn = st.button("⬆️", key="game_up", disabled=st.session_state.game_over, use_container_width=True)
            with r1[2]:
                st.write("")
            # Row 2: left, down, right
            r2 = st.columns(3)
            with r2[0]:
                left_btn = st.button("⬅️", key="game_left", disabled=st.session_state.game_over, use_container_width=True)
            with r2[1]:
                down_btn = st.button("⬇️", key="game_down", disabled=st.session_state.game_over, use_container_width=True)
            with r2[2]:
                right_btn = st.button("➡️", key="game_right", disabled=st.session_state.game_over, use_container_width=True)

            if up_btn:
                move_robot(-1, 0)
            if down_btn:
                move_robot(1, 0)
            if left_btn:
                move_robot(0, -1)
            if right_btn:
                move_robot(0, 1)
            if up_btn or down_btn or left_btn or right_btn:
                if not st.session_state.game_over:
                    st.rerun()

            # ── Game status ──
            st.markdown("---")
            st.markdown("### 📊 Status")
            st.info(st.session_state.game_msg)

            boxes_total = len(env.boxes)
            boxes_delivered = len(env.filled_slots)
            st.metric("📦 Boxes Delivered", f"{boxes_delivered} / {boxes_total}")
            st.metric("👣 Steps Taken", st.session_state.game_steps)
            st.metric("🎒 Carrying Box", "✅ Yes" if env.carrying_box else "❌ No")
            st.metric("🏆 Score", st.session_state.game_score)
            # ── Reset button ──
            st.markdown("---")
            if st.button("🔄 Reset Game", use_container_width=True, type="primary"):
                reset_game()
                st.rerun()

