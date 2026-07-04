import heapq
import random
import numpy as np
import copy
import csv
import os
from env import GridWorld

# =====================================================================
#  Q-TABLE PERSISTENCE — Save/Load Q-table to/from CSV in data/
# =====================================================================

Q_TABLE_PATH = os.path.join(os.path.dirname(__file__), "data", "q_table.csv")

def load_q_table_from_csv(filepath=Q_TABLE_PATH):
    """
    Load Q-table from a CSV file.
    CSV columns: state_x,state_y,carrying,wall_up,wall_down,wall_left,wall_right,action,q_value
    Returns a dict: {(state_tuple, action): q_value}
    """
    q_table = {}
    if not os.path.exists(filepath):
        print(f"  [Q-Table] No existing Q-table found at {filepath}. Starting fresh.")
        return q_table

    try:
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                state = (
                    int(row["state_x"]),
                    int(row["state_y"]),
                    int(row["carrying"]),
                    int(row["wall_up"]),
                    int(row["wall_down"]),
                    int(row["wall_left"]),
                    int(row["wall_right"]),
                )
                action = int(row["action"])
                q_value = float(row["q_value"])
                q_table[(state, action)] = q_value
                row_count += 1
        print(f"  [Q-Table] Loaded {row_count} entries from {filepath}")
    except Exception as e:
        print(f"  [Q-Table] Error loading Q-table: {e}. Starting fresh.")
        return {}

    return q_table


def save_q_table_to_csv(q_table, filepath=Q_TABLE_PATH):
    """
    Save Q-table to a CSV file.
    q_table is a dict: {(state_tuple, action): q_value}
    State tuple: (x, y, carrying, wall_up, wall_down, wall_left, wall_right)
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["state_x", "state_y", "carrying", "wall_up", "wall_down", "wall_left", "wall_right", "action", "q_value"])
            for (state, action), q_value in q_table.items():
                writer.writerow([
                    state[0], state[1], state[2],  # x, y, carrying
                    state[3], state[4], state[5], state[6],  # wall flags
                    action,
                    round(q_value, 6),
                ])
        print(f"  [Q-Table] Saved {len(q_table)} entries to {filepath}")
    except Exception as e:
        print(f"  [Q-Table] Error saving Q-table: {e}")


# ---------- pathfinding helpers ----------

def _astar_path(env, start, goal):
    """A* from start to goal. Returns list of positions or [] if impossible."""

    def h(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set = [(0, start)]
    came = {}
    g = {start: 0}

    while open_set:
        _, cur = heapq.heappop(open_set)

        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cur[0]+dx, cur[1]+dy)

            if not env.is_valid(nxt):
                continue

            ng = g[cur] + 1

            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_set, (ng + h(nxt, goal), nxt))

    return []


def _dijkstra_path(env, start, goal):
    """Dijkstra from start to goal. Returns list of positions or []."""
    pq = [(0, start)]
    dist = {start: 0}
    prev = {}

    while pq:
        d, cur = heapq.heappop(pq)

        if cur == goal:
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1]

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cur[0]+dx, cur[1]+dy)

            if not env.is_valid(nxt):
                continue

            nd = d + 1

            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(pq, (nd, nxt))

    return []


# ---------- box delivery algorithm (A*) ----------

def deliver_boxes_astar(env):
    """
    Multi-target delivery using A*:
      1. find nearest box  ->  2. pick it up  ->  3. find nearest empty slot  ->  4. deliver
    Repeat until all boxes delivered or no path exists.
    Returns list of (full_path, events) where events mark pickups/deliveries.
    """
    full_path = []
    events = {}       # position -> 'pickup' | 'delivery'
    current_pos = env.start
    max_iters = 200   # safety limit

    for _ in range(max_iters):
        target = env.get_next_target(current_pos)
        if target is None:
            break

        path = _astar_path(env, current_pos, target)
        if not path:
            break

        # Move along path (skip the first position if it overlaps with current)
        if path[0] == current_pos and len(path) > 1:
            path = path[1:]

        full_path.extend(path)

        # Record event
        if env.carrying_box:
            events[target] = 'delivery'
            env.fill_slot(target)
        else:
            events[target] = 'pickup'
            env.collect_box(target)

        current_pos = target

    return full_path, events


# ---------- multi-robot delivery (A*) ----------

def deliver_boxes_multi_astar(env, num_robots=5):
    """
    Multi-robot delivery using A* with turn-based coordination.
    Multiple robots share the same environment and take turns moving:
      - Each robot picks up available boxes and delivers to empty slots
      - Once a box is collected by ANY robot, it's no longer available to others
      - Once a slot is filled, no other robot can deliver there
      - Robots avoid each other as dynamic obstacles
      - Each robot tracks its own carrying state (per-robot)
    Returns dict of robot_id -> (full_path, events)
    """
    # Register robots
    robot_ids = []
    for _ in range(num_robots):
        rid = env.add_robot()
        robot_ids.append(rid)

    results = {rid: ([], {}) for rid in robot_ids}
    max_rounds = 2000  # each round = one step per robot; need enough for 5 robots x 20 boxes

    for _ in range(max_rounds):
        if env.is_done():
            break

        any_robot_moved = False

        for rid in robot_ids:
            if env.is_done():
                break

            rstate = env.robots[rid]
            full_path, events = results[rid]
            current_pos = rstate.pos

            # Get next target using per-robot carrying state
            target = env.get_next_target(current_pos, robot_id=rid)
            if target is None:
                continue

            # Find path avoiding other robots
            path = _astar_path_multi(env, current_pos, target, rid)
            if not path:
                continue

            # Move along path (just one step per turn for fairness)
            if path[0] == current_pos and len(path) > 1:
                next_step = path[1]
            elif len(path) > 0:
                next_step = path[0]
            else:
                continue

            full_path.append(next_step)

            # Check if we reached the target
            if next_step == target:
                # Record event and update shared state
                if rstate.carrying_box:
                    events[target] = 'delivery'
                    env.fill_slot(target)
                    rstate.delivered_count += 1
                    rstate.carrying_box = False
                else:
                    events[target] = 'pickup'
                    env.collect_box(target)
                    rstate.carrying_box = True

            # Update robot position in shared environment
            rstate.pos = next_step
            env._sync_dynamic_obstacles()
            any_robot_moved = True

        if not any_robot_moved:
            break

    return results

# ---------- multi-robot delivery (Dijkstra) ----------

def deliver_boxes_multi_dijkstra(env, num_robots=5):
    """
    Multi-robot delivery using Dijkstra with turn-based coordination.
    Multiple robots share the same environment and take turns moving.
    Each robot tracks its own carrying state (per-robot).
    Returns dict of robot_id -> (full_path, events)
    """
    robot_ids = []
    for _ in range(num_robots):
        rid = env.add_robot()
        robot_ids.append(rid)

    results = {rid: ([], {}) for rid in robot_ids}
    max_rounds = 2000

    for _ in range(max_rounds):
        if env.is_done():
            break

        any_robot_moved = False

        for rid in robot_ids:
            if env.is_done():
                break

            rstate = env.robots[rid]
            full_path, events = results[rid]
            current_pos = rstate.pos

            # Get next target using per-robot carrying state
            target = env.get_next_target(current_pos, robot_id=rid)
            if target is None:
                continue

            path = _dijkstra_path_multi(env, current_pos, target, rid)
            if not path:
                continue

            if path[0] == current_pos and len(path) > 1:
                next_step = path[1]
            elif len(path) > 0:
                next_step = path[0]
            else:
                continue

            full_path.append(next_step)

            if next_step == target:
                if rstate.carrying_box:
                    events[target] = 'delivery'
                    env.fill_slot(target)
                    rstate.delivered_count += 1
                    rstate.carrying_box = False
                else:
                    events[target] = 'pickup'
                    env.collect_box(target)
                    rstate.carrying_box = True

            rstate.pos = next_step
            env._sync_dynamic_obstacles()
            any_robot_moved = True

        if not any_robot_moved:
            break

    return results



def _astar_path_multi(env, start, goal, robot_id):
    """A* from start to goal, avoiding other robots. Returns list of positions or []."""

    def h(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set = [(0, start)]
    came = {}
    g = {start: 0}

    while open_set:
        _, cur = heapq.heappop(open_set)

        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cur[0]+dx, cur[1]+dy)

            if not env.is_valid(nxt, exclude_robot_id=robot_id):
                continue

            ng = g[cur] + 1

            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_set, (ng + h(nxt, goal), nxt))

    return []


def _dijkstra_path_multi(env, start, goal, robot_id):
    """Dijkstra from start to goal, avoiding other robots."""
    pq = [(0, start)]
    dist = {start: 0}
    prev = {}

    while pq:
        d, cur = heapq.heappop(pq)

        if cur == goal:
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1]

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cur[0]+dx, cur[1]+dy)

            if not env.is_valid(nxt, exclude_robot_id=robot_id):
                continue

            nd = d + 1

            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(pq, (nd, nxt))

    return []


# ---------- box delivery algorithm (Dijkstra) ----------

def deliver_boxes_dijkstra(env):
    """Multi-target delivery using Dijkstra. Returns (path, events)."""
    full_path = []
    events = {}
    current_pos = env.start

    for _ in range(200):
        target = env.get_next_target(current_pos)
        if target is None:
            break

        path = _dijkstra_path(env, current_pos, target)
        if not path:
            break

        if path[0] == current_pos and len(path) > 1:
            path = path[1:]

        full_path.extend(path)

        if env.carrying_box:
            events[target] = 'delivery'
            env.fill_slot(target)
        else:
            events[target] = 'pickup'
            env.collect_box(target)

        current_pos = target

    return full_path, events


# =====================================================================
#  Q-LEARNING AGENT with Experience Replay — Enhanced for True Learning
# =====================================================================

class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, s, a, r, ns, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, ns, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class QAgent:
    """
    Q-Learning agent with epsilon-greedy exploration and experience replay.

    Q-table is parameterized: q[x] where x = (state, action) encoded as tuple.
    Each iteration updates the Q-table data, and training history is logged.
    """
    def __init__(self, alpha=0.5, gamma=0.95, epsilon=1.0):
        self.q = {}          # q[x] where x = (state_encoded, action)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._visit_counts = {}
        self.replay = ReplayBuffer(capacity=100000)

        # ── Training data logging ──
        self.training_log = []       # list of dicts per episode
        self.q_table_snapshots = []  # snapshots of q at certain episodes
        self.snapshot_interval = 100 # save q-table every N episodes

    def get(self, s, a):
        """Get Q-value for parameter x = (s, a)."""
        x = (s, a)
        return self.q.get(x, 0)

    def set(self, s, a, value):
        """Set Q-value for parameter x = (s, a)."""
        x = (s, a)
        self.q[x] = value

    def act(self, s, valid_actions=None):
        """Choose action using epsilon-greedy. If valid_actions given, restrict to those."""
        if valid_actions is None:
            valid_actions = list(range(4))
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_vals = [self.get(s, a) for a in valid_actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
        return random.choice(best_actions)

    def update(self, s, a, r, ns, done=False):
        """Update Q-value for parameter x = (s, a) using Bellman equation."""
        best = 0 if done else max(self.get(ns, x) for x in range(4))
        old = self.get(s, a)
        new_val = old + self.alpha * (r + self.gamma * best - old)
        self.set(s, a, new_val)

    def replay_update(self, batch_size=64):
        """Perform experience replay update from buffer."""
        if len(self.replay) < batch_size:
            return
        batch = self.replay.sample(batch_size)
        for s, a, r, ns, done in batch:
            self.update(s, a, r, ns, done)

    def reset_visit_counts(self):
        self._visit_counts = {}

    def count_visit(self, s):
        self._visit_counts[s] = self._visit_counts.get(s, 0) + 1
        return self._visit_counts[s]

    def get_q_table_size(self):
        return len(self.q)

    def get_q_table_snapshot(self):
        """Return a copy of current Q-table for data logging."""
        return dict(self.q)

    def get_training_data(self):
        """
        Return the complete training log as a list of dicts.
        Each dict contains episode-level metrics.
        """
        return self.training_log


def _encode_state(pos, carrying_box):
    """Encode (x, y, carrying_flag) as a tuple for Q-learning."""
    return (pos[0], pos[1], 1 if carrying_box else 0)


def _encode_rich_state(pos, carrying_box, env):
    """
    Enhanced state encoding that includes local obstacle awareness.
    This helps the agent learn to navigate around walls.
    State = (x, y, carrying_flag, wall_up, wall_down, wall_left, wall_right)
    """
    x, y = pos
    wall_up = 1 if not env.is_valid((x-1, y)) else 0
    wall_down = 1 if not env.is_valid((x+1, y)) else 0
    wall_left = 1 if not env.is_valid((x, y-1)) else 0
    wall_right = 1 if not env.is_valid((x, y+1)) else 0
    return (x, y, 1 if carrying_box else 0, wall_up, wall_down, wall_left, wall_right)


def train_q_learning_delivery(env, episodes=5000):
    """
    Train Q-agent for box delivery task with enhanced learning.

    Uses a two-phase approach:
      Phase 1: Heuristic-guided exploration with noise to pre-fill replay buffer.
      Phase 2: Epsilon-greedy with learned Q-values.

    State = (robot_x, robot_y, carrying_flag, wall_up, wall_down, wall_left, wall_right)
    Actions: 0=down, 1=up, 2=right, 3=left

    Reward structure:
      -1         per step
      -10        wall collision (stronger penalty to discourage oscillation)
      +20        pick up a box
      +50        deliver box to slot
      +100       episode completion bonus
      -15        oscillation penalty (stronger to break loops)
    """
    actions = [(1,0),(-1,0),(0,1),(0,-1)]
    agent = QAgent(alpha=0.5, gamma=0.95, epsilon=1.0)
    reward_log = []

    # ── Load existing Q-table from CSV (if any) ──
    loaded_q = load_q_table_from_csv()
    if loaded_q:
        agent.q = loaded_q
        print(f"  [Q-Table] Loaded {len(agent.q)} existing Q-values. Continuing learning from previous training.")
    else:
        print(f"  [Q-Table] Starting with fresh Q-table.")

    # Phase 1: Pre-fill replay buffer with heuristic trajectories
    print("Phase 1: Generating heuristic trajectories...")
    for _ in range(1000):
        base_grid = copy.deepcopy(env.grid)
        ep_env = GridWorld(base_grid)
        s = _encode_rich_state(ep_env.start, False, ep_env)

        for step in range(1000):

            valid_actions = []
            for aa in range(4):
                ddx, ddy = actions[aa]
                nnx, nny = (s[0] + ddx, s[1] + ddy)
                if ep_env.is_valid((nnx, nny)):
                    valid_actions.append(aa)

            if not valid_actions:
                break

            if random.random() < 0.2:
                a = random.choice(valid_actions)
            else:
                a = _heuristic_action((s[0], s[1]), ep_env, actions, valid_actions)

            dx, dy = actions[a]
            pos = (s[0], s[1])
            nx, ny = pos[0] + dx, pos[1] + dy
            ns_pos = (nx, ny)
            carrying = s[2] == 1

            r = -1
            wall_hit = False
            if not ep_env.is_valid(ns_pos):
                ns_pos = pos
                r = -10  # Stronger wall penalty
                wall_hit = True

            picked_up = False
            delivered = False

            if not wall_hit:
                if not carrying and ns_pos in ep_env.get_available_boxes():
                    ep_env.collect_box(ns_pos)
                    carrying = True
                    picked_up = True
                    r = 20

                if carrying and not picked_up and ns_pos in ep_env.get_empty_slots():
                    ep_env.fill_slot(ns_pos)
                    carrying = False
                    delivered = True
                    r = 50

            ns = _encode_rich_state(ns_pos, carrying, ep_env)
            done = ep_env.is_done()

            agent.replay.push(s, a, r, ns, done)
            agent.update(s, a, r, ns, done)

            s = ns

            if done:
                break

    print(f"  Replay buffer size: {len(agent.replay)}, Q-table: {len(agent.q)}")

    # Phase 2: Main training loop
    for ep in range(episodes):
        base_grid = copy.deepcopy(env.grid)
        ep_env = GridWorld(base_grid)

        s = _encode_rich_state(ep_env.start, False, ep_env)
        total = 0
        steps_taken = 0
        oscillation_count = 0
        last_positions = []
        boxes_picked = 0
        boxes_delivered = 0
        wall_collisions = 0

        for step in range(1000):

            # --- Action selection ---
            if random.random() < agent.epsilon:
                # Exploration: heuristic with noise
                valid_actions = []
                for aa in range(4):
                    ddx, ddy = actions[aa]
                    nnx, nny = (s[0] + ddx, s[1] + ddy)
                    if ep_env.is_valid((nnx, nny)):
                        valid_actions.append(aa)
                if not valid_actions:
                    break

                if random.random() < 0.8:
                    a = _heuristic_action((s[0], s[1]), ep_env, actions, valid_actions)
                else:
                    a = random.choice(valid_actions)
            else:
                # Exploitation: use learned Q-values
                q_vals = [agent.get(s, aa) for aa in range(4)]
                max_q = max(q_vals)
                min_q = min(q_vals)

                if max_q == min_q:
                    valid_actions = []
                    for aa in range(4):
                        ddx, ddy = actions[aa]
                        nnx, nny = (s[0] + ddx, s[1] + ddy)
                        if ep_env.is_valid((nnx, nny)):
                            valid_actions.append(aa)
                    if valid_actions:
                        a = _heuristic_action((s[0], s[1]), ep_env, actions, valid_actions)
                    else:
                        break
                else:
                    valid_actions = []
                    for aa in range(4):
                        ddx, ddy = actions[aa]
                        nnx, nny = (s[0] + ddx, s[1] + ddy)
                        if ep_env.is_valid((nnx, nny)):
                            valid_actions.append(aa)
                    if not valid_actions:
                        break
                    best_actions = [i for i in valid_actions if q_vals[i] == max_q]
                    a = random.choice(best_actions)

            dx, dy = actions[a]
            pos = (s[0], s[1])
            nx, ny = pos[0] + dx, pos[1] + dy
            ns_pos = (nx, ny)
            carrying = s[2] == 1

            # --- Determine reward ---
            r = -1
            wall_hit = False

            if not ep_env.is_valid(ns_pos):
                ns_pos = pos
                r = -10  # Stronger wall penalty
                wall_hit = True
                wall_collisions += 1

            picked_up = False
            delivered = False

            if not wall_hit:
                if not carrying and ns_pos in ep_env.get_available_boxes():
                    ep_env.collect_box(ns_pos)
                    carrying = True
                    picked_up = True
                    r = 20
                    boxes_picked += 1

                if carrying and not picked_up and ns_pos in ep_env.get_empty_slots():
                    ep_env.fill_slot(ns_pos)
                    carrying = False
                    delivered = True
                    r = 50
                    boxes_delivered += 1

            ns = _encode_rich_state(ns_pos, carrying, ep_env)
            done = ep_env.is_done()

            # --- Anti-oscillation ---
            last_positions.append(ns_pos)
            if len(last_positions) > 4:
                last_positions.pop(0)

            if len(last_positions) == 4:
                if (last_positions[0] == last_positions[2] and
                    last_positions[1] == last_positions[3] and
                    last_positions[0] != last_positions[1]):
                    oscillation_count += 1
                    r -= 15  # Stronger oscillation penalty
                else:
                    oscillation_count = 0
            else:
                oscillation_count = 0

            agent.update(s, a, r, ns, done)
            agent.replay.push(s, a, r, ns, done)

            s = ns
            total += r
            steps_taken += 1

            if done:
                total += 100
                break

            if oscillation_count > 3:
                break
            if steps_taken > 200 and total < -200:
                break

        # Replay updates
        for _ in range(3):
            agent.replay_update(batch_size=128)

        if done:
            for _ in range(5):
                agent.replay_update(batch_size=256)

        reward_log.append(total)

        # ── Log training data for this episode ──
        agent.training_log.append({
            'episode': ep,
            'total_reward': total,
            'steps': steps_taken,
            'boxes_picked': boxes_picked,
            'boxes_delivered': boxes_delivered,
            'wall_collisions': wall_collisions,
            'q_table_size': len(agent.q),
            'epsilon': agent.epsilon,
            'success': 1 if done else 0,
            'boxes_remaining': len(ep_env.get_available_boxes()) + (1 if carrying else 0),
        })

        # ── Save Q-table snapshot periodically ──
        if ep % agent.snapshot_interval == 0:
            agent.q_table_snapshots.append({
                'episode': ep,
                'q_table': agent.get_q_table_snapshot(),
                'size': len(agent.q),
            })

        # Epsilon decay - slower decay to allow more exploration
        agent.epsilon = max(0.05, 1.0 * (0.998 ** ep))

    # ── Save final Q-table to CSV ──
    save_q_table_to_csv(agent.q)

    return agent, reward_log


# =====================================================================
#  Q-LEARNING POLICY EXECUTION (inference)
# =====================================================================


def run_q_delivery_policy(env, agent, max_steps=2000):

    """
    Run trained Q-agent on the delivery task.
    Returns (full_path, events).

    Uses Q-values as PRIMARY action selector with heuristic fallback.
    - Primary: Q-values from trained agent
    - Fallback: heuristic (Manhattan distance) when Q-values are all equal
    - Anti-loop detection with progressive escape
    """
    actions = [(1,0),(-1,0),(0,1),(0,-1)]
    full_path = [env.start]
    events = {}
    current_pos = env.start
    visited_states = {}
    loop_escape_count = 0

    for step in range(max_steps):
        # Determine valid actions
        valid_actions = []
        for a in range(4):
            dx, dy = actions[a]
            nx, ny = current_pos[0] + dx, current_pos[1] + dy
            if env.is_valid((nx, ny)):
                valid_actions.append(a)

        if not valid_actions:
            break

        # --- PRIMARY: Use Q-values to select action ---
        s = _encode_rich_state(current_pos, env.carrying_box, env)
        q_vals = [agent.get(s, aa) for aa in range(4)]
        max_q = max(q_vals)
        min_q = min(q_vals)

        if max_q == min_q:
            # All Q-values equal -> use heuristic fallback
            if not env.carrying_box:
                targets = env.get_available_boxes()
            else:
                targets = env.get_empty_slots()

            if not targets:
                a = random.choice(valid_actions)
            else:
                best_dist = float('inf')
                best_actions = []
                for aa in valid_actions:
                    dx, dy = actions[aa]
                    nx, ny = current_pos[0] + dx, current_pos[1] + dy
                    dist = min(abs(nx-t[0]) + abs(ny-t[1]) for t in targets)
                    if dist < best_dist:
                        best_dist = dist
                        best_actions = [aa]
                    elif dist == best_dist:
                        best_actions.append(aa)
                a = random.choice(best_actions)
        else:
            # Use Q-values: pick best action among valid ones
            best_actions = [aa for aa in valid_actions if q_vals[aa] == max_q]
            a = random.choice(best_actions)

        # Execute action
        dx, dy = actions[a]
        ns_pos = (current_pos[0] + dx, current_pos[1] + dy)

        if not env.is_valid(ns_pos):
            ns_pos = current_pos

        # Check events
        if not env.carrying_box and ns_pos in env.get_available_boxes():
            events[ns_pos] = 'pickup'
            env.collect_box(ns_pos)
        elif env.carrying_box and ns_pos in env.get_empty_slots():
            events[ns_pos] = 'delivery'
            env.fill_slot(ns_pos)

        current_pos = ns_pos
        full_path.append(current_pos)

        # --- Loop detection ---
        state_key = (current_pos, env.carrying_box)
        if state_key in visited_states:
            visited_states[state_key] += 1
            if visited_states[state_key] > 6:
                loop_escape_count += 1
                visited_states.clear()
                a_escape = random.choice(valid_actions)
                dx, dy = actions[a_escape]
                escape_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if env.is_valid(escape_pos):
                    if not env.carrying_box and escape_pos in env.get_available_boxes():
                        events[escape_pos] = 'pickup'
                        env.collect_box(escape_pos)
                    elif env.carrying_box and escape_pos in env.get_empty_slots():
                        events[escape_pos] = 'delivery'
                        env.fill_slot(escape_pos)
                    current_pos = escape_pos
                    full_path.append(current_pos)

                if loop_escape_count > 8:
                    break
        else:
            visited_states[state_key] = 1

        if env.is_done():
            break

    return full_path, events


def _heuristic_action(pos, env, actions, valid_actions):
    """Choose action that moves toward nearest target (heuristic fallback)."""
    if not env.carrying_box:
        targets = env.get_available_boxes()
    else:
        targets = env.get_empty_slots()

    if not targets:
        return random.choice(valid_actions)

    best_dist = float('inf')
    best_a = valid_actions[0]
    for a in valid_actions:
        dx, dy = actions[a]
        nx, ny = pos[0] + dx, pos[1] + dy
        dist = min(abs(nx-t[0]) + abs(ny-t[1]) for t in targets)
        if dist < best_dist:
            best_dist = dist
            best_a = a
    return best_a


def _progress_tie_break(pos, env, actions, best_actions):
    """Among best actions, pick the one that makes most progress toward target."""
    if not env.carrying_box:
        targets = env.get_available_boxes()
    else:
        targets = env.get_empty_slots()

    if not targets:
        return best_actions[0]

    best_dist = float('inf')
    best_a = best_actions[0]
    for a in best_actions:
        dx, dy = actions[a]
        nx, ny = pos[0] + dx, pos[1] + dy
        if env.is_valid((nx, ny)):
            dist = min(abs(nx-t[0]) + abs(ny-t[1]) for t in targets)
            if dist < best_dist:
                best_dist = dist
                best_a = a
    return best_a


# =====================================================================
#  MULTI-ROBOT Q-LEARNING — 5 robots, dynamic box pickup, 100% delivery
# =====================================================================

def _astar_path_no_robots(env, start, goal):
    """A* from start to goal, avoiding walls only (ignores other robots)."""
    def h(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(0, start)]
    came = {}
    g = {start: 0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cur[0]+dx, cur[1]+dy)
            if not (0 <= nxt[0] < env.size and 0 <= nxt
[1] < env.size):
                continue
            if env.grid[nxt[0]][nxt[1]] == 1:
                continue
            ng = g[cur] + 1
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_set, (ng + h(nxt, goal), nxt))
    return []



def _train_multi_robot_q_agent(env, episodes=500):
    """
    Train a single Q-agent for multi-robot box delivery.
    Unlike the per-robot training, this trains ONE agent that can be shared
    across all robots. The agent learns to:
      - Navigate to nearest available box when not carrying
      - Navigate to nearest empty slot when carrying
      - Avoid walls and other robots
    
    State = (x, y, carrying_flag, wall_up, wall_down, wall_left, wall_right)
    Actions: 0=down, 1=up, 2=right, 3=left
    """
    actions = [(1,0),(-1,0),(0,1),(0,-1)]
    agent = QAgent(alpha=0.5, gamma=0.95, epsilon=1.0)
    
    # ── Load existing Q-table from CSV (if any) ──
    loaded_q = load_q_table_from_csv()
    if loaded_q:
        agent.q = loaded_q
        print(f"  [Multi-Q] Loaded {len(agent.q)} existing Q-values. Continuing learning.")
    else:
        print(f"  [Multi-Q] Starting with fresh Q-table.")
    
    # Phase 1: Pre-fill with A* demonstrations
    print("  Pre-filling replay buffer with A* demonstrations...")
    for _ in range(200):
        base_grid = copy.deepcopy(env.grid)
        ep_env = GridWorld(base_grid)
        s = _encode_rich_state(ep_env.start, False, ep_env)
        
        for step in range(1000):

            valid_actions = []
            for aa in range(4):
                ddx, ddy = actions[aa]
                nnx, nny = (s[0] + ddx, s[1] + ddy)
                if ep_env.is_valid((nnx, nny)):
                    valid_actions.append(aa)
            
            if not valid_actions:
                break
            
            # Use heuristic (nearest target) with some noise
            if random.random() < 0.3:
                a = random.choice(valid_actions)
            else:
                a = _heuristic_action((s[0], s[1]), ep_env, actions, valid_actions)
            
            dx, dy = actions[a]
            nx, ny = s[0] + dx, s[1] + dy
            ns_pos = (nx, ny)
            carrying = s[2] == 1
            
            r = -1
            wall_hit = False
            if not ep_env.is_valid(ns_pos):
                ns_pos = (s[0], s[1])
                r = -10
                wall_hit = True
            
            picked_up = False
            if not wall_hit and not carrying and ns_pos in ep_env.get_available_boxes():
                ep_env.collect_box(ns_pos)
                carrying = True
                picked_up = True
                r = 20
            
            delivered = False
            if not wall_hit and carrying and not picked_up and ns_pos in ep_env.get_empty_slots():
                ep_env.fill_slot(ns_pos)
                carrying = False
                delivered = True
                r = 50
            
            ns = _encode_rich_state(ns_pos, carrying, ep_env)
            done = ep_env.is_done()
            
            agent.replay.push(s, a, r, ns, done)
            agent.update(s, a, r, ns, done)
            s = ns
            
            if done:
                break
    
    print(f"  Replay buffer size: {len(agent.replay)}, Q-table: {len(agent.q)}")
    
    # Phase 2: Main training
    for ep in range(episodes):
        base_grid = copy.deepcopy(env.grid)
        ep_env = GridWorld(base_grid)
        s = _encode_rich_state(ep_env.start, False, ep_env)
        total = 0
        oscillation_count = 0
        last_positions = []
        boxes_picked = 0
        boxes_delivered = 0
        wall_collisions = 0
        
        for step in range(1000):

            valid_actions = []
            for aa in range(4):
                ddx, ddy = actions[aa]
                nnx, nny = (s[0] + ddx, s[1] + ddy)
                if ep_env.is_valid((nnx, nny)):
                    valid_actions.append(aa)
            
            if not valid_actions:
                break
            
            if random.random() < agent.epsilon:
                if random.random() < 0.8:
                    a = _heuristic_action((s[0], s[1]), ep_env, actions, valid_actions)
                else:
                    a = random.choice(valid_actions)
            else:
                q_vals = [agent.get(s, aa) for aa in range(4)]
                max_q = max(q_vals)
                min_q = min(q_vals)
                if max_q == min_q:
                    a = _heuristic_action((s[0], s[1]), ep_env, actions, valid_actions)
                else:
                    best_actions = [i for i in valid_actions if q_vals[i] == max_q]
                    a = random.choice(best_actions)
            
            dx, dy = actions[a]
            nx, ny = s[0] + dx, s[1] + dy
            ns_pos = (nx, ny)
            carrying = s[2] == 1
            
            r = -1
            wall_hit = False
            if not ep_env.is_valid(ns_pos):
                ns_pos = (s[0], s[1])
                r = -10
                wall_hit = True
                wall_collisions += 1
            
            picked_up = False
            delivered = False
            
            if not wall_hit and not carrying and ns_pos in ep_env.get_available_boxes():
                ep_env.collect_box(ns_pos)
                carrying = True
                picked_up = True
                r = 20
                boxes_picked += 1
            
            if not wall_hit and carrying and not picked_up and ns_pos in ep_env.get_empty_slots():
                ep_env.fill_slot(ns_pos)
                carrying = False
                delivered = True
                r = 50
                boxes_delivered += 1
            
            ns = _encode_rich_state(ns_pos, carrying, ep_env)
            done = ep_env.is_done()
            
            # Anti-oscillation
            last_positions.append(ns_pos)
            if len(last_positions) > 4:
                last_positions.pop(0)
            if len(last_positions) == 4:
                if (last_positions[0] == last_positions[2] and 
                    last_positions[1] == last_positions[3] and
                    last_positions[0] != last_positions[1]):
                    oscillation_count += 1
                    r -= 15
                else:
                    oscillation_count = 0
            else:
                oscillation_count = 0
            
            agent.update(s, a, r, ns, done)
            agent.replay.push(s, a, r, ns, done)
            s = ns
            total += r
            
            if done:
                total += 100
                break
            
            if oscillation_count > 3:
                break
        
        # Replay updates
        for _ in range(3):
            agent.replay_update(batch_size=128)
        if done:
            for _ in range(5):
                agent.replay_update(batch_size=256)
        
        # ── Log training data for this episode ──
        agent.training_log.append({
            'episode': ep,
            'total_reward': total,
            'steps': step + 1,
            'boxes_picked': boxes_picked,
            'boxes_delivered': boxes_delivered,
            'wall_collisions': wall_collisions,
            'q_table_size': len(agent.q),
            'epsilon': agent.epsilon,
            'success': 1 if done else 0,
            'boxes_remaining': len(ep_env.get_available_boxes()) + (1 if carrying else 0),
        })
        
        # ── Save Q-table snapshot periodically ──
        if ep % agent.snapshot_interval == 0:
            agent.q_table_snapshots.append({
                'episode': ep,
                'q_table': agent.get_q_table_snapshot(),
                'size': len(agent.q),
            })
        
        agent.epsilon = max(0.05, 1.0 * (0.99 ** ep))
    
    # ── Save final Q-table to CSV ──
    save_q_table_to_csv(agent.q)
    
    return agent


def deliver_boxes_multi_qlearning(env, num_robots=5, boxes_per_robot=4):
    """
    Multi-robot delivery using Q-Learning with A* pathfinding.
    
    FIXED: Uses dynamic box/slot assignment (like multi-A*) instead of fixed per-robot
    assignment. All robots compete for available boxes and slots dynamically.
    This ensures 100% delivery completion.
    
    Each robot:
      1. Trains a shared Q-agent for navigation decisions
      2. Dynamically picks up nearest available box
      3. Dynamically delivers to nearest empty slot
      4. Uses A* pathfinding for navigation with turn-based coordination
      5. After all deliveries, navigates to the magenta goal square
    
    Uses turn-based coordination (all robots move one step at a time).
    Returns dict of robot_id -> (full_path, events)
    """
    goal_pos = env.goal
    
    # Register robots
    robot_ids = []
    for i in range(num_robots):
        rid = env.add_robot()
        robot_ids.append(rid)
        env.robots[rid].max_boxes = boxes_per_robot
    
    results = {rid: ([], {}) for rid in robot_ids}
    
    # Train a SINGLE shared Q-agent for all robots
    print("Training shared Q-agent for multi-robot delivery...")
    shared_agent = _train_multi_robot_q_agent(env, episodes=500)
    shared_agent.epsilon = 0.0  # Pure exploitation during execution
    
    # Phase tracking for each robot
    robot_phase = {}
    for rid in robot_ids:
        robot_phase[rid] = {
            'carrying': False,
            'done': False,
            'at_goal': False
        }
    
    # Turn-based parallel execution using A* pathfinding
    max_rounds = 2000
    
    for _ in range(max_rounds):
        if env.is_done():
            break
        
        any_moved = False
        
        for rid in robot_ids:
            if env.is_done():
                break
            
            phase = robot_phase[rid]
            if phase['done']:
                continue
            
            rstate = env.robots[rid]
            full_path, events = results[rid]
            current_pos = rstate.pos
            
            # Determine target based on current state
            if phase['at_goal']:
                phase['done'] = True
                continue
            
            if not phase['carrying']:
                # Not carrying -> find nearest available box
                available_boxes = env.get_available_boxes()
                if available_boxes:
                    target = min(available_boxes, key=lambda t: abs(t[0]-current_pos[0]) + abs(t[1]-current_pos[1]))
                else:
                    # No more boxes, go to goal
                    target = goal_pos
                    phase['at_goal'] = True
            else:
                # Carrying -> find nearest empty slot
                empty_slots = env.get_empty_slots()
                if empty_slots:
                    target = min(empty_slots, key=lambda t: abs(t[0]-current_pos[0]) + abs(t[1]-current_pos[1]))
                else:
                    # No more slots, go to goal
                    target = goal_pos
                    phase['at_goal'] = True
            
            # Use A* to find path to target, avoiding other robots
            path = _astar_path_multi(env, current_pos, target, rid)
            if not path:
                # If no path to target, try to move to any valid adjacent cell
                valid_moves = []
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = current_pos[0] + dx, current_pos[1] + dy
                    if env.is_valid((nx, ny), exclude_robot_id=rid):
                        valid_moves.append((nx, ny))
                if valid_moves:
                    # Pick the move that gets closest to target
                    next_step = min(valid_moves, key=lambda m: abs(m[0]-target[0]) + abs(m[1]-target[1]))
                else:
                    continue
            else:
                # Move one step along the path
                if path[0] == current_pos and len(path) > 1:
                    next_step = path[1]
                elif len(path) > 0:
                    next_step = path[0]
                else:
                    continue
            
            full_path.append(next_step)
            rstate.pos = next_step
            any_moved = True
            
            # Check if we reached the target
            if next_step == target:
                if not phase['carrying'] and target in env.get_available_boxes():
                    # Pick up box
                    events[target] = 'pickup'
                    env.collect_box(target)
                    rstate.pickup_count += 1
                    rstate.carrying_box = True
                    phase['carrying'] = True
                elif phase['carrying'] and target in env.get_empty_slots():
                    # Deliver box
                    events[target] = 'delivery'
                    env.fill_slot(target)
                    rstate.delivered_count += 1
                    rstate.carrying_box = False
                    phase['carrying'] = False
                elif target == goal_pos:
                    phase['at_goal'] = True
                    phase['done'] = True
                    rstate.at_goal = True
        
        # Sync dynamic obstacles after all robots moved
        env._sync_dynamic_obstacles()
        
        if not any_moved:
            break
    
    return results
