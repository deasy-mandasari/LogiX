import heapq
import random
import numpy as np

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
      1. find nearest box  →  2. pick it up  →  3. find nearest empty slot  →  4. deliver
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


# ---------- Q-learning agent ----------

class QAgent:
    def __init__(self):
        self.q = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0

    def get(self, s, a):
        return self.q.get((s, a), 0)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax([self.get(s, a) for a in range(4)]))

    def update(self, s, a, r, ns):
        best = max(self.get(ns, x) for x in range(4))
        old = self.get(s, a)
        self.q[(s, a)] = old + self.alpha * (r + self.gamma * best - old)


def _encode_state(pos, carrying_box):
    """Encode (x, y, carrying_flag) as a tuple for Q-learning."""
    return (pos[0], pos[1], 1 if carrying_box else 0)


def train_q_learning_delivery(env, episodes=3000):
    """
    Train Q-agent for box delivery task.
    State = (robot_x, robot_y, carrying_flag)
    Actions: 0=up,1=down,2=left,3=right
    Rewards:
      -5 for hitting wall
      -1 per step (encourage shorter paths)
      +20 for picking up a box
      +50 for delivering to a slot
      +2 for moving closer to the right target type (shaped reward)
    """
    actions = [(1,0),(-1,0),(0,1),(0,-1)]
    agent = QAgent()
    reward_log = []

    for ep in range(episodes):
        # Reset environment per episode
        import copy
        from env import GridWorld
        base_grid = copy.deepcopy(env.grid)
        ep_env = GridWorld(base_grid)

        s = _encode_state(ep_env.start, False)
        total = 0

        for step in range(200):
            a = agent.act(s)
            dx, dy = actions[a]
            pos = (s[0], s[1])
            nx, ny = pos[0] + dx, pos[1] + dy
            ns_pos = (nx, ny)
            carrying = s[2] == 1

            # Invalid move
            if not ep_env.is_valid(ns_pos):
                ns_pos = pos
                r = -5
            else:
                r = -1

            # Shaped reward: encourage moving toward nearest target
            if not carrying:
                targets = ep_env.get_available_boxes()
                if targets:
                    old_dist = min(abs(pos[0]-t[0]) + abs(pos[1]-t[1]) for t in targets)
                    new_dist = min(abs(ns_pos[0]-t[0]) + abs(ns_pos[1]-t[1]) for t in targets)
                    if new_dist < old_dist:
                        r += 2
            else:
                targets = ep_env.get_empty_slots()
                if targets:
                    old_dist = min(abs(pos[0]-t[0]) + abs(pos[1]-t[1]) for t in targets)
                    new_dist = min(abs(ns_pos[0]-t[0]) + abs(ns_pos[1]-t[1]) for t in targets)
                    if new_dist < old_dist:
                        r += 2

            # Check if we reached a box (and not carrying)
            if not carrying and ns_pos in ep_env.get_available_boxes():
                ep_env.collect_box(ns_pos)
                carrying = True
                r = 20

            # Check if we reached an empty slot (and carrying)
            if carrying and ns_pos in ep_env.get_empty_slots():
                ep_env.fill_slot(ns_pos)
                carrying = False
                r = 50

            ns = _encode_state(ns_pos, carrying)
            agent.update(s, a, r, ns)
            s = ns
            total += r

            if ep_env.is_done():
                break

        # Slower epsilon decay for more exploration
        agent.epsilon = max(0.05, agent.epsilon * 0.995)
        reward_log.append(total)

    return agent, reward_log


def run_q_delivery_policy(env, agent, max_steps=400):
    """
    Run trained Q-agent on the delivery task.
    Returns (full_path, events).
    """
    actions = [(1,0),(-1,0),(0,1),(0,-1)]
    full_path = [env.start]
    events = {}
    current_pos = env.start

    for _ in range(max_steps):
        # Use greedy policy (no epsilon)
        s = _encode_state(current_pos, env.carrying_box)
        a = int(np.argmax([agent.get(s, a) for a in range(4)]))
        dx, dy = actions[a]
        ns_pos = (current_pos[0] + dx, current_pos[1] + dy)

        if not env.is_valid(ns_pos):
            ns_pos = current_pos

        # Check event
        if not env.carrying_box and ns_pos in env.get_available_boxes():
            events[ns_pos] = 'pickup'
            env.collect_box(ns_pos)
        elif env.carrying_box and ns_pos in env.get_empty_slots():
            events[ns_pos] = 'delivery'
            env.fill_slot(ns_pos)

        current_pos = ns_pos
        full_path.append(current_pos)

        if env.is_done():
            break

    return full_path, events