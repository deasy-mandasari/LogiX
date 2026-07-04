class RobotState:
    """Tracks state for a single robot."""
    def __init__(self, robot_id, start_pos, max_boxes=4):
        self.robot_id = robot_id
        self.pos = start_pos
        self.carrying_box = False
        self.path = [start_pos]
        self.events = {}
        self.delivered_count = 0
        self.pickup_count = 0
        self.max_boxes = max_boxes  # max boxes this robot should deliver
        self.done = False           # True when robot has delivered max_boxes
        self.at_goal = False        # True when robot has reached the goal after finishing


class GridWorld:
    def __init__(self, grid):
        self.grid = grid
        self.size = len(grid)

        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)

        self.dynamic_obstacles = set()

        # Parse rack slots (value 2) and boxes (value 3)
        self.rack_slots = set()
        self.boxes = set()
        for x in range(self.size):
            for y in range(self.size):
                if grid[x][y] == 2:
                    self.rack_slots.add((x, y))
                elif grid[x][y] == 3:
                    self.boxes.add((x, y))

        # Track state
        self.collected_boxes = set()   # boxes already picked up
        self.filled_slots = set()      # rack slots already filled
        self.carrying_box = False      # whether robot is currently carrying a box

        # Multi-robot support
        self.robots = {}               # robot_id -> RobotState
        self._next_robot_id = 1

    def add_robot(self, start_pos=None):
        """Register a new robot and return its ID."""
        if start_pos is None:
            start_pos = self.start
        robot_id = f"AGV-{self._next_robot_id}"
        self._next_robot_id += 1
        self.robots[robot_id] = RobotState(robot_id, start_pos)
        # Update dynamic obstacles with all robot positions
        self._sync_dynamic_obstacles()
        return robot_id

    def remove_robot(self, robot_id):
        """Remove a robot from the environment."""
        if robot_id in self.robots:
            del self.robots[robot_id]
            self._sync_dynamic_obstacles()

    def _sync_dynamic_obstacles(self):
        """Update dynamic_obstacles to include all robot positions (except the one moving)."""
        self.dynamic_obstacles = set()
        for rid, rstate in self.robots.items():
            self.dynamic_obstacles.add(rstate.pos)

    def get_robot_positions(self, exclude_robot_id=None):
        """Get positions of all robots, optionally excluding one."""
        positions = set()
        for rid, rstate in self.robots.items():
            if exclude_robot_id is not None and rid == exclude_robot_id:
                continue
            positions.add(rstate.pos)
        return positions

    def is_valid(self, pos, exclude_robot_id=None):
        x, y = pos

        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return False

        # Walls (1) are blocked
        if self.grid[x][y] == 1:
            return False

        # Dynamic obstacles (other robots)
        if exclude_robot_id is not None:
            other_positions = self.get_robot_positions(exclude_robot_id)
            if pos in other_positions:
                return False
        elif pos in self.dynamic_obstacles:
            return False

        return True

    def get_available_boxes(self):
        """Return boxes that haven't been collected yet."""
        return self.boxes - self.collected_boxes

    def get_empty_slots(self):
        """Return rack slots that haven't been filled yet."""
        return self.rack_slots - self.filled_slots

    def collect_box(self, pos):
        """Mark a box as collected."""
        if pos in self.boxes:
            self.collected_boxes.add(pos)
            self.carrying_box = True

    def fill_slot(self, pos):
        """Mark a rack slot as filled."""
        if pos in self.rack_slots:
            self.filled_slots.add(pos)
            self.carrying_box = False

    def is_done(self):
        """All boxes have been delivered to slots."""
        return len(self.filled_slots) == len(self.boxes)

    def get_next_target(self, current_pos, robot_id=None):
        """
        Determine the next target for the robot.
        If not carrying a box, find nearest available box.
        If carrying a box, find nearest empty rack slot.
        Uses per-robot carrying state if robot_id is provided.
        Returns the target position, or None if no target available.
        """
        # Use per-robot carrying state if available
        if robot_id is not None and robot_id in self.robots:
            carrying = self.robots[robot_id].carrying_box
        else:
            carrying = self.carrying_box

        if carrying:
            targets = self.get_empty_slots()
        else:
            targets = self.get_available_boxes()

        if not targets:
            return None

        # Find nearest target by Manhattan distance
        return min(targets, key=lambda t: abs(t[0] - current_pos[0]) + abs(t[1] - current_pos[1]))

