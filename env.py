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

    def is_valid(self, pos):
        x, y = pos

        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return False

        # Walls (1) are blocked
        if self.grid[x][y] == 1:
            return False

        # Dynamic obstacles
        if pos in self.dynamic_obstacles:
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

    def get_next_target(self, current_pos):
        """
        Determine the next target for the robot.
        If not carrying a box, find nearest available box.
        If carrying a box, find nearest empty rack slot.
        Returns the target position, or None if no target available.
        """
        if self.carrying_box:
            targets = self.get_empty_slots()
        else:
            targets = self.get_available_boxes()

        if not targets:
            return None

        # Find nearest target by Manhattan distance
        return min(targets, key=lambda t: abs(t[0] - current_pos[0]) + abs(t[1] - current_pos[1]))