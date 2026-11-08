class LineCounter:
    def __init__(self, x1, x2, y, dead_band=8, max_gap=0):
        if x1 >= x2:
            raise ValueError("x1 must be smaller than x2.")

        dead_band = max(0, dead_band)
        max_gap   = max(1, max_gap)

        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.dead_band = dead_band
        self.max_gap = max_gap

        self.previous = {}
        self.counted_ids = set()
        self.counts = {"up": 0, "down": 0}

    def update(self, track_id, x, y, frame_index):
        if track_id in self.counted_ids:
            return None

        distance = y - self.y

        if abs(distance) <= self.dead_band:
            return None

        side = 1 if distance > 0 else -1
        previous = self.previous.get(track_id)

        self.previous[track_id] = (x,y,side,frame_index)

        if previous is None:
            return

        old_x, old_y, old_side, old_frame = previous
        gap = frame_index - old_frame
        if gap <= 0 or gap > self.max_gap or side == old_side:
            return None
        
        # Interpolate where the movement intersects the horizontal line.
        fraction = (self.y - old_y) / (y - old_y)
        crossing_x = old_x + fraction * (x - old_x)

        if not self.x1 <= crossing_x <= self.x2:
            return None

        direction = "down" if side == 1 else "up"
        self.counted_ids.add(track_id)
        self.counts[direction] += 1
        return direction