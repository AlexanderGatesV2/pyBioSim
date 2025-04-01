import math

# Challenge type constants
CHALLENGE_CIRCLE = 0              # Survive inside a circular area
CHALLENGE_RIGHT_HALF = 1          # Survive in right half of arena
CHALLENGE_RIGHT_QUARTER = 2       # Survive in right quarter of arena
CHALLENGE_STRING = 3              # Survive with specific number of neighbors
CHALLENGE_CENTER_WEIGHTED = 4     # Survive near center, weighted by distance
CHALLENGE_CENTER_UNWEIGHTED = 4   # Survive near center, equal weighting
CHALLENGE_CENTER_SPARSE = 8       # Survive near center with specific neighbor count
CHALLENGE_CORNER = 5              # Survive near any corner
CHALLENGE_CORNER_WEIGHTED = 6     # Survive near any corner, weighted by distance
CHALLENGE_MIGRATE_DISTANCE = 7    # Score based on distance traveled
CHALLENGE_LEFT_EIGHTH = 9         # Survive in left eighth of arena
CHALLENGE_RADIOACTIVE_WALLS = 10  # Survive radioactive walls challenge
CHALLENGE_AGAINST_ANY_WALL = 11   # Survive by touching any wall
CHALLENGE_TOUCH_ANY_WALL = 12     # Survive by having touched a wall during lifetime
CHALLENGE_EAST_WEST_EIGHTHS = 13  # Survive in leftmost or rightmost eighth
CHALLENGE_NEAR_BARRIER = 14       # Survive near barriers
CHALLENGE_PAIRS = 15              # Survive in pairs with specific neighbor configuration
CHALLENGE_LOCATION_SEQUENCE = 16  # Score based on number of locations visited
CHALLENGE_ALTRUISM = 17           # Altruism challenge - NW safe zone
CHALLENGE_ALTRUISM_SACRIFICE = 18 # Altruism sacrifice - NE sacrifice zone

class SurvivalCriteria:
    """
    Implementation of all survival criteria for the evolutionary simulation.
    
    Each criterion evaluates whether a creature survives to the next generation
    and assigns a fitness score that influences reproduction probability.
    
    Each criterion returns (passed, score) where:
    - passed is a boolean indicating if the creature passed the criterion
    - score is a float 0.0-1.0 indicating how well it performed
    """

    def __init__(self, params, grid):
        self.params = params
        self.grid = grid

    def check_criterion(self, creature, challenge_type):
        """
        Check if a creature meets a specific survival criterion

        Args:
            creature: The creature to evaluate
            challenge_type: Which challenge/criterion to apply

        Returns:
            Tuple of (passed, score) where passed is boolean and score is 0.0-1.0
        """
        if not creature.alive:
            return (False, 0.0)

        # Call the appropriate criterion function based on challenge type
        result = None
        if challenge_type == CHALLENGE_CIRCLE:
            result = self.challenge_circle(creature)
        elif challenge_type == CHALLENGE_RIGHT_HALF:
            result = self.challenge_right_half(creature)
        elif challenge_type == CHALLENGE_RIGHT_QUARTER:
            result = self.challenge_right_quarter(creature)
        elif challenge_type == CHALLENGE_LEFT_EIGHTH:
            result = self.challenge_left_eighth(creature)
        elif challenge_type == CHALLENGE_STRING:
            result = self.challenge_string(creature)
        elif challenge_type == CHALLENGE_CENTER_WEIGHTED:
            result = self.challenge_center_weighted(creature)
        elif challenge_type == CHALLENGE_CENTER_UNWEIGHTED:
            result = self.challenge_center_unweighted(creature)
        elif challenge_type == CHALLENGE_CENTER_SPARSE:
            result = self.challenge_center_sparse(creature)
        elif challenge_type == CHALLENGE_CORNER:
            result = self.challenge_corner(creature)
        elif challenge_type == CHALLENGE_CORNER_WEIGHTED:
            result = self.challenge_corner_weighted(creature)
        elif challenge_type == CHALLENGE_RADIOACTIVE_WALLS:
            result = self.challenge_radioactive_walls(creature)
        elif challenge_type == CHALLENGE_AGAINST_ANY_WALL:
            result = self.challenge_against_any_wall(creature)
        elif challenge_type == CHALLENGE_TOUCH_ANY_WALL:
            result = self.challenge_touch_any_wall(creature)
        elif challenge_type == CHALLENGE_MIGRATE_DISTANCE:
            result = self.challenge_migrate_distance(creature)
        elif challenge_type == CHALLENGE_EAST_WEST_EIGHTHS:
            result = self.challenge_east_west_eighths(creature)
        elif challenge_type == CHALLENGE_NEAR_BARRIER:
            result = self.challenge_near_barrier(creature)
        elif challenge_type == CHALLENGE_PAIRS:
            result = self.challenge_pairs(creature)
        elif challenge_type == CHALLENGE_LOCATION_SEQUENCE:
            result = self.challenge_location_sequence(creature)
        elif challenge_type == CHALLENGE_ALTRUISM:
            result = self.challenge_altruism(creature)
        elif challenge_type == CHALLENGE_ALTRUISM_SACRIFICE:
            result = self.challenge_altruism_sacrifice(creature)
        else:
            # Default to passing all creatures with same score
            result = (True, 1.0)
            
        # Always calculate a fallback score based on energy
        # This ensures at least some creatures survive each generation
        energy_factor = min(1.0, creature.energy / 1000.0)
        
        # Calculate a score that favors creatures with higher energy
        fallback_score = energy_factor
        
        # If the creature passed the original criterion, return that result
        if result[0]:
            return result
        else:
            # Otherwise return the fallback score
            return (False, fallback_score)

    def challenge_circle(self, creature):
        """
        Survivors are those inside a circular area in the top-left quadrant.
        Score is higher for creatures closer to the center of the circle.
        """
        safe_center = (self.params['world_size'][0] // 4, self.params['world_size'][1] // 4)
        radius = self.params['world_size'][0] // 4

        offset = (creature.position[0] - safe_center[0], creature.position[1] - safe_center[1])
        distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2)

        if distance <= radius:
            # Score is better the closer to center (normalized to 0-1)
            score = (radius - distance) / radius
            return (True, score)
        else:
            return (False, 0.0)

    def challenge_right_half(self, creature):
        """
        Survivors are all those on the right half of the arena.
        All survivors receive equal score.
        """
        if creature.position[0] > self.params['world_size'][0] // 2:
            return (True, 1.0)
        else:
            return (False, 0.0)

    def challenge_right_quarter(self, creature):
        """
        Survivors are all those on the rightmost quarter of the arena.
        All survivors receive equal score.
        """
        if creature.position[0] > self.params['world_size'][0] // 2 + self.params['world_size'][0] // 4:
            return (True, 1.0)
        else:
            return (False, 0.0)

    def challenge_left_eighth(self, creature):
        """
        Survivors are all those on the leftmost eighth of the arena.
        All survivors receive equal score.
        """
        if creature.position[0] < self.params['world_size'][0] // 8:
            return (True, 1.0)
        else:
            return (False, 0.0)

    def challenge_string(self, creature):
        """
        Survivors have 2-3 neighbors within a radius of 1.5 cells,
        and are not touching the border of the arena.
        All survivors receive equal score.
        """
        min_neighbors = 2  # Adjust as needed
        max_neighbors = 3  # Adjust as needed
        radius = 1.5

        # Check if touching border
        x, y = int(creature.position[0]), int(creature.position[1])
        if (x == 0 or x == self.params['world_size'][0] - 1 or
                y == 0 or y == self.params['world_size'][1] - 1):
            return (False, 0.0)

        # Count neighbors
        count = 0
        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq <= radius * radius:
                    nx = min(self.params['world_size'][0] - 1, max(0, x + dx))
                    ny = min(self.params['world_size'][1] - 1, max(0, y + dy))

                    if self.grid.data[nx, ny, 0] > 0:  # Contains a creature
                        count += 1

        # Subtract self from count
        count -= 1

        if min_neighbors <= count <= max_neighbors:
            return (True, 1.0)
        else:
            return (False, 0.0)

    def challenge_center_weighted(self, creature):
        """
        Survivors are those within a radius of the center of the arena.
        Score is higher for creatures closer to the center.
        """
        safe_center = (self.params['world_size'][0] // 2, self.params['world_size'][1] // 2)
        radius = self.params['world_size'][0] // 3

        offset = (creature.position[0] - safe_center[0], creature.position[1] - safe_center[1])
        distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2)

        if distance <= radius:
            # Score is better the closer to center
            score = (radius - distance) / radius
            return (True, score)
        else:
            return (False, 0.0)

    def challenge_center_unweighted(self, creature):
        """
        Survivors are those within a radius of the center of the arena.
        All survivors receive equal score regardless of distance from center.
        """
        safe_center = (self.params['world_size'][0] // 2, self.params['world_size'][1] // 2)
        radius = self.params['world_size'][0] // 3

        offset = (creature.position[0] - safe_center[0], creature.position[1] - safe_center[1])
        distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2)

        if distance <= radius:
            return (True, 1.0)
        else:
            return (False, 0.0)

    def challenge_center_sparse(self, creature):
        """
        Survivors are within outer radius of center and have 5-8 neighbors
        (including self) within an inner radius of 1.5 cells.
        All survivors receive equal score.
        """
        safe_center = (self.params['world_size'][0] // 2, self.params['world_size'][1] // 2)
        outer_radius = self.params['world_size'][0] // 4
        inner_radius = 1.5
        min_neighbors = 5  # Includes self
        max_neighbors = 8

        # Check if within outer radius
        x, y = int(creature.position[0]), int(creature.position[1])
        offset = (x - safe_center[0], y - safe_center[1])
        distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2)

        if distance <= outer_radius:
            # Count neighbors within inner radius
            count = 0
            for dx in range(-int(inner_radius), int(inner_radius) + 1):
                for dy in range(-int(inner_radius), int(inner_radius) + 1):
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= inner_radius * inner_radius:
                        nx = min(self.params['world_size'][0] - 1, max(0, x + dx))
                        ny = min(self.params['world_size'][1] - 1, max(0, y + dy))

                        if self.grid.data[nx, ny, 0] > 0:  # Contains a creature
                            count += 1

            if min_neighbors <= count <= max_neighbors:
                return (True, 1.0)

        return (False, 0.0)

    def challenge_corner(self, creature):
        """
        Survivors are those within a radius of any corner of the arena.
        All survivors receive equal score.
        """
        radius = self.params['world_size'][0] // 8
        x, y = creature.position

        # Check each corner
        corners = [
            (0, 0),
            (0, self.params['world_size'][1] - 1),
            (self.params['world_size'][0] - 1, 0),
            (self.params['world_size'][0] - 1, self.params['world_size'][1] - 1)
        ]

        for corner in corners:
            offset = (x - corner[0], y - corner[1])
            distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2)

            if distance <= radius:
                return (True, 1.0)

        return (False, 0.0)

    def challenge_corner_weighted(self, creature):
        """
        Survivors are those within a radius of any corner of the arena.
        Score is higher for creatures closer to a corner.
        """
        radius = self.params['world_size'][0] // 4
        x, y = creature.position

        # Check each corner
        corners = [
            (0, 0),
            (0, self.params['world_size'][1] - 1),
            (self.params['world_size'][0] - 1, 0),
            (self.params['world_size'][0] - 1, self.params['world_size'][1] - 1)
        ]

        for corner in corners:
            offset = (x - corner[0], y - corner[1])
            distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2)

            if distance <= radius:
                # Score is better the closer to corner
                score = (radius - distance) / radius
                return (True, score)

        return (False, 0.0)

    def challenge_radioactive_walls(self, creature):
        """
        Radioactive walls challenge - creatures die if they touch walls.
        This is primarily handled in the simulator's step function.
        All survivors receive equal score.
        """
        return (True, 1.0)

    def challenge_against_any_wall(self, creature):
        """
        Survivors are those currently touching any wall of the arena.
        All survivors receive equal score.
        """
        x, y = int(creature.position[0]), int(creature.position[1])

        if (x == 0 or x == self.params['world_size'][0] - 1 or
                y == 0 or y == self.params['world_size'][1] - 1):
            return (True, 1.0)
        else:
            return (False, 0.0)

    def challenge_touch_any_wall(self, creature):
        """
        Survivors are those that touched a wall at any time during their lifetime.
        This is tracked by the creature's challengeBits flag.
        All survivors receive equal score.
        """
        if creature.challengeBits:
            return (True, 1.0)
        else:
            return (False, 0.0)

    def challenge_migrate_distance(self, creature):
        """
        All creatures survive, but are scored by distance traveled from birth location.
        Score is normalized by the maximum possible distance in the arena.
        """
        dx = creature.position[0] - creature.birth_position[0]
        dy = creature.position[1] - creature.birth_position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Normalize by maximum possible distance
        max_distance = max(self.params['world_size'][0], self.params['world_size'][1])
        score = distance / max_distance

        return (True, score)

    def challenge_east_west_eighths(self, creature):
        """
        Survivors are those in the leftmost or rightmost eighth of the arena.
        All survivors receive equal score.
        """
        x = creature.position[0]

        if (x < self.params['world_size'][0] // 8 or
                x >= self.params['world_size'][0] - self.params['world_size'][0] // 8):
            return (True, 1.0)
        else:
            return (False, 0.0)

    def challenge_near_barrier(self, creature):
        """
        Survivors are those within a radius of any barrier center.
        Score is higher for creatures closer to a barrier center.
        """
        radius = self.params['world_size'][0] // 2
        x, y = creature.position

        # Find nearest barrier center
        min_distance = float('inf')
        for center in self.grid.barrier_centers:
            dx = x - center[0]
            dy = y - center[1]
            distance = math.sqrt(dx * dx + dy * dy)
            min_distance = min(min_distance, distance)

        if min_distance <= radius:
            score = 1.0 - (min_distance / radius)
            return (True, score)
        else:
            return (False, 0.0)

    def challenge_pairs(self, creature):
        """
        Survivors are creatures that form exclusive pairs.
        To survive, a creature must:
        1. Not be on the border
        2. Have exactly one neighbor
        3. That neighbor must have no other neighbors (except the original creature)
        All survivors receive equal score.
        """
        x, y = int(creature.position[0]), int(creature.position[1])

        # Check if on border
        if (x == 0 or x == self.params['world_size'][0] - 1 or
                y == 0 or y == self.params['world_size'][1] - 1):
            return (False, 0.0)

        # Count neighbors and check if they have other neighbors
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue  # Skip self

                nx = x + dx
                ny = y + dy
                if (0 <= nx < self.params['world_size'][0] and
                        0 <= ny < self.params['world_size'][1]):
                    creature_id = self.grid.data[nx, ny, 0]
                    if creature_id > 0:
                        neighbors.append((nx, ny, creature_id))

        # Must have exactly one neighbor
        if len(neighbors) != 1:
            return (False, 0.0)

        # Check if that neighbor has any other neighbors
        nx, ny, _ = neighbors[0]
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue  # Skip the neighbor itself

                nnx = nx + dx
                nny = ny + dy
                if (0 <= nnx < self.params['world_size'][0] and
                        0 <= nny < self.params['world_size'][1] and
                        (nnx != x or nny != y)):  # Not the original creature
                    if self.grid.data[nnx, nny, 0] > 0:
                        return (False, 0.0)  # Neighbor has another neighbor

        return (True, 1.0)

    def challenge_location_sequence(self, creature):
        """
        Survivors are scored by how many distinct locations they visited
        in sequence during their lifetime, tracked by the challengeBits field.
        Score is proportional to the number of locations visited.
        """
        count = bin(creature.challengeBits).count('1')
        if count > 0:
            max_bits = 8 * 4  # Assuming challengeBits is a 32-bit integer
            score = count / max_bits
            return (True, score)
        else:
            return (False, 0.0)

    def challenge_altruism(self, creature):
        """
        Altruism challenge - creatures in the northwest safe zone have higher score.
        Score is higher for creatures closer to the center of the safe zone.
        This is part of the altruism experiment paired with altruism_sacrifice.
        """
        safe_center = (self.params['world_size'][0] // 4, self.params['world_size'][1] // 4)
        radius = self.params['world_size'][0] // 4

        offset = (creature.position[0] - safe_center[0], creature.position[1] - safe_center[1])
        distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2)

        if distance <= radius:
            # Score is better the closer to center
            score = (radius - distance) / radius
            return (True, score)
        else:
            return (False, 0.0)

    def challenge_altruism_sacrifice(self, creature):
        """
        Altruism sacrifice - creatures in the northeast sacrifice zone
        are selected for kinship-based reproduction.
        Score is higher for creatures closer to the center of the sacrifice zone.
        This is part of the altruism experiment paired with altruism.
        """
        sacrifice_center = (self.params['world_size'][0] - self.params['world_size'][0] // 4,
                            self.params['world_size'][1] - self.params['world_size'][1] // 4)
        radius = self.params['world_size'][0] // 4

        offset = (creature.position[0] - sacrifice_center[0], creature.position[1] - sacrifice_center[1])
        distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2)

        if distance <= radius:
            score = (radius - distance) / radius
            return (True, score)
        else:
            return (False, 0.0)
