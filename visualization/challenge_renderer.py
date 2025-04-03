import pygame
import math
from core.survival_criteria import (
    CHALLENGE_CIRCLE, CHALLENGE_RIGHT_HALF, CHALLENGE_RIGHT_QUARTER, CHALLENGE_LEFT_EIGHTH,
    CHALLENGE_CENTER_WEIGHTED, CHALLENGE_CENTER_UNWEIGHTED, CHALLENGE_CENTER_SPARSE,
    CHALLENGE_CORNER, CHALLENGE_CORNER_WEIGHTED, CHALLENGE_RADIOACTIVE_WALLS,
    CHALLENGE_AGAINST_ANY_WALL, CHALLENGE_TOUCH_ANY_WALL, CHALLENGE_EAST_WEST_EIGHTHS,
    CHALLENGE_NEAR_BARRIER, CHALLENGE_ALTRUISM, CHALLENGE_ALTRUISM_SACRIFICE,
    CHALLENGE_STRING, CHALLENGE_MIGRATE_DISTANCE, CHALLENGE_PAIRS, CHALLENGE_LOCATION_SEQUENCE
)

class ChallengeRenderer:
    """
    Handles rendering of challenge areas on the grid.
    This class visualizes the areas where creatures need to be to survive
    based on the current challenge type.
    """
    def __init__(self, display_scale):
        """
        Initialize the challenge renderer

        Args:
            display_scale: The scale factor for rendering grid cells
        """
        self.display_scale = display_scale
        self.challenge_colors = {
            'safe': (0, 255, 0),  # Green for safe areas
            'hazard': (255, 0, 0),  # Red for hazardous areas
            'neutral': (100, 100, 255),  # Blue for neutral/special areas
        }

    def render_challenge_area(self, screen, grid, challenge_type, params):
        """
        Render the challenge area based on the challenge type

        Args:
            screen: The pygame surface to render on
            grid: The grid object containing world data
            challenge_type: The type of challenge being run
            params: Parameters including transparency level
        """
        # Get transparency level from params (default to 40)
        transparency = params.get('challenge_highlight_transparency', 40)
        
        # Call the appropriate rendering method based on challenge type
        if challenge_type == CHALLENGE_RIGHT_HALF:
            self._render_right_half(screen, grid, transparency)
        elif challenge_type == CHALLENGE_RIGHT_QUARTER:
            self._render_right_quarter(screen, grid, transparency)
        elif challenge_type == CHALLENGE_LEFT_EIGHTH:
            self._render_left_eighth(screen, grid, transparency)
        elif challenge_type == CHALLENGE_CIRCLE:
            self._render_circle(screen, grid, transparency)
        elif challenge_type == CHALLENGE_CENTER_WEIGHTED or challenge_type == CHALLENGE_CENTER_UNWEIGHTED:
            self._render_center_circle(screen, grid, transparency)
        elif challenge_type == CHALLENGE_CENTER_SPARSE:
            self._render_center_sparse(screen, grid, transparency)
        elif challenge_type == CHALLENGE_CORNER or challenge_type == CHALLENGE_CORNER_WEIGHTED:
            self._render_corners(screen, grid, transparency)
        elif challenge_type == CHALLENGE_AGAINST_ANY_WALL or challenge_type == CHALLENGE_TOUCH_ANY_WALL:
            self._render_walls(screen, grid, transparency)
        elif challenge_type == CHALLENGE_EAST_WEST_EIGHTHS:
            self._render_east_west_eighths(screen, grid, transparency)
        elif challenge_type == CHALLENGE_NEAR_BARRIER:
            self._render_near_barriers(screen, grid, transparency)
        elif challenge_type == CHALLENGE_RADIOACTIVE_WALLS:
            self._render_radioactive_walls(screen, grid, transparency, params)
        elif challenge_type == CHALLENGE_ALTRUISM:
            self._render_altruism_zones(screen, grid, transparency)
        # Add more challenge types as needed

    def _render_right_half(self, screen, grid, transparency):
        """Render the right half of the grid as a safe area"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Calculate the midpoint of the grid
        mid_x = grid.size[0] // 2
        
        # Draw a rectangle covering the right half
        pygame.draw.rect(
            screen, color,
            (mid_x * scale, 0, grid.size[0] * scale - mid_x * scale, grid.size[1] * scale)
        )

    def _render_right_quarter(self, screen, grid, transparency):
        """Render the right quarter of the grid as a safe area"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Calculate the starting point (3/4 of the grid width)
        start_x = grid.size[0] // 2 + grid.size[0] // 4
        
        # Draw a rectangle covering the right quarter
        pygame.draw.rect(
            screen, color,
            (start_x * scale, 0, grid.size[0] * scale - start_x * scale, grid.size[1] * scale)
        )

    def _render_left_eighth(self, screen, grid, transparency):
        """Render the left eighth of the grid as a safe area"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Calculate the end point (1/8 of the grid width)
        end_x = grid.size[0] // 8
        
        # Draw a rectangle covering the left eighth
        pygame.draw.rect(
            screen, color,
            (0, 0, end_x * scale, grid.size[1] * scale)
        )

    def _render_circle(self, screen, grid, transparency):
        """Render a circle in the top-left quadrant as a safe area"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Calculate the center and radius
        center_x = grid.size[0] // 4
        center_y = grid.size[1] // 4
        radius = grid.size[0] // 4
        
        # Draw a filled circle
        pygame.draw.circle(
            screen, color,
            (center_x * scale, center_y * scale), radius * scale
        )

    def _render_center_circle(self, screen, grid, transparency):
        """Render a circle in the center as a safe area"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Calculate the center and radius
        center_x = grid.size[0] // 2
        center_y = grid.size[1] // 2
        radius = grid.size[0] // 3
        
        # Draw a filled circle
        pygame.draw.circle(
            screen, color,
            (center_x * scale, center_y * scale), radius * scale
        )

    def _render_center_sparse(self, screen, grid, transparency):
        """Render a circle in the center for the center sparse challenge"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Calculate the center and radius
        center_x = grid.size[0] // 2
        center_y = grid.size[1] // 2
        radius = grid.size[0] // 4
        
        # Draw a filled circle
        pygame.draw.circle(
            screen, color,
            (center_x * scale, center_y * scale), radius * scale
        )

    def _render_corners(self, screen, grid, transparency):
        """Render circles in each corner as safe areas"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Calculate the radius
        radius = grid.size[0] // 8
        
        # Draw circles in each corner
        corners = [
            (0, 0),
            (0, grid.size[1] - 1),
            (grid.size[0] - 1, 0),
            (grid.size[0] - 1, grid.size[1] - 1)
        ]
        
        for corner in corners:
            pygame.draw.circle(
                screen, color,
                (corner[0] * scale, corner[1] * scale), radius * scale
            )

    def _render_walls(self, screen, grid, transparency):
        """Render the walls as safe areas"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Draw rectangles for each wall
        # Top wall
        pygame.draw.rect(
            screen, color,
            (0, 0, grid.size[0] * scale, scale)
        )
        # Bottom wall
        pygame.draw.rect(
            screen, color,
            (0, (grid.size[1] - 1) * scale, grid.size[0] * scale, scale)
        )
        # Left wall
        pygame.draw.rect(
            screen, color,
            (0, 0, scale, grid.size[1] * scale)
        )
        # Right wall
        pygame.draw.rect(
            screen, color,
            ((grid.size[0] - 1) * scale, 0, scale, grid.size[1] * scale)
        )

    def _render_east_west_eighths(self, screen, grid, transparency):
        """Render the leftmost and rightmost eighths as safe areas"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Calculate the width of an eighth
        eighth_width = grid.size[0] // 8
        
        # Draw rectangles for the leftmost and rightmost eighths
        # Leftmost eighth
        pygame.draw.rect(
            screen, color,
            (0, 0, eighth_width * scale, grid.size[1] * scale)
        )
        # Rightmost eighth
        pygame.draw.rect(
            screen, color,
            ((grid.size[0] - eighth_width) * scale, 0, eighth_width * scale, grid.size[1] * scale)
        )

    def _render_near_barriers(self, screen, grid, transparency):
        """Render areas near barriers as safe areas"""
        scale = self.display_scale
        color = self.challenge_colors['safe'] + (transparency,)  # Add transparency
        
        # Define the proximity range (1-3 cells) - same as in survival_criteria.py
        proximity_range = 3
        
        # Draw rectangles around each barrier
        for bx, by in grid.barrier_locations:
            # Draw a rectangle around this barrier cell
            pygame.draw.rect(
                screen, color,
                ((bx - proximity_range) * scale, 
                 (by - proximity_range) * scale, 
                 (proximity_range * 2 + 1) * scale, 
                 (proximity_range * 2 + 1) * scale)
            )
            
        # Draw a border around the world to indicate that creatures near borders won't survive
        border_color = self.challenge_colors['hazard'] + (transparency,)  # Red with transparency
        border_width = 0  # Width of the border in cells
        
        # Top border
        pygame.draw.rect(
            screen, border_color,
            (0, 0, grid.size[0] * scale, border_width * scale)
        )
        # Bottom border
        pygame.draw.rect(
            screen, border_color,
            (0, (grid.size[1] - border_width) * scale, grid.size[0] * scale, border_width * scale)
        )
        # Left border
        pygame.draw.rect(
            screen, border_color,
            (0, 0, border_width * scale, grid.size[1] * scale)
        )
        # Right border
        pygame.draw.rect(
            screen, border_color,
            ((grid.size[0] - border_width) * scale, 0, border_width * scale, grid.size[1] * scale)
        )

    def _render_radioactive_walls(self, screen, grid, transparency, params):
        """Render the radioactive walls"""
        scale = self.display_scale
        color = self.challenge_colors['hazard'] + (transparency,)  # Add transparency
        
        # Determine which wall is radioactive based on the current step
        step = params.get('current_step', 0)
        steps_per_generation = params.get('steps_per_generation', 300)
        
        if step < steps_per_generation // 2:
            # Left wall is radioactive in the first half
            pygame.draw.rect(
                screen, color,
                (0, 0, scale * 3, grid.size[1] * scale)
            )
        else:
            # Right wall is radioactive in the second half
            pygame.draw.rect(
                screen, color,
                ((grid.size[0] - 3) * scale, 0, scale * 3, grid.size[1] * scale)
            )

    def _render_altruism_zones(self, screen, grid, transparency):
        """Render the altruism challenge zones"""
        scale = self.display_scale
        
        # Safe zone in NW quadrant (green)
        safe_color = self.challenge_colors['safe'] + (transparency,)
        safe_center = (grid.size[0] // 4, grid.size[1] // 4)
        safe_radius = grid.size[0] // 4
        
        pygame.draw.circle(
            screen, safe_color,
            (safe_center[0] * scale, safe_center[1] * scale), safe_radius * scale
        )
        
        # Sacrifice zone in NE quadrant (blue)
        sacrifice_color = self.challenge_colors['neutral'] + (transparency,)
        sacrifice_center = (grid.size[0] - grid.size[0] // 4, grid.size[1] // 4)
        sacrifice_radius = grid.size[0] // 4
        
        pygame.draw.circle(
            screen, sacrifice_color,
            (sacrifice_center[0] * scale, sacrifice_center[1] * scale), sacrifice_radius * scale
        )

    def get_challenge_name(self, challenge_type):
        """Return a human-readable name for the challenge type"""
        challenge_names = {
            CHALLENGE_CIRCLE: "Circle (NW Quadrant)",
            CHALLENGE_RIGHT_HALF: "Right Half",
            CHALLENGE_RIGHT_QUARTER: "Right Quarter",
            CHALLENGE_LEFT_EIGHTH: "Left Eighth",
            CHALLENGE_STRING: "String Formation",
            CHALLENGE_CENTER_WEIGHTED: "Center (Weighted)",
            CHALLENGE_CENTER_UNWEIGHTED: "Center (Unweighted)",
            CHALLENGE_CENTER_SPARSE: "Center (Sparse)",
            CHALLENGE_CORNER: "Corners",
            CHALLENGE_CORNER_WEIGHTED: "Corners (Weighted)",
            CHALLENGE_RADIOACTIVE_WALLS: "Radioactive Walls",
            CHALLENGE_AGAINST_ANY_WALL: "Against Any Wall",
            CHALLENGE_TOUCH_ANY_WALL: "Touch Any Wall",
            CHALLENGE_EAST_WEST_EIGHTHS: "East/West Eighths",
            CHALLENGE_NEAR_BARRIER: "Near Barriers",
            CHALLENGE_PAIRS: "Pairs",
            CHALLENGE_ALTRUISM: "Altruism",
            CHALLENGE_ALTRUISM_SACRIFICE: "Altruism Sacrifice",
        }
        return challenge_names.get(challenge_type, f"Challenge {challenge_type}")
