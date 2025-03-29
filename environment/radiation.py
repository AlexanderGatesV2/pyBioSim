import numpy as np
import math
import random


class RadiationManager:
    def __init__(self, grid, params):
        """
        Initialize the radiation manager.
        
        Args:
            grid: The simulation grid
            params: Simulation parameters containing radiation settings
        """
        self.grid = grid
        self.params = params
        self.radioactive_wall = 'west'
        self.radiation_step_counter = 0

    def setup_radiation(self):
        """
        Setup the alternating radioactive walls environment.
        
        This initializes radiation levels with an exponential falloff
        from the west wall as the initial radioactive source.
        """
        if not self.params['enable_radioactive_environment']:
            return

        # Reset counter
        self.radiation_step_counter = 0

        # Start with west wall being radioactive
        self.radioactive_wall = 'west'

        # Reset radiation levels
        self.grid.data[:, :, 3] = 0

        # Set radiation levels (exponential falloff from west wall)
        for x in range(self.grid.size[0]):
            for y in range(self.grid.size[1]):
                # Distance from west wall
                west_dist = x
                # Calculate radiation level - higher near wall
                radiation = math.exp(-west_dist / self.params['radiation_falloff_factor'])
                self.grid.data[x, y, 3] = radiation

    def update_radiation(self):
        """
        Update the radioactive environment and switch walls if needed.
        
        Radiation alternates between the east and west walls based on
        the radiation_switch_steps parameter.
        """
        if not self.params['enable_radioactive_environment']:
            return

        # Increment counter
        self.radiation_step_counter += 1

        # Check if it's time to switch walls
        if self.radiation_step_counter >= self.params['radiation_switch_steps']:
            self.radiation_step_counter = 0
            if self.radioactive_wall == 'west':
                self.radioactive_wall = 'east'
            else:
                self.radioactive_wall = 'west'

            # Reset radiation levels
            self.grid.data[:, :, 3] = 0

            # Set new radiation levels
            for x in range(self.grid.size[0]):
                for y in range(self.grid.size[1]):
                    if self.radioactive_wall == 'west':
                        # Distance from west wall
                        dist = x
                    else:
                        # Distance from east wall
                        dist = self.grid.size[0] - 1 - x

                    # Calculate radiation level - higher near active wall
                    radiation = math.exp(-dist / self.params['radiation_falloff_factor'])
                    self.grid.data[x, y, 3] = radiation

    def apply_radiation_effects(self, creatures):
        """
        Apply radiation effects to creatures.
        
        Creatures in higher radiation areas have a higher probability
        of being killed by radiation.
        
        Args:
            creatures: List of creatures to check for radiation effects
            
        Returns:
            List of creatures killed by radiation
        """
        if not self.params['enable_radioactive_environment']:
            return []

        killed_creatures = []
        for creature in creatures:
            x, y = int(creature.position[0]), int(creature.position[1])
            radiation_level = self.grid.data[x, y, 3]

            # Higher radiation level means higher chance of death
            if radiation_level > 0 and random.random() < radiation_level:
                killed_creatures.append(creature)

        return killed_creatures
