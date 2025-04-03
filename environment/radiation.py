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

        # Print radiation settings
        radiation_intensity = self.params.get('radiation_intensity', 1.0)
        radiation_falloff = self.params.get('radiation_falloff_factor', 10.0)
        print(f"RADIATION SETTINGS: Intensity = {radiation_intensity}, Falloff = {radiation_falloff}")

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
            old_wall = self.radioactive_wall
            if self.radioactive_wall == 'west':
                self.radioactive_wall = 'east'
            else:
                self.radioactive_wall = 'west'
            print(f"RADIATION WALL SWITCHED: {old_wall} -> {self.radioactive_wall}")

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
        of being killed by radiation. The radiation intensity parameter
        scales the effect of radiation.
        
        Args:
            creatures: List of creatures to check for radiation effects
            
        Returns:
            List of creatures killed by radiation
        """
        if not self.params['enable_radioactive_environment']:
            return []

        # Print the number of creatures being checked
        alive_creatures = [c for c in creatures if c.alive]
        print(f"RADIATION CHECK: Checking {len(alive_creatures)} alive creatures for radiation effects")

        # Get radiation intensity from parameters (default to 1.0 if not specified)
        radiation_intensity = self.params.get('radiation_intensity', 1.0)
        
        killed_creatures = []
        for creature in creatures:
            if not creature.alive:
                continue
                
            x, y = int(creature.position[0]), int(creature.position[1])
            radiation_level = self.grid.data[x, y, 3]

            # Scale radiation level by intensity
            scaled_radiation_level = radiation_level * radiation_intensity
            
            # Higher radiation level means higher chance of death
            if scaled_radiation_level > 0 and random.random() < scaled_radiation_level:
                killed_creatures.append(creature)
                # Print every radiation death
                print(f"RADIATION DEATH: Creature {creature.id} killed at position ({x}, {y}) with radiation level {scaled_radiation_level:.2f}")

        # Print summary of radiation deaths
        if killed_creatures:
            print(f"RADIATION SUMMARY: {len(killed_creatures)} creatures killed by radiation this step")
            
        return killed_creatures
