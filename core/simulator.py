import math
import random
import time
import pygame

from agents.creature import Creature
from agents.genome import Gene, Genome
from agents.population import Population
from core.types import Action, Sensor
from environment.zones import ZoneManager
from environment.barriers import BarrierManager
from environment.radiation import RadiationManager
from utils.random_generator import random_generator
from visualization.renderer import Renderer
from visualization.logger import Logger
from core.survival_criteria import SurvivalCriteria, CHALLENGE_RADIOACTIVE_WALLS, CHALLENGE_TOUCH_ANY_WALL, \
    CHALLENGE_LOCATION_SEQUENCE, CHALLENGE_ALTRUISM_SACRIFICE, CHALLENGE_ALTRUISM


class Simulator:
    def __init__(self, params, grid, signals):
        """
        Initialize the simulator with all required components.
        
        Args:
            params: Simulation parameters dictionary
            grid: The world grid
            signals: Pheromone signal manager
        """
        self.params = params
        self.grid = grid
        self.signals = signals  # Use the signals object passed in

        # Create components
        self.population = Population(params['population_size'], params)
        self.zone_manager = ZoneManager(grid, params)
        self.barrier_manager = BarrierManager(grid, params)
        self.radiation_manager = RadiationManager(grid, params)
        self.renderer = Renderer(params)
        self.logger = Logger(params)

        # Create survival criteria checker
        self.survival_criteria = SurvivalCriteria(params, grid)

        # Initialize state
        self.generation = 0
        self.step = 0
        self.running = True
        self.paused = False

        # Tracking statistics
        self.murder_count = 0

    def initialize(self):
        """
        Initialize the simulation by setting up the environment and population.
        
        This resets the grid, creates barriers and zones, and places the initial
        population of creatures.
        """
        # Reset grid
        self.grid.reset()

        # Create barriers
        self.barrier_manager.create_barriers(self.params.get('barrierType', 0))

        # Create zones
        self.zone_manager.place_random_zones()

        # Setup radiation if enabled
        if self.params.get('enable_radioactive_environment', False):
            self.radiation_manager.setup_radiation()

        # Initialize population
        self.population.initialize(self.grid)

        # Reset counters
        self.generation = 0
        self.step = 0
        self.murder_count = 0


    def update(self):
        """
        Update one simulation step.
        
        This updates all creatures, processes deaths and movements,
        updates environmental effects, and checks if the generation is complete.
        """
        if self.paused:
            return

        # Create a dictionary of creatures for quick lookup
        creatures_dict = {creature.id: creature for creature in self.population.creatures}

        # Debug output to verify creatures are alive at the start of the update
        alive_count = sum(1 for c in self.population.creatures if c.alive)
        if self.step == 0 or self.step == self.params['steps_per_generation'] - 1:
            print(f"Step {self.step}: {alive_count} alive creatures at start of update")
            # Print positions of all creatures
            for creature in self.population.creatures:
                if creature.alive:
                    print(f"Creature {creature.id} at position {creature.position}")

        # Update all creatures
        for creature in self.population.creatures:
            if creature.alive:
                # Pass the current simulation step to the creature update
                creature.update(self.grid, self.population.creatures, self.signals, self.step)

        # Debug code to verify creatures are detecting zones
        safe_creatures = sum(1 for c in self.population.creatures if c.in_safe_zone)
        # print(f"Safe creatures: {safe_creatures}/{len(self.population.creatures)}")

        # Apply challenge-specific effects
        self._handle_challenge_specific_effects()

        # Process death queue
        self.murder_count += len(self.grid.death_queue)
        self.grid.process_death_queue(creatures_dict)
        
        # Process move queue
        self.grid.process_move_queue(creatures_dict)
        
        # Check for duplicate creatures
        duplicates = self.grid.check_for_duplicates()
        if duplicates:
            print(f"WARNING: Found {len(duplicates)} positions with multiple creatures:")
            for pos, creature_ids in duplicates:
                print(f"  Position {pos} has creatures: {creature_ids}")
                
                # Fix duplicates by moving all but one creature to a new position
                first_creature_id = creature_ids[0]
                for creature_id in creature_ids[1:]:
                    if creature_id in creatures_dict and creatures_dict[creature_id].alive:
                        # Find a new empty location
                        try:
                            new_pos = self.grid.find_empty_location()
                            # Update grid
                            self.grid.data[pos[0], pos[1], 0] = first_creature_id
                            self.grid.data[int(new_pos[0]), int(new_pos[1]), 0] = creature_id
                            # Update creature position
                            creatures_dict[creature_id].position = new_pos
                            print(f"    Moved creature {creature_id} to {new_pos}")
                        except Exception as e:
                            print(f"    Failed to move creature {creature_id}: {e}")

        # Fade pheromones
        for layer in range(self.signals.num_layers):
            self.signals.fade(layer)

        # Update radiation if enabled
        if self.params.get('enable_radioactive_environment', False):
            self.radiation_manager.update_radiation()
            self.radiation_manager.apply_radiation_effects(self.population.creatures)

        # Debug output to verify creatures are alive at the end of the update
        alive_count = sum(1 for c in self.population.creatures if c.alive)
        if self.step == 0 or self.step == self.params['steps_per_generation'] - 1:
            print(f"Step {self.step}: {alive_count} alive creatures at end of update")

        # Increment step counter
        self.step += 1

        # Check if generation is complete
        if self.step >= self.params['steps_per_generation']:
            self.end_generation()

    def _handle_challenge_specific_effects(self):
        """
        Handle challenge-specific effects that need to be processed
        during the simulation step.
        
        This applies special rules for certain challenges like radioactive walls,
        wall-touching detection, and location sequence tracking.
        """
        challenge_type = self.params.get('challenge', 0)

        # CHALLENGE_RADIOACTIVE_WALLS
        if challenge_type == CHALLENGE_RADIOACTIVE_WALLS:
            # During the first half of the generation, the west wall is radioactive.
            # In the second half, the east wall is radioactive.
            radioactive_x = 0 if self.step < self.params['steps_per_generation'] / 2 else self.params['world_size'][
                                                                                              0] - 1

            # Apply radiation effects with exponential falloff
            for creature in self.population.creatures:
                if creature.alive:
                    # Calculate distance from radioactive wall
                    distance = abs(int(creature.position[0]) - radioactive_x)

                    # Only apply within half the arena width
                    if distance < self.params['world_size'][0] / 2:
                        # Chance of death increases closer to wall
                        chance_of_death = 1.0 / distance if distance > 0 else 1.0
                        if random_generator.random_float() < chance_of_death:
                            self.grid.queue_for_death(creature.id)

        # CHALLENGE_TOUCH_ANY_WALL
        elif challenge_type == CHALLENGE_TOUCH_ANY_WALL:
            # If a creature touches any wall, it gets flagged with challengeBits
            for creature in self.population.creatures:
                if creature.alive:
                    x, y = int(creature.position[0]), int(creature.position[1])
                    if (x == 0 or x == self.params['world_size'][0] - 1 or
                            y == 0 or y == self.params['world_size'][1] - 1):
                        creature.challengeBits = True

        # CHALLENGE_LOCATION_SEQUENCE
        elif challenge_type == CHALLENGE_LOCATION_SEQUENCE:
            radius = 9.0  # Distance to count as "visited"
            for creature in self.population.creatures:
                if creature.alive:
                    # Check each barrier center as a location to visit
                    for n, center in enumerate(self.grid.barrier_centers):
                        bit = 1 << n  # Bit position for this location

                        # If creature hasn't visited this location yet
                        if (creature.challengeBits & bit) == 0:
                            # Check if creature is at the location now
                            dx = creature.position[0] - center[0]
                            dy = creature.position[1] - center[1]
                            if math.sqrt(dx * dx + dy * dy) <= radius:
                                # Mark location as visited
                                creature.challengeBits |= bit
                                break  # Only visit one new location per step

    def end_generation(self):
        """
        Handle the end of a generation.
        
        This logs statistics, applies survival criteria, creates a new generation
        through selection and reproduction, and places the new creatures on the grid.
        """
        # Log statistics
        self.logger.log_generation(self.generation, self.population.creatures, self.murder_count)

        # Display sample genomes if configured
        if self.params.get('displaySampleGenomes', 0) > 0 and self.generation % self.params.get('genomeAnalysisStride',
                                                                                                1) == 0:
            self._display_sample_genomes(self.params['displaySampleGenomes'])

        # Use the population's natural selection method to create the next generation
        # This handles survival criteria, selection, and reproduction
        new_creatures = self.population.natural_selection_tournament(self.grid)
        
        # Replace current population with new generation
        self.population.creatures = new_creatures

        # Increment generation counter and reset step
        self.generation += 1
        self.step = 0
        self.murder_count = 0

        # Clear the grid and place new creatures
        self._place_new_generation()

    def _apply_survival_criteria(self):
        """
        Apply survival criteria to determine which creatures survive.
        
        For the altruism challenge, this also handles kinship-based selection
        where sacrificed creatures can save genetically similar creatures.

        Returns:
            List of (creature, score) tuples that passed the criteria
        """
        survivors = []

        for creature in self.population.creatures:
            if creature.alive and creature.brain.connections:
                passed, score = self.survival_criteria.check_criterion(
                    creature, self.params.get('challenge', 0))

                if passed:
                    survivors.append((creature, score))

        # Handle special case for the altruism challenge
        if self.params.get('challenge', 0) == CHALLENGE_ALTRUISM and self.generation >= 10:
            # Get sacrificed creatures
            sacrifices = []
            for creature in self.population.creatures:
                if creature.alive and creature.brain.connections:
                    passed, score = self.survival_criteria.check_criterion(
                        creature, CHALLENGE_ALTRUISM_SACRIFICE)

                    if passed:
                        sacrifices.append((creature, score))

            # Apply kinship-based selection if there are sacrifices
            if sacrifices:
                altruism_factor = 10  # How many saved per sacrifice

                # Find genetic matches
                saved_kin = []
                for sacrifice, _ in sacrifices:
                    # For each sacrifice, try to save related survivors
                    for _ in range(altruism_factor):
                        best_match = None
                        highest_similarity = 0.7  # Threshold

                        for survivor, survivor_score in survivors:
                            similarity = sacrifice.genome.calculate_similarity(survivor.genome)
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match = (survivor, survivor_score)

                        if best_match:
                            saved_kin.append(best_match)

                # Use saved kin if available, otherwise use regular survivors
                if saved_kin:
                    survivors = saved_kin

        # Sort by score (highest first)
        survivors.sort(key=lambda x: x[1], reverse=True)

        return survivors

    def _spawn_new_generation(self, survivors):
        """
        Create a new generation from survivors.
        
        This applies elitism to keep the top performers and fills the rest
        of the population with offspring created through crossover and mutation.

        Args:
            survivors: List of (creature, score) tuples

        Returns:
            List of new creatures for the next generation
        """
        # If no survivors, restart with random genomes
        if not survivors:
            # print(f"Generation {self.generation}: No survivors, restarting with random genomes")
            new_creatures = []
            for _ in range(self.params['population_size']):
                new_genome = Genome(length=self.params['genome_length'], params=self.params)
                new_creature = Creature(genome=new_genome, params=self.params)
                new_creatures.append(new_creature)
            return new_creatures

        # print(f"Generation {self.generation}: {len(survivors)} survivors")

        # Create new generation from survivors
        new_creatures = []

        # Apply elitism - keep top 5% of survivors
        elite_count = max(1, int(self.params['population_size'] * 0.05))
        elites = [s[0] for s in survivors[:elite_count]]

        # Add elite clones to new generation
        for elite in elites:
            # Clone the genome
            elite_genome = Genome(
                genes=[Gene(g.hex_value, params=self.params) for g in elite.genome.genes],
                params=self.params
            )

            # Create new creature with cloned genome
            new_creature = Creature(genome=elite_genome, params=self.params)
            new_creatures.append(new_creature)

        # Fill the rest with offspring from crossover
        while len(new_creatures) < self.params['population_size']:
            # Select parents using tournament selection
            parent1 = self._tournament_selection(survivors)
            parent2 = self._tournament_selection(survivors)

            # Try to ensure different parents when possible
            attempts = 0
            while parent1 == parent2 and len(survivors) > 1 and attempts < 3:
                parent2 = self._tournament_selection(survivors)
                attempts += 1

            # Create child genome through crossover and mutation
            child_genome = parent1.genome.crossover(parent2.genome)
            child_genome.mutate(self.params['mutation_rate'])

            # Create child creature with the new genome
            child = Creature(genome=child_genome, params=self.params)
            new_creatures.append(child)

        return new_creatures

    def _tournament_selection(self, scored_creatures, tournament_size=3):
        """
        Select a parent using tournament selection.
        
        This randomly selects a subset of creatures and returns the one
        with the highest score, providing selection pressure while
        maintaining diversity.

        Args:
            scored_creatures: List of (creature, score) tuples
            tournament_size: Size of tournament

        Returns:
            Selected creature
        """
        if len(scored_creatures) <= tournament_size:
            contestants = scored_creatures
        else:
            # Select random contestants
            indices = random_generator.random_uint(0, len(scored_creatures) - 1, tournament_size)
            contestants = [scored_creatures[i] for i in indices]

        # Return contestant with highest score
        return max(contestants, key=lambda x: x[1])[0]

    def _place_new_generation(self):
        """
        Place the new generation on the grid.
        
        This clears the grid of previous creatures and places each new
        creature at a random empty location.
        """
        # Clear the grid (just the creature layer)
        for x in range(self.grid.size[0]):
            for y in range(self.grid.size[1]):
                if self.grid.data[x, y, 0] > 0:  # Only clear creature IDs
                    self.grid.data[x, y, 0] = 0

        # Track occupied positions to avoid duplicates
        occupied_positions = set()
        
        # Place creatures at random locations
        for creature in self.population.creatures:
            # Find an empty location
            position = self.grid.find_empty_location()
            
            # Ensure the position is not already occupied
            attempts = 0
            while (int(position[0]), int(position[1])) in occupied_positions and attempts < 10:
                position = self.grid.find_empty_location()
                attempts += 1
                
            # Mark position as occupied
            occupied_positions.add((int(position[0]), int(position[1])))
            
            # Update creature position
            creature.position = position

            # Update grid
            x, y = int(position[0]), int(position[1])
            self.grid.data[x, y, 0] = creature.id
            
        # Verify no duplicates
        duplicates = self.grid.check_for_duplicates()
        if duplicates:
            print(f"WARNING: Found {len(duplicates)} positions with multiple creatures after placement:")
            for pos, creature_ids in duplicates:
                print(f"  Position {pos} has creatures: {creature_ids}")

    def _display_sample_genomes(self, count):
        """
        Display information about sample genomes for analysis.
        
        This prints details about a subset of creatures including their
        position, energy, neural network structure, and sample genes.

        Args:
            count: Number of sample genomes to display
        """
        # Take the first 'count' living creatures
        display_count = min(count, len(self.population.creatures))
        displayed = 0

        # print("\n---------------------------")
        # print(f"Sample Genomes (Generation {self.generation})")
        # print("---------------------------")

        for creature in self.population.creatures:
            if displayed >= display_count:
                break

            if creature.alive:
                # print(f"Creature ID: {creature.id}")
                # print(f"Position: {creature.position}")
                # print(f"Energy: {creature.energy:.2f}")
                # print(f"Neural Network: {creature.brain.active_internal_neurons} neurons, "
                #       f"{creature.brain.total_connections} connections")

                # Add a few genes for reference
                num_genes = min(5, len(creature.genome.genes))
                # print(f"Sample Genes ({num_genes} of {len(creature.genome.genes)}):")
                for i in range(num_genes):
                    gene = creature.genome.genes[i]
                    # print(f"  {gene.hex_value}: {gene.source_type}->{gene.source_id} to "
                    #       f"{gene.sink_type}->{gene.sink_id} (w={gene.weight:.3f})")

                # print("---------------------------")
                displayed += 1

        # Display sensor/action usage statistics
        self._display_sensor_action_stats()

    def _display_sensor_action_stats(self):
        """
        Display statistics about sensor and action usage.
        
        This counts how many connections use each sensor and action type
        across the population, providing insight into which neural inputs
        and outputs are most commonly used.
        """
        sensor_counts = {i: 0 for i in range(self.params['num_sensory_neurons'])}
        action_counts = {i: 0 for i in range(self.params['num_output_neurons'])}

        # Count connections for each sensor and action
        for creature in self.population.creatures:
            if creature.alive:
                for conn in creature.brain.connections:
                    if conn['source_type'] == 0:  # SENSOR
                        sensor_counts[conn['source_num']] += 1
                    if conn['sink_type'] == 1:  # ACTION
                        action_counts[conn['sink_num']] += 1

        # Print sensor usage
        # print("Sensors in use:")
        for sensor_id, count in sorted(sensor_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                sensor_name = self._get_sensor_name(sensor_id)
                # print(f"  {count} - {sensor_name}")

        # Print action usage
        # print("Actions in use:")
        for action_id, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                action_name = self._get_action_name(action_id)
                # print(f"  {count} - {action_name}")

        # print("---------------------------")

    def _get_sensor_name(self, sensor_id):
        """Get the name of a sensor from its ID"""
        for sensor in Sensor:
            if sensor.value == sensor_id:
                return sensor.name
        return f"SENSOR_{sensor_id}"

    def _get_action_name(self, action_id):
        """Get the name of an action from its ID"""
        for action in Action:
            if action.value == action_id:
                return action.name
        return f"ACTION_{action_id}"

    def run(self):
        """
        Run the main simulation loop.
        
        This initializes the simulation and then enters the main loop that
        handles events, updates the simulation state, and renders the world
        until the simulation is terminated.
        """
        # Initialize before starting
        self.initialize()

        try:
            # Main simulation loop
            while self.running:
                # Handle events
                self.running, self.paused = self.renderer.handle_events()

                # Update simulation
                if not self.paused:
                    self.update()

                # Update renderer with current state
                self.renderer.set_generation(self.generation)
                self.renderer.set_step(self.step)
                
                # Update params with current step for challenge visualization
                self.params['current_step'] = self.step
                
                self.renderer.render_world(self.grid, self.population.creatures)

        finally:
            # Clean up when simulation ends
            self.logger.close()
            pygame.quit()
