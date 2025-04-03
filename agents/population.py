import random
import csv
import os
import datetime
import math

from agents.creature import Creature
from agents.genome import Genome, Gene
from core.survival_criteria import SurvivalCriteria, CHALLENGE_ALTRUISM_SACRIFICE, CHALLENGE_ALTRUISM
from core.types import calculate_genetic_similarity
from utils.genetic import analyze_genome_complexity, calculate_genetic_diversity

class Population:
    def __init__(self, size, params):
        """
        Initialize a population of creatures.
        
        Args:
            size: Number of creatures in the population
            params: Simulation parameters dictionary
        """
        self.timestamp = None
        self.params = params
        self.creatures = []
        self.size = size
        self.generation = 0

    def initialize(self, grid):
        """Initialize the first generation with uniform placement"""
        self.creatures = []

        # Create a grid-based placement strategy for more uniform distribution
        world_width = self.params['world_size'][0]
        world_height = self.params['world_size'][1]

        # Calculate ideal spacing
        spacing = max(1, min(world_width, world_height) // int(math.sqrt(self.size)))

        # Generate positions grid
        positions = []
        for x in range(spacing // 2, world_width, spacing):
            for y in range(spacing // 2, world_height, spacing):
                positions.append((x, y))

        # Shuffle positions
        random.shuffle(positions)

        # If we need more positions than our grid provides, add random ones
        while len(positions) < self.size:
            x = random.randint(0, world_width - 1)
            y = random.randint(0, world_height - 1)
            positions.append((x, y))

        # Track occupied positions to avoid duplicates
        occupied_positions = set()
        
        # Create creatures at these positions
        for i in range(min(self.size, len(positions))):
            # Create a genome with higher initial diversity
            genome = Genome(length=self.params['genome_length'], params=self.params)

            # Apply normalization but make it less aggressive
            if hasattr(Genome, 'normalize_genome_weights_mild'):
                Genome.normalize_genome_weights_mild(genome)
            else:
                Genome.normalize_genome_weights(genome)

            # If position is already occupied or is a barrier, find a new one
            position = positions[i]
            if not grid.is_valid_move_target(int(position[0]), int(position[1])) or (int(position[0]), int(position[1])) in occupied_positions:
                position = grid.find_empty_location()
                
                # Ensure the position is not already occupied
                attempts = 0
                while (int(position[0]), int(position[1])) in occupied_positions and attempts < 10:
                    position = grid.find_empty_location()
                    attempts += 1
            
            # Mark position as occupied
            occupied_positions.add((int(position[0]), int(position[1])))

            creature = Creature(genome=genome, position=position, params=self.params)
            self.creatures.append(creature)

            # Register creature in the world grid
            grid.data[int(creature.position[0]), int(creature.position[1]), 0] = creature.id
            
        # Verify no duplicates
        duplicates = []
        position_counts = {}
        for creature in self.creatures:
            pos = (int(creature.position[0]), int(creature.position[1]))
            if pos in position_counts:
                position_counts[pos].append(creature.id)
            else:
                position_counts[pos] = [creature.id]
                
        for pos, creature_ids in position_counts.items():
            if len(creature_ids) > 1:
                duplicates.append((pos, creature_ids))
                
        if duplicates:
            print(f"WARNING: Found {len(duplicates)} positions with multiple creatures after initialization:")
            for pos, creature_ids in duplicates:
                print(f"  Position {pos} has creatures: {creature_ids}")
                
                # Fix duplicates by moving all but one creature to a new position
                first_creature_id = creature_ids[0]
                for creature_id in creature_ids[1:]:
                    for creature in self.creatures:
                        if creature.id == creature_id:
                            # Find a new empty location
                            try:
                                new_pos = grid.find_empty_location()
                                # Update grid
                                grid.data[pos[0], pos[1], 0] = first_creature_id
                                grid.data[int(new_pos[0]), int(new_pos[1]), 0] = creature_id
                                # Update creature position
                                creature.position = new_pos
                                print(f"    Moved creature {creature_id} to {new_pos}")
                            except Exception as e:
                                print(f"    Failed to move creature {creature_id}: {e}")

    def natural_selection_tournament(self, grid):
        """
        Perform natural selection based on survival criteria

        Args:
            grid: The world grid

        Returns:
            List of offspring creatures for next generation
        """
        if len(self.creatures) == 0:
            # Create a new random population
            self.initialize(grid)
            return self.creatures

        # Create survival criteria checker
        criteria = SurvivalCriteria(self.params, grid)

        # Get survivors based on challenge type
        survivors = []
        for creature in self.creatures:
            # Get the result from the criterion check
            result = criteria.check_criterion(creature, self.params['challenge'])
            passed, score = result
            
            # Check if the creature is in a safe zone based on the challenge type
            in_safe_zone = False
            if self.params['challenge'] == 0:  # CHALLENGE_CIRCLE
                # Circle in the top-left quadrant
                center_x = self.params['world_size'][0] // 4
                center_y = self.params['world_size'][1] // 4
                radius = self.params['world_size'][0] // 4
                dx = creature.position[0] - center_x
                dy = creature.position[1] - center_y
                distance = math.sqrt(dx * dx + dy * dy)
                in_safe_zone = distance <= radius
            elif self.params['challenge'] == 1:  # CHALLENGE_RIGHT_HALF
                # Right half of the arena
                in_safe_zone = creature.position[0] > self.params['world_size'][0] // 2
            elif self.params['challenge'] == 4:  # CHALLENGE_CENTER_WEIGHTED
                # Circle in the center
                center_x = self.params['world_size'][0] // 2
                center_y = self.params['world_size'][1] // 2
                radius = self.params['world_size'][0] // 3
                dx = creature.position[0] - center_x
                dy = creature.position[1] - center_y
                distance = math.sqrt(dx * dx + dy * dy)
                in_safe_zone = distance <= radius
            elif self.params['challenge'] == 5:  # CHALLENGE_CORNER
                # Corners of the arena
                radius = self.params['world_size'][0] // 8
                corners = [
                    (0, 0),
                    (0, self.params['world_size'][1] - 1),
                    (self.params['world_size'][0] - 1, 0),
                    (self.params['world_size'][0] - 1, self.params['world_size'][1] - 1)
                ]
                
                for corner in corners:
                    dx = creature.position[0] - corner[0]
                    dy = creature.position[1] - corner[1]
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance <= radius:
                        in_safe_zone = True
                        break
            elif self.params['challenge'] == 10:  # CHALLENGE_RADIOACTIVE_WALLS
                # For radioactive walls challenge, check if creature is far enough from the radioactive wall
                x = int(creature.position[0])
                world_width = self.params['world_size'][0]
                
                # Determine which wall is radioactive based on current step
                current_step = self.params.get('current_step', 0)
                steps_per_generation = self.params.get('steps_per_generation', 100)
                radioactive_x = 0 if current_step < steps_per_generation / 2 else world_width - 1
                
                # Calculate distance from radioactive wall
                distance = abs(x - radioactive_x)
                
                # Only creatures that are far enough from the radioactive wall are in the safe zone
                in_safe_zone = distance >= world_width / 3  # Increased from 1/4 to 1/3 to match survival_criteria.py
            elif self.params['challenge'] == 8:  # CHALLENGE_CENTER_SPARSE
                # For center sparse challenge, check if creature is near center with specific neighbor count
                # This should match the logic in survival_criteria.py
                safe_center = (self.params['world_size'][0] // 2, self.params['world_size'][1] // 2)
                outer_radius = self.params['world_size'][0] // 4
                inner_radius = 1.5
                min_neighbors = 5  # Includes self
                max_neighbors = 8
                
                # Check if within outer radius of center
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
                                
                                if grid.data[nx, ny, 0] > 0:  # Contains a creature
                                    count += 1
                    
                    # Check if the neighbor count is within the required range
                    in_safe_zone = min_neighbors <= count <= max_neighbors
                else:
                    in_safe_zone = False
            else:
                # For other challenges, use the passed result directly
                in_safe_zone = passed
            
            # Only count creatures that pass the criterion, are in a safe zone,
            # and have valid neural connections
            if passed and in_safe_zone and creature.brain.connections:
                survivors.append((creature, score))
        
        # If no survivors but creatures are alive, check if they're in the right half
        if not survivors:
            alive_count = sum(1 for c in self.creatures if c.alive)
            if alive_count > 0:
                for creature in self.creatures:
                    if creature.alive and creature.position[0] > self.params['world_size'][0] / 2:
                        # This creature should have survived - it's in the right half
                        survivors.append((creature, 1.0))
        
        # Sort survivors by score (highest first)
        survivors.sort(key=lambda x: x[1], reverse=True)

        # Special case for altruism challenge
        if self.params['challenge'] == CHALLENGE_ALTRUISM:
            sacrifices = []
            for creature in self.creatures:
                passed, score = criteria.check_criterion(
                    creature, CHALLENGE_ALTRUISM_SACRIFICE)
                if passed and creature.brain.connections:
                    sacrifices.append((creature, score))

            # Handle kinship-based reproduction
            if self.generation > 10 and sacrifices:  # Apply after 10 generations
                altruism_factor = 10  # How many saved per sacrifice

                # Find genetically similar creatures to sacrifices
                saved_kin = []
                for sacrifice, _ in sacrifices:
                    # For each sacrifice, find related survivors
                    for _ in range(altruism_factor):
                        most_similar = None
                        highest_similarity = 0.7  # Threshold for kinship

                        for survivor, survivor_score in survivors:
                            similarity = calculate_genetic_similarity(
                                sacrifice.genome, survivor.genome)

                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                most_similar = (survivor, survivor_score)

                        if most_similar:
                            saved_kin.append(most_similar)

                # Replace survivors with saved kin
                if saved_kin:
                    survivors = saved_kin

        # Create offspring population
        offspring = []

        # Make sure we have survivors
        if not survivors:
            # Instead of creating a random population, select the top-scoring creatures
            # based on the fallback scoring mechanism in SurvivalCriteria.check_criterion
            fallback_survivors = []
            criteria = SurvivalCriteria(self.params, grid)
            
            for creature in self.creatures:
                if creature.alive:
                    # Get the fallback score (the second element in the tuple)
                    _, score = criteria.check_criterion(creature, self.params['challenge'])
                    # Only add if the creature has a valid brain with connections
                    if hasattr(creature.brain, 'connections') and creature.brain.connections:
                        fallback_survivors.append((creature, score))
            
            # Sort by score (highest first)
            fallback_survivors.sort(key=lambda x: x[1], reverse=True)
            
            # Take the top 20% as survivors
            survivor_count = max(10, int(len(fallback_survivors) * 0.2))
            survivors = fallback_survivors[:survivor_count]
            
            # If we still have no survivors, create a random population
            if not survivors:
                return [Creature(genome=Genome(length=self.params['genome_length'], params=self.params),
                               params=self.params) for _ in range(self.size)]

        # Apply elitism - keep top performers
        elite_count = max(1, int(self.size * 0.10))  # Save top 10%
        elites = [s[0] for s in survivors[:elite_count]]

        # Add elite clones to offspring
        for elite in elites:
            genome_copy = Genome([Gene(g.hex_value, params=self.params) for g in elite.genome.genes], params=self.params)
            child = Creature(genome=genome_copy, params=self.params)
            offspring.append(child)

        # Fill the rest with crossover and mutation
        while len(offspring) < self.size:
            # Select parents using tournament selection
            parent1 = self.tournament_selection(survivors)
            parent2 = self.tournament_selection(survivors)

            # Ensure different parents when possible
            attempts = 0
            while parent1 == parent2 and len(survivors) > 1 and attempts < 3:
                parent2 = self.tournament_selection(survivors)
                attempts += 1

            # Create child through crossover and mutation
            child_genome = parent1.genome.crossover(parent2.genome)
            child_genome.mutate(self.params['mutation_rate'])

            # Create child creature with parent information
            child = Creature(
                genome=child_genome, 
                params=self.params,
                parent1_id=parent1.id,
                parent2_id=parent2.id
            )
            offspring.append(child)

        # Update generation counter
        self.generation += 1

        # Export parent genome information
        parent_creatures = [creature for creature, _ in survivors]
        self.export_parent_genomes(parent_creatures)
        
        # Count unique parents that contributed to reproduction
        # This is a set of parent IDs that were used for reproduction
        reproducer_ids = set()
        
        # We need to track which parents were actually used for reproduction
        # This happens in the tournament selection process
        # For simplicity, we'll consider all survivors as potential reproducers
        # and count the actual number of unique parents used
        
        # Count survivors
        survivors_count = len(survivors)
        
        # For reproduction count, we'll use the number of unique parents
        # that contributed to the offspring (excluding elites)
        reproduction_count = len(parent_creatures)
        
        print(f"Generation {self.generation}: {survivors_count} survivors, {reproduction_count} reproducers")
        
        return offspring, survivors_count, reproduction_count

    def tournament_selection(self, scored_creatures, tournament_size=5):
        """
        Select a creature using tournament selection

        Args:
            scored_creatures: List of (creature, score) tuples
            tournament_size: Number of creatures in tournament

        Returns:
            Selected creature
        """
        # Select random contestants
        if len(scored_creatures) <= tournament_size:
            contestants = scored_creatures
        else:
            contestants = random.sample(scored_creatures, tournament_size)

        # Return creature with highest score
        return max(contestants, key=lambda x: x[1])[0]

    def update(self, grid, signals=None):
        """
        Update all creatures in the population
        
        This method calls move() on each creature, which queues movement
        requests that will be processed at the end of the step.
        """
        # Create a dictionary of creatures for quick lookup
        creatures_dict = {creature.id: creature for creature in self.creatures}
        
        # The main creature update logic (sensors, feed-forward, movement queuing)
        # is now handled in Simulator.update calling creature.update.
        # This loop is likely redundant and was causing the TypeError.
        # for creature in self.creatures:
        #     if creature.alive and creature.age <= self.params['max_age']:
        #         creature.move(grid, self.creatures, signals)
        
        # Process queued operations
        grid.process_death_queue(creatures_dict)
        grid.process_move_queue(creatures_dict)

    # Helper method to cluster similar genomes
    def cluster_similar_genomes(self, creatures, similarity_threshold=0.8):
        clusters = []
        unclustered = creatures.copy()

        while unclustered:
            # Start a new cluster with the first creature
            current = unclustered.pop(0)
            current_cluster = [current]

            # Find all similar creatures
            i = 0
            while i < len(unclustered):
                similarity = calculate_genetic_similarity(
                    current.genome, unclustered[i].genome
                )
                if similarity > similarity_threshold:
                    current_cluster.append(unclustered.pop(i))
                else:
                    i += 1

            clusters.append(current_cluster)

        return clusters

    def export_parent_genomes(self, reproducers):
        """
        Export detailed information about parent genomes that survived and reproduced.

        This method provides a comprehensive log of parent creatures, capturing
        key genomic, neural, and behavioral characteristics.

        Args:
            reproducers (list): List of creatures selected for reproduction
        """
        # Validate logging configuration
        if not self.params.get('log_to_csv', False):
            return

        # Ensure log directory exists
        log_folder = self.params.get('log_folder', 'logs')
        os.makedirs(log_folder, exist_ok=True)

        # Generate unique timestamp if not already present
        if not hasattr(self, "timestamp"):
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct detailed filename
        genome_filename = os.path.join(
            log_folder,
            f"parent_genomes_gen{self.generation}_{self.timestamp}.csv"
        )

        try:
            with open(genome_filename, 'w', newline='') as file:
                writer = csv.writer(file)

                # Comprehensive header capturing multiple aspects of parent creatures
                header = [
                    'Parent_ID',
                    'Generation',
                    'In_Safe_Zone',
                    'Energy',
                    'Age',
                    'Brain_Active_Neurons',
                    'Brain_Total_Connections',
                    'Genome_Length',
                    'Genetic_Complexity_Score',
                    'Parent1_ID',
                    'Parent2_ID'
                ]

                # Add detailed gene information to header
                max_genes = max(len(parent.genome.genes) for parent in reproducers) if reproducers else 0
                for i in range(max_genes):
                    header.extend([
                        f'Gene_{i}_Hex_Value',
                        f'Gene_{i}_Source_Type',
                        f'Gene_{i}_Sink_Type',
                        f'Gene_{i}_Weight'
                    ])

                writer.writerow(header)

                # Log details for each reproducing parent
                for parent in reproducers:
                    # Compute genetic complexity score
                    try:
                        complexity_score = self._compute_genome_complexity_score(parent.genome)
                    except Exception:
                        complexity_score = 0.0

                    # Prepare base row with parent information
                    row = [
                        parent.id,
                        self.generation,
                        int(parent.in_safe_zone),  # Convert boolean to integer
                        f"{parent.energy:.2f}",
                        0,  # Age removed, using 0 as placeholder
                        parent.brain.active_internal_neurons,
                        parent.brain.total_connections,
                        len(parent.genome.genes),
                        f"{complexity_score:.4f}",
                        parent.parent1_id if hasattr(parent, 'parent1_id') and parent.parent1_id is not None else "None",
                        parent.parent2_id if hasattr(parent, 'parent2_id') and parent.parent2_id is not None else "None"
                    ]

                    # Add detailed gene information
                    for gene in parent.genome.genes:
                        row.extend([
                            gene.hex_value,
                            gene.source_type,
                            gene.sink_type,
                            f"{gene.weight:.4f}"
                        ])

                    writer.writerow(row)

        except Exception as e:
            pass

    def _compute_genome_complexity_score(self, genome):
        """
        Compute a complexity score for a genome based on various characteristics.

        Args:
            genome (Genome): Genome to analyze

        Returns:
            float: Complexity score
        """
        try:
            # Use existing genome analysis function
            complexity = analyze_genome_complexity(genome)

            # Combine multiple complexity metrics
            complexity_score = (
                    complexity['gene_diversity'] * 0.3 +
                    complexity['computational_capacity'] * 0.3 +
                    complexity['recurrent_ratio'] * 0.2 +
                    (len(genome.genes) / self.params.get('genome_max_length', 300)) * 0.2
            )

            return complexity_score
        except Exception:
            # Fallback if computation fails
            return 0.0

    def analyze_population(self):
        """
        Analyze the current population and return statistics

        Returns:
            Dictionary with population statistics
        """
        if not self.creatures:
            return {"population_size": 0}

        # Extract genome stats for all creatures
        genome_stats = []
        for creature in self.creatures:
            genome_stats.append(analyze_genome_complexity(creature.genome))

        # Calculate population-level statistics
        avg_gene_diversity = sum(stats["gene_diversity"] for stats in genome_stats) / len(genome_stats)
        avg_computational_capacity = sum(stats["computational_capacity"] for stats in genome_stats) / len(genome_stats)
        avg_recurrent_ratio = sum(stats["recurrent_ratio"] for stats in genome_stats) / len(genome_stats)

        # Calculate metrics about behavioral diversity
        energy_values = [c.energy for c in self.creatures]
        neuron_counts = [c.brain.active_internal_neurons for c in self.creatures]
        connection_counts = [c.brain.total_connections for c in self.creatures]
        zone_distribution = sum(1 for c in self.creatures if c.in_safe_zone) / len(self.creatures)

        # Calculate genetic diversity
        genetic_diversity = calculate_genetic_diversity(self.creatures)

        # Identify most successful genomes
        creatures_by_energy = sorted(self.creatures, key=lambda c: c.energy, reverse=True)
        top_creature = creatures_by_energy[0] if creatures_by_energy else None

        return {
            "population_size": len(self.creatures),
            "avg_gene_diversity": avg_gene_diversity,
            "avg_computational_capacity": avg_computational_capacity,
            "avg_recurrent_ratio": avg_recurrent_ratio,
            "genetic_diversity": genetic_diversity,
            "zone_distribution": zone_distribution,
            "avg_energy": sum(energy_values) / len(energy_values) if energy_values else 0,
            "min_energy": min(energy_values) if energy_values else 0,
            "max_energy": max(energy_values) if energy_values else 0,
            "avg_neurons": sum(neuron_counts) / len(neuron_counts) if neuron_counts else 0,
            "avg_connections": sum(connection_counts) / len(connection_counts) if connection_counts else 0,
            "top_creature_id": top_creature.id if top_creature else None,
            "top_creature_energy": top_creature.energy if top_creature else 0,
            "top_creature_metrics": analyze_genome_complexity(top_creature.genome) if top_creature else None
        }
