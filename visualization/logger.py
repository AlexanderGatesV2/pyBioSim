import os
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
from utils.genetic import calculate_genetic_diversity # Import the function


class Logger:
    def __init__(self, params):
        """
        Initialize the logger for recording simulation statistics.
        
        Args:
            params: Simulation parameters containing logging configuration
        """
        self.params = params
        self.log_folder = params['log_folder']
        self.file = None
        self.writer = None

        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_folder):
            try:
                os.makedirs(self.log_folder)
            except Exception as e:
                print(f"Failed to create log folder '{self.log_folder}': {e}")

        # Create file and write header
        if params['log_to_csv']:
            try:
                # Create unique filename with timestamp
                self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.filename = os.path.join(self.log_folder, f"evolution_log_{self.timestamp}.csv")

                self.file = open(self.filename, 'w', newline='')
                self.writer = csv.writer(self.file)
                self.writer.writerow([
                    'Generation',
                    'Population',
                    'Creatures_In_Safe_Zone',
                    'Safe_Zone_Percentage',
                    'Min_Energy',
                    'Max_Energy',
                    'Avg_Energy',
                    'Avg_Internal_Neurons_Used',
                    'Avg_Connections_Used',
                    'Genetic_Diversity',
                    'Kills',
                    'Selection_Method'
                ])
            except Exception as e:
                print(f"Failed to create CSV log file: {e}")
                self.file = None
                self.writer = None

    def log_generation(self, generation, creatures, murder_count=0):
        """
        Log statistics about the current generation to CSV.
        
        Records population metrics including safe zone percentages,
        energy statistics, neural network complexity, and genetic diversity.
        
        Args:
            generation: Current generation number
            creatures: List of creatures in the current generation
            murder_count: Number of creatures killed in this generation
        """
        if not self.params['log_to_csv'] or self.writer is None or not creatures:
            return

        # Skip logging some generations to reduce I/O overhead
        # Only log every 5th generation or generation 1 for performance
        if generation > 1 and generation % 5 != 0:
            return

        # Calculate statistics
        population = len(creatures)
        safe_creatures = [c for c in creatures if c.in_safe_zone]
        safe_percentage = (len(safe_creatures) / population * 100) if population > 0 else 0

        # Calculate average number of internal neurons used
        neuron_counts = [c.brain.active_internal_neurons for c in creatures]
        avg_neurons = sum(neuron_counts) / len(neuron_counts) if neuron_counts else 0

        # Calculate average number of connections used
        connection_counts = [c.brain.total_connections for c in creatures]
        avg_connections = sum(connection_counts) / len(connection_counts) if connection_counts else 0

        # Calculate genetic diversity using pairwise similarity
        genetic_diversity = calculate_genetic_diversity(creatures)

        # Count kills
        kills = sum(1 for c in creatures if c.has_killed)

        # Calculate energy statistics
        energies = [c.energy for c in creatures]
        min_energy = min(energies) if energies else 0
        max_energy = max(energies) if energies else 0
        avg_energy = sum(energies) / len(energies) if energies else 0

        # Write data to CSV
        self.writer.writerow([
            generation,
            population,
            len(safe_creatures),
            f"{safe_percentage:.2f}",
            f"{min_energy:.2f}",
            f"{max_energy:.2f}",
            f"{avg_energy:.2f}",
            f"{avg_neurons:.2f}",
            f"{avg_connections:.2f}",
            f"{genetic_diversity:.3f}",
            kills,
            self.params['selection_method']
        ])
        # Flush to ensure data is written
        self.file.flush()

    def close(self):
        """
        Close the log file to ensure all data is saved.
        
        Should be called when the simulation ends.
        """
        if self.file:
            self.file.close()

    def plot_statistics(self):
        """
        Plot statistics from the log file.
        
        Creates a figure with four subplots showing:
        - Safe zone percentage over generations
        - Average internal neurons used over generations
        - Average connections used over generations
        - Genetic diversity over generations
        
        Saves the figure to the log folder.
        """
        if not os.path.exists(self.filename):
            print("Log file not found")
            return

        # Read data from CSV
        data = []
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                data.append(row)

        if not data:
            print("No data to plot")
            return

        # Convert data to numpy arrays
        generations = np.array([int(row[0]) for row in data])
        safe_percentages = np.array([float(row[3]) for row in data])
        avg_neurons = np.array([float(row[7]) for row in data])
        avg_connections = np.array([float(row[8]) for row in data])
        genetic_diversity = np.array([float(row[9]) for row in data])

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot safe zone percentage
        plt.subplot(2, 2, 1)
        plt.plot(generations, safe_percentages)
        plt.title('Safe Zone Percentage')
        plt.xlabel('Generation')
        plt.ylabel('Percentage')

        # Plot average neurons
        plt.subplot(2, 2, 2)
        plt.plot(generations, avg_neurons)
        plt.title('Average Internal Neurons Used')
        plt.xlabel('Generation')
        plt.ylabel('Count')

        # Plot average connections
        plt.subplot(2, 2, 3)
        plt.plot(generations, avg_connections)
        plt.title('Average Connections Used')
        plt.xlabel('Generation')
        plt.ylabel('Count')

        # Plot genetic diversity
        plt.subplot(2, 2, 4)
        plt.plot(generations, genetic_diversity)
        plt.title('Genetic Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Diversity (0-1)')

        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.log_folder, f"statistics_{self.timestamp}.png"))
        plt.close()
