import os
import csv
import datetime

class CreatureLineageLogger:
    """
    Logger for tracking parent-child relationships between creatures.
    
    This class provides functionality to log the lineage of creatures across generations,
    making it possible to trace the ancestry of any creature in the simulation.
    """
    
    def __init__(self, params):
        """
        Initialize the lineage logger.
        
        Args:
            params: Simulation parameters containing logging configuration
        """
        self.params = params
        self.log_folder = params.get('log_folder', 'evolution_logs')
        self.file = None
        self.writer = None
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_folder):
            try:
                os.makedirs(self.log_folder)
            except Exception as e:
                print(f"Failed to create log folder '{self.log_folder}': {e}")
        
        # Create lineage log file
        if params.get('log_to_csv', False):
            try:
                self.filename = os.path.join(self.log_folder, f"creature_lineage_{self.timestamp}.csv")
                self.file = open(self.filename, 'w', newline='')
                self.writer = csv.writer(self.file)
                self.writer.writerow([
                    'Generation',
                    'Creature_ID',
                    'Parent1_ID',
                    'Parent2_ID',
                    'In_Safe_Zone',
                    'Energy',
                    'Brain_Neurons',
                    'Brain_Connections'
                ])
            except Exception as e:
                print(f"Failed to create lineage log file: {e}")
                self.file = None
                self.writer = None
    
    def log_generation(self, generation, creatures):
        """
        Log the lineage information for a generation of creatures.
        
        Args:
            generation: Current generation number
            creatures: List of creatures in the current generation
        """
        if not self.params.get('log_to_csv', False) or self.writer is None:
            return
        
        for creature in creatures:
            # Get parent IDs, defaulting to "None" if not available
            parent1_id = creature.parent1_id if hasattr(creature, 'parent1_id') and creature.parent1_id is not None else "None"
            parent2_id = creature.parent2_id if hasattr(creature, 'parent2_id') and creature.parent2_id is not None else "None"
            
            # Write creature lineage information
            self.writer.writerow([
                generation,
                creature.id,
                parent1_id,
                parent2_id,
                int(creature.in_safe_zone),  # Convert boolean to integer
                f"{creature.energy:.2f}",
                creature.brain.active_internal_neurons,
                creature.brain.total_connections
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
