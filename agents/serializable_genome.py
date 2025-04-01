import logging
from utils.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__, log_file='serializable_genome_debug.log', console_level=logging.CRITICAL, file_level=logging.CRITICAL)

class SerializableGene:
    """
    A serializable representation of a Gene that can be pickled for multiprocessing.
    
    This class contains only the essential data needed for gene processing in parallel processes.
    """
    
    def __init__(self, gene=None):
        """
        Initialize a serializable gene, optionally from an existing Gene.
        
        Args:
            gene: Optional Gene object to convert from
        """
        # Gene data
        self.hex_value = '00000000'
        self.source_type = 0
        self.source_id = 0
        self.sink_type = 0
        self.sink_id = 0
        self.weight = 0.0
        
        # If a gene is provided, convert it
        if gene is not None:
            self.from_gene(gene)
    
    def from_gene(self, gene):
        """
        Convert a Gene object to a SerializableGene.
        
        Args:
            gene: The Gene object to convert
            
        Returns:
            self: The updated SerializableGene object
        """
        try:
            # Copy gene data
            self.hex_value = gene.hex_value
            self.source_type = gene.source_type
            self.source_id = gene.source_id
            self.sink_type = gene.sink_type
            self.sink_id = gene.sink_id
            self.weight = gene.weight
            
            return self
        except Exception as e:
            logger.error(f"Error converting Gene to SerializableGene: {e}", exc_info=True)
            raise
    
    def to_gene(self, params=None):
        """
        Convert a SerializableGene back to a Gene.
        
        Args:
            params: Optional simulation parameters
            
        Returns:
            Gene: A new Gene object with the properties from this SerializableGene
        """
        try:
            from agents.genome import Gene
            
            # Create a new Gene with the hex value and params
            gene = Gene(self.hex_value, params)
            
            # Update gene fields
            gene.source_type = self.source_type
            gene.source_id = self.source_id
            gene.sink_type = self.sink_type
            gene.sink_id = self.sink_id
            gene.weight = self.weight
            
            return gene
        except Exception as e:
            logger.error(f"Error converting SerializableGene to Gene: {e}", exc_info=True)
            raise


class SerializableGenome:
    """
    A serializable representation of a Genome that can be pickled for multiprocessing.
    
    This class contains only the essential data needed for genome processing in parallel processes.
    It does not contain complex objects like functions or class methods, which are replaced
    with serializable representations.
    """
    
    def __init__(self, genome=None):
        """
        Initialize a serializable genome, optionally from an existing Genome.
        
        Args:
            genome: Optional Genome object to convert from
        """
        # Genome parameters
        self.genome_initial_length_min = 24
        self.genome_initial_length_max = 48
        self.genome_max_length = 300
        self.genes_insertion_deletion_rate = 0.0
        self.deletion_ratio = 0.5
        
        # Serializable genes
        self.serializable_genes = []
        
        # If a genome is provided, convert it
        if genome is not None:
            self.from_genome(genome)
    
    def from_genome(self, genome):
        """
        Convert a Genome object to a SerializableGenome.
        
        Args:
            genome: The Genome object to convert
            
        Returns:
            self: The updated SerializableGenome object
        """
        try:
            # Copy genome parameters
            self.genome_initial_length_min = genome.genome_initial_length_min
            self.genome_initial_length_max = genome.genome_initial_length_max
            self.genome_max_length = genome.genome_max_length
            self.genes_insertion_deletion_rate = genome.genes_insertion_deletion_rate
            self.deletion_ratio = genome.deletion_ratio
            
            # Convert genes to serializable genes
            self.serializable_genes = []
            for gene in genome.genes:
                self.serializable_genes.append(SerializableGene(gene))
            
            return self
        except Exception as e:
            logger.error(f"Error converting Genome to SerializableGenome: {e}", exc_info=True)
            raise
    
    def to_genome(self, params=None):
        """
        Convert a SerializableGenome back to a Genome.
        
        Args:
            params: Optional simulation parameters
            
        Returns:
            Genome: A new Genome object with the properties from this SerializableGenome
        """
        try:
            from agents.genome import Genome
            
            # Convert serializable genes to regular genes
            genes = []
            for serializable_gene in self.serializable_genes:
                genes.append(serializable_gene.to_gene(params))
            
            # Create a new Genome with the genes and params
            genome = Genome(genes=genes, params=params)
            
            # Update genome parameters
            genome.genome_initial_length_min = self.genome_initial_length_min
            genome.genome_initial_length_max = self.genome_initial_length_max
            genome.genome_max_length = self.genome_max_length
            genome.genes_insertion_deletion_rate = self.genes_insertion_deletion_rate
            genome.deletion_ratio = self.deletion_ratio
            
            return genome
        except Exception as e:
            logger.error(f"Error converting SerializableGenome to Genome: {e}", exc_info=True)
            raise
    
    def update_from_genome(self, genome):
        """
        Update this SerializableGenome with the latest state from a Genome.
        
        Args:
            genome: The Genome object to update from
            
        Returns:
            self: The updated SerializableGenome object
        """
        try:
            # Update genome parameters
            self.genome_initial_length_min = genome.genome_initial_length_min
            self.genome_initial_length_max = genome.genome_initial_length_max
            self.genome_max_length = genome.genome_max_length
            self.genes_insertion_deletion_rate = genome.genes_insertion_deletion_rate
            self.deletion_ratio = genome.deletion_ratio
            
            # Update serializable genes
            self.serializable_genes = []
            for gene in genome.genes:
                self.serializable_genes.append(SerializableGene(gene))
            
            return self
        except Exception as e:
            logger.error(f"Error updating SerializableGenome from Genome: {e}", exc_info=True)
            raise
