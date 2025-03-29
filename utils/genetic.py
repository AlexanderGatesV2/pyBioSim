import random
import numpy as np
from agents.genome import Genome, Gene


def calculate_genetic_similarity(genome1, genome2):
    """
    Calculate genetic similarity between two genomes.
    
    Similarity is measured as the proportion of genes that match exactly
    between the two genomes.
    
    Args:
        genome1: First genome to compare
        genome2: Second genome to compare
        
    Returns:
        Float: Similarity score between 0.0 (no similarity) and 1.0 (identical)
    """
    if len(genome1.genes) != len(genome2.genes):
        return 0.0

    matching_genes = 0
    for i in range(len(genome1.genes)):
        if genome1.genes[i].hex_value == genome2.genes[i].hex_value:
            matching_genes += 1

    return matching_genes / len(genome1.genes)


def calculate_genetic_diversity(population):
    """
    Calculate genetic diversity in a population.
    
    Diversity is measured as 1 minus the average pairwise similarity
    between genomes in the population.
    
    Args:
        population: List of creatures to analyze
        
    Returns:
        Float: Diversity score between 0.0 (identical genomes) and 1.0 (maximum diversity)
    """
    if len(population) <= 1:
        return 0.0

    # Sample pairs for performance reasons if population is large
    num_samples = min(1000, len(population) * (len(population) - 1) // 2)
    total_similarity = 0.0

    if len(population) <= 50:  # For small populations, check all pairs
        count = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                total_similarity += calculate_genetic_similarity(
                    population[i].genome, population[j].genome)
                count += 1
        average_similarity = total_similarity / count if count > 0 else 0
    else:  # For larger populations, sample random pairs
        for _ in range(num_samples):
            i, j = random.sample(range(len(population)), 2)
            total_similarity += calculate_genetic_similarity(
                population[i].genome, population[j].genome)
        average_similarity = total_similarity / num_samples

    # Return diversity (1 - similarity)
    return 1.0 - average_similarity


def generate_child_genome(parent1, parent2, mutation_rate=0.01):
    """
    Generate a child genome from two parents with possible mutations.
    
    Args:
        parent1: First parent creature
        parent2: Second parent creature
        mutation_rate: Probability of mutation for each gene (default: 0.01)
        
    Returns:
        Genome: A new genome created by combining parent genomes with mutations
    """
    # Create child genome through crossover
    child_genome = parent1.genome.crossover(parent2.genome)

    # Apply mutations
    child_genome.mutate(mutation_rate)

    return child_genome


def analyze_genome_complexity(genome):
    """
    Analyze genome complexity and return comprehensive metrics.

    Calculates various metrics including gene diversity, connection types,
    and computational capacity estimates.

    Args:
        genome: The genome to analyze

    Returns:
        Dictionary with complexity metrics including:
        - length: Total number of genes
        - unique_genes: Number of unique gene hex values
        - gene_diversity: Proportion of unique genes
        - Connection type counts (sensory_to_internal, etc.)
        - recurrent_connections: Number of self-connections
        - computational_capacity: Estimated computational power
    """
    # Count unique genes
    unique_genes = set()
    for gene in genome.genes:
        unique_genes.add(gene.hex_value)

    # Count different connection types
    sensory_to_internal = 0
    sensory_to_output = 0
    internal_to_internal = 0
    internal_to_output = 0

    for gene in genome.genes:
        if gene.source_type == 0 and gene.sink_type == 0:
            sensory_to_internal += 1
        elif gene.source_type == 0 and gene.sink_type == 1:
            sensory_to_output += 1
        elif gene.source_type == 1 and gene.sink_type == 0:
            internal_to_internal += 1
        elif gene.source_type == 1 and gene.sink_type == 1:
            internal_to_output += 1

    # Count recurrent connections (self-connections)
    recurrent_connections = 0
    for gene in genome.genes:
        if gene.source_type == 1 and gene.sink_type == 0 and gene.source_id == gene.sink_id:
            recurrent_connections += 1

    # Calculate metrics
    gene_diversity = len(unique_genes) / len(genome.genes) if genome.genes else 0
    internal_connection_ratio = (internal_to_internal + internal_to_output) / len(genome.genes) if genome.genes else 0
    sensory_connection_ratio = (sensory_to_internal + sensory_to_output) / len(genome.genes) if genome.genes else 0
    recurrent_ratio = recurrent_connections / len(genome.genes) if genome.genes else 0

    # Estimate computational capacity based on connection patterns
    computational_capacity = (
                                     internal_to_internal * 2.0 +  # Internal-to-internal connections contribute most to computational power
                                     sensory_to_internal * 1.0 +  # Sensory-to-internal connections add capacity
                                     internal_to_output * 0.5  # Internal-to-output connections add less capacity
                             ) / len(genome.genes) if genome.genes else 0

    return {
        "length": len(genome.genes),
        "unique_genes": len(unique_genes),
        "gene_diversity": gene_diversity,
        "sensory_to_internal": sensory_to_internal,
        "sensory_to_output": sensory_to_output,
        "internal_to_internal": internal_to_internal,
        "internal_to_output": internal_to_output,
        "recurrent_connections": recurrent_connections,
        "recurrent_ratio": recurrent_ratio,
        "internal_connection_ratio": internal_connection_ratio,
        "sensory_connection_ratio": sensory_connection_ratio,
        "computational_capacity": computational_capacity
    }


def visualize_genome(genome, filename=None):
    """
    Create a visual representation of a genome as a heatmap.

    Generates a 2D heatmap showing the neural network connections
    encoded by the genome, with weights represented by colors.

    Args:
        genome: The genome to visualize
        filename: If provided, save visualization to this file
                 If None, display interactively if possible
                 
    Requires:
        matplotlib package for visualization
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a heatmap representation of the genome
        plt.figure(figsize=(12, 8))

        # Analyze genome for visualization
        metrics = analyze_genome_complexity(genome)

        # Create a 2D representation of genes
        # X axis: source type and ID
        # Y axis: sink type and ID
        # Color: weight
        max_neurons = 64  # Arbitrary max for visualization

        # Initialize connection matrix
        connection_matrix = np.zeros((2 * max_neurons, 2 * max_neurons))

        # Fill in connection matrix
        for gene in genome.genes:
            # Source coordinate (sensory=0-63, internal=64-127)
            source_x = gene.source_id % max_neurons
            if gene.source_type == 1:  # Internal
                source_x += max_neurons

            # Sink coordinate (internal=0-63, output=64-127)
            sink_y = gene.sink_id % max_neurons
            if gene.sink_type == 1:  # Output
                sink_y += max_neurons

            # Add weight (scaled to -1..1 range)
            connection_matrix[sink_y, source_x] = gene.weight

        # Create heatmap
        plt.imshow(connection_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Connection Weight')

        # Add labels
        plt.axvline(x=max_neurons - 0.5, color='black', linestyle='--')
        plt.axhline(y=max_neurons - 0.5, color='black', linestyle='--')

        plt.xticks([max_neurons // 2, max_neurons + max_neurons // 2], ['Sensory', 'Internal'])
        plt.yticks([max_neurons // 2, max_neurons + max_neurons // 2], ['Internal', 'Output'])

        # Add title with metrics
        plt.title(f"Genome Visualization\nLength: {len(genome.genes)}, Diversity: {metrics['gene_diversity']:.2f}\n"
                  f"Computational Capacity: {metrics['computational_capacity']:.2f}")

        # Add annotations for different connection types
        plt.annotate(f"S→I: {metrics['sensory_to_internal']}",
                     xy=(0.05, 0.05), xycoords='axes fraction')
        plt.annotate(f"S→O: {metrics['sensory_to_output']}",
                     xy=(0.05, 0.65), xycoords='axes fraction')
        plt.annotate(f"I→I: {metrics['internal_to_internal']}",
                     xy=(0.65, 0.05), xycoords='axes fraction')
        plt.annotate(f"I→O: {metrics['internal_to_output']}",
                     xy=(0.65, 0.65), xycoords='axes fraction')
        plt.annotate(f"Recurrent: {metrics['recurrent_connections']}",
                     xy=(0.4, 0.5), xycoords='axes fraction')

        plt.tight_layout()

        # Save or display
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("Genome visualization requires matplotlib package.")
