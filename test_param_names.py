import json
import os
from core.params import load_parameters

# Create a temporary config file with old camelCase parameter names
config = {
    'responsivenessCurveKFactor': 5,
    'longProbeDistance': 20,
    'populationSensorRadius': 3.0,
    'geneInsertionDeletionRate': 0.02
}

temp_config_path = 'temp_config.json'

with open(temp_config_path, 'w') as f:
    json.dump(config, f)

# Load the parameters
params = load_parameters(temp_config_path)

# Check if the old parameter names are recognized
print('Old Parameter Name Test:')
print(f'responsiveness_curve_k_factor: {params.get("responsiveness_curve_k_factor", "Not found")}')
print(f'longprobe_dist: {params.get("longprobe_dist", "Not found")}')
print(f'population_sensor_radius: {params.get("population_sensor_radius", "Not found")}')
print(f'genes_insertion_deletion_rate: {params.get("genes_insertion_deletion_rate", "Not found")}')

# Check if the original camelCase names are in the params
print('\nOriginal Parameter Names in Params:')
print(f'responsivenessCurveKFactor: {params.get("responsivenessCurveKFactor", "Not found")}')
print(f'longProbeDistance: {params.get("longProbeDistance", "Not found")}')
print(f'populationSensorRadius: {params.get("populationSensorRadius", "Not found")}')
print(f'geneInsertionDeletionRate: {params.get("geneInsertionDeletionRate", "Not found")}')

# Clean up
os.remove(temp_config_path)
