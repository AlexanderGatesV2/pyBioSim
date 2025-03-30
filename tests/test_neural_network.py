import unittest
import numpy as np
import math
from agents.neural_network import NeuralNetwork
from agents.genome import Genome, Gene
from core.params import load_parameters
from core.types import Sensor, Action

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.params = load_parameters()
        self.params['num_sensory_neurons'] = 24
        self.params['num_internal_neurons'] = 20
        self.params['num_output_neurons'] = 17
        
        # Create a sample genome
        genes = [
            Gene(hex_value="00000000"),  # Sensor 0 -> Neuron 0, weight 0
            Gene(hex_value="01010101"),  # Sensor 1 -> Neuron 1, weight 0.000122...
            Gene(hex_value="80808080"),  # Neuron 0 -> Action 0, weight -1.0
            Gene(hex_value="81818181"),  # Neuron 1 -> Action 1, weight -0.984...
            Gene(hex_value="42424242"),  # Sensor 2 -> Neuron 2, weight 0.516...
            Gene(hex_value="C3C3C3C3"),  # Neuron 3 -> Action 3, weight -0.472...
        ]
        self.genome = Genome(genes=genes, params=self.params)
        
        # Create neural network
        self.nn = NeuralNetwork(self.genome, self.params)
    
    def test_feed_forward_basic(self):
        """Test basic feed-forward operation"""
        sensory_inputs = np.zeros(self.params['num_sensory_neurons'])
        sensory_inputs[0] = 0.5
        sensory_inputs[1] = 1.0
        sensory_inputs[2] = 0.2
        
        action_values, state_updates = self.nn.feed_forward(sensory_inputs, sim_step=0)
        
        # Check action values (these are approximate based on the weights)
        # Action 0 should be influenced by Neuron 0 (driven by Sensor 0)
        # Action 1 should be influenced by Neuron 1 (driven by Sensor 1)
        # Action 3 should be influenced by Neuron 3 (not driven in this genome)
        
        # Neuron 0 input = Sensor 0 * weight = 0.5 * 0 = 0
        # Neuron 0 output = tanh(0) = 0
        # Action 0 input = Neuron 0 * weight = 0 * (-1.0 / 8192.0) = 0
        # Action 0 output = (tanh(0) + 1) / 2 = 0.5
        self.assertAlmostEqual(action_values[0], 0.5, places=5)
        
        # Neuron 1 input = Sensor 1 * weight = 1.0 * (257 / 8192.0) = 0.03137...
        # Neuron 1 output = tanh(0.03137...) = 0.03136...
        # Action 1 input = Neuron 1 * weight = 0.03136... * (-32383 / 8192.0) = -0.1239...
        # Action 1 output = (tanh(-0.1239...) + 1) / 2 = (-0.1233... + 1) / 2 = 0.4383...
        self.assertAlmostEqual(action_values[1], 0.4383, places=3) # Reverted expected value
        
        # Neuron 3 (remapped to 2) is not driven, so its output remains 0.5
        # Weight: int('C3C3', 16) = 50115. Negative: 50115 - 65536 = -15421
        # Sink Num: 67 % 17 = 16. Connection is N3 -> A16
        # Action 16 input = Neuron 2 * weight = 0.5 * (-15421 / 8192.0) = -0.941...
        # Action 16 output = (tanh(-0.941...) + 1) / 2 = (-0.735... + 1) / 2 = 0.132...
        # Note: This depends on the initial neuron output value (0.5)
        self.assertAlmostEqual(action_values[16], 0.132, places=3) # Check Action 16
    
    def test_responsiveness_curve(self):
        """Test the responsiveness curve calculation"""
        # Test with responsiveness = 0.5 (mid-level)
        r = 0.5
        k = self.params['responsiveness_curve_k_factor']
        expected = math.pow((r - 2.0), -2.0 * k) - math.pow(2.0, -2.0 * k) * (1.0 - r)
        calculated = self.nn._apply_responsiveness_curve(r)
        self.assertAlmostEqual(calculated, expected, places=5)
        
        # Test with responsiveness = 1.0 (max)
        r = 1.0
        expected = math.pow((r - 2.0), -2.0 * k) - math.pow(2.0, -2.0 * k) * (1.0 - r)
        calculated = self.nn._apply_responsiveness_curve(r)
        self.assertAlmostEqual(calculated, expected, places=5)
        
        # Test with responsiveness = 0.0 (min)
        r = 0.0
        expected = math.pow((r - 2.0), -2.0 * k) - math.pow(2.0, -2.0 * k) * (1.0 - r)
        calculated = self.nn._apply_responsiveness_curve(r)
        self.assertAlmostEqual(calculated, expected, places=5)
    
    def test_post_process_actions(self):
        """Test the post-processing of actions"""
        action_values = np.zeros(self.params['num_output_neurons'])
        action_values[Action.SET_RESPONSIVENESS.value] = 0.7  # Raw value, will be scaled
        action_values[Action.SET_OSCILLATOR_PERIOD.value] = 0.2 # Raw value
        action_values[Action.EMIT_SIGNAL0.value] = 0.8 # Raw value
        action_values[Action.KILL_FORWARD.value] = 0.9 # Raw value
        
        state_updates = self.nn._post_process_actions(action_values)
        
        # Check responsiveness
        self.assertAlmostEqual(state_updates['responsiveness'], 0.7, places=5)
        
        # Check oscillator period
        period_01 = (math.tanh(0.2) + 1.0) / 2.0
        expected_period = 1 + int(1.5 + math.exp(7.0 * period_01))
        self.assertEqual(state_updates['oscPeriod'], expected_period)
        
        # Check signal emission (probabilistic)
        # We can't test the exact outcome, but check if the key exists
        # when the level is above threshold
        adjusted_responsiveness = self.nn._apply_responsiveness_curve(0.7)
        emit_level = 0.8 * adjusted_responsiveness
        if emit_level > 0.5:
            # It's possible 'emit_signal0' is not set due to randomness
            pass
        else:
            self.assertNotIn('emit_signal0', state_updates)
            
        # Check kill attempt (probabilistic)
        kill_level = 0.9 * adjusted_responsiveness
        if kill_level > 0.5:
            # It's possible 'attempt_kill' is not set due to randomness
            pass
        else:
            self.assertNotIn('attempt_kill', state_updates)

if __name__ == "__main__":
    unittest.main()
