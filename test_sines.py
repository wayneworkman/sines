import unittest
import numpy as np
import pyopencl as cl
import os
import shutil
import tempfile
from unittest import mock
from unittest.mock import patch, MagicMock
from sines import (
    generate_sine_wave,
    load_data,
    load_previous_waves,
    refine_candidates,
    brute_force_sine_wave_search,
    setup_opencl
)
import json
import logging

class TestSines(unittest.TestCase):
    def setUp(self):
        # Set up OpenCL context and command queue
        platforms = cl.get_platforms()
        if platforms:
            devices = platforms[0].get_devices()
            if devices:
                self.context = cl.Context(devices=[devices[0]])
                self.queue = cl.CommandQueue(self.context)
            else:
                self.context = None
                self.queue = None
        else:
            self.context = None
            self.queue = None

        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    # Existing tests...
    def test_generate_sine_wave_basic(self):
        # Testing the basic generation of a sine wave
        params = {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        sine_wave = generate_sine_wave(params, num_points=10)
        self.assertEqual(len(sine_wave), 10)
        expected_wave = np.sin(2 * np.pi * 1 * np.arange(10) + 0)
        np.testing.assert_array_almost_equal(sine_wave, expected_wave, decimal=5)

    def test_generate_sine_wave_edge_cases(self):
        # Zero amplitude test
        params_zero_amplitude = {"amplitude": 0, "frequency": 1e-5, "phase_shift": 0}
        sine_wave_zero_amplitude = generate_sine_wave(params_zero_amplitude, num_points=10)
        self.assertTrue(np.allclose(sine_wave_zero_amplitude, 0, atol=1e-6))

        # Testing edge case with phase shift of 2*pi, should be same as no phase shift
        params_no_shift = {"amplitude": 1, "frequency": 1e-5, "phase_shift": 0}
        params_full_shift = {"amplitude": 1, "frequency": 1e-5, "phase_shift": 2 * np.pi}
        sine_wave_no_shift = generate_sine_wave(params_no_shift, num_points=10)
        sine_wave_full_shift = generate_sine_wave(params_full_shift, num_points=10)
        self.assertTrue(np.allclose(sine_wave_no_shift, sine_wave_full_shift, atol=1e-6))

    def test_load_data_valid(self):
        # Creating a temporary test file with valid data
        file_path = os.path.join(self.test_dir, "test_data.csv")
        with open(file_path, "w") as f:
            f.write("date,value\n2020-01-01,100\n2020-01-02,200\n")
        
        loaded_data = load_data(file_path, "date", "value")
        self.assertAlmostEqual(loaded_data[1], 200)

    def test_load_previous_waves_edge_cases(self):
        # Creating a sample previous waves file for loading
        wave_file_path = os.path.join(self.test_dir, "wave_1.json")
        wave_params = {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        with open(wave_file_path, "w") as f:
            json.dump(wave_params, f)

        loaded_waves = load_previous_waves(num_points=10, output_dir=self.test_dir)
        expected_wave = generate_sine_wave(wave_params, 10)
        np.testing.assert_array_almost_equal(loaded_waves, expected_wave, decimal=5)

    def test_refine_candidates_edge_cases(self):
        top_candidates = [
            ({"amplitude": 1.0, "frequency": 0.001, "phase_shift": 0.5}, 0.1),
            ({"amplitude": 1.5, "frequency": 0.002, "phase_shift": 0.3}, 0.2)
        ]
        observed_data = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        combined_wave = np.zeros(len(observed_data), dtype=np.float32)


        refined_best_params, refined_best_score = refine_candidates(
            top_candidates, observed_data, combined_wave, self.context, self.queue, None, wave_count=5, desired_refinement_step_size="fast"
        )
        self.assertIsInstance(refined_best_params, dict)
        self.assertTrue(isinstance(refined_best_score, np.floating))

    def test_brute_force_sine_wave_search_edge_cases(self):
        observed_data = np.array([0, 1, 0, -1] * 5, dtype=np.float32)
        combined_wave = np.zeros(len(observed_data), dtype=np.float32)

        candidates = brute_force_sine_wave_search(
            observed_data, combined_wave, self.context, self.queue, None, wave_count=5
        )
        self.assertIsInstance(candidates, list)

    def test_integration_load_generate_search_refine(self):
        params = {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        sine_wave = generate_sine_wave(params, num_points=10)
        observed_data = sine_wave + np.random.normal(0, 0.1, len(sine_wave)).astype(np.float32)
        combined_wave = np.zeros(len(observed_data), dtype=np.float32)

        candidates = brute_force_sine_wave_search(
            observed_data, combined_wave, self.context, self.queue, None, wave_count=5
        )

        refined_best_params, refined_best_score = refine_candidates(
            candidates, observed_data, combined_wave, self.context, self.queue, None, wave_count=5, desired_refinement_step_size="fine"
        )
        self.assertIsInstance(refined_best_params, dict)
        self.assertGreater(refined_best_score, 0)


    def test_load_data_json_valid(self):
        # Test loading data from a valid JSON file
        file_path = os.path.join(self.test_dir, "test_data.json")
        data = [
            {"date": "2020-01-01", "value": 100},
            {"date": "2020-01-02", "value": 200}
        ]
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        loaded_data = load_data(file_path, "date", "value")
        self.assertEqual(len(loaded_data), 2)
        self.assertAlmostEqual(loaded_data[0], 100)
        self.assertAlmostEqual(loaded_data[1], 200)

    def test_load_data_json_invalid_fallback(self):
        # Test loading data from an invalid JSON file, expecting fallback to CSV
        # First, create an invalid JSON file
        json_file_path = os.path.join(self.test_dir, "invalid_data.json")
        with open(json_file_path, "w") as f:
            f.write("{invalid json}")
        
        # Also create a valid CSV file with the same name (to simulate fallback)
        csv_file_path = os.path.join(self.test_dir, "invalid_data.json")
        with open(csv_file_path, "w") as f:
            f.write("date,value\n2020-01-01,100\n2020-01-02,200\n")
        
        loaded_data = load_data(json_file_path, "date", "value")
        self.assertEqual(len(loaded_data), 2)
        self.assertAlmostEqual(loaded_data[0], 100)
        self.assertAlmostEqual(loaded_data[1], 200)

    def test_load_data_json_invalid_no_csv_fallback(self):
        # Test loading data from an invalid JSON file without a valid CSV fallback
        json_file_path = os.path.join(self.test_dir, "invalid_data.json")
        with open(json_file_path, "w") as f:
            f.write("{invalid json}")
        
        # Ensure no corresponding CSV file exists
        with self.assertRaises(ValueError):
            load_data(json_file_path, "date", "value")

    def test_load_data_missing_value_col(self):
        # Test loading data with missing value column
        file_path = os.path.join(self.test_dir, "test_data.csv")
        with open(file_path, "w") as f:
            f.write("date,wrong_value\n2020-01-01,100\n2020-01-02,200\n")
        
        with self.assertRaises(ValueError):
            load_data(file_path, "date", "value")  # 'value' column does not exist

    @patch('sines.cl.get_platforms')
    def test_setup_opencl_no_nvidia_platform(self, mock_get_platforms):
        # Mock OpenCL platforms to exclude NVIDIA
        mock_platform = MagicMock()
        mock_platform.name = "AMD Platform"
        mock_get_platforms.return_value = [mock_platform]

        with self.assertRaises(RuntimeError) as context:
            setup_opencl()
        self.assertIn("NVIDIA platform not found", str(context.exception))

    @patch('sines.cl.get_platforms')
    def test_setup_opencl_no_gpu_devices(self, mock_get_platforms):
        # Mock OpenCL platforms with NVIDIA but no GPU devices
        mock_platform = MagicMock()
        mock_platform.name = "NVIDIA CUDA"
        mock_platform.get_devices.return_value = []  # No GPU devices
        mock_get_platforms.return_value = [mock_platform]

        with self.assertRaises(RuntimeError) as context:
            setup_opencl()
        self.assertIn("No GPU devices found on NVIDIA platform", str(context.exception))

    def test_load_previous_waves_multiple_files(self):
        # Create multiple wave files
        wave_params1 = {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        wave_params2 = {"amplitude": 2, "frequency": 0.5, "phase_shift": np.pi / 2}
        
        wave_file1 = os.path.join(self.test_dir, "wave_1.json")
        wave_file2 = os.path.join(self.test_dir, "wave_2.json")
        
        with open(wave_file1, "w") as f:
            json.dump(wave_params1, f)
        with open(wave_file2, "w") as f:
            json.dump(wave_params2, f)
        
        num_points = 10
        loaded_waves = load_previous_waves(num_points=num_points, output_dir=self.test_dir)
        
        expected_wave1 = generate_sine_wave(wave_params1, num_points)
        expected_wave2 = generate_sine_wave(wave_params2, num_points)
        expected_combined = expected_wave1 + expected_wave2
        
        np.testing.assert_array_almost_equal(loaded_waves, expected_combined, decimal=5)

    def test_load_previous_waves_no_files(self):
        # Ensure that loading waves with no wave files returns a zeroed array
        num_points = 10
        loaded_waves = load_previous_waves(num_points=num_points, output_dir=self.test_dir)
        expected_wave = np.zeros(num_points, dtype=np.float32)
        np.testing.assert_array_equal(loaded_waves, expected_wave)

    def test_load_previous_waves_corrupted_files(self):
        # Create a corrupted wave file
        corrupted_wave_file = os.path.join(self.test_dir, "wave_1.json")
        with open(corrupted_wave_file, "w") as f:
            f.write("{invalid json}")

        # Create a valid wave file alongside
        valid_wave_file = os.path.join(self.test_dir, "wave_2.json")
        wave_params = {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        with open(valid_wave_file, "w") as f:
            json.dump(wave_params, f)
        
        # Capture logging warnings
        with self.assertLogs(level='WARNING') as log:
            loaded_waves = load_previous_waves(num_points=10, output_dir=self.test_dir)
        
        # Verify that corrupted file was skipped and valid wave was loaded
        expected_wave = generate_sine_wave(wave_params, 10)
        np.testing.assert_array_almost_equal(loaded_waves, expected_wave, decimal=5)
        # Check that a warning was logged
        self.assertTrue(any("Error loading wave file" in message for message in log.output))

    def test_refine_candidates_empty_top_candidates(self):
        # Test refine_candidates with empty top_candidates
        top_candidates = []
        observed_data = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        combined_wave = np.zeros(len(observed_data), dtype=np.float32)

        refined_best_params, refined_best_score = refine_candidates(
            top_candidates, observed_data, combined_wave, self.context, self.queue, None, wave_count=1, desired_refinement_step_size="fast"
        )
        self.assertIsNone(refined_best_params)
        self.assertEqual(refined_best_score, np.inf)

    def test_brute_force_sine_wave_search_no_combinations(self):
        # Mock STEP_SIZES to have empty ranges
        with patch.dict('sines.STEP_SIZES', {'fast': {'amplitude': np.array([]), 'frequency': np.array([]), 'phase_shift': np.array([])}}):
            observed_data = np.array([0, 1, 0, -1], dtype=np.float32)
            combined_wave = np.zeros(len(observed_data), dtype=np.float32)
            
            candidates = brute_force_sine_wave_search(
                observed_data, combined_wave, self.context, self.queue, None, wave_count=1
            )
            self.assertEqual(len(candidates), 0)

    def test_generate_sine_wave_negative_params(self):
        # Test generate_sine_wave with negative amplitude
        params_negative_amplitude = {"amplitude": -1, "frequency": 1, "phase_shift": 0}
        sine_wave_neg_amp = generate_sine_wave(params_negative_amplitude, num_points=10)
        expected_wave = -1 * np.sin(2 * np.pi * 1 * np.arange(10) + 0)
        np.testing.assert_array_almost_equal(sine_wave_neg_amp, expected_wave, decimal=5)

        # Test generate_sine_wave with negative frequency
        params_negative_freq = {"amplitude": 1, "frequency": -1, "phase_shift": 0}
        sine_wave_neg_freq = generate_sine_wave(params_negative_freq, num_points=10)
        # Negative frequency should behave the same as positive frequency with phase shift inverted
        expected_wave_neg_freq = np.sin(2 * np.pi * -1 * np.arange(10) + 0)
        np.testing.assert_array_almost_equal(sine_wave_neg_freq, expected_wave_neg_freq, decimal=5)

    def test_load_data_with_moving_average(self):
        # Test load_data with moving average applied
        file_path = os.path.join(self.test_dir, "test_data.csv")
        with open(file_path, "w") as f:
            f.write("date,value\n2020-01-01,100\n2020-01-02,200\n2020-01-03,300\n2020-01-04,400\n")
        
        loaded_data = load_data(file_path, "date", "value", moving_average=2)
        expected = np.array([100, 150, 250, 350], dtype=np.float32)
        np.testing.assert_array_almost_equal(loaded_data, expected, decimal=5)

    def test_load_data_unsorted_dates(self):
        # Test load_data with unsorted dates to ensure sorting
        file_path = os.path.join(self.test_dir, "test_data.csv")
        with open(file_path, "w") as f:
            f.write("date,value\n2020-01-03,300\n2020-01-01,100\n2020-01-02,200\n")
        
        loaded_data = load_data(file_path, "date", "value")
        expected = np.array([100, 200, 300], dtype=np.float32)
        np.testing.assert_array_almost_equal(loaded_data, expected, decimal=5)

    def test_load_previous_waves_with_set_negatives_zero(self):
        # Create a wave file with negative values
        wave_params = {"amplitude": 1, "frequency": 1, "phase_shift": 3 * np.pi / 2}  # Sine wave will have negative values
        wave_file = os.path.join(self.test_dir, "wave_1.json")
        with open(wave_file, "w") as f:
            json.dump(wave_params, f)
        
        num_points = 10
        combined_wave = load_previous_waves(num_points=num_points, output_dir=self.test_dir)
        # Since set_negatives_zero is False by default, negatives are kept
        expected_wave = generate_sine_wave(wave_params, num_points)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)


    def test_generate_sine_wave_set_negatives_zero(self):
        # Test generate_sine_wave with set_negatives_zero=True
        params = {"amplitude": 1, "frequency": 1, "phase_shift": 3 * np.pi / 2}
        sine_wave = generate_sine_wave(params, num_points=4, set_negatives_zero=True)
        expected_wave = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(sine_wave, expected_wave, decimal=5)



if __name__ == "__main__":
    # Configure logging to capture warnings during tests
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
