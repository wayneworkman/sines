import unittest
import numpy as np
import pyopencl as cl
import os
import shutil
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock
from sines import (
    generate_sine_wave,
    load_data,
    load_previous_waves,
    refine_candidates,
    brute_force_sine_wave_search,
    setup_opencl,
    STEP_SIZES  # Import STEP_SIZES for modification
)
from extrapolator import (
    load_observed_data,
    load_sine_waves,
    calculate_average_timespan,
    generate_combined_sine_wave,
    plot_data
)
import json
import logging

class TestSines(unittest.TestCase):
    def setUp(self):
        # Set logging level to suppress non-critical messages during tests
        logging.basicConfig(level=logging.ERROR)

        # Existing setup code...
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

        # Initialize STEP_SIZES with default values for testing
        max_observed = 1.0  # Assuming a default max_observed value
        scaling_factor = 1.5
        amplitude_upper_limit = max_observed * scaling_factor

        # Define STEP_SIZES similar to main() in sines.py
        STEP_SIZES['ultrafine'] = {
            'amplitude': np.arange(0.1, amplitude_upper_limit, 0.01 * max_observed),
            'frequency': np.arange(0.00001, 0.001, 0.0000075),
            'phase_shift': np.arange(0, 2 * np.pi, 0.025)
        }
        STEP_SIZES['fine'] = {
            'amplitude': np.arange(0.1, amplitude_upper_limit, 0.02 * max_observed),
            'frequency': np.arange(0.00001, 0.001, 0.000015),
            'phase_shift': np.arange(0, 2 * np.pi, 0.05)
        }
        STEP_SIZES['normal'] = {
            'amplitude': np.arange(0.1, amplitude_upper_limit, 0.05 * max_observed),
            'frequency': np.arange(0.00001, 0.001, 0.00003),
            'phase_shift': np.arange(0, 2 * np.pi, 0.15)
        }
        STEP_SIZES['fast'] = {
            'amplitude': np.arange(0.1, amplitude_upper_limit, 0.1 * max_observed),
            'frequency': np.arange(0.00001, 0.001, 0.00006),
            'phase_shift': np.arange(0, 2 * np.pi, 0.3)
        }

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

        # **Updated: Replace 'output_dir' with 'waves_dir'**
        loaded_waves = load_previous_waves(num_points=10, waves_dir=self.test_dir)
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
        # **Updated: Replace 'output_dir' with 'waves_dir'**
        loaded_waves = load_previous_waves(num_points=num_points, waves_dir=self.test_dir)
        
        expected_wave1 = generate_sine_wave(wave_params1, num_points)
        expected_wave2 = generate_sine_wave(wave_params2, num_points)
        expected_combined = expected_wave1 + expected_wave2
        
        np.testing.assert_array_almost_equal(loaded_waves, expected_combined, decimal=5)

    def test_load_previous_waves_no_files(self):
        # Ensure that loading waves with no wave files returns a zeroed array
        num_points = 10
        # **Updated: Replace 'output_dir' with 'waves_dir'**
        loaded_waves = load_previous_waves(num_points=num_points, waves_dir=self.test_dir)
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
            # **Updated: Replace 'output_dir' with 'waves_dir'**
            loaded_waves = load_previous_waves(num_points=10, waves_dir=self.test_dir)
        
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
        
        num_points = 4
        # **Updated: Replace 'output_dir' with 'waves_dir'**
        combined_wave = load_previous_waves(num_points=num_points, waves_dir=self.test_dir, set_negatives_zero=False)
        expected_wave = generate_sine_wave(wave_params, num_points, set_negatives_zero=False)

        # Test without setting negatives to zero
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

        # Test with setting negatives to zero
        combined_wave_zero_neg = load_previous_waves(num_points=num_points, waves_dir=self.test_dir, set_negatives_zero=True)
        expected_wave_zero_neg = np.maximum(expected_wave, 0)
        np.testing.assert_array_almost_equal(combined_wave_zero_neg, expected_wave_zero_neg, decimal=5)

    def test_generate_sine_wave_set_negatives_zero(self):
        # Test generate_sine_wave with set_negatives_zero=True
        params = {"amplitude": 1, "frequency": 1, "phase_shift": 3 * np.pi / 2}
        sine_wave = generate_sine_wave(params, num_points=4, set_negatives_zero=True)
        # Sine wave at phase_shift=3*pi/2: sin(-pi/2) = -1
        expected_wave = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(sine_wave, expected_wave, decimal=5)

class TestExtrapolator(unittest.TestCase):
    def setUp(self):
        # Set logging level to suppress non-critical messages during tests
        logging.basicConfig(level=logging.ERROR)

        # Existing setup code...
        self.test_dir = tempfile.mkdtemp()
        self.patcher = patch('extrapolator.plt.show')
        self.mock_show = self.patcher.start()

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)
        self.patcher.stop()

    def test_load_observed_data_valid_csv(self):
        # Creating a temporary test CSV file with valid data
        file_path = os.path.join(self.test_dir, "test_data.csv")
        with open(file_path, "w") as f:
            f.write("date,sunspot\n2020-01-01,100\n2020-01-02,200\n")
        
        dates, indices, data_values = load_observed_data(file_path, date_col="date", value_col="sunspot")
        self.assertEqual(len(indices), 2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(dates))
        np.testing.assert_array_almost_equal(data_values, np.array([100, 200]))

    def test_load_observed_data_invalid_file(self):
        # Test loading data from a non-existent file
        file_path = os.path.join(self.test_dir, "non_existent.csv")
        with self.assertRaises(FileNotFoundError) as context:
            load_observed_data(file_path, date_col="date", value_col="sunspot")
        self.assertIn("No such file or directory", str(context.exception))

    def test_load_observed_data_missing_value_col(self):
        # Test loading data with missing value column
        file_path = os.path.join(self.test_dir, "test_data.csv")
        with open(file_path, "w") as f:
            f.write("date,wrong_column\n2020-01-01,100\n2020-01-02,200\n")
        
        with self.assertRaises(ValueError):
            load_observed_data(file_path, date_col="date", value_col="sunspot")

    def test_load_sine_waves_valid(self):
        # Create valid sine wave JSON files
        wave_params1 = {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        wave_params2 = {"amplitude": 2, "frequency": 0.5, "phase_shift": np.pi / 2}
        
        wave_file1 = os.path.join(self.test_dir, "wave_1.json")
        wave_file2 = os.path.join(self.test_dir, "wave_2.json")
        
        with open(wave_file1, "w") as f:
            json.dump(wave_params1, f)
        with open(wave_file2, "w") as f:
            json.dump(wave_params2, f)
        
        sine_waves = load_sine_waves(self.test_dir)
        self.assertEqual(len(sine_waves), 2)
        self.assertDictEqual(sine_waves[0], wave_params1)
        self.assertDictEqual(sine_waves[1], wave_params2)

    @patch('logging.warning')
    def test_load_sine_waves_invalid_json(self, mock_logging_warning):
        # Create an invalid sine wave JSON file
        invalid_wave_file = os.path.join(self.test_dir, "wave_1.json")
        with open(invalid_wave_file, "w") as f:
            f.write("{invalid json}")

        # Create a valid sine wave file alongside
        valid_wave_file = os.path.join(self.test_dir, "wave_2.json")
        wave_params = {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        with open(valid_wave_file, "w") as f:
            json.dump(wave_params, f)
        
        # Run the function
        sine_waves = load_sine_waves(self.test_dir)
        
        # Assertions
        self.assertEqual(len(sine_waves), 1)
        self.assertDictEqual(sine_waves[0], wave_params)
        mock_logging_warning.assert_any_call("Wave file 'wave_1.json' is not a valid JSON. Skipping.")

    @patch('logging.warning')
    def test_load_sine_waves_missing_parameters(self, mock_logging_warning):
        # Create a sine wave JSON file missing required parameters
        incomplete_wave_file = os.path.join(self.test_dir, "wave_1.json")
        incomplete_wave = {"amplitude": 1, "frequency": 1}  # Missing 'phase_shift'
        with open(incomplete_wave_file, "w") as f:
            json.dump(incomplete_wave, f)
        
        # Create a valid wave file
        valid_wave_file = os.path.join(self.test_dir, "wave_2.json")
        wave_params = {"amplitude": 2, "frequency": 0.5, "phase_shift": np.pi / 2}
        with open(valid_wave_file, "w") as f:
            json.dump(wave_params, f)
        
        # Run the function
        sine_waves = load_sine_waves(self.test_dir)
        
        # Assertions
        self.assertEqual(len(sine_waves), 1)
        self.assertDictEqual(sine_waves[0], wave_params)
        mock_logging_warning.assert_any_call("Could not load wave file 'wave_1.json': Wave file 'wave_1.json' is missing required parameters.. Skipping.")

    def test_calculate_average_timespan(self):
        # Test with sorted dates
        file_path = os.path.join(self.test_dir, "test_data.csv")
        with open(file_path, "w") as f:
            f.write("date,sunspot\n2020-01-01,100\n2020-01-02,200\n2020-01-03,300\n")
        
        dates, _, _ = load_observed_data(file_path, date_col="date", value_col="sunspot")
        avg_timespan = calculate_average_timespan(dates)
        self.assertEqual(avg_timespan, 1.0)

        # Test with unsorted dates
        file_path_unsorted = os.path.join(self.test_dir, "test_data_unsorted.csv")
        with open(file_path_unsorted, "w") as f:
            f.write("date,sunspot\n2020-01-03,300\n2020-01-01,100\n2020-01-02,200\n")
        
        dates_unsorted, _, _ = load_observed_data(file_path_unsorted, date_col="date", value_col="sunspot")
        avg_timespan_unsorted = calculate_average_timespan(dates_unsorted)
        self.assertEqual(avg_timespan_unsorted, 1.0)

        # Test with insufficient data
        file_path_single = os.path.join(self.test_dir, "test_data_single.csv")
        with open(file_path_single, "w") as f:
            f.write("date,sunspot\n2020-01-01,100\n")
        
        dates_single, _, _ = load_observed_data(file_path_single, date_col="date", value_col="sunspot")
        avg_timespan_single = calculate_average_timespan(dates_single)
        self.assertIsNone(avg_timespan_single)

    def test_generate_combined_sine_wave_after_sum(self):
        # Create multiple sine waves
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 0},  # sin(2*pi*1*t + 0)
            {"amplitude": 2, "frequency": 0.5, "phase_shift": np.pi / 2}  # 2*sin(pi*t + pi/2)
        ]
        indices = np.arange(10)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        
        # Manually compute expected combined wave
        wave1 = generate_sine_wave(sine_waves[0], 10)
        wave2 = generate_sine_wave(sine_waves[1], 10)
        expected_combined = wave1 + wave2
        expected_combined = np.maximum(expected_combined, 0)  # after_sum
        
        np.testing.assert_array_almost_equal(combined_wave, expected_combined, decimal=5)

    def test_generate_combined_sine_wave_per_wave(self):
        # Create multiple sine waves
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 0},  # sin(2*pi*1*t + 0)
            {"amplitude": 2, "frequency": 0.5, "phase_shift": np.pi / 2}  # 2*sin(pi*t + pi/2)
        ]
        indices = np.arange(10)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        
        # Manually compute expected combined wave
        wave1 = generate_sine_wave(sine_waves[0], 10, set_negatives_zero=True)
        wave2 = generate_sine_wave(sine_waves[1], 10, set_negatives_zero=True)
        expected_combined = wave1 + wave2
        
        np.testing.assert_array_almost_equal(combined_wave, expected_combined, decimal=5)

    def test_generate_combined_sine_wave_empty_waves(self):
        # Test with no sine waves
        sine_waves = []
        indices = np.arange(10)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        expected_wave = np.zeros(10, dtype=np.float64)
        np.testing.assert_array_equal(combined_wave, expected_wave)

    def test_generate_combined_sine_wave_with_negatives_after_sum(self):
        # Create sine waves that sum to negative values
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": np.pi},  # sin(2*pi*t + pi) = -sin(2*pi*t)
            {"amplitude": 1, "frequency": 1, "phase_shift": np.pi}   # sin(2*pi*t + pi) = -sin(2*pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        # Each wave is -sin(2*pi*t), sum is -2*sin(2*pi*t)
        expected_wave = np.zeros(4, dtype=np.float64)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

    def test_plot_data(self):
        # Test plotting function (mock plt.show)
        dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
        indices = np.arange(3)
        data_values = np.array([100, 200, 150])
        combined_wave = np.array([90, 210, 160])
        
        with patch('extrapolator.plt.show') as mock_show:
            plot_data(dates, indices, data_values, combined_wave)
            mock_show.assert_called_once()

    def test_generate_combined_sine_wave_with_set_negatives_zero_after_sum(self):
        # Create sine waves that sum to negative values
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 3 * np.pi / 2},  # sin(2*pi*t + 3pi/2) = -cos(2*pi*t)
            {"amplitude": 1, "frequency": 1, "phase_shift": 3 * np.pi / 2}   # sin(2*pi*t + 3pi/2) = -cos(2*pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        
        # Each wave is -cos(2*pi*t), sum is -2*cos(2*pi*t)
        # After max(combined_wave, 0), all values should be 0
        expected_combined = np.zeros(4, dtype=np.float64)
        np.testing.assert_array_almost_equal(combined_wave, expected_combined, decimal=5)

    def test_generate_combined_sine_wave_with_set_negatives_zero_per_wave(self):
        # Create sine waves where some individual waves have negative values
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 3 * np.pi / 2},  # sin(2*pi*t + 3pi/2) = -cos(2*pi*t)
            {"amplitude": 2, "frequency": 0.5, "phase_shift": 0}           # 2*sin(pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        
        # wave1: max(-cos(2*pi*t), 0), wave2: 2*sin(pi*t)
        wave1 = generate_sine_wave(sine_waves[0], 4, set_negatives_zero=True)
        wave2 = generate_sine_wave(sine_waves[1], 4, set_negatives_zero=True)
        expected_combined = wave1 + wave2
        np.testing.assert_array_almost_equal(combined_wave, expected_combined, decimal=5)

    def test_generate_combined_sine_wave_negative_amplitude(self):
        # Create sine wave with negative amplitude
        sine_waves = [
            {"amplitude": -1, "frequency": 1, "phase_shift": 0}  # -sin(2*pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        
        # Expected wave: max(-sin(2*pi*t), 0)
        expected_wave = np.maximum(-np.sin(2 * np.pi * 1 * indices + 0), 0)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

    def test_generate_combined_sine_wave_negative_frequency(self):
        # Create sine wave with negative frequency
        sine_waves = [
            {"amplitude": 1, "frequency": -1, "phase_shift": 0}  # sin(-2*pi*t) = -sin(2*pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        
        # Expected wave: max(-sin(2*pi*t), 0)
        expected_wave = np.maximum(-np.sin(2 * np.pi * 1 * indices + 0), 0)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

    def test_generate_combined_sine_wave_all_negatives_after_sum(self):
        # Create sine waves that sum to negative values
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 3 * np.pi / 2},  # sin(2*pi*t + 3pi/2) = -cos(2*pi*t)
            {"amplitude": 1, "frequency": 1, "phase_shift": 3 * np.pi / 2}   # sin(2*pi*t + 3pi/2) = -cos(2*pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        
        # wave1: -cos(2*pi*t), wave2: -cos(2*pi*t), sum = -2*cos(2*pi*t)
        # For integer t, cos(2*pi*t) = 1, so sum = -2
        # After max(combined_wave, 0), all values should be 0
        expected_combined = np.zeros(4, dtype=np.float64)
        np.testing.assert_array_almost_equal(combined_wave, expected_combined, decimal=5)


    def test_generate_combined_sine_wave_partial_negatives_after_sum(self):
        # Create sine waves where some sums are negative and some are positive
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 0},   # sin(2*pi*t)
            {"amplitude": 1, "frequency": 1, "phase_shift": np.pi}  # sin(2*pi*t + pi) = -sin(2*pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        
        # Sum is sin(2*pi*t) - sin(2*pi*t) = 0 for all t, max with 0 remains 0
        expected_wave = np.zeros(4, dtype=np.float64)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

    def test_generate_combined_sine_wave_multiple_waves_after_sum(self):
        # Create multiple sine waves with varying phases and frequencies
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 0},
            {"amplitude": 2, "frequency": 0.5, "phase_shift": np.pi / 2},
            {"amplitude": 0.5, "frequency": 2, "phase_shift": np.pi}
        ]
        indices = np.arange(5)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        
        # Manually compute expected combined wave
        wave1 = generate_sine_wave(sine_waves[0], 5)
        wave2 = generate_sine_wave(sine_waves[1], 5)
        wave3 = generate_sine_wave(sine_waves[2], 5)
        expected_combined = wave1 + wave2 + wave3
        expected_combined = np.maximum(expected_combined, 0)  # after_sum
        
        np.testing.assert_array_almost_equal(combined_wave, expected_combined, decimal=5)

    def test_generate_combined_sine_wave_large_number_of_points(self):
        # Create multiple sine waves
        sine_waves = [
            {"amplitude": 1, "frequency": 0.1, "phase_shift": 0},
            {"amplitude": 0.5, "frequency": 0.2, "phase_shift": np.pi / 4},
            {"amplitude": 2, "frequency": 0.05, "phase_shift": np.pi / 2}
        ]
        indices = np.arange(1000)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        
        # Manually compute expected combined wave
        wave1 = generate_sine_wave(sine_waves[0], 1000, set_negatives_zero=True)
        wave2 = generate_sine_wave(sine_waves[1], 1000, set_negatives_zero=True)
        wave3 = generate_sine_wave(sine_waves[2], 1000, set_negatives_zero=True)
        expected_combined = wave1 + wave2 + wave3
        
        # Use decimal=4 to allow minor differences
        np.testing.assert_array_almost_equal(combined_wave, expected_combined, decimal=4)


    def test_generate_combined_sine_wave_single_wave_after_sum(self):
        # Create a single sine wave
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        
        # Expected wave: sin(2*pi*1*t), max with 0
        expected_wave = np.maximum(np.sin(2 * np.pi * 1 * indices + 0), 0)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

    def test_generate_combined_sine_wave_zero_waves_after_sum(self):
        # Test with no sine waves
        sine_waves = []
        indices = np.arange(5)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        expected_wave = np.zeros(5, dtype=np.float64)
        np.testing.assert_array_equal(combined_wave, expected_wave)

    def test_generate_combined_sine_wave_zero_waves_per_wave(self):
        # Test with no sine waves
        sine_waves = []
        indices = np.arange(5)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        expected_wave = np.zeros(5, dtype=np.float64)
        np.testing.assert_array_equal(combined_wave, expected_wave)

    def test_generate_combined_sine_wave_invalid_set_negatives_zero(self):
        # Test with invalid set_negatives_zero value
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        ]
        indices = np.arange(4)
        with self.assertRaises(ValueError):
            combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='invalid_option')


    def test_generate_combined_sine_wave_non_boolean_set_negatives_zero(self):
        # Test passing a non-string type to set_negatives_zero
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 0}
        ]
        indices = np.arange(4)
        with self.assertRaises(ValueError):
            combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero=123)


    def test_generate_combined_sine_wave_large_amplitude(self):
        # Create sine wave with large amplitude
        sine_waves = [
            {"amplitude": 1000, "frequency": 1, "phase_shift": 0}
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        
        # Expected wave: sin(2*pi*1*t) scaled by 1000, max with 0
        expected_wave = np.maximum(1000 * np.sin(2 * np.pi * 1 * indices + 0), 0)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)



    def test_generate_combined_sine_wave_zero_amplitude(self):
        # Create sine wave with zero amplitude
        sine_waves = [
            {"amplitude": 0, "frequency": 1, "phase_shift": 0}  # 0*sin(2*pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        
        # Expected wave: zeros
        expected_wave = np.zeros(4, dtype=np.float64)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

    def test_generate_combined_sine_wave_phase_shift_over_2pi(self):
        # Create sine wave with phase shift greater than 2*pi
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 5 * np.pi}  # Equivalent to pi phase shift
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        
        # sin(2*pi*t + 5*pi) = sin(2*pi*t + pi) = -sin(2*pi*t)
        expected_wave = np.maximum(-np.sin(2 * np.pi * 1 * indices + 0), 0)  # per_wave
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

    def test_generate_combined_sine_wave_multiple_waves_per_wave(self):
        # Create multiple sine waves with 'per_wave' negative handling
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 0},   # sin(2*pi*t)
            {"amplitude": 1, "frequency": 1, "phase_shift": np.pi}  # sin(2*pi*t + pi) = -sin(2*pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        
        # wave1: sin(2*pi*t), wave2: max(-sin(2*pi*t), 0)
        wave1 = generate_sine_wave(sine_waves[0], 4, set_negatives_zero=False)
        wave2 = generate_sine_wave(sine_waves[1], 4, set_negatives_zero=True)
        expected_combined = wave1 + wave2
        np.testing.assert_array_almost_equal(combined_wave, expected_combined, decimal=5)

    def test_generate_combined_sine_wave_high_frequency(self):
        # Create sine wave with high frequency
        sine_waves = [
            {"amplitude": 1, "frequency": 10, "phase_shift": 0}  # sin(20*pi*t)
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='per_wave')
        
        # Expected wave: sin(20*pi*t), max with 0
        expected_wave = np.maximum(np.sin(20 * np.pi * indices + 0), 0)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

    def test_generate_combined_sine_wave_non_integer_indices(self):
        # Create sine wave with non-integer indices
        sine_waves = [
            {"amplitude": 1, "frequency": 0.1, "phase_shift": 0}
        ]
        # Using floating-point indices
        indices = np.linspace(0, 1, 5)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        
        # Expected wave: sin(2*pi*0.1*t), max with 0
        expected_wave = np.maximum(np.sin(2 * np.pi * 0.1 * indices + 0), 0)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

    def test_generate_combined_sine_wave_large_phase_shift(self):
        # Create sine wave with large phase shift
        sine_waves = [
            {"amplitude": 1, "frequency": 1, "phase_shift": 10 * np.pi}  # Equivalent to 0 phase shift
        ]
        indices = np.arange(4)
        combined_wave = generate_combined_sine_wave(sine_waves, indices, set_negatives_zero='after_sum')
        
        # sin(2*pi*t + 10*pi) = sin(2*pi*t), since sin(x + 2*pi*n) = sin(x)
        expected_wave = np.maximum(np.sin(2 * np.pi * 1 * indices + 10 * np.pi), 0)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

if __name__ == "__main__":
    unittest.main()
