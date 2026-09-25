import unittest
import numpy as np
import pyopencl as cl
import os
import shutil
import tempfile
import pandas as pd
from unittest.mock import patch
import json
import logging
import multi_sines
from extrapolator import load_sine_waves

class TestMultiSines(unittest.TestCase):
    def setUp(self):
        # Set logging level to suppress non-critical messages during tests
        logging.basicConfig(level=logging.ERROR)

        # Use the first OpenCL device (a GPU, or the PoCL CPU driver on machines without one)
        device = cl.get_platforms()[0].get_devices()[0]
        self.context = cl.Context(devices=[device])
        self.queue = cl.CommandQueue(self.context)

        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.waves_dir = os.path.join(self.test_dir, "waves")
        self.log_dir = os.path.join(self.test_dir, "logs")

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def write_data_file(self, values):
        file_path = os.path.join(self.test_dir, "data.csv")
        dates = pd.date_range("2020-01-01", periods=len(values), freq="D")
        pd.DataFrame({"date": dates, "value": values}).to_csv(file_path, index=False)
        return file_path

    def run_main(self, data_file, *extra_args):
        # Run multi_sines.main() on the test's OpenCL device instead of requiring an NVIDIA GPU.
        # main() opens a log file handler even though logging is already configured by setUp;
        # stub it out so the handler's file is not left open.
        argv = ["multi_sines.py", "--data-file", data_file, "--waves-dir", self.waves_dir,
                "--log-dir", self.log_dir, "--no-plot", *extra_args]
        with patch("sys.argv", argv), \
             patch("multi_sines.setup_opencl", return_value=(self.context, self.queue)), \
             patch("logging.FileHandler", return_value=logging.NullHandler()):
            multi_sines.main()

    def saved_waves(self):
        waves = {}
        for filename in sorted(os.listdir(self.waves_dir)):
            with open(os.path.join(self.waves_dir, filename)) as f:
                waves[filename] = json.load(f)
        return waves

    def test_refine_candidates_refines_every_candidate(self):
        # Observed data is exactly one sine wave. The first candidate is far from it and the
        # second is close, so the best refined result only comes from refining the second one.
        t = np.arange(1000)
        observed_data = (10 * np.sin(2 * np.pi * 0.0005 * t + 1.0)).astype(np.float32)
        combined_wave = np.zeros_like(observed_data)
        top_candidates = [
            {"waves": [{"amplitude": 5.0, "frequency": 0.0002, "phase_shift": 3.0}], "score": 1.0},
            {"waves": [{"amplitude": 10.0, "frequency": 0.0005, "phase_shift": 1.0}], "score": 2.0},
        ]
        # Coarse steps keep the refinement grid small (4 values per parameter)
        coarse_steps = {"amplitude_step_ratio": 0.25, "frequency_step": 0.0000025, "phase_shift_step": 0.05}

        with patch.dict(multi_sines.REFINEMENT_STEP_SIZES_BASE, {"fast": coarse_steps}):
            refined = multi_sines.refine_candidates(
                top_candidates, observed_data, combined_wave, self.context, self.queue, None, 1,
                desired_refinement_step_size="fast", max_observed=10.0, num_waves=1, top_n=3
            )

        best = min(refined, key=lambda candidate: candidate["score"])
        self.assertAlmostEqual(best["waves"][0]["amplitude"], 10.0, places=4)
        self.assertAlmostEqual(best["waves"][0]["frequency"], 0.0005, places=6)
        self.assertLess(best["score"], 1.0)

    def test_refine_candidates_no_candidates(self):
        observed_data = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        combined_wave = np.zeros_like(observed_data)

        refined = multi_sines.refine_candidates(
            [], observed_data, combined_wave, self.context, self.queue, None, 1,
            desired_refinement_step_size="fast", num_waves=1
        )
        self.assertEqual(refined, [])

    def test_main_single_wave(self):
        # Large values make main() choose the coarse 'fast' search grid, keeping the search quick
        t = np.arange(300)
        data_file = self.write_data_file(3000 * np.sin(2 * np.pi * 0.0005 * t + 1.0))

        self.run_main(data_file, "--wave-count", "1", "--num-waves", "1")

        waves = self.saved_waves()
        self.assertEqual(list(waves), ["wave_1.json"])
        self.assertEqual(set(waves["wave_1.json"]), {"amplitude", "frequency", "phase_shift"})
        # The saved wave can be loaded by the extrapolator
        self.assertEqual(len(load_sine_waves(self.waves_dir)), 1)

    def test_main_multiple_waves_saved_to_separate_files(self):
        data_file = self.write_data_file(np.arange(50, dtype=float))
        wave1 = {"amplitude": 2.0, "frequency": 0.001, "phase_shift": 0.5}
        wave2 = {"amplitude": 3.0, "frequency": 0.002, "phase_shift": 1.5}
        candidates = [{"waves": [wave1, wave2], "score": np.float32(10.0)}]

        with patch("multi_sines.brute_force_sine_wave_search", return_value=candidates):
            self.run_main(data_file, "--wave-count", "2", "--num-waves", "2")

        # One file per wave, in the same format as sines.py writes
        self.assertEqual(self.saved_waves(), {"wave_1.json": wave1, "wave_2.json": wave2})
        self.assertEqual(load_sine_waves(self.waves_dir), [wave1, wave2])

    def test_main_uses_refined_candidate(self):
        data_file = self.write_data_file(np.arange(50, dtype=float))
        brute_force_wave = {"amplitude": 2.0, "frequency": 0.001, "phase_shift": 0.5}
        refined_wave = {"amplitude": 2.5, "frequency": 0.0011, "phase_shift": 0.6}

        with patch("multi_sines.brute_force_sine_wave_search",
                   return_value=[{"waves": [brute_force_wave], "score": np.float32(50.0)}]), \
             patch("multi_sines.refine_candidates",
                   return_value=[{"waves": [refined_wave], "score": np.float32(5.0)}]) as mock_refine:
            self.run_main(data_file, "--wave-count", "1", "--num-waves", "1", "--desired-refinement-step-size", "fast")

        mock_refine.assert_called_once()
        # The refined candidate scored better, so it is the one saved
        self.assertEqual(self.saved_waves(), {"wave_1.json": refined_wave})

if __name__ == "__main__":
    unittest.main()
