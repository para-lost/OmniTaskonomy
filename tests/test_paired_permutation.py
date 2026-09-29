from fractions import Fraction
from itertools import product
from math import comb
import unittest

import numpy as np

from omnitaskonomy.analysis.paired_permutation import exact_weighted_swap, paired_test


class PairedPermutationTests(unittest.TestCase):
    def test_small_cases_match_exhaustive_question_swaps(self):
        for seeds in (1, 2, 3):
            for differences in product(range(-seeds, seeds + 1), repeat=3):
                source = [[int(seed < max(value, 0)) for value in differences] for seed in range(seeds)]
                baseline = [[int(seed < max(-value, 0)) for value in differences] for seed in range(seeds)]
                observed = abs(sum(differences))
                null = [sum(sign * value for sign, value in zip(signs, differences))
                        for signs in product((-1, 1), repeat=3)]
                expected = Fraction(sum(abs(value) >= observed for value in null), len(null))
                with self.subTest(seeds=seeds, differences=differences):
                    result = paired_test(source, baseline)
                    self.assertEqual(Fraction(int(result["p_numerator"]), int(result["p_denominator"])), expected)
                    self.assertEqual(result["p_value"], float(expected))
                    self.assertAlmostEqual(result["gain_pp"], 100 * sum(differences) / (seeds * 3))

    def test_single_seed_matches_exact_mcnemar_including_agreements(self):
        for discordant in (0, 1, 5, 9, 20):
            for wins in range(discordant + 1):
                losses = discordant - wins
                source = [[1] * wins + [0] * losses + [1, 0]]
                baseline = [[0] * wins + [1] * losses + [1, 0]]
                expected = min(Fraction(1), Fraction(
                    2 * sum(comb(discordant, k) for k in range(min(wins, losses) + 1)),
                    2 ** discordant,
                ))
                result = paired_test(source, baseline)
                self.assertEqual(Fraction(int(result["p_numerator"]), int(result["p_denominator"])), expected)
                self.assertEqual(result["n_questions"], discordant + 2)
                self.assertEqual(result["n_seeds"], 1)
                self.assertAlmostEqual(result["gain_pp"], 100 * (wins - losses) / (discordant + 2))

    def test_seed_cancellation_happens_before_question_swaps(self):
        result = paired_test(
            [[1, 0, 1], [0, 1, 0], [1, 1, 0]],
            [[0, 1, 0], [1, 0, 1], [1, 1, 0]],
        )
        self.assertEqual(result, {
            "n_questions": 3, "n_seeds": 3, "gain_pp": 0.0, "p_value": 1.0,
            "p_numerator": "1", "p_denominator": "1", "significant": False, "direction": "zero",
        })

    def test_duplicating_seed_predictions_does_not_reduce_p_value(self):
        source, baseline = [[1] * 6 + [0, 1]], [[0] * 6 + [1, 1]]
        single = paired_test(source, baseline)
        for repeats in (2, 3):
            result = paired_test(source * repeats, baseline * repeats)
            self.assertEqual(result["n_questions"], single["n_questions"])
            self.assertEqual(result["gain_pp"], single["gain_pp"])
            self.assertEqual(result["p_value"], single["p_value"])
            self.assertEqual(result["p_numerator"], single["p_numerator"])
            self.assertEqual(result["p_denominator"], single["p_denominator"])

    def test_significance_threshold_and_effect_direction(self):
        # Exact sign-flip probabilities have power-of-two denominators, so 0.05
        # itself is unattainable; these are attainable tails on either side.
        for questions, expected_p, significant in ((5, 0.0625, False), (6, 0.03125, True)):
            for source, baseline, direction, gain in (
                ([[1] * questions], [[0] * questions], "positive", 100.0),
                ([[0] * questions], [[1] * questions], "negative", -100.0),
            ):
                result = paired_test(source, baseline)
                self.assertEqual(result["p_value"], expected_p)
                self.assertEqual(result["significant"], significant)
                self.assertEqual(result["direction"], direction)
                self.assertEqual(result["gain_pp"], gain)

    def test_boolean_unsigned_and_float_hits_preserve_negative_differences(self):
        for dtype in (np.bool_, np.uint8, np.float64):
            result = paired_test(np.array([[0, 0, 1]], dtype=dtype), np.array([[1, 1, 1]], dtype=dtype))
            self.assertEqual(result["direction"], "negative")
            self.assertAlmostEqual(result["gain_pp"], -200 / 3)
            self.assertEqual(result["p_value"], 0.5)

    def test_invalid_shapes_seed_counts_and_correctness_fail(self):
        invalid = [
            ([], []), ([[]], [[]]), ([0, 1], [0, 1]),
            ([[0]], [[0, 1]]), ([[[0]]], [[[0]]]),
            (np.empty((0, 1)), np.empty((0, 1))),
            ([[0]] * 4, [[0]] * 4),
            ([[0], [0, 1]], [[0], [0, 1]]),
        ]
        for source, baseline in invalid:
            with self.subTest(source=source, baseline=baseline), self.assertRaises(ValueError):
                paired_test(source, baseline)
        for value in (-1, 2, 0.5, float("nan"), float("inf"), "1", None, 1 + 0j):
            for source, baseline in (([[value]], [[0]]), ([[0]], [[value]])):
                with self.subTest(source=source, baseline=baseline), self.assertRaises(ValueError):
                    paired_test(source, baseline)
        with self.assertRaises(ValueError):
            paired_test(np.array([[1]], dtype=object), [[0]])

    def test_exact_helper_rejects_invalid_count_support(self):
        for args in ((-1, 0, 0, 0), (1.0, 0, 0, 1), (True, 0, 0, 1),
                     (0, 0, 1, 4), (1, 2, 3, 13)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                exact_weighted_swap(*args)


if __name__ == "__main__":
    unittest.main()
