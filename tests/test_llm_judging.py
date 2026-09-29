"""Offline checks for LLM-first matching and its API-failure fallback."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
API_FAILURE = 'Failed to obtain answer via API.'


class FakeJudge:
    def __init__(self, responses, available=True):
        self.responses = iter(responses)
        self.available = available
        self.prompts = []

    def working(self):
        return self.available

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class LLMJudgingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'VLMEvalKit'))
        import pandas as pd
        from vlmeval.dataset import image_mcq
        from vlmeval.dataset.utils import multiple_choice
        cls.pandas = pd
        cls.datasets = image_mcq
        cls.matching = multiple_choice

    def item(self, **changes):
        return {'index': 0, 'question': 'Which object is closer?', 'A': 'trailer',
                'B': 'truck', 'prediction': 'A', 'answer': 'B', 'GT': 'B', **changes}

    def extract(self, judge, **changes):
        return self.matching.extract_answer_from_item(
            judge, self.item(**changes), dataset_name='CV-Bench-3D', llm_first=True,
        )

    def test_healthy_judge_precedes_exact_match_without_ground_truth(self):
        judge = FakeJudge(['B'])
        result = self.extract(judge, answer='ground-truth-must-not-be-sent', GT='also-private')
        self.assertEqual(result, {'opt': 'B', 'log': 'B'})
        self.assertEqual(len(judge.prompts), 1)
        self.assertIn('only valid output labels are A, B, Z', judge.prompts[0])
        self.assertNotIn('ground-truth-must-not-be-sent', judge.prompts[0])
        self.assertNotIn('also-private', judge.prompts[0])

    def test_legacy_mode_keeps_prefetch_and_exact_matching(self):
        judge = FakeJudge([])
        result = self.matching.extract_answer_from_item(judge, self.item(), 'CV-Bench-3D')
        self.assertEqual(result, {'opt': 'A', 'log': 'A'})
        self.assertEqual(judge.prompts, [])
        unmatched = self.matching.extract_answer_from_item(None, self.item(prediction='unclear'), 'MMStar')
        self.assertEqual(unmatched['opt'], 'Z')
        self.assertIn('Failed in Prefetch', unmatched['log'])

    def test_unavailable_api_falls_back_to_original_prediction(self):
        for prediction, expected in [('A', 'A'), ('unclear', 'Z')]:
            with self.subTest(prediction=prediction):
                result = self.extract(None, prediction=prediction)
                self.assertEqual(result['opt'], expected)
                self.assertIn('Exact matching fallback: API unavailable.', result['log'])

    def test_only_api_failure_responses_allow_request_fallback(self):
        judge = FakeJudge([API_FAILURE] * 3)
        result = self.extract(judge)
        self.assertEqual(result['opt'], 'A')
        self.assertEqual(len(judge.prompts), 3)
        self.assertIn('Exact matching fallback: API request failed.', result['log'])

    def test_invalid_judge_responses_never_use_exact_matching_or_random_labels(self):
        for responses in [['D'] * 3, [API_FAILURE, 'D', API_FAILURE]]:
            with self.subTest(responses=responses):
                judge = FakeJudge(responses)
                with patch.object(self.matching.rd, 'choice', side_effect=AssertionError('Random guess')):
                    result = self.extract(judge)
                self.assertEqual(result['opt'], 'Z')
                self.assertIn('Unresolved LLM judge response.', result['log'])
                self.assertNotIn('Exact matching fallback:', result['log'])
                self.assertEqual(len(judge.prompts), 3)

    def test_valid_z_is_a_normal_zero_score(self):
        judge = FakeJudge(['(Z)'])
        result = self.matching.eval_vanilla(judge, self.item(), 'CV-Bench-3D', llm_first=True)
        self.assertEqual(result['hit'], 0)
        self.assertIn('(Z)', result['log'])
        self.assertNotIn('Unresolved', result['log'])
        self.assertNotIn('fallback', result['log'])
        self.assertEqual(len(judge.prompts), 1)

    def test_unrelated_judge_errors_propagate(self):
        with self.assertRaisesRegex(ValueError, 'implementation error'):
            self.extract(FakeJudge([ValueError('implementation error')]))

    def test_real_dataset_classes_propagate_policy_and_isolate_cached_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, dataset_class, alias in [
                ('MMStar', self.datasets.ImageMCQDataset, 'openai'),
                ('CV-Bench-3D', self.datasets.CVBench, 'chatgpt-0125'),
            ]:
                with self.subTest(benchmark=name):
                    frame = self.pandas.DataFrame([self.item(split='3D' if name.startswith('CV') else 'test')])
                    dataset = dataset_class.__new__(dataset_class)
                    dataset.dataset_name = name
                    dataset.data = frame.drop(columns=['prediction', 'GT'])
                    prediction = root / f'fixture_{name}.xlsx'
                    frame.drop(columns='GT').to_excel(prediction, index=False)
                    legacy = FakeJudge([])
                    with patch.object(self.datasets, 'gpt_key_set', return_value=True), \
                            patch.object(self.datasets, 'build_judge', return_value=legacy):
                        old_scores = dataset.evaluate(str(prediction), model='chatgpt-0125', nproc=1)
                    self.assertEqual(float(old_scores['Overall'].iloc[0]), 0)
                    self.assertEqual(legacy.prompts, [])
                    old_result = root / f'{prediction.stem}_{alias}_result.xlsx'
                    old_bytes = old_result.read_bytes()
                    judge = FakeJudge(['B'])
                    with patch.object(self.datasets, 'gpt_key_set', return_value=True), \
                            patch.object(self.datasets, 'build_judge', return_value=judge) as build:
                        scores = dataset.evaluate(str(prediction), model='chatgpt-0125', nproc=1, llm_first=True)
                    self.assertEqual(float(scores['Overall'].iloc[0]), 1)
                    self.assertEqual(len(judge.prompts), 1)
                    self.assertNotIn('llm_first', build.call_args.kwargs)
                    self.assertEqual(old_result.read_bytes(), old_bytes)
                    result = root / f'{prediction.stem}_{alias}_llm_first_result.xlsx'
                    self.assertEqual(self.pandas.read_excel(result)['hit'].tolist(), [1])
                    self.assertTrue(result.with_suffix('.pkl').exists())
                    first_acc = root / f'{prediction.stem}_{alias}_llm_first_acc.csv'
                    self.assertTrue(first_acc.exists())
                    second = FakeJudge(['A'])
                    with patch.object(self.datasets, 'gpt_key_set', return_value=True), \
                            patch.object(self.datasets, 'build_judge', return_value=second):
                        second_scores = dataset.evaluate(str(prediction), model='gpt-4-0125', nproc=1, llm_first=True)
                    self.assertEqual(float(second_scores['Overall'].iloc[0]), 0)
                    self.assertEqual(len(second.prompts), 1)
                    second_alias = 'gpt-4-0125' if name.startswith('CV') else 'gpt4'
                    second_result = root / f'{prediction.stem}_{second_alias}_llm_first_result.xlsx'
                    self.assertEqual(self.pandas.read_excel(second_result)['hit'].tolist(), [0])
                    self.assertEqual(float(self.pandas.read_csv(first_acc)['Overall'].iloc[0]), 1)

    def test_missing_key_and_failed_health_check_record_unavailability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, dataset_class, key_present in [
                ('MMStar', self.datasets.ImageMCQDataset, False),
                ('CV-Bench-3D', self.datasets.CVBench, True),
            ]:
                with self.subTest(benchmark=name):
                    frame = self.pandas.DataFrame([self.item(split='3D')])
                    dataset = dataset_class.__new__(dataset_class)
                    dataset.dataset_name = name
                    dataset.data = frame.drop(columns=['prediction', 'GT'])
                    prediction = root / f'{name}.xlsx'
                    frame.drop(columns='GT').to_excel(prediction, index=False)
                    judge = FakeJudge([], available=False)
                    with patch.object(self.datasets, 'gpt_key_set', return_value=key_present), \
                            patch.object(self.datasets, 'build_judge', return_value=judge):
                        dataset.evaluate(str(prediction), model='chatgpt-0125', nproc=1, llm_first=True)
                    alias = 'chatgpt-0125' if name.startswith('CV') else 'openai'
                    result = self.pandas.read_excel(root / f'{name}_{alias}_llm_first_result.xlsx')
                    self.assertEqual(result['hit'].tolist(), [0])
                    self.assertIn('Exact matching fallback: API unavailable.', result['log'].iloc[0])
                    self.assertEqual(judge.prompts, [])

    def test_circular_datasets_reject_unsupported_policy_before_reading_predictions(self):
        dataset = self.datasets.ImageMCQDataset.__new__(self.datasets.ImageMCQDataset)
        dataset.dataset_name = 'MMBench'
        with self.assertRaisesRegex(ValueError, 'circular datasets'):
            dataset.evaluate('/does/not/exist.xlsx', model='chatgpt-0125', llm_first=True)


if __name__ == '__main__':
    unittest.main()
