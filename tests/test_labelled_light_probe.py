# -*- coding: utf-8 -*-
"""光照诊断的GT忽略、实例等权、错误转移、固定投票和几何反变换。"""
import unittest

import numpy as np
import torch

from data.direct_dual_head_dataset import _spatial_transform
from tools.labelled_light_probe import (
    aggregate_report, classification_metrics, error_transitions, evaluate_native, illuminated_input, inverse_geometry,
    make_core_instances, paired_bootstrap, prepare_regions, region_votes,
)
from tools.semantic_crossover import revote_instances


class LabelledLightProbeTests(unittest.TestCase):
    def test_unknown_labels_do_not_contribute_to_errors_or_bce(self):
        target = np.array([0, 1, -1, -2])
        probability = np.array([.2, .8, np.nan, np.inf])
        result = classification_metrics(target, probability)
        self.assertEqual(result['confusion'], [[1, 0], [0, 1]])
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['accuracy'], 1)
        self.assertAlmostEqual(result['bce'], -np.log(.8))

    def test_instance_weighting_differs_from_pixel_weighting(self):
        ids = np.array([[1, 1, 1, 2]], dtype=np.uint16)
        probability = np.array([[.1, .1, .1, .1]], dtype=np.float32)
        lookup = np.array([-1, 0, 1], dtype=np.float32)
        result = evaluate_native(probability, ids, lookup, ids)
        self.assertEqual(result['pixels']['accuracy'], .75)
        self.assertEqual(result['instances']['accuracy'], .5)
        self.assertEqual(result['instances']['balanced_accuracy'], .5)
        self.assertEqual(result['instances']['n'], 2)
        self.assertEqual(result['instances']['confusion'], [[1, 0], [1, 0]])

    def test_error_transitions_separate_harm_and_correction(self):
        target = np.array([0, 1, 0, 1, -1])
        base = np.array([.1, .9, .9, .1, .1])
        after = np.array([.9, .1, .1, .1, .9])
        result = error_transitions(target, base, after)
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['changed'], 3)
        self.assertEqual(result['correct_to_wrong'], 2)
        self.assertEqual(result['wrong_to_correct'], 1)
        self.assertEqual(result['net_new_errors'], 1)

    def test_no_clip_core_subset_has_matched_baseline(self):
        ids = np.array([[1, 1, 2, 2]], dtype=np.uint16)
        core = np.array([[0, 1, 2, 0]], dtype=np.uint16)
        lookup = np.array([-1, 0, 1], dtype=np.float32)
        baseline = np.array([[.1, .1, .9, .9]], dtype=np.float32)
        probability = np.array([[.9, .9, .1, .1]], dtype=np.float32)
        clipped = np.array([[False, True, False, False]])
        result = evaluate_native(probability, ids, lookup, core,
                                 baseline_probability=baseline, clipped=clipped)
        self.assertEqual(result['instances']['transitions']['correct_to_wrong'], 2)
        self.assertEqual(result['unclipped_core_instances']['n'], 1)
        self.assertEqual(result['unclipped_core_instances']['transitions']['correct_to_wrong'], 1)
        self.assertEqual(result['unclipped_core_instances']['ferrite_recall'], 0)

    def test_bbox_float32_votes_exactly_match_existing_deployment_vote(self):
        ids = np.zeros((30, 40), dtype=np.uint16)
        ids[1:12, 2:14] = 1
        ids[10:25, 20:33] = 5
        ids[5:8, 8:10] = 0
        probability = np.random.default_rng(14).random(ids.shape).astype(np.float32)
        probability[ids == 5] = .5
        regions = prepare_regions(ids, ids)
        scores, core_scores = region_votes(probability, regions)
        classes, expected = revote_instances(ids, probability)
        self.assertEqual(scores, {int(k): v for k, v in expected.items()})
        self.assertEqual(core_scores, scores)
        self.assertEqual(classes['5'], 0)

    def test_empty_small_cores_are_reported_without_losing_primary_instances(self):
        ids = np.zeros((12, 12), dtype=np.uint16)
        ids[1:11, 1:9] = 1
        ids[1:11, 10] = 2
        core = make_core_instances(ids, radius=2, minimum_pixels=12)
        self.assertTrue(np.any(core == 1))
        self.assertFalse(np.any(core == 2))
        lookup = np.array([-1, 0, 1], dtype=np.float32)
        result = evaluate_native(np.full(ids.shape, .2, dtype=np.float32), ids, lookup, core)
        self.assertEqual(result['instances']['n'], 2)
        self.assertEqual(result['core_instances']['n'], 1)
        self.assertEqual(result['core_excluded'], {'instances': 1, 'pearlite': 0, 'ferrite': 1})

    def test_inverse_geometry_covers_all_dihedral_combinations(self):
        original = torch.arange(2 * 7 * 9).reshape(2, 7, 9)
        for h in (False, True):
            for v in (False, True):
                for k in range(4):
                    sample = {'horizontal_flip': h, 'vertical_flip': v, 'rotation_k': k}
                    changed = _spatial_transform(original, h, v, k)
                    self.assertTrue(torch.equal(inverse_geometry(changed, sample), original))

    def test_bootstrap_treats_images_as_units(self):
        rows = [{'current': {'accuracy': .5}, 'baseline': {'accuracy': .75}},
                {'current': {'accuracy': .2}, 'baseline': {'accuracy': .2}},
                {'current': {'accuracy': 1.}, 'baseline': {'accuracy': .75}}]
        result = paired_bootstrap(rows, 'accuracy', draws=1000, seed=10)
        self.assertEqual(result['images'], 3)
        self.assertEqual(result['mean_delta'], 0)
        self.assertEqual(result['positive_images'], 1)
        self.assertEqual(result['negative_images'], 1)
        self.assertEqual(result['unchanged_images'], 1)
        self.assertLess(result['ci95'][0], 0)
        self.assertGreater(result['ci95'][1], 0)

    def test_report_aggregation_preserves_counts_and_ignores_empty_core(self):
        ids = np.array([[1, 1, 2]], dtype=np.uint16)
        lookup = np.array([-1, 0, 1], dtype=np.float32)
        values = evaluate_native(np.array([[.1, .1, .9]], dtype=np.float32),
                                 ids, lookup, np.zeros_like(ids))
        condition = {'models': {'full': {'offset_0': values}}}
        report = {'models': {'full': {}}, 'variants': [{'key': 'offset_0'}],
                  'images': {str(i): {'conditions': {'clean': condition, 'blur': condition}} for i in range(2)}}
        summary = aggregate_report(report)['clean']['full']['offset_0']
        self.assertEqual(summary['instances']['n'], 4)
        self.assertEqual(summary['instances']['confusion'], [[2, 0], [0, 2]])
        self.assertEqual(summary['instances']['paired_image_accuracy']['ci95'], [0., 0.])
        self.assertIsNone(summary['core_instances']['accuracy'])

    def test_input_metrics_exclude_padding_and_identity_is_exact(self):
        image = torch.full((1, 3, 6, 6), .5)
        image[:, :, 4:] = 1.
        zero, clipped, metrics = illuminated_input(image, {'kind': 'offset', 'value': 0.}, (4, 6))
        self.assertIs(zero, image)
        shifted, clipped, metrics = illuminated_input(image, {'kind': 'offset', 'value': .1}, (4, 6))
        self.assertTrue(clipped[:, :, 4:].all())
        self.assertEqual(metrics['clipped_pixel_fraction'], 0.)
        self.assertAlmostEqual(metrics['mean_y_change'], .1, places=6)


if __name__ == '__main__':
    unittest.main()
