import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lesion_metrics import match_lesions, grouped_counts


class LesionTests(unittest.TestCase):
    def test_maximum_cardinality_beats_greedy_overlap(self):
        g = np.zeros((9, 10), bool)
        g[1:3, 1:8] = True
        g[5:7, 1:8] = True
        p = np.zeros_like(g)
        p[1:7, 1:4] = True
        p[1:3, 6] = True
        r = match_lesions(g, p)
        self.assertEqual((r['tp'], r['fn'], r['fp']), (2, 0, 0))
        self.assertEqual({(x['gt_id'], x['pred_id']) for x in r['matches']}, {(1, 2), (2, 1)})

    def test_iou_threshold_is_inclusive_not_any_contact(self):
        g = np.ones((1, 10), bool)
        p = np.zeros_like(g); p[0, 0] = True
        self.assertEqual(match_lesions(g, p)['tp'], 1)
        g = np.ones((1, 11), bool)
        p = np.zeros_like(g); p[0, 0] = True
        self.assertEqual(match_lesions(g, p)['tp'], 0)

    def test_merged_prediction_only_matches_one_gt(self):
        g = np.zeros((10, 15), bool)
        g[2:5, 2:5] = True
        g[2:5, 8:11] = True
        p = np.zeros_like(g)
        p[2:5, 2:11] = True
        r = match_lesions(g, p)
        self.assertEqual((r['tp'], r['fn'], r['fp']), (1, 1, 0))

    def test_split_prediction_has_extra_false_positive(self):
        g = np.zeros((10, 15), bool)
        g[2:5, 2:11] = True
        p = g.copy()
        p[:, 6] = False
        r = match_lesions(g, p)
        self.assertEqual((r['tp'], r['fn'], r['fp']), (1, 0, 1))

    def test_single_pixel_and_diagonal_connectivity(self):
        g = np.eye(3, dtype=bool)
        r = match_lesions(g, g)
        self.assertEqual((r['gt_count'], r['tp'], r['gt_areas']), (1, 1, [3]))
        g = np.ones((1, 1), bool)
        self.assertEqual(match_lesions(g, g)['tp'], 1)

    def test_empty_masks_and_ignored_pixels(self):
        g = np.zeros((3, 3), bool)
        self.assertEqual(match_lesions(g, g)['tp'], 0)
        p = g.copy()
        p[1, 1] = True
        self.assertEqual(match_lesions(g, p)['fp'], 1)
        self.assertEqual(match_lesions(g, p, ~p)['fp'], 0)

    def test_size_bins_include_equal_boundaries(self):
        r = {'gt_areas': [32, 33, 74, 75], 'matches': [{'gt_id': 1}, {'gt_id': 4}]}
        self.assertEqual(grouped_counts(r, [32, 74]), {
            'small': {'gt': 1, 'detected': 1}, 'medium': {'gt': 2, 'detected': 0},
            'large': {'gt': 1, 'detected': 1}})


if __name__ == '__main__':
    unittest.main()
