"""Regression checks for per-rank KV bytes and TP-instance capacity."""

import unittest

from RL_Framework.infra.cost_model.model import RolloutCostModel


class RolloutKVCapacityTests(unittest.TestCase):
    def setUp(self):
        self.model = RolloutCostModel(
            num_params=14e9, d_model=4096, n_layers=40, n_kv_heads=8,
            head_dim=128, dtype_bytes=2, flops_peak=1e15, mem_bw=2e12,
            mem_capacity=80e9, tp_comm_overhead=8e-6,
            kv_frag_rate=0.04, act_workspace_bytes=1e9,
        )

    def test_capacity_matches_per_rank_memory_budget(self):
        for tp, expected in ((1, 298828), (2, 767578), (4, 1705078)):
            with self.subTest(tp=tp):
                capacity = self.model.compute_token_capacity(tp)
                self.assertEqual(capacity, expected)
                available_per_rank = (80e9 - 29e9 / tp) * 0.96
                bytes_per_token_per_rank = 2 * 40 * (8 // tp) * 128 * 2
                self.assertLessEqual(capacity * bytes_per_token_per_rank,
                                     available_per_rank)
                self.assertGreater((capacity + 1) * bytes_per_token_per_rank,
                                   available_per_rank)

    def test_capacity_accounts_for_replicated_kv_heads(self):
        tp = 16
        available_per_rank = (80e9 - 29e9 / tp) * 0.96
        bytes_per_token_per_rank = 2 * 40 * 128 * 2
        expected = int(available_per_rank / bytes_per_token_per_rank)
        self.assertEqual(self.model.compute_token_capacity(tp), expected)

    def test_oom_and_batch_use_corrected_capacity(self):
        self.assertFalse(self.model.check_oom(767578, tp=2))
        self.assertTrue(self.model.check_oom(767579, tp=2))
        self.assertAlmostEqual(
            self.model.compute_steady_state_batch(2, 32768, 0),
            767578 / 32768,
        )

    def test_exhausted_kv_pool_has_zero_capacity(self):
        self.model.mem_capacity = 1e9
        self.assertEqual(self.model.compute_token_capacity(2), 0)
        self.assertTrue(self.model.check_oom(1, tp=2))


if __name__ == "__main__":
    unittest.main()
