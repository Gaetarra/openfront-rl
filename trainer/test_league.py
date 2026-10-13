import collections
import unittest

import numpy as np

from trainer.league import League, panel_table, ratings


class LeagueTest(unittest.TestCase):
    def test_opponents_that_beat_the_learner_are_picked_more_often(self):
        np.random.seed(0)
        league = League(["strong", "weak", "new"], slots=4)
        league.results["strong"] = [0] * 30 + [1] * 10  # the learner wins a quarter
        league.results["weak"] = [1] * 48 + [0] * 2
        w = dict(zip(league.pool, league.weights()))
        self.assertAlmostEqual(league.win_rate("new"), 0.5, msg="a new member starts at 1/2")
        self.assertGreater(w["strong"], w["new"])
        self.assertGreater(w["new"], w["weak"])
        picks = collections.Counter()
        for _ in range(4000):
            league.pick()
            picks[league.current] += 1
        for p in league.pool:
            self.assertAlmostEqual(picks[p] / 4000, w[p], delta=0.03)

    def test_only_the_last_games_set_the_weight(self):
        league = League(["a"], slots=1)
        league.results["a"] = [0] * 100 + [1] * 50
        self.assertAlmostEqual(league.win_rate("a"), 51 / 52)

    def test_a_finished_game_counts_against_its_slots_opponent_and_the_next_is_against_the_pick(self):
        league = League(["a", "b"], slots=3)
        league.slot, league.current = ["a", "a", "b"], "b"
        league.finished([{"env": 0, "rank": 1}, {"env": 2, "rank": 2}])
        self.assertEqual(league.results, {"a": [1], "b": [0]})
        self.assertEqual(league.slot, ["b", "a", "b"], "slot 1's game goes on against its opponent")

    def test_a_resume_restarts_every_slot_against_the_saved_pick(self):
        league = League(["a", "b"], slots=3)
        league.add("c")
        league.add("a")
        league.slot, league.current = ["a", "b", "c"], "c"
        league.results["a"].append(1)
        resumed = League(["a", "b"], slots=3)
        resumed.load(league.state())
        self.assertEqual((resumed.pool, resumed.results), (["a", "b", "c"], {"a": [1], "b": [], "c": []}))
        self.assertEqual(resumed.slot, ["c"] * 3)


PANEL = ["checkpoints/bc3b-v4/latest.pt", "checkpoints/bc/update0060000.pt", "checkpoints/ppo-bc2/update005000.pt"]


def evals(true, games=400, seed=0):
    """Frozen evaluations of snapshots with known Elo ratings ({update: rating}) against a panel at 0, -100 and +50."""
    rng = np.random.default_rng(seed)
    out = []
    for u, r in true.items():
        vs = {m: {"win_rate": rng.binomial(games, 1 / (1 + 10 ** ((rm - r) / 400))) / games, "episodes": games}
              for m, rm in zip(PANEL, (0, -100, 50))}
        out.append({"update": u, "minutes": 17.0, "vs": vs})
    return out


class RatingsTest(unittest.TestCase):
    def test_known_ratings_come_back_with_the_first_panel_member_at_0(self):
        true = {0: 100, 250: 150, 500: 200, 750: 250}
        fit, trend = ratings(evals(true), PANEL)
        self.assertEqual(fit[PANEL[0]], (0.0, 0.0))
        for p, r in {**true, PANEL[1]: -100, PANEL[2]: 50}.items():
            self.assertLess(abs(fit[p][0] - r), 3 * fit[p][1], p)
            self.assertLess(fit[p][1], 40, "400 games a pair")
        self.assertLess(abs(trend[0] - 50), 3 * 15)
        self.assertGreater(trend[1], 0.95)

    def test_a_flat_rating_does_not_read_as_rising(self):
        fit, trend = ratings(evals({u: 100 for u in range(0, 1250, 250)}, seed=1), PANEL)
        self.assertLess(abs(trend[0]), 20)  # 50 a step above

    def test_a_perfect_score_stays_finite_and_one_snapshot_has_no_trend(self):
        perfect = [{"update": 0, "minutes": 17.0, "vs": {m: {"win_rate": 1.0, "episodes": 40} for m in PANEL}}]
        fit, trend = ratings(perfect, PANEL)
        self.assertTrue(400 < fit[0][0] < 5000)
        self.assertIsNone(trend)

    def test_the_control_page_table(self):
        e = evals({0: 100, 250: 150})
        rows = panel_table(e, PANEL, *ratings(e, PANEL))
        self.assertEqual([r[0] for r in rows], ["update 250 (17 min)", "update 0 (17 min)", "bc3b-v4/latest",
                                                "bc/update0060000", "ppo-bc2/update005000",
                                                "Rating trend, last 2 snapshots"])
        self.assertEqual(rows[2][1], "0, the anchor")


if __name__ == "__main__":
    unittest.main()
