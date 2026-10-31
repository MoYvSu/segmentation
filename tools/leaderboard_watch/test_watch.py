# -*- coding: utf-8 -*-
"""使用虚构榜单验证采集契约，不访问官方接口。"""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location("leaderboard_app", Path(__file__).with_name("app.py"))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

STAGE = "阶段甲"


def raw(team_id="DEMO-A", rank=1, score="9.20", **extra):
    return {"CSBH_": team_id, "TDMC_": "演示队伍" + team_id, "XH_": rank, "FS_": score,
            "SDSTBH_": app.TRACK_ID, "DQJD_": STAGE, "ZPZHTJSJ_": "2026-01-01 08:00:00",
            "DFSJ_": "2026-01-01 09:00:00", "createTime": "2026-01-01 10:00:00", **extra}


def response(*rows):
    return {"success": True, "data": list(rows), "total": len(rows)}


def normalized():
    return app.normalize_rows(response(raw(), raw("DEMO-B", 2, "8.0")), STAGE)


class ParseTests(unittest.TestCase):
    def test_reject_partial_pages_even_when_api_reports_success(self):
        data = response(raw()); data["total"] = 39
        with self.assertRaisesRegex(app.SourceError, "不完整"):
            app.normalize_rows(data, STAGE)

    def test_more_than_twenty_teams_are_kept(self):
        data = response(*(raw(f"DEMO-{i}", i, str(100 - i)) for i in range(1, 40)))
        self.assertEqual(len(app.normalize_rows(data, STAGE)), 39)

    def test_volatile_read_time_and_order_do_not_create_change(self):
        before = response(raw(), raw("DEMO-B", 2, "8.00"))
        after = copy.deepcopy(before)
        after["data"].reverse()
        for row in after["data"]:
            row["createTime"] = "2026-01-02 10:00:00"
        left = app.normalize_rows(before, STAGE)
        right = app.normalize_rows(after, STAGE)
        self.assertEqual(app.digest_rows(left), app.digest_rows(right))

    def test_unknown_score_is_not_zero_and_zero_is_valid(self):
        rows = app.normalize_rows(response(raw(score=None), raw("DEMO-B", 2, 0)), STAGE)
        self.assertIsNone(rows[0]["score"])
        self.assertEqual(rows[1]["score"], "0")

    def test_wrong_stage_duplicate_id_failure_and_nan_are_rejected(self):
        for data in [response(raw(DQJD_="阶段乙")), response(raw(), raw()),
                     {"success": False, "data": [], "total": 0}, response(raw(score="NaN"))]:
            with self.subTest(data=data), self.assertRaises(app.SourceError):
                app.normalize_rows(data, STAGE)

    def test_official_tie_order_is_not_replaced_with_own_sort(self):
        rows = app.normalize_rows(response(raw("DEMO-Z", 1, "9"), raw("DEMO-A", 2, "9")), STAGE)
        self.assertEqual([r["team_id"] for r in rows], ["DEMO-Z", "DEMO-A"])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = app.Store(Path(self.temp.name) / "history.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_unchanged_checks_do_not_create_snapshots_or_alerts(self):
        self.store.record(STAGE, normalized(), at="2026-01-01T10:00:00+08:00")
        self.store.record(STAGE, normalized(), at="2026-01-01T10:05:00+08:00")
        state = self.store.state(STAGE)
        self.assertEqual(state["checks_count"], 2)
        self.assertEqual(len(state["events"]), 1)
        self.assertEqual(state["events"][0]["kind"], "baseline")
        self.assertIsNone(state["last_changed_at"])
        history = self.store.history(STAGE, 0)
        self.assertEqual(len(history["observations"]), 2)
        self.assertEqual(len(history["snapshots"]), 1)

    def test_failed_check_preserves_scores_and_records_gap(self):
        self.store.record(STAGE, normalized(), at="2026-01-01T10:00:00+08:00")
        self.store.record(STAGE, None, error="网络断开", at="2026-01-01T10:05:00+08:00")
        state = self.store.state(STAGE)
        self.assertEqual(state["rows"][0]["score"], "9.2")
        self.assertEqual(state["last_check"]["success"], 0)
        self.assertEqual(state["last_success_at"], "2026-01-01T10:00:00+08:00")
        self.assertEqual(self.store.history(STAGE, 0)["observations"][-1]["success"], 0)

    def test_stage_isolation_changes_reversions_and_restart(self):
        rows = normalized()
        self.store.record(STAGE, rows)
        improved = copy.deepcopy(rows)
        improved[1].update(score="10", rank=1)
        improved[0]["rank"] = 2
        improved.reverse()
        self.store.record(STAGE, improved)
        self.store.record("阶段乙", rows)
        state = self.store.state(STAGE)
        self.assertEqual(state["rows"][0]["score_delta"], "2")
        self.assertEqual(state["rows"][0]["rank_delta"], 1)
        self.assertEqual(len(self.store.state("阶段乙")["events"]), 1)
        self.store.record(STAGE, rows)
        reopened = app.Store(self.store.path)
        self.assertEqual(len(reopened.history(STAGE, 0)["snapshots"]), 3)
        self.assertEqual(reopened.state(STAGE)["rows"][0]["rank_delta"], 1)

    def test_missing_team_is_observation_without_invented_reason(self):
        events = app.changes(normalized(), normalized()[:1])
        self.assertEqual(events[0]["kind"], "missing")
        self.assertIn("原因未知", events[0]["title"])

    def test_history_limit_is_disclosed_and_export_can_read_all(self):
        for _ in range(5):
            self.store.record(STAGE, normalized())
        limited = self.store.history(STAGE, 0, limit=2)
        self.assertTrue(limited["truncated"])
        self.assertEqual(len(limited["observations"]), 2)
        self.assertEqual(limited["total_checks"], 5)


class HttpTests(unittest.TestCase):
    setUp = StoreTests.setUp
    tearDown = StoreTests.tearDown
    def test_loopback_api_and_control_guards(self):
        self.store.record(STAGE, normalized())
        self.store.set_meta("source", {"title": "演示榜单", "stages": [STAGE], "checked_at": app.now()})
        collector = app.Collector(self.store, 300)
        server = app.make_server(self.store, collector, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/state") as result:
                self.assertEqual(len(json.load(result)["rows"]), 2)
            bad = Request(base + "/api/refresh", data=b"")
            with self.assertRaises(HTTPError) as error:
                urlopen(bad)
            self.assertEqual(error.exception.code, 403)
            bad = Request(base + "/api/refresh", data=b"", headers={"X-Leaderboard-Control": "1", "Origin": "https://elsewhere.example"})
            with self.assertRaises(HTTPError) as error:
                urlopen(bad)
            self.assertEqual(error.exception.code, 403)
            bad = Request(base + "/api/health", headers={"Host": "elsewhere.example"})
            with self.assertRaises(HTTPError) as error:
                urlopen(bad)
            self.assertEqual(error.exception.code, 403)
            with urlopen(base + "/api/export.json?hours=0") as result:
                self.assertIn("attachment", result.headers["Content-Disposition"])
                self.assertEqual(len(json.load(result)["observations"]), 1)
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/api/history?hours=-1")
            self.assertEqual(error.exception.code, 400)
            req = Request(base + "/api/refresh", data=b"", headers={"X-Leaderboard-Control": "1"})
            collector.last_started = __import__("time").monotonic()
            with urlopen(req) as result:
                self.assertFalse(json.load(result)["accepted"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
