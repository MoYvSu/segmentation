# -*- coding: utf-8 -*-
"""公开排行榜的本地采集器与看板；仅使用 Python 标准库。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import signal
import sqlite3
import threading
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent / "web"
DEFAULT_DATA = ROOT / "output" / "leaderboard_watch"
SOURCE_URL = (
    "https://reg.aicomp.cn/special/phb/detail?id=4839619591879659587"
    "&rwId=4839619548191789112&stbh=4839619562087518304"
)
API_URL = "https://jluat-smart-app-api.yuntu.cn/third/jsphb"
BOARD_ID = "4839619591879659587"
TASK_ID = "4839619548191789112"
TRACK_ID = "4839619562087518304"
APP_ID = "aicomp-leaderboard-watch"
TZ = timezone(timedelta(hours=8))
LOG = logging.getLogger(APP_ID)


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class SourceError(ValueError):
    """接口失败或结构不完整；不得用它覆盖上一份有效榜单。"""


def score_value(value) -> str | None:
    if value is None or str(value).strip() in ("", "--", "—"):
        return None
    try:
        score = Decimal(str(value))
    except InvalidOperation as exc:
        raise SourceError("接口包含无法识别的分数") from exc
    if not score.is_finite():
        raise SourceError("接口分数不是有限数字")
    return format(score.normalize(), "f")


def normalize_rows(response: dict, stage: str) -> list[dict]:
    if not isinstance(response, dict) or response.get("success") is not True:
        raise SourceError("排行接口未返回成功状态")
    raw = response.get("data")
    total = response.get("total")
    if not isinstance(raw, list) or not isinstance(total, int) or isinstance(total, bool):
        raise SourceError("排行接口结构改变：缺少队伍列表或总数")
    # 官网 pageNo=0 返回全部队伍，20 是前端表格的显示页长。
    # 必须核对 total；未来接口改成分页时，不能把第一页冒充全榜。
    if len(raw) != total:
        raise SourceError(f"榜单不完整：收到 {len(raw)} 队，接口声明 {total} 队")
    result, seen = [], set()
    for row in raw:
        if not isinstance(row, dict):
            raise SourceError("队伍条目不是对象")
        team_id = row.get("CSBH_")
        if not isinstance(team_id, str) or not team_id.strip() or team_id in seen:
            raise SourceError("参赛编号缺失或重复")
        seen.add(team_id)
        if row.get("SDSTBH_") != TRACK_ID or row.get("DQJD_") != stage:
            raise SourceError("接口返回了其他赛题或阶段的数据")
        rank = row.get("XH_")
        if rank is not None:
            try:
                number = Decimal(str(rank))
                if isinstance(rank, bool) or not number.is_finite() or number < 1 or number != int(number):
                    raise ValueError()
                rank = int(number)
            except (ValueError, InvalidOperation, OverflowError) as exc:
                raise SourceError("名次不是正整数") from exc
        result.append({
            "team_id": team_id,
            "name": str(row.get("TDMC_") or team_id),
            "rank": rank,
            "score": score_value(row.get("FS_")),
            "submitted_at": row.get("ZPZHTJSJ_") or None,
            "scored_at": row.get("DFSJ_") or None,
        })
    return sorted(result, key=lambda x: (x["rank"] is None, x["rank"] or 0, x["team_id"]))


def digest_rows(rows: list[dict]) -> str:
    # 不包含接口每次读取时重新生成的 createTime、返回顺序等易变信息。
    stable = sorted(rows, key=lambda x: x["team_id"])
    return hashlib.sha256(json_text(stable).encode("utf-8")).hexdigest()


def changes(previous: list[dict] | None, current: list[dict]) -> list[dict]:
    if previous is None:
        return [{"kind": "baseline", "title": f"开始记录：{len(current)} 支队伍", "team_id": None}]
    old, new = {r["team_id"]: r for r in previous}, {r["team_id"]: r for r in current}
    events = []
    for row in current:
        team_id = row["team_id"]
        before = old.get(team_id)
        event = {"team_id": team_id, "name": row["name"], "rank": row["rank"], "score": row["score"]}
        if before is None:
            events.append(dict(event, kind="new", title=f"{row['name']} 出现在榜单中"))
            continue
        if before == row:
            continue
        score_delta = None
        rank_delta = None
        if row["score"] is not None and before["score"] is not None:
            score_delta = str(Decimal(row["score"]) - Decimal(before["score"]))
        if row["rank"] is not None and before["rank"] is not None:
            rank_delta = before["rank"] - row["rank"]
        parts = []
        if row["score"] != before["score"]:
            parts.append(f"分数 {before['score'] or '未公布'} → {row['score'] or '未公布'}")
        if rank_delta:
            parts.append(f"名次 {before['rank']} → {row['rank']}")
        if row["name"] != before["name"]:
            parts.append("队名变化")
        if not parts:
            parts.append("公开提交或打分时间变化，分数及名次未变")
        kind = "score" if row["score"] != before["score"] else "rank" if rank_delta else "update"
        events.append(dict(event, kind=kind, title=f"{row['name']}：{'；'.join(parts)}",
                           score_delta=score_delta, rank_delta=rank_delta))
    for team_id in old.keys() - new.keys():
        events.append({"kind": "missing", "team_id": team_id, "name": old[team_id]["name"],
                       "title": f"{old[team_id]['name']} 本次未出现在公开榜单中（原因未知）"})
    return events


class Source:
    def post(self, payload: dict) -> dict:
        req = Request(API_URL, data=json_text(payload).encode("utf-8"), headers={
            "Content-Type": "application/json",
            "Origin": "https://reg.aicomp.cn",
            "Referer": SOURCE_URL,
            "User-Agent": "LocalLeaderboardWatch/1.0",
        })
        try:
            with urlopen(req, timeout=18) as response:
                body = response.read(4 * 1024 * 1024 + 1)
            if len(body) > 4 * 1024 * 1024:
                raise SourceError("接口响应超出大小限制")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise SourceError("接口返回的不是 JSON 对象")
            return data
        except HTTPError as exc:
            raise SourceError(f"公开接口 HTTP {exc.code}") from exc
        except (OSError, ValueError) as exc:
            if isinstance(exc, SourceError):
                raise
            raise SourceError(f"公开接口读取失败：{type(exc).__name__}") from exc

    def metadata(self) -> dict:
        response = self.post({"type": "JSJD", "bdId": BOARD_ID, "stbh": TRACK_ID})
        data = response.get("data")
        if response.get("success") is not True or not isinstance(data, dict):
            raise SourceError("阶段接口读取失败")
        if data.get("SDSTBH_") != TRACK_ID or data.get("RWID_") != TASK_ID:
            raise SourceError("阶段接口返回了其他赛题")
        stages = [s.strip() for s in (data.get("JDMC_") or "").split(",") if s.strip()]
        if not stages or len(stages) != len(set(stages)):
            raise SourceError("接口未提供有效的比赛阶段")
        # 只保留看板需要的公开信息，排除后台操作者和联系方式。
        return {"title": data.get("STMC_") or "竞赛排行榜", "stages": stages,
                "configured_update_slots": data.get("GXSJ_") or "", "checked_at": now()}

    def ranking(self, stage: str) -> list[dict]:
        response = self.post({"pageNo": 0, "pageSize": 20, "type": "JSDF", "rwId": TASK_ID,
                              "stbh": TRACK_ID, "jd": stage})
        return normalize_rows(response, stage)


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY, stage TEXT NOT NULL, observed_at TEXT NOT NULL,
                    digest TEXT NOT NULL, rows_json TEXT NOT NULL, events_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS snapshot_stage ON snapshots(stage,id);
                CREATE TABLE IF NOT EXISTS checks (
                    id INTEGER PRIMARY KEY, stage TEXT NOT NULL, checked_at TEXT NOT NULL,
                    success INTEGER NOT NULL, snapshot_id INTEGER, error TEXT);
                CREATE INDEX IF NOT EXISTS check_stage ON checks(stage,id);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def set_meta(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json_text(value)))

    def get_meta(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def record(self, stage: str, rows: list[dict] | None, error: str | None = None, at: str | None = None):
        at = at or now()
        with self.connect() as db:
            last = db.execute("SELECT * FROM snapshots WHERE stage=? ORDER BY id DESC LIMIT 1", (stage,)).fetchone()
            snapshot_id = last["id"] if last else None
            changed = False
            if rows is not None:
                digest = digest_rows(rows)
                if last is None or last["digest"] != digest:
                    events = changes(json.loads(last["rows_json"]) if last else None, rows)
                    cursor = db.execute("INSERT INTO snapshots(stage,observed_at,digest,rows_json,events_json) VALUES (?,?,?,?,?)",
                                        (stage, at, digest, json_text(rows), json_text(events)))
                    snapshot_id, changed = cursor.lastrowid, True
                db.execute("INSERT INTO checks(stage,checked_at,success,snapshot_id) VALUES (?,?,1,?)", (stage, at, snapshot_id))
            else:
                db.execute("INSERT INTO checks(stage,checked_at,success,snapshot_id,error) VALUES (?,?,0,?,?)",
                           (stage, at, snapshot_id, error or "未知错误"))
        return {"snapshot_id": snapshot_id, "changed": changed}

    def state(self, stage: str) -> dict:
        with self.connect() as db:
            db.execute("BEGIN")
            recent = db.execute("SELECT * FROM snapshots WHERE stage=? ORDER BY id DESC LIMIT 2", (stage,)).fetchall()
            last_check = db.execute("SELECT * FROM checks WHERE stage=? ORDER BY id DESC LIMIT 1", (stage,)).fetchone()
            good = db.execute("SELECT checked_at FROM checks WHERE stage=? AND success=1 ORDER BY id DESC LIMIT 1", (stage,)).fetchone()
            first = db.execute("SELECT checked_at FROM checks WHERE stage=? AND success=1 ORDER BY id LIMIT 1", (stage,)).fetchone()
            count = db.execute("SELECT count(*) FROM checks WHERE stage=?", (stage,)).fetchone()[0]
            since = datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")
            today = db.execute("SELECT count(*) FROM snapshots WHERE stage=? AND observed_at>=? AND id != (SELECT min(id) FROM snapshots WHERE stage=?)",
                               (stage, since, stage)).fetchone()[0]
            event_rows = db.execute("SELECT id,observed_at,events_json FROM snapshots WHERE stage=? ORDER BY id DESC LIMIT 60", (stage,)).fetchall()
        latest = recent[0] if recent else None
        rows = json.loads(latest["rows_json"]) if latest else []
        before = {r["team_id"]: r for r in json.loads(recent[1]["rows_json"])} if len(recent) > 1 else {}
        scored = [r for r in rows if r["score"] is not None]
        leader = scored[0] if scored else None
        for i, row in enumerate(rows):
            previous = before.get(row["team_id"])
            row["score_delta"] = str(Decimal(row["score"]) - Decimal(previous["score"])) if previous and row["score"] is not None and previous["score"] is not None else None
            row["rank_delta"] = previous["rank"] - row["rank"] if previous and previous["rank"] is not None and row["rank"] is not None else None
            row["gap_leader"] = str(Decimal(leader["score"]) - Decimal(row["score"])) if leader and row["score"] is not None else None
            ahead = rows[i - 1] if i else None
            row["gap_ahead"] = str(Decimal(ahead["score"]) - Decimal(row["score"])) if ahead and ahead["score"] is not None and row["score"] is not None else None
        events = []
        for record in event_rows:
            for index, event in enumerate(json.loads(record["events_json"])):
                events.append(dict(event, at=record["observed_at"], id=f"{record['id']}-{index}"))
        gap = str(Decimal(scored[0]["score"]) - Decimal(scored[1]["score"])) if len(scored) > 1 else None
        insights = []
        if leader:
            insights.append(f"{leader['name']} 当前列第 {leader['rank']}，公开分数 {leader['score']}。")
        if gap is not None:
            insights.append(f"前两名相差 {Decimal(gap):.4f} 分；榜单并列名次仍以官方为准。")
        if len(scored) > 1:
            close = sum(1 for r in scored[1:] if abs(Decimal(leader["score"]) - Decimal(r["score"])) <= Decimal("1"))
            insights.append(f"另有 {close} 支队伍与榜首的分差不超过 1 分。")
        if first:
            insights.append(f"从 {first[0].replace('T', ' ')[:19]} 开始观测；未读取到的历史不作推断。")
        return {"stage": stage, "rows": rows, "snapshot_id": latest["id"] if latest else None,
                "last_changed_at": latest["observed_at"] if latest and len(event_rows) > 1 else None,
                "started_at": first[0] if first else None,
                "last_check": dict(last_check) if last_check else None,
                "last_success_at": good[0] if good else None,
                "checks_count": count, "changes_today": today, "events": events[:100],
                "insights": insights, "leading_gap": gap,
                "baseline_at": first[0] if first else None}

    def history(self, stage: str, hours: int = 24, limit: int = 5000) -> dict:
        cutoff = (datetime.now(TZ) - timedelta(hours=hours)).isoformat(timespec="seconds") if hours else ""
        with self.connect() as db:
            db.execute("BEGIN")
            count = db.execute("SELECT count(*) FROM checks WHERE stage=? AND checked_at>=?", (stage, cutoff)).fetchone()[0]
            checks = db.execute("SELECT checked_at,success,snapshot_id,error FROM checks WHERE stage=? AND checked_at>=? ORDER BY id DESC LIMIT ?",
                                (stage, cutoff, limit)).fetchall()[::-1]
            ids = {c["snapshot_id"] for c in checks if c["snapshot_id"] is not None}
            snapshots = []
            # 分批读取，避免某些 SQLite 版本的 SQL 参数个数限制。
            ordered = sorted(ids)
            for start in range(0, len(ordered), 400):
                batch = ordered[start:start + 400]
                query = f"SELECT id,observed_at,rows_json FROM snapshots WHERE id IN ({','.join('?' for _ in batch)})"
                for record in db.execute(query, batch):
                    snapshots.append({"id": record["id"], "observed_at": record["observed_at"], "rows": json.loads(record["rows_json"])})
        return {"observations": [dict(c) for c in checks], "snapshots": snapshots,
                "total_checks": count, "truncated": count > limit, "hours": hours}


class Collector:
    def __init__(self, store: Store, interval: int, source=None):
        self.store, self.interval, self.source = store, interval, source or Source()
        self.wake, self.stop = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.busy = False
        self.next_at = None
        self.error = None
        self.failures = 0
        self.last_started = 0.0
        self.thread = threading.Thread(target=self.run, name="leaderboard-collector", daemon=True)

    def status(self):
        with self.lock:
            return {"collecting": self.busy, "next_check_at": self.next_at, "interval_seconds": self.interval,
                    "error": self.error, "consecutive_failures": self.failures,
                    "manual_cooldown_seconds": max(0, int(60 - (time.monotonic() - self.last_started)))}

    def refresh(self):
        with self.lock:
            remaining = max(0, int(60 - (time.monotonic() - self.last_started)))
            if self.busy or remaining:
                return {"accepted": False, "busy": self.busy, "retry_after": remaining}
            self.wake.set()
            return {"accepted": True}

    def close(self):
        self.stop.set()
        self.wake.set()

    def run(self):
        while not self.stop.is_set():
            self.wake.clear()
            with self.lock:
                self.busy, self.next_at, self.last_started = True, None, time.monotonic()
            failed, errors = False, []
            metadata = self.store.get_meta("source", {})
            try:
                age = (datetime.now(TZ) - datetime.fromisoformat(metadata["checked_at"])).total_seconds() if metadata else float("inf")
                if age >= 3600:
                    metadata = self.source.metadata()
                    self.store.set_meta("source", metadata)
            except Exception as exc:
                failed = True
                errors.append(str(exc))
                LOG.warning("阶段信息读取失败：%s", exc)
            for stage in metadata.get("stages", []):
                if self.stop.is_set():
                    break
                try:
                    rows = self.source.ranking(stage)
                    result = self.store.record(stage, rows)
                    LOG.info("%s：%s 队，%s", stage, len(rows), "新榜单" if result["changed"] else "无变化")
                except Exception as exc:
                    failed = True
                    message = str(exc) or type(exc).__name__
                    errors.append(f"{stage}：{message}")
                    self.store.record(stage, None, error=message)
                    LOG.warning("%s 采集失败：%s", stage, message)
            with self.lock:
                self.busy = False
                self.failures = self.failures + 1 if failed else 0
                delay = min(1800, self.interval * (2 ** min(self.failures, 3))) if failed else self.interval
                self.error = "；".join(errors) or None
                self.next_at = (datetime.now(TZ) + timedelta(seconds=delay)).isoformat(timespec="seconds")
            self.wake.wait(delay)


def make_server(store: Store, collector: Collector, port: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            LOG.debug(fmt, *args)

        def send(self, code, body, content_type="application/json; charset=utf-8", attachment=False):
            if isinstance(body, (dict, list)):
                body = json_text(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            if attachment:
                self.send_header("Content-Disposition", 'attachment; filename="leaderboard-history.json"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.allowed_host():
                self.send(403, {"error": "仅允许本机访问"})
                return
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            metadata = store.get_meta("source", {})
            stages = metadata.get("stages", [])
            stage = query.get("stage", [stages[0] if stages else ""])[0]
            if parsed.path == "/api/health":
                self.send(200, {"app": APP_ID, "status": "running", "collector": collector.status()})
            elif parsed.path == "/api/state":
                if stages and stage not in stages:
                    self.send(400, {"error": "未知比赛阶段"})
                    return
                state = store.state(stage)
                self.send(200, dict(state, source=metadata, source_url=SOURCE_URL, collector=collector.status(), server_time=now()))
            elif parsed.path in ("/api/history", "/api/export.json"):
                if stage not in stages:
                    self.send(400, {"error": "未知比赛阶段"})
                    return
                try:
                    hours = int(query.get("hours", ["24"])[0])
                    if hours < 0 or hours > 87600:
                        raise ValueError()
                except ValueError:
                    self.send(400, {"error": "时间范围无效"})
                    return
                history = store.history(stage, hours, limit=1000000 if parsed.path.endswith("export.json") else 5000)
                self.send(200, dict(history, source_url=SOURCE_URL, stage=stage, exported_at=now()), attachment=parsed.path.endswith("export.json"))
            else:
                static = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/chart-core.js": ("chart-core.js", "text/javascript; charset=utf-8"),
                          "/styles.css": ("styles.css", "text/css; charset=utf-8")}
                if parsed.path not in static:
                    self.send(404, {"error": "未找到页面"})
                    return
                file, mime = static[parsed.path]
                self.send(200, (WEB / file).read_bytes(), mime)

        def do_POST(self):
            if not self.allowed_host() or self.headers.get("X-Leaderboard-Control") != "1":
                self.send(403, {"error": "本机控制请求无效"})
                return
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}:
                self.send(403, {"error": "请求来源无效"})
                return
            if self.path == "/api/refresh":
                self.send(200, collector.refresh())
            elif self.path == "/api/shutdown":
                self.send(200, {"stopping": True})
                collector.close()
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send(404, {"error": "未知操作"})
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="AIC 公开排行榜本地监控")
    parser.add_argument("command", choices=["serve", "once", "stop"], nargs="?", default="serve")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if args.command == "stop":
        req = Request(f"http://127.0.0.1:{args.port}/api/shutdown", data=b"", headers={"X-Leaderboard-Control": "1"})
        with urlopen(req, timeout=5) as response:
            print(response.read().decode())
        return
    if args.interval < 60:
        parser.error("自动采集间隔至少 60 秒，建议 300 秒")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.data_dir / "watch.log", encoding="utf-8"), logging.StreamHandler()])
    store = Store(args.data_dir / "history.sqlite3")
    if args.command == "once":
        source = Source()
        metadata = source.metadata()
        store.set_meta("source", metadata)
        for stage in metadata["stages"]:
            rows = source.ranking(stage)
            result = store.record(stage, rows)
            print(json_text({"stage": stage, "teams": len(rows), **result}))
        return
    collector = Collector(store, args.interval)
    server = make_server(store, collector, args.port)
    def shutdown(signum, frame):
        collector.close()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    collector.thread.start()
    LOG.info("看板 http://127.0.0.1:%s/，每 %s 秒采集一次；数据 %s", args.port, args.interval, args.data_dir)
    try:
        server.serve_forever(poll_interval=0.3)
    finally:
        collector.close()
        server.server_close()
        collector.thread.join(timeout=25)


if __name__ == "__main__":
    main()
