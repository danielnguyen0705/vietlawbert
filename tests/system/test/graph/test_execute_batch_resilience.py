"""Tier-2: execute_batch chịu lỗi — không mất bản ghi khi kết nối chết tạm thời,
và ĐẾM đúng số bản ghi bị bỏ khi lỗi vĩnh viễn. Dùng fake driver, không cần Neo4j."""
from database.neo4j_client import Neo4jClient


class _Res:
    def consume(self): return None

class _Sess:
    def __init__(self, drv): self.drv = drv
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, cypher, parameters=None):
        self.drv.calls += 1
        if self.drv.calls in self.drv.fail_calls:
            raise RuntimeError("Failed to read from defunct connection (timed out)")
        return _Res()

class _Driver:
    def __init__(self): self.calls = 0; self.fail_calls = set(); self.closed = 0
    def session(self): return _Sess(self)
    def close(self): self.closed += 1
    def verify_connectivity(self): return True


def _mk(driver):
    c = Neo4jClient.__new__(Neo4jClient)
    c.uri, c.user, c.password = "bolt://x", "u", "p"
    c.driver = driver
    c.last_dropped = 0
    c._dropped_total = 0
    return c


def test_transient_connection_recovers_no_loss(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    drv = _Driver(); drv.fail_calls = {1}      # lần chạy đầu: lỗi kết nối rồi phục hồi
    c = _mk(drv)
    def fake_reconnect():
        c.driver = _Driver()                   # driver mới sạch (mô phỏng tái kết nối)
        c._reconnected = getattr(c, "_reconnected", 0) + 1
    monkeypatch.setattr(c, "_reconnect", fake_reconnect)

    total = c.execute_batch("CYPHER", [{"i": i} for i in range(10)], batch_size=5, max_retries=5)
    assert total == 10            # nạp đủ sau khi tái kết nối
    assert c.last_dropped == 0
    assert getattr(c, "_reconnected", 0) >= 1


def test_permanent_failure_counts_dropped(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    drv = _Driver(); drv.fail_calls = {1, 2, 3, 4, 5}   # sub-batch đầu lỗi mọi lần thử
    c = _mk(drv)
    monkeypatch.setattr(c, "_reconnect", lambda: None)

    total = c.execute_batch("CYPHER", [{"i": i} for i in range(10)], batch_size=5, max_retries=5)
    assert total == 5             # sub-batch sau vẫn nạp được
    assert c.last_dropped == 5    # 5 bản ghi lỗi bị đếm, KHÔNG âm thầm
    assert c._dropped_total == 5


# ---- G1-c: drop_database ưu tiên server-side, fallback khi cú pháp cũ ----
class _Counters:
    nodes_deleted = 123

class _Summary:
    counters = _Counters()

class _DropSess:
    def __init__(self, drv): self.drv = drv
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, cypher, **kw):
        self.drv.queries.append(cypher)
        if "IN TRANSACTIONS" in cypher:
            if self.drv.server_fails:
                raise RuntimeError("Neo.ClientError.Statement.SyntaxError: CALL {} IN TRANSACTIONS")
            return _DropConsume()
        # fallback loop: xóa hết trong 1 lô rồi 0
        self.drv.loop_calls += 1
        n = 5 if self.drv.loop_calls == 1 else 0
        return _DropSingle(n)

class _DropConsume:
    def consume(self): return _Summary()

class _DropSingle:
    def __init__(self, n): self.n = n
    def single(self): return {"c": self.n}

class _DropDriver:
    def __init__(self, server_fails): self.server_fails=server_fails; self.queries=[]; self.loop_calls=0
    def session(self): return _DropSess(self)

def _drop_client(driver):
    c = Neo4jClient.__new__(Neo4jClient); c.driver = driver; return c

def test_drop_uses_server_side_when_available():
    d = _DropDriver(server_fails=False)
    _drop_client(d).drop_database()
    assert any("IN TRANSACTIONS" in q for q in d.queries)
    assert d.loop_calls == 0  # KHÔNG rơi vào fallback

def test_drop_falls_back_on_old_neo4j():
    d = _DropDriver(server_fails=True)
    _drop_client(d).drop_database()
    assert d.loop_calls >= 1  # đã fallback vòng lặp client
