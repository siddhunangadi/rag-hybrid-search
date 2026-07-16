from datetime import datetime, timedelta, timezone

from rag_hybrid_search.audit import AuditEvent, AuditLog


def _event(**overrides) -> AuditEvent:
    defaults = dict(
        event_id="evt-1",
        event_type="query",
        timestamp=datetime.now(timezone.utc),
        request_id="req-1",
        key_id="key-1",
        role="reader",
        endpoint="/answer",
        action="answer",
        status="success",
    )
    defaults.update(overrides)
    return AuditEvent(**defaults)


def test_record_and_query_roundtrip(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(_event(event_id="evt-1"))
    log.record(_event(event_id="evt-2"))

    events, total = log.query()
    assert total == 2
    assert {e.event_id for e in events} == {"evt-1", "evt-2"}


def test_query_filters_by_event_type(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(_event(event_id="evt-1", event_type="query"))
    log.record(_event(event_id="evt-2", event_type="upload", document_id="doc-1"))

    events, total = log.query(event_type="upload")
    assert total == 1
    assert events[0].event_id == "evt-2"


def test_query_filters_by_document_id_and_status(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(_event(event_id="evt-1", event_type="upload", document_id="doc-1", status="success"))
    log.record(_event(event_id="evt-2", event_type="upload", document_id="doc-1", status="failure"))
    log.record(_event(event_id="evt-3", event_type="upload", document_id="doc-2", status="success"))

    events, total = log.query(document_id="doc-1", status="failure")
    assert total == 1
    assert events[0].event_id == "evt-2"


def test_query_time_range_filters(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    now = datetime.now(timezone.utc)
    log.record(_event(event_id="old", timestamp=now - timedelta(days=2)))
    log.record(_event(event_id="new", timestamp=now))

    events, total = log.query(start=now - timedelta(hours=1))
    assert total == 1
    assert events[0].event_id == "new"


def test_query_sort_and_pagination(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    now = datetime.now(timezone.utc)
    for i in range(5):
        log.record(_event(event_id=f"evt-{i}", timestamp=now + timedelta(seconds=i)))

    page, total = log.query(sort="asc", offset=0, limit=2)
    assert total == 5
    assert [e.event_id for e in page] == ["evt-0", "evt-1"]

    page_desc, _ = log.query(sort="desc", offset=0, limit=2)
    assert [e.event_id for e in page_desc] == ["evt-4", "evt-3"]


def test_events_are_appended_not_rewritten(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.record(_event(event_id="evt-1"))
    log.record(_event(event_id="evt-2"))

    lines = path.read_text().splitlines()
    assert len(lines) == 2
