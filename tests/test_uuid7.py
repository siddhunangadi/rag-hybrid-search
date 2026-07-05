import re
import time

from rag_hybrid_search.uuid7 import uuid7


def test_uuid7_format():
    value = uuid7()
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        value,
    )


def test_uuid7_is_time_ordered():
    first = uuid7()
    time.sleep(0.002)
    second = uuid7()
    assert first < second


def test_uuid7_unique():
    values = {uuid7() for _ in range(1000)}
    assert len(values) == 1000
