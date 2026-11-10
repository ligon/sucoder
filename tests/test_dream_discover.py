"""Project navigation and handoff boundaries without any provider calls."""
import hashlib

import pytest

from sucoder import dream, dream_discover as discover, dream_explore as explore


class Source:
    name = "sample"

    def __init__(self, files, extra=()):
        self.files, self.extra = files, extra
        self.blobs = {hashlib.sha1(data).hexdigest(): data for data in files.values()}

    def git(self, *args, **kwargs):
        rows = [f"100644 blob {hashlib.sha1(data).hexdigest()} {len(data)}\t{name}".encode()
            for name, data in self.files.items()]
        return b"\0".join(rows + list(self.extra)) + b"\0"

    def blob(self, oid, *args):
        return self.blobs[oid]


def request(action, path="", **kwargs):
    return dict(dict(action=action, path=path, query="", offset=0, start=0, end=0), **kwargs)


def test_project_navigation_reaches_unrelated_deep_directories_and_reports_pagination():
    inventory = discover.Inventory(Source({**{f"area-{i:03}/deep/note.txt": b"Evidence\n" for i in range(90)},
        "root.txt": b"Root\n"}), "pinned")
    first = inventory.lookup(request("list"), 131072)
    second_request = request("list")
    second_request["offset"] = first["next_offset"]
    second = inventory.lookup(second_request, 131072)
    assert first["state"] == "incomplete" and second["state"] == "complete"
    assert len(first["entries"]) + len(second["entries"]) == 91
    deep = inventory.lookup(request("read", "area-089/deep/note.txt"), 131072)
    assert deep["text"] == "Evidence\n" and deep["byte_end"] == 9


def test_git_line_ranges_preserve_lf_semantics_crlf_and_multibyte_evidence():
    raw = "one\r\nvertical\vcontrol\nthree caf\u00e9\n".encode()
    inventory = discover.Inventory(Source({"note.txt": raw}), "pinned")
    result = inventory.lookup(request("read", "note.txt", start=2, end=3), 131072)
    assert result["text"].encode() == raw[5:]
    selected = inventory.selected([dict(path="note.txt", start=2, end=3)])
    assert selected[0]["locator"]["lines"] == [2, 3]
    assert selected[0]["locator"]["commit"] == "pinned"


def test_search_advances_past_oversized_blobs_and_records_omissions():
    inventory = discover.Inventory(Source({"a-large.txt": b"x" * (2 * 1024 * 1024 + 1),
        "b-small.txt": b"needle\n"}), "pinned")
    value = request("search")
    value["query"] = "needle"
    result = inventory.lookup(value, 131072)
    assert result["state"] == "incomplete" and result["next_offset"] == 2
    assert result["matches"][0]["path"] == "b-small.txt"
    assert "a-large.txt" in result["omitted"]


def test_search_advances_when_one_file_has_more_matches_than_fit():
    inventory = discover.Inventory(Source({"a.txt": b"needle repeat\n" * 3000,
        "b.txt": b"needle distinct qualification\n"}), "pinned")
    first = inventory.lookup(request("search", query="needle"), 131072)
    assert first["state"] == "incomplete" and first["next_offset"] == 1
    assert first["truncated_matches"][0]["path"] == "a.txt"
    second = inventory.lookup(request("search", query="needle", offset=first["next_offset"]), 131072)
    assert second["matches"][0]["text"] == "needle distinct qualification"


@pytest.mark.parametrize("path", ["../private", "/private", "a/../b", "a\\b"])
def test_project_lookup_rejects_traversal(path):
    inventory = discover.Inventory(Source({"ok.txt": b"ok\n"}), "pinned")
    with pytest.raises(dream.DreamError):
        inventory.lookup(request("read", path), 131072)


def test_nonregular_and_binary_files_are_not_transmitted():
    inventory = discover.Inventory(Source({"binary": b"\0credential", "text": b"text\n"},
        extra=[b"120000 blob abc 12\tlink", b"160000 commit abc -\tsubmodule"]), "pinned")
    assert {row["path"] for row in inventory.omitted} == {"link", "submodule"}
    for name in ("link", "submodule", "binary"):
        with pytest.raises(dream.DreamError):
            inventory.lookup(request("read", name), 131072)


def test_discovery_delivery_and_selected_evidence_are_bounded():
    inventory = discover.Inventory(Source({"long": b"x" * 65537 + b"\n"}), "pinned")
    assert inventory.lookup(request("read", "long"), 131072)["state"] == "range_too_large"
    assert inventory.lookup(request("list"), 0)["state"] == "discovery_delivery_exhausted"
    with pytest.raises(dream.DreamError, match="65536"):
        inventory.selected([dict(path="long", start=0, end=0)])


@pytest.mark.parametrize("purpose", ["A" * 4096, "Keep useful understanding"])
def test_editorial_handoff_bounds_long_multiline_reasoning_without_mutating_the_record(purpose):
    outcome = dict(rationale="Long\nreason " * 400, questions=["Question\n" * 1000] * 8)
    saved = list(outcome["questions"])
    result = explore._purpose(dict(purpose=purpose), outcome)
    dream._text(result, "purpose")
    assert result.startswith(purpose) and len(result) <= 4096
    assert outcome["questions"] == saved and "\n" in outcome["rationale"]
