"""Project navigation and handoff boundaries without any provider calls."""
import hashlib
import time

import pytest

from sucoder import dream, dream_discover as discover, dream_explore as explore
from tests.test_dream import git, workspace  # noqa: F401


class Source:
    name = "sample"

    def __init__(self, files, extra=()):
        self.files, self.extra = files, extra
        self.blobs = {hashlib.sha1(data).hexdigest(): data for data in files.values()}
        self.greps = 0

    def git(self, *args, **kwargs):
        if args[0] == "grep":  # git grep -c -z at a commit, as Inventory._match_counts calls it.
            self.greps += 1
            query, commit = args[args.index("-e") + 1].encode(), args[args.index("-e") + 2]
            scope = args[args.index("--") + 1] if "--" in args else ""
            rows = [f"{commit}:{name}".encode() + b"\0" + str(count).encode()
                for name, data in sorted(self.files.items())
                if (not scope or name == scope or name.startswith(scope + "/")) and b"\0" not in data[:8000]
                for count in [sum(query in line for line in data.split(b"\n"))] if count]
            return b"".join(row + b"\n" for row in rows)
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


def test_line_ends_past_the_file_clamp_but_starts_past_it_do_not():
    raw = b"".join(b"line %d\n" % n for n in range(1, 110))  # 109 lines, as in the live run.
    inventory = discover.Inventory(Source({"handoff.org": raw}), "pinned")
    result = inventory.lookup(request("read", "handoff.org", start=100, end=220), 131072)
    assert (result["start"], result["end"], result["total_lines"]) == (100, 109, 109)
    assert result["text"].encode() == raw[raw.index(b"line 100\n"):]
    assert inventory.selected([dict(path="handoff.org", start=1, end=220)])[0]["locator"]["lines"] == [1, 109]
    with pytest.raises(dream.DreamError, match="outside the pinned file"):
        inventory.lookup(request("read", "handoff.org", start=110, end=220), 131072)
    with pytest.raises(dream.DreamError, match="outside the pinned file"):
        inventory.selected([dict(path="handoff.org", start=110, end=220)])


@pytest.mark.parametrize("given, expected", [
    ("reply-evidence", "reply-evidence"), ("Reply_Evidence", "Reply_Evidence"), ("", ""),
    ("Confined launch verification and local merge", "confined-launch-verification-and-local-merge"),
    ("  Dream: budgets & replay (v2)!  ", "dream-budgets-replay-v2"), ("___", ""), ("x" * 200, "x" * 128),
])
def test_explorer_topic_phrases_become_simple_names(given, expected):
    assert discover.topic_name(given) == expected
    assert not expected or discover.TOPIC.fullmatch(expected)


@pytest.mark.parametrize("navigation", [1, 2])
def test_search_advances_past_oversized_blobs_and_records_omissions(navigation):
    inventory = discover.Inventory(Source({"a-large.txt": b"x" * (2 * 1024 * 1024 + 1) + b"needle\n",
        "b-small.txt": b"needle\n"}), "pinned", navigation=navigation)
    value = request("search")
    value["query"] = "needle"
    result = inventory.lookup(value, 131072)
    assert result["state"] == "incomplete" and result["next_offset"] == 2
    assert result["matches"][0]["path"] == "b-small.txt"
    assert "a-large.txt" in result["omitted"]


LONG = b"".join(b"def needle_gate(self):\n" if n == 4941 else b"line %d\n" % n for n in range(1, 10635))
NOISE = {f"lsms_library/countries/c{i:03}.py": b"import pandas\n" * 40 for i in range(300)}


def test_one_git_grep_finds_a_deep_match_without_reading_files_that_lack_it():
    # The second LSMS_Library dream (2026-11-13): the function sits at line 4941 of a
    # 10,634-line country.py, and 128 files per call never reached it across the tree.
    source = Source({**NOISE, "lsms_library/country.py": LONG})
    inventory = discover.Inventory(source, "pinned")
    for path in ("", "lsms_library", "lsms_library/country.py"):
        result = inventory.lookup(request("search", path, query="def needle_gate"), 131072)
        assert result["state"] == "complete" and result["searched_files"] == 1, path
        assert result["matches"] == [dict(path="lsms_library/country.py", line=4941, text="def needle_gate(self):")]
        assert (result["matching_files"], result["total_matches"]) == (1, 1)
    common = inventory.lookup(request("search", "lsms_library/countries", query="import pandas"), 16384)
    assert common["state"] == "incomplete" and (common["matching_files"], common["total_matches"]) == (300, 12000)


def test_navigation_1_keeps_the_frozen_directory_only_semantics():
    inventory = discover.Inventory(Source({**NOISE, "lsms_library/country.py": LONG}), "pinned", navigation=1)
    by_file = inventory.lookup(request("search", "lsms_library/country.py", query="def needle_gate"), 131072)
    assert by_file == dict(state="complete", matches=[], next_offset=0, searched_bytes=0, searched_files=0, omitted=[])
    tree = inventory.lookup(request("search", query="def needle_gate"), 131072)
    assert tree["state"] == "incomplete" and tree["searched_files"] == 128 and not tree["matches"]
    assert inventory.lookup(request("list", "nowhere"), 131072)["entries"] == []
    assert inventory.source.greps == 0


def test_a_path_naming_a_file_or_nothing_is_reported_as_such():
    inventory = discover.Inventory(Source({"docs/guide.md": b"needle\n", "root.txt": b"x\n"}), "pinned")
    listed = inventory.lookup(request("list", "docs/guide.md"), 131072)
    assert listed["state"] == "complete" and [e["path"] for e in listed["entries"]] == ["docs/guide.md"]
    assert listed["entries"][0]["kind"] == "file"
    for action, path in (("list", "nowhere"), ("search", "nowhere"), ("search", "docs/guide")):
        result = inventory.lookup(request(action, path, query="needle" if action == "search" else ""), 131072)
        assert result["state"] == "no_such_path" and result["path"] == path, (action, path)
    absent = inventory.lookup(request("search", "docs", query="not there"), 131072)
    assert absent["state"] == "complete" and absent["matches"] == [] and absent["matching_files"] == 0
    assert inventory.lookup(request("search", query="needle\nx"), 131072)["matching_files"] == 0


def test_search_reads_the_pinned_commit_not_the_working_tree(workspace):
    repo, _, config, _ = workspace
    (repo / "deep").mkdir()
    (repo / "deep" / "long.txt").write_bytes(LONG)
    git(repo, "add", "deep/long.txt")
    git(repo, "commit", "-qm", "deep file")
    commit = git(repo, "rev-parse", "HEAD")
    (repo / "deep" / "long.txt").write_bytes(b"edited away\n")
    (repo / "untracked.txt").write_bytes(b"def needle_gate(self):\n")
    source = dream.GitSource(config, "sample", time.monotonic() + 30)
    inventory = discover.Inventory(source, commit)
    found = inventory.lookup(request("search", query="def needle_gate"), 131072)
    assert found["matches"] == [dict(path="deep/long.txt", line=4941, text="def needle_gate(self):")]
    assert (found["state"], found["searched_files"], found["matching_files"]) == ("complete", 1, 1)
    none = inventory.lookup(request("search", "deep/long.txt", query="edited away"), 131072)
    assert (none["state"], none["matches"], none["matching_files"]) == ("complete", [], 0)


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
