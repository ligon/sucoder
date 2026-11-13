"""Inline Org operations preserve history, identity and link destinations."""
import json
from pathlib import Path
import subprocess
import time

import pytest

from sucoder import dream, dream_editorial as edit, dream_wiki


TOPIC = ".sucoder/wiki/topics/replies.org"
ARCHIVE = ".sucoder/wiki/archive/replies.org"
OTHER = ".sucoder/wiki/topics/overview.org"
STAMP = "2026-11-09T08:00:00Z"
PAGE = ("#+title: Replies\n#+TODO: TODO NEXT ACTIVE WAITING REVIEW | DONE MOOT\n"
    "* Article\n:PROPERTIES:\n:CUSTOM_ID: rule\n:END:\n"
    "Native delivery and assistant text establish a reply; a summary alone does not. Caf\u00e9.\n"
    "[[#task][Decision history]]\n"
    "** Talk :editorial:\n*** TODO Correct the note\n:PROPERTIES:\n:CUSTOM_ID: task\n:ID: unique-task-id\n:OWNER: Sue\n:EFFORT: 0:15\n:END:\n"
    ":LOGBOOK:\n- State \"TODO\" from \"\" [2026-11-08 Sun 08:00]\n:END:\n"
    "Preserve [[file:../../../evidence.org][original evidence]], [[#rule][the active rule]] and [[file:overview.org::#navigation][related article]].\n"
    "**** DONE Checked detail\nCLOSED: [2026-11-08 Sun 09:00]\n:PROPERTIES:\n:CUSTOM_ID: detail\n:END:\nA retained qualification.\n")


def task(details, target="task", path=TOPIC):
    return next(heading for heading in details[path]["headings"] if heading["target"] == target)


def transition(files, state, *, note="Sue: recorded proposed task transition.", properties=None):
    before = task(edit.inspect(files))
    return dict(kind="transition", path=TOPIC, target="task", before_sha256=before["sha256"],
        state=state, note=note, properties=properties or {})


def test_rich_inspection_matches_original_hashes_and_preserves_utf8_ranges():
    details = edit.inspect({TOPIC: PAGE})
    original = dream_wiki.inventory({TOPIC: PAGE}, time.monotonic() + 30)
    assert details[TOPIC]["targets"] == original[TOPIC]["targets"]
    heading = task(details)
    assert heading["properties"]["OWNER"] == "Sue" and heading["properties"]["EFFORT"] == "0:15"
    assert heading["properties"]["CATEGORY"] == "replies"
    assert heading["inherited_tags"] == ["editorial"]
    assert heading["outline"] == ["Article", "Talk"]
    assert dream.digest(PAGE.encode()[heading["begin"]:heading["end"]]) == heading["sha256"]
    assert "2026-11-08" in heading["logbook"]


def test_all_seven_states_returned_review_reopening_and_logbook_history():
    files = {TOPIC: PAGE}
    for state in ["NEXT", "ACTIVE", "WAITING", "NEXT", "ACTIVE", "REVIEW", "ACTIVE", "REVIEW", "DONE", "ACTIVE", "MOOT"]:
        action = transition(files, state, properties={"REVIEWER": "Independent reviewer"} if state == "REVIEW" else {})
        result = edit.native(files, [action], at=STAMP)
        files.update(result["files"])
        heading = task(result["inventory"])
        assert heading["state"] == state
        assert heading["properties"]["OWNER"] == "Sue" and heading["properties"]["EFFORT"] == "0:15"
        assert 'State "TODO" from ""' in files[TOPIC]
        assert "[2026-11-09 Mon 08:00]" in heading["logbook"]
        assert "Sue: recorded proposed task transition." in heading["logbook"]
        assert ("CLOSED:" in files[TOPIC].split("**** DONE")[0]) == (state in edit.TERMINAL)
    assert 'State "ACTIVE" from "REVIEW"' in files[TOPIC]
    assert 'State "MOOT" from "ACTIVE"' in files[TOPIC]
    assert files[TOPIC].count(":LOGBOOK:") == 1


def completed():
    files = {TOPIC: PAGE, ARCHIVE: "#+title: Archive\n",
        OTHER: "* Navigation\n:PROPERTIES:\n:CUSTOM_ID: navigation\n:END:\n[[file:replies.org::#task][Recorded review]] and [[file:replies.org::#detail][detail]].\n"}
    result = edit.native(files, [transition(files, "DONE")], at=STAMP)
    files.update(result["files"])
    return files


def test_archive_preserves_properties_logbook_links_tags_and_forwarding_anchors():
    files = completed()
    original = task(edit.inspect(files))
    action = dict(kind="archive", path=TOPIC, target="task", before_sha256=original["sha256"], destination=ARCHIVE)
    result = edit.native(files, [action], at=STAMP)
    active, archive = result["files"][TOPIC], result["files"][ARCHIVE]
    archived = task(result["inventory"], path=ARCHIVE)
    assert archived["state"] == "DONE" and archived["level"] == 1
    assert archived["properties"]["ID"] == "unique-task-id"
    assert archived["properties"]["OWNER"] == "Sue" and archived["properties"]["EFFORT"] == "0:15"
    assert archived["properties"]["ARCHIVE_FILE"] == TOPIC
    assert archived["properties"]["ARCHIVE_OLPATH"] == "Article/Talk"
    assert archived["properties"]["ARCHIVE_ITAGS"] == "editorial"
    assert archived["properties"]["ARCHIVE_TIME"] == "2026-11-09 Mon 08:00"
    assert archived["logbook"] == original["logbook"]
    assert "unique-task-id" not in active and task(result["inventory"])["state"] is None
    assert task(result["inventory"], "detail")["state"] is None
    assert "Caf\u00e9" in active and "A retained qualification." in archive
    assert "file:../topics/replies.org::#rule" in archive
    assert "../../../evidence.org" in archive
    assert "file:../topics/overview.org::#navigation" in archive
    assert "file:../archive/replies.org::#task" in result["files"][OTHER]
    assert "file:../archive/replies.org::#detail" in result["files"][OTHER]
    assert "file:../archive/replies.org::#task" in active
    assert result["moves"][0]["targets"] == ["task", "detail"]
    replay = edit.native(files, [action], at=STAMP)
    assert replay == result


def test_archive_rejects_open_descendants_and_wrong_frozen_hash():
    files = completed()
    files[TOPIC] = files[TOPIC].replace("**** DONE Checked detail", "**** WAITING Checked detail")
    before = task(edit.inspect(files))
    action = dict(kind="archive", path=TOPIC, target="task", before_sha256=before["sha256"], destination=ARCHIVE)
    with pytest.raises(dream.DreamError, match="open descendant"):
        edit.native(files, [action], at=STAMP)
    action["before_sha256"] = "0" * 64
    with pytest.raises(dream.DreamError, match="frozen subtree"):
        edit.native(files, [action], at=STAMP)


def test_closing_then_archiving_one_frozen_task_is_one_inline_transaction():
    files = {TOPIC: PAGE, ARCHIVE: ""}
    closing = transition(files, "DONE")
    moving = dict(kind="archive", path=TOPIC, target="task", before_sha256=closing["before_sha256"], destination=ARCHIVE)
    result = edit.native(files, [closing, moving], at=STAMP)
    assert task(result["inventory"])["state"] is None
    assert task(result["inventory"], path=ARCHIVE)["state"] == "DONE"
    assert 'State "DONE" from "TODO"' in result["files"][ARCHIVE]


def test_inspection_does_not_follow_setup_files_or_evaluate_inline_code(tmp_path):
    canary = tmp_path / "must-not-exist"
    text = (f'#+SETUPFILE: /ssh:invalid:/private\n#+BIND: org-export-use-babel t\n'
        f'#+begin_src emacs-lisp\n(write-region "bad" nil "{canary}")\n#+end_src\n' + PAGE)
    result = edit.inspect({TOPIC: text})
    assert not canary.exists() and task(result)["properties"]["OWNER"] == "Sue"


@pytest.mark.parametrize("properties", [{"CUSTOM_ID": "replace"}, {"ARCHIVE_FILE": "/private"}, {"OWNER": ""}])
def test_actions_cannot_rewrite_identity_or_inject_archive_metadata(properties):
    files = {TOPIC: PAGE}
    action = transition(files, "NEXT", properties=properties)
    with pytest.raises(dream.DreamError):
        edit.native(files, [action], at=STAMP)


def test_archive_context_and_history_match_native_org_archiving(tmp_path):
    files = completed()
    before = task(edit.inspect(files))
    result = edit.native(files, [dict(kind="archive", path=TOPIC, target="task",
        before_sha256=before["sha256"], destination=ARCHIVE)], at=STAMP)
    source = tmp_path / "replies.org"
    archive = tmp_path / "archived.org"
    source.write_text(files[TOPIC])
    script = tmp_path / "native-archive.el"
    script.write_text("(require 'org)\n(require 'org-archive)\n"
        "(let ((org-mode-hook nil) (org-inhibit-startup t) (make-backup-files nil) "
        "(org-archive-subtree-save-file-p t) "
        f"(org-archive-location {json.dumps(str(archive) + '::')}))\n"
        f"(with-current-buffer (find-file-noselect {json.dumps(str(source))})\n"
        "(goto-char (org-find-property \"CUSTOM_ID\" \"task\"))\n(org-back-to-heading t)\n"
        "(org-archive-subtree)\n(save-buffer)))\n")
    subprocess.run(["emacs", "--batch", "-Q", "--script", str(script)], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    native = task(edit.inspect({ARCHIVE: archive.read_text()}), path=ARCHIVE)
    ours = task(result["inventory"], path=ARCHIVE)
    for key in ["ID", "CUSTOM_ID", "OWNER", "EFFORT", "ARCHIVE_OLPATH", "ARCHIVE_ITAGS", "ARCHIVE_TODO", "ARCHIVE_CATEGORY"]:
        assert ours["properties"][key] == native["properties"][key]
    assert ours["closed"] == native["closed"] and ours["state"] == native["state"] == "DONE"
    assert ours["logbook"] == native["logbook"] == before["logbook"]
    assert ours["properties"]["ARCHIVE_FILE"] == TOPIC
    assert native["properties"]["ARCHIVE_FILE"] == str(source)
    assert ours["properties"]["ARCHIVE_TIME"] == "2026-11-09 Mon 08:00"


def test_same_frozen_actions_reject_moved_or_duplicate_target_without_partial_outputs():
    files = completed()
    action = dict(kind="archive", path=TOPIC, target="task", before_sha256=task(edit.inspect(files))["sha256"], destination=ARCHIVE)
    with pytest.raises(dream.DreamError, match="Duplicate editorial action"):
        edit.native(files, [action, action], at=STAMP)
    action["target"] = "not-in-snapshot"
    with pytest.raises(dream.DreamError, match="frozen subtree"):
        edit.native(files, [action], at=STAMP)


def test_owner_changes_preserve_the_old_attribution_in_the_timestamped_note():
    files = {TOPIC: PAGE}
    result = edit.native(files, [transition(files, "NEXT", properties={"OWNER": "Casey"})], at=STAMP)
    changed = task(result["inventory"])
    assert changed["properties"]["OWNER"] == "Casey"
    assert "OWNER: Sue -> Casey" in changed["logbook"]


def test_forwarder_title_cannot_be_reinterpreted_as_a_new_open_task():
    files = completed()
    files[TOPIC] = files[TOPIC].replace("DONE Correct the note", "DONE TODO follow-up wording")
    action = dict(kind="archive", path=TOPIC, target="task", before_sha256=task(edit.inspect(files))["sha256"], destination=ARCHIVE)
    result = edit.native(files, [action], at=STAMP)
    assert task(result["inventory"])["state"] is None
    assert task(result["inventory"], path=ARCHIVE)["state"] == "DONE"


def test_frozen_timestamps_replay_identically_under_another_host_timezone(monkeypatch):
    files = {TOPIC: PAGE}
    action = transition(files, "DONE")
    first = edit.native(files, [action], at=STAMP)
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    second = edit.native(files, [action], at=STAMP)
    assert second == first
    assert task(second["inventory"])["closed"] == "[2026-11-09 Mon 08:00]"


def test_archive_history_links_keep_their_original_destinations():
    files = completed()
    files[TOPIC] = files[TOPIC].replace("Sue: recorded proposed task transition.", "Sue: retained [[#rule][accepted active rule]].")
    before = task(edit.inspect(files))
    result = edit.native(files, [dict(kind="archive", path=TOPIC, target="task",
        before_sha256=before["sha256"], destination=ARCHIVE)], at=STAMP)
    logbook = task(result["inventory"], path=ARCHIVE)["logbook"]
    assert "Sue: retained" in logbook and "accepted active rule" in logbook
    assert "file:../topics/replies.org::#rule" in logbook


def test_state_changes_do_not_adopt_source_directed_setup_settings(tmp_path):
    settings = tmp_path / "untrusted.org"
    settings.write_text("#+CATEGORY: SOURCE_SETTINGS_CANARY\n")
    files = {TOPIC: f"#+SETUPFILE: {settings}\n" + PAGE}
    result = edit.native(files, [transition(files, "NEXT")], at=STAMP)
    assert task(result["inventory"])["properties"]["CATEGORY"] == "replies"
    assert "SOURCE_SETTINGS_CANARY" not in json.dumps(result)
