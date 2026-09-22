"""``sucoder message``: the pure half.

The rules under test are the ones issue 26 set out: every message says who
it is from; nothing is ever typed into a pane that is a shell, because Enter
there runs the text as a command; an unprobed pane is refused unless
forced; recipients are the sessions the listing shows, jobs and login
sessions alike, and a job is reached through its allocation.
"""

from __future__ import annotations

import subprocess

import pytest

from sucoder.messaging import (
    Recipient, capture_pane_command, extract_reply, frame, new_message_id,
    pane_tail, parse_send_output, plan_recipients, send_keys_command,
)
from sucoder.sessions_report import (
    JobRow, LoginSession, Report, SessionEntry, TargetGroup,
)


def _job(job_id, token, node="n1", pane="claude"):
    return SessionEntry(
        job=JobRow(job_id, f"sucoder-{token}", "p", "a", "q", "RUNNING", "1:00", node),
        mirror=token, target="carleton-htc", session_keys=(f"{token}--carleton-htc",),
        pane=pane,
    )


def _report(*entries, logins=(), unmatched=()):
    return Report(
        groups=[TargetGroup("carleton-htc", "p / a / q", list(entries))],
        logins=list(logins), unmatched=list(unmatched),
    )


PLAN = dict(
    cluster_hosts={"hpc.brc": ("ln001.brc", object())},
    target_cluster={"carleton-htc": "hpc.brc", "savio-node": "hpc.brc"},
    confined_targets={"carleton-htc"},
)


# -- framing ---------------------------------------------------------------------

def test_frame_names_the_sender_and_flattens_to_one_line():
    line = frame("ln001 Lustre is wedged,\n  stop repairing", "ligon@lyn", "2026-09-22 03:00")
    assert line == "[message from ligon@lyn via sucoder, 2026-09-22 03:00] ln001 Lustre is wedged, stop repairing"
    assert "\n" not in line


# -- the command that types it --------------------------------------------------

def test_a_job_is_reached_through_its_allocation_on_its_own_socket():
    r = Recipient("x", "ln001.brc", "sucoder-M", socket="sucoder-M", job_id=42)
    cmd = send_keys_command(r, "hello")
    assert "srun --jobid=42 --overlap --quiet --chdir=/tmp" in cmd
    assert "TMPDIR=/tmp" in cmd
    assert "tmux -L sucoder-M send-keys -t sucoder-M -l hello" in cmd
    assert "send-keys -t sucoder-M Enter" in cmd
    assert "printf 'SENT\\t%s\\n' sucoder-M" in cmd


def test_a_login_session_is_reached_on_the_default_server_without_srun():
    r = Recipient("x", "ln001.brc", "sucoder-M")
    cmd = send_keys_command(r, "hello")
    assert "srun" not in cmd and " -L " not in cmd
    assert cmd.startswith("if tmux send-keys -t sucoder-M -l hello")


def test_the_text_is_sent_literally_not_as_key_names():
    r = Recipient("x", "h", "s")
    cmd = send_keys_command(r, "press C-c; then $HOME 'quoted'")
    # -l keeps tmux from reading C-c as a key; the shell quoting keeps the
    # rest from being expanded on the way there.
    assert "send-keys -t s -l 'press C-c; then $HOME '\"'\"'quoted'\"'\"''" in cmd


@pytest.mark.skipif(subprocess.run(["which", "tmux"], capture_output=True).returncode != 0,
                    reason="tmux not installed")
def test_the_command_types_into_a_real_tmux_session(tmp_path):
    """Run it for real: a detached tmux on a private socket, running `cat`,
    receives the line and echoes it back into the pane."""
    sock = f"sucoder-test-{tmp_path.name}"
    subprocess.run(["tmux", "-L", sock, "new-session", "-d", "-s", "sucoder-M", "cat"], check=True)
    try:
        r = Recipient("x", "local", "sucoder-M", socket=sock)
        out = subprocess.run(["bash", "-c", send_keys_command(r, "ping from the test")],
                             capture_output=True, text=True)
        assert parse_send_output(out.stdout) == (["sucoder-M"], [])
        subprocess.run(["sleep", "0.5"])
        pane = subprocess.run(["tmux", "-L", sock, "capture-pane", "-p", "-t", "sucoder-M"],
                              capture_output=True, text=True).stdout
        # Typed once, echoed once by cat.
        assert pane.count("ping from the test") == 2, pane
    finally:
        subprocess.run(["tmux", "-L", sock, "kill-server"], capture_output=True)


def test_parse_send_output_splits_sent_from_failed():
    assert parse_send_output("SENT\ta\nFAILED\tb\nnoise\n") == (["a"], ["b"])


# -- who gets it -------------------------------------------------------------------

def test_one_mirror_selects_its_job_and_frames_it_as_confined():
    chosen, skipped = plan_recipients(_report(_job(1, "M"), _job(2, "N")), mirror="M", **PLAN)
    assert skipped == []
    [r] = chosen
    assert r.job_id == 1 and r.session == "sucoder-M" and r.socket == "sucoder-M"
    assert r.host == "ln001.brc" and r.label == "M (job 1 on n1, carleton-htc)"


def test_an_unconfined_target_uses_the_default_tmux_server():
    report = Report(groups=[TargetGroup("savio-node", "p", [
        SessionEntry(job=JobRow(3, "sucoder-M", "p", "a", "q", "RUNNING", "1:00", "n2"),
                     mirror="M", target="savio-node", pane="claude"),
    ])])
    [r], _ = plan_recipients(report, mirror="M", **PLAN)
    assert r.socket == "" and r.job_id == 3


def test_a_mirror_no_config_names_is_selected_by_its_token():
    entry = _job(1, "K_Agg")
    entry.mirror = None                       # the listing could not map the token
    [r], _ = plan_recipients(_report(entry), mirror="K Agg", **PLAN)
    assert r.session == "sucoder-K_Agg"


def test_a_shell_pane_is_never_typed_into_even_when_forced():
    chosen, skipped = plan_recipients(_report(_job(1, "M", pane="bash")), mirror="M", force=True, **PLAN)
    assert chosen == []
    assert skipped == ["M (job 1 on n1, carleton-htc): agent exited; its pane is a shell, "
                       "and a message typed there would run as a command"]


def test_an_unprobed_pane_is_refused_unless_forced():
    chosen, skipped = plan_recipients(_report(_job(1, "M", pane=None)), mirror="M", **PLAN)
    assert chosen == [] and "--force sends anyway" in skipped[0]
    chosen, skipped = plan_recipients(_report(_job(1, "M", pane=None)), mirror="M", force=True, **PLAN)
    assert len(chosen) == 1 and skipped == []


def test_everyone_takes_every_live_session_jobs_and_logins_alike():
    logins = [LoginSession("ln003.brc", "sucoder-L", "savio-login", pane="claude"),
              LoginSession("ln003.brc", "sucoder-Z", "savio-login", pane="bash")]
    chosen, skipped = plan_recipients(
        _report(_job(1, "M"), _job(2, "N", pane="bash"), logins=logins), everyone=True, **PLAN,
    )
    assert [r.session for r in chosen] == ["sucoder-M", "sucoder-L"]
    assert len(skipped) == 2 and all("agent exited" in s for s in skipped)
    login = chosen[1]
    assert login.job_id is None and login.host == "ln003.brc" and login.socket == ""


def test_a_target_narrows_the_selection():
    chosen, _ = plan_recipients(_report(_job(1, "M")), mirror="M", target="savio-node", **PLAN)
    assert chosen == []
    chosen, _ = plan_recipients(_report(_job(1, "M")), mirror="M", target="carleton-htc", **PLAN)
    assert len(chosen) == 1


def test_a_job_on_a_cluster_that_did_not_answer_is_named_not_dropped():
    chosen, skipped = plan_recipients(
        _report(_job(1, "M")), mirror="M",
        cluster_hosts={}, target_cluster=PLAN["target_cluster"], confined_targets=set(),
    )
    assert chosen == [] and "cluster did not answer" in skipped[0]


def test_a_job_matching_no_target_is_named_not_guessed_at():
    stray = SessionEntry(job=JobRow(9, "sucoder-M", "p", "a", "q", "RUNNING", "1:00", "n9"), pane="claude")
    chosen, skipped = plan_recipients(_report(unmatched=[stray]), mirror="M", **PLAN)
    assert chosen == [] and "matches no configured target" in skipped[0]


def test_nothing_selected_without_a_mirror_or_everyone():
    chosen, skipped = plan_recipients(_report(_job(1, "M")), **PLAN)
    assert chosen == [] and skipped == []


def test_a_session_seen_via_the_gateway_and_its_own_node_is_one_recipient():
    """The gateway is a round-robin alias for the login nodes, so one tmux
    session can be reported under both names.  Two sends would be two
    deliveries."""
    logins = [LoginSession("hpc.brc", "sucoder-L", "savio", pane="claude"),
              LoginSession("ln003.brc", "sucoder-L", "savio", pane="claude")]
    chosen, skipped = plan_recipients(
        _report(logins=logins), mirror="L", gateway_hosts={"hpc.brc"}, **PLAN,
    )
    assert [r.host for r in chosen] == ["ln003.brc"] and skipped == []
    # With no named node reporting it, the gateway's copy is the only one.
    chosen, _ = plan_recipients(
        _report(logins=logins[:1]), mirror="L", gateway_hosts={"hpc.brc"}, **PLAN,
    )
    assert [r.host for r in chosen] == ["hpc.brc"]


# -- the reply, pulled from the pane ----------------------------------------------

def test_frame_carries_an_id_a_reply_can_name():
    line = frame("hi", "me@lyn", "2026-09-22 11:11", "ab12")
    assert line == "[message ab12 from me@lyn via sucoder, 2026-09-22 11:11] hi"
    assert len(new_message_id()) == 4 and new_message_id() != new_message_id()


def test_capture_goes_through_the_allocation_like_the_send():
    r = Recipient("x", "ln001.brc", "sucoder-M", socket="sucoder-M", job_id=42)
    cmd = capture_pane_command(r, lines=150)
    assert "srun --jobid=42 --overlap" in cmd
    assert "tmux -L sucoder-M capture-pane -p -S -150 -t sucoder-M" in cmd
    assert "srun" not in capture_pane_command(Recipient("x", "h", "sucoder-L"))


PANE = """\
 ✨ [message ab12 from ligon@lyn via sucoder, 2026-09-22 11:11] Hi there, please
    confirm.
 ● Working on it.
 ● REPLY ab12: got it, all good here; the framing arrived intact and I am
   carrying on with the tests.
 ╭────────╮
 │ > Well │
 ╰────────╯
"""


def test_extract_reply_joins_the_reflowed_rows_and_stops_at_the_prompt_box():
    assert extract_reply(PANE, "ab12") == (
        "got it, all good here; the framing arrived intact and I am carrying on with the tests."
    )


def test_extract_reply_ignores_the_message_itself_and_other_ids():
    # The framed line mentions the id too; it is not a reply.
    assert extract_reply(PANE.replace("REPLY ab12", "REPLY zz99"), "ab12") is None
    assert extract_reply("", "ab12") is None


def test_extract_reply_takes_the_last_reply_when_the_agent_corrects_itself():
    pane = PANE + "\n ● REPLY ab12: correction, one test failed.\n\n"
    assert extract_reply(pane, "ab12") == "correction, one test failed."


def test_pane_tail_drops_blank_rows_and_keeps_the_last_n():
    assert pane_tail("a\n\nb\n c \n\n", 2) == "b\n c"


def test_a_peek_may_read_a_shell_pane():
    chosen, skipped = plan_recipients(
        _report(_job(1, "M", pane="bash")), mirror="M", for_reading=True, **PLAN,
    )
    assert len(chosen) == 1 and skipped == []
