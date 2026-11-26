"""Bounded JSONL traversal; raw scans are not admitted evidence (ledger 3-5)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import time

from . import dream, dream_isolate, dream_transcript, dream_view
from .dream import DreamError, canonical, digest


LIMITS = dict(chunk_bytes=65536, alignment_bytes=65536, scan_bytes=16 * 1024 * 1024,
    record_bytes=16 * 1024 * 1024, records=4096, sessions=8,
    inventory_entries=8192, inventory_metadata=16384, inventory_candidates=1024,
    tail_file_bytes=1024 * 1024, tail_total_bytes=32 * 1024 * 1024)
QUALIFICATION = "unfrozen; traversal completion does not establish query absence in a coherent source revision"


def validate_limits(value):
    dream._object(value, tuple(LIMITS))
    if any(type(value[k]) is not int or not 0 < value[k] <= LIMITS[k] for k in LIMITS):
        raise DreamError("Invalid frozen navigation resource limits", 3)
    return value


class Reader:
    """Chunked complete records with separate alignment and raw-work counters."""

    def __init__(self, row, position, deadline, limits, *, align=False, skip_start=-1):
        self.row, self.deadline, self.limits = row, deadline, limits
        self.position, self.pending, self.skip_start = position, b"", skip_start
        self.scan_bytes, self.alignment_bytes, self.parts = 0, 0, []
        self.stopped, self.aligning = False, align
        path = Path(row["path"])
        if not path.is_absolute() or path != path.resolve():
            raise DreamError("source_identity_changed", 3)
        self.stream = os.fdopen(dream_transcript._open_source(path), "rb", buffering=0)
        self.identity = os.fstat(self.stream.fileno())
        expected = row.get("source_identity", [self.identity.st_dev, self.identity.st_ino])
        if (not stat.S_ISREG(self.identity.st_mode) or self.identity.st_size < row["bytes"] or
                expected != [self.identity.st_dev, self.identity.st_ino]):
            self.stream.close()
            raise DreamError("source_identity_changed", 3)
        self.stream.seek(position)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                after = os.stat(self.row["path"], follow_symlinks=False)
                if ((after.st_dev, after.st_ino) != (self.identity.st_dev, self.identity.st_ino) or
                        not stat.S_ISREG(after.st_mode) or after.st_size < self.row["bytes"] or
                        Path(self.row["path"]).resolve() != Path(self.row["path"])):
                    raise DreamError("source_identity_changed", 3)
        finally:
            self.stream.close()

    def _read(self, size, alignment=False):
        key = "alignment_bytes" if alignment else "scan_bytes"
        left = self.limits[key] - getattr(self, key)
        size = min(size, self.limits["chunk_bytes"], left, self.row["bytes"] - self.stream.tell())
        if size <= 0 or time.monotonic() >= self.deadline - 0.025:
            self.stopped = True
            return b""
        start = self.stream.tell()
        data = self.stream.read(size)
        if not data:
            raise DreamError("source_shrank", 3)
        setattr(self, key, getattr(self, key) + len(data))
        if self.parts and self.parts[-1]["end"] == start:
            part = self.parts[-1]
        else:
            part = dict(start=start, end=start, hash=hashlib.sha256())
            self.parts.append(part)
        part["hash"].update(data)
        part["end"] += len(data)
        return data

    def align(self):
        if not self.aligning:
            return True
        if self.position == 0:
            self.aligning = False
            return True
        self.stream.seek(self.position - 1)
        if self._read(1, True) == b"\n":
            self.aligning = False
            return True
        while self.stream.tell() < self.row["bytes"]:
            start = self.stream.tell()
            data = self._read(self.limits["chunk_bytes"], True)
            if not data:
                self.position = self.stream.tell()
                return False
            at = data.find(b"\n")
            if at >= 0:
                self.position = start + at + 1
                self.stream.seek(self.position)
                self.aligning = False
                return True
        self.position = self.row["bytes"]
        self.aligning = False
        return True

    def next(self):
        if not self.align():
            return None
        while True:
            at = self.pending.find(b"\n")
            if at >= 0:
                raw, self.pending = self.pending[:at + 1], self.pending[at + 1:]
                lo, self.position = self.position, self.position + len(raw)
                if self.skip_start >= 0 or len(raw) > self.limits["record_bytes"]:
                    if self.skip_start < 0:
                        self.skip_start = lo
                    lo, self.skip_start = self.skip_start, -1
                    return dict(start=lo, end=self.position, gap="oversized_record")
                return dict(start=lo, end=self.position, raw=raw)
            if self.position + len(self.pending) >= self.row["bytes"]:
                if self.pending:
                    lo = self.skip_start if self.skip_start >= 0 else self.position
                    self.position, self.pending = self.row["bytes"], b""
                    self.skip_start = -1
                    return dict(start=lo, end=self.position, gap="unterminated_pinned_record")
                return None
            data = self._read(self.limits["chunk_bytes"])
            if not data:
                if self.skip_start >= 0 or len(self.pending) >= self.limits["record_bytes"]:
                    if self.skip_start < 0:
                        self.skip_start = self.position
                    self.position += len(self.pending)
                    self.pending = b""
                return None
            self.pending += data
            if len(self.pending) > self.limits["record_bytes"] and b"\n" not in self.pending:
                if self.skip_start < 0:
                    self.skip_start = self.position
                self.position += len(self.pending)
                self.pending = b""

    def state(self):
        return dict(position=self.position, align=self.aligning, skip_start=self.skip_start)

    def receipts(self):
        return [dict(path=self.row["path"], byte_start=p["start"], byte_end=p["end"], sha256=p["hash"].hexdigest())
                for p in self.parts]


def normalize(row, start, end, cap, deadline, limits, contract=dream_view.CONTRACT_V1):
    """The old complete-record selection rule, without a whole-prefix read."""
    if not 0 <= start < row["bytes"] or (end and not start < end <= row["bytes"]):
        raise DreamError("Session window is outside the pinned prefix")
    target = end or row["bytes"]
    raw, lo, hi = bytearray(), None, None
    with Reader(row, start, deadline, limits, align=True) as reader:
        if not reader.align() or reader.position >= target:
            raise DreamError("No complete record starts within the selected range")
        for _ in range(limits["records"]):
            record = reader.next()
            if record is None:
                if reader.stopped:
                    raise DreamError("selection_processing_limit")
                break
            if "gap" in record:
                raise DreamError("selection_record_limit")
            if raw and record["end"] > target:
                break
            if lo is None:
                lo = record["start"]
            raw.extend(record["raw"])
            hi = record["end"]
            if hi - lo > limits["scan_bytes"]:
                raise DreamError("Selected session span exceeds its bounded range")
            if hi >= target:
                break
        else:
            raise DreamError("selection_record_limit")
        if not raw:
            raise DreamError("No complete session record fits the selected range")
        text, _ = dream_view.render(bytes(raw), row["thread_id"], lo, contract)
        if len(text.encode()) > cap:
            raise DreamError("Selected evidence exceeds 65536 bytes; select narrower windows")
        return dict(start=lo, end=hi, text=text, receipts=reader.receipts(), raw_sha256=digest(raw))


def _observation(state, scanned, aligned, gaps, omitted, earlier, *, row=None, lo=None, hi=None,
                 position=0, body="", dropped=0, target=0, next_cursor="", contract=dream_view.CONTRACT_V1):
    """One public shape for both response fitting and actual delivery."""
    value = dict(state=state, source_consistency=QUALIFICATION, raw_scanned_bytes=scanned,
        alignment_bytes=aligned, gaps=gaps, omitted_gap_details=omitted, earlier_gaps=earlier,
        inventory_complete=False, next_cursor=next_cursor,
        render_surface=contract + "; redaction, tool clipping and metadata omissions apply")
    if row is not None:
        header = ("SESSION " + row["thread_id"] + " raw bytes " + str(lo) + "-" + str(hi) +
                  "; " + str(dropped) + " records without conversation text omitted\n\n")
        value.update(path="@sessions/" + row["thread_id"], thread_id=row["thread_id"], byte_start=lo, byte_end=hi,
            session_bytes=row["bytes"], next_offset=position, text=header + body if lo is not None else "",
            range_expanded=hi is not None and hi > target)
    return value


def page(rows, action, query, state, cap, deadline, limits, contract=dream_view.CONTRACT_V1):
    """Return a public observation and private continuation/read receipts."""
    state = dict(state)
    index, position = state["session"], state["position"]
    found, body, receipts, gaps = [], [], [], []
    scanned = aligned = records = visited = dropped = 0
    lo = hi = None
    stopped = False
    skip_record_digest = state.get("record_sha256", "")
    block_index = state.get("block", 0)
    had_gaps = state.get("had_gaps", False)
    first = index
    target = state["range_end"]
    while index < len(rows) and visited < limits["sessions"]:
        row = rows[index]
        path = "@sessions/" + row["thread_id"]
        visited += 1
        if row["state"] != "finished":
            gaps.append(dict(path=path, reason="session_" + row["state"]))
            had_gaps = True
            index, position = index + 1, 0
            continue
        remaining = dict(limits, scan_bytes=limits["scan_bytes"] - scanned,
                         alignment_bytes=limits["alignment_bytes"] - aligned)
        if remaining["scan_bytes"] <= 0 or remaining["alignment_bytes"] <= 0:
            stopped = True
            break
        with Reader(row, position, deadline, remaining, align=state.get("align", False),
                    skip_start=state.get("skip_start", -1)) as reader:
            range_complete = False
            if action == "read":
                reader.align()
            while records < limits["records"]:
                if action == "read" and reader.position >= target:
                    range_complete = True
                    break
                record = reader.next()
                if record is None:
                    stopped = reader.stopped
                    if reader.skip_start >= 0:
                        gaps.append(dict(path=path, reason="oversized_record", byte_start=reader.skip_start,
                                         byte_end=reader.position, partial=True))
                        had_gaps = True
                    break
                records += 1
                start, end = record["start"], record["end"]
                if "gap" in record:
                    if skip_record_digest:
                        raise DreamError("continued_record_changed", 3)
                    gaps.append(dict(path=path, reason=record["gap"], byte_start=start, byte_end=end))
                    had_gaps = True
                    continue
                raw = record["raw"]
                raw_digest = digest(raw)
                if skip_record_digest and raw_digest != skip_record_digest:
                    raise DreamError("continued_record_changed", 3)
                if action == "read" and body and end > target:
                    reader.position, stopped = start, False
                    reader.pending = b""
                    range_complete = True
                    break
                entry = dream._json(raw)
                if not isinstance(entry, dict):
                    raise DreamError("Transcript record must be a JSON object")
                if entry.get("sessionId") is None and not list(dream_view._blocks(entry, start, contract)):
                    text, mapping = "", dict(blocks=[], dropped_records=1)
                else:
                    text, mapping = dream_view.render(raw, row["thread_id"], start, contract)
                if action == "read":
                    piece = text.split("\n\n", 1)[1] if text else ""
                    candidate = "".join(body) + piece
                    if end >= target:
                        fit = _observation("complete_with_gaps" if had_gaps else "complete", scanned + reader.scan_bytes,
                            aligned + reader.alignment_bytes, [], len(gaps), state.get("had_gaps", False), row=row,
                            lo=lo if lo is not None else start, hi=end, position=end, body=candidate,
                            dropped=dropped + mapping["dropped_records"], target=target, contract=contract)
                    else:
                        # Use bounded worst-case counters and the same public shape;
                        # gap details may be omitted, but record text may not be clipped.
                        pin = row["bytes"]
                        fit = _observation("complete_with_gaps", limits["scan_bytes"], limits["alignment_bytes"], [],
                            limits["records"] + limits["sessions"], False, row=row, lo=pin, hi=pin, position=pin,
                            body=candidate, dropped=limits["records"], target=pin, next_cursor="0" * 64, contract=contract)
                    if len(canonical(fit)) > cap:
                        if body or lo is not None:
                            reader.position, reader.pending = start, b""
                            skip_record_digest = raw_digest
                            stopped = True
                            break
                        gaps.append(dict(path=path, reason="rendered_record_too_large", byte_start=start, byte_end=end))
                        had_gaps = True
                        lo, hi = start, end
                        block_index, skip_record_digest = 0, ""
                        stopped = end < target
                        break
                    if lo is None:
                        lo = start
                    hi = end
                    body.append(piece)
                    dropped += mapping["dropped_records"]
                else:
                    data = text.encode()
                    blocks = mapping["blocks"]
                    for number in range(block_index, len(blocks)):
                        block = blocks[number]
                        value = data[block["start"]:block["end"]].decode()
                        if query not in value:
                            continue
                        at = value.find(query)
                        hit = dict(path=path, offset=start, end=end, text=value[max(0, at - 200):at + 312])
                        if len(canonical(found + [hit])) > cap - 768:
                            if not found:
                                gaps.append(dict(path=path, reason="snippet_exceeds_delivery", byte_start=start, byte_end=end))
                                had_gaps = True
                                continue
                            reader.position, reader.pending = start, b""
                            block_index, skip_record_digest, stopped = number, raw_digest, True
                            break
                        found.append(hit)
                    if stopped:
                        break
                block_index, skip_record_digest = 0, ""
            else:
                stopped = reader.position < row["bytes"]
            scanned += reader.scan_bytes
            aligned += reader.alignment_bytes
            receipts.extend(reader.receipts())
            position = reader.position
            resume = reader.state()
            if action == "read":
                target_done = range_complete or reader.position >= target or reader.position >= row["bytes"]
                if target_done:
                    index = len(rows)
                break
            if stopped:
                break
            index, position = index + 1, 0
            state.update(align=False, skip_start=-1)
    unfinished = index < len(rows)
    if unfinished:
        next_state = dict(state, session=index, position=position, block=block_index,
                          record_sha256=skip_record_digest, had_gaps=had_gaps)
        if 'resume' in locals():
            next_state.update(resume)
        # At a session boundary, a reader state from the previous session is stale.
        if index != first and position == 0:
            next_state.update(position=0, align=False, skip_start=-1)
        if next_state == state:
            return dict(observation=dict(state="processing_limit", unsearched_scope=True, source_consistency=QUALIFICATION,
                raw_scanned_bytes=scanned, alignment_bytes=aligned), continuation=None, receipts=receipts)
    else:
        next_state = None
    public_gaps = gaps[:4]
    observation = _observation("incomplete" if unfinished else ("complete_with_gaps" if had_gaps else
        "complete_unfrozen" if action == "search" else "complete"), scanned, aligned, public_gaps,
        max(0, len(gaps) - len(public_gaps)), state.get("had_gaps", False),
        row=rows[0] if action == "read" else None, lo=lo, hi=hi, position=position,
        body="".join(body), dropped=dropped, target=target, next_cursor="0" * 64 if next_state else "", contract=contract)
    if action == "search":
        observation["matches"] = found
    else:
        if lo is None and not unfinished:
            observation["state"] = "complete_with_gaps" if had_gaps else "no_complete_record"
        if gaps and gaps[-1]["reason"] == "rendered_record_too_large":
            observation["state"] = "range_too_large"
    while len(canonical(observation)) > cap and observation["gaps"]:
        observation["gaps"].pop()
        observation["omitted_gap_details"] += 1
    return dict(observation=observation, continuation=next_state, receipts=receipts)
