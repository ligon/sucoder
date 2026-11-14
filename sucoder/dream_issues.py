"""Explicit correction-task export; never invoked by an unattended dream."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.parse import quote

from . import dream, dream_editorial, dream_explore, dream_memory, dream_support
from .dream import DreamError, canonical, digest


LABEL = "dream-memory"
MAX_SCAN_BYTES = 2 * 1024 * 1024


def _gh(*args, data=None):
    try:
        result = subprocess.run(["gh", *args], input=data, text=True, capture_output=True,
            timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DreamError("Dream issue export requires an authenticated gh command: " + str(exc), 3) from exc
    if result.returncode:
        raise DreamError("GitHub issue operation failed; inspect the marker before retrying: " + result.stderr.strip()[:2048], 3)
    if len(result.stdout.encode()) > 16 * 1024 * 1024:
        raise DreamError("GitHub issue listing exceeds its inspection allowance", 3)
    return result.stdout


def existing_issues(repository):
    pages = json.loads(_gh("api", "--paginate", "--slurp", f"repos/{repository}/issues?state=all&per_page=100"))
    if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
        raise DreamError("Invalid GitHub issue listing", 3)
    return [row for page in pages for row in page if "pull_request" not in row]


def repository_name(source, explicit=None):
    if explicit is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", explicit):
            raise DreamError("Use --repository OWNER/REPO")
        return explicit
    found = set()
    for remote in source.git("remote").decode().splitlines():
        value = source.git("remote", "get-url", remote).decode().strip()
        match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:)([^/]+/[^/]+?)(?:\.git)?/?", value)
        if match:
            found.add(match[1])
    if len(found) != 1:
        raise DreamError("Cannot identify one project GitHub repository; supply --repository OWNER/REPO")
    return repository_name(source, found.pop())


def _inventory(source, commit):
    rows = source.git("ls-tree", "-r", "-z", commit, "--", dream_memory.PREFIX)
    pages, receipts, used = {}, [], 0
    for row in filter(None, rows.split(b"\0")):
        info, name = row.split(b"\t", 1)
        mode, kind, oid = info.decode().split()
        path = name.decode()
        selected = path.startswith(".sucoder/wiki/topics/") and path.endswith(".org")
        receipt = bool(re.fullmatch(r"\.sucoder/wiki/receipts/[0-9a-f]{64}\.json", path))
        if not selected and not receipt:
            continue
        if kind != "blob" or mode not in ("100644", "100755"):
            raise DreamError("Correction inventory contains a non-regular memory entry", 3)
        size = int(source.git("cat-file", "-s", oid).strip())
        if used + size > MAX_SCAN_BYTES:
            raise DreamError("Correction inventory exceeds its evidence allowance; no issues filed", 3)
        raw = source.blob(oid, MAX_SCAN_BYTES - used)
        used += len(raw)
        if selected:
            pages[path] = raw.decode()
        else:
            value = dream._json(raw)
            dream_support.check_receipt(value)
            receipts.append(value)
    if len(pages) > 128:
        raise DreamError("Correction inventory exceeds the native page allowance", 3)
    return pages, receipts


def _provenance(receipts, path, target, page_hash, repository_uuid):
    found = []
    for receipt in receipts:
        review = receipt.get("review", {})
        if (not isinstance(review, dict) or review.get("verdict") != "accept" or review.get("failures") or
                any(row.get("severity") == "blocking" for row in review.get("findings", []))):
            continue
        if receipt.get("repository_uuid") != repository_uuid or not any(o.get("path") == path and
                o.get("sha256") == page_hash and o.get("rule") == "file-utf8-v1" for o in receipt.get("outputs", [])):
            continue
        units = {u["unit_id"]: u for u in receipt.get("units", [])}
        groups = [([units.get(c.get("unit_id")) for c in ref.get("citations", [])], receipt.get("project_commit"))
            for ref in receipt.get("references", []) if ref.get("path") == path and ref.get("target") == target]
        groups += [([c.get("unit") for c in row.get("citations", [])], None)
            for row in receipt.get("inherited_support", []) if row.get("path") == path and
                row.get("target") == target and row.get("page_sha256") == page_hash]
        for cited, project_commit in groups:
            private = not cited or any(u is None or u.get("kind") != "git" or
                u.get("repository_uuid") != repository_uuid for u in cited)
            commit = project_commit or next((u.get("commit") for u in cited
                if u is not None and u.get("kind") == "git" and u.get("repository_uuid") == repository_uuid), None)
            found.append(dict(commit=commit, private=private, units=cited))
    return found


def drafts(source, commit, repository, existing=(), *, allow_private=False):
    """Inspect reviewed correction tasks, with no model, GitHub or git writes."""
    identity = dream._repository_uuid(source, commit)
    if identity["provisional"]:
        raise DreamError("Correction export needs a reviewed memory source registry for stable markers", 3)
    pages, receipts = _inventory(source, commit)
    details = dream_editorial.inspect(pages, source.deadline) if pages else {}
    records, pending = [], []
    for path, page in details.items():
        raw = pages[path].encode()
        page_hash = digest(raw)
        for heading in page["headings"]:
            target = heading["target"]
            props = heading["properties"]
            if not heading["state"] or not target or not props.get("TARGET"):
                continue
            relative = path[len(dream_memory.PREFIX):]
            marker = f"<!-- sucoder-dream: {identity['repository_uuid']}/{relative}#{target} -->"
            matches = [row for row in existing if marker in (row.get("body") or "")]
            opened = heading["state"] not in dream_editorial.TERMINAL
            issue_open = any(row.get("state", "").lower() == "open" for row in matches)
            state = ("open_task_without_issue" if not matches else
                "open_task_with_open_issue" if issue_open else "closed_issue_with_open_task") if opened else (
                "terminal_task_with_open_issue" if issue_open else "terminal_task_without_open_issue")
            record = dict(page=relative, target=target, task_state=heading["state"], marker=marker, state=state,
                issues=[dict(number=row["number"], state=row["state"], url=row.get("html_url", row.get("url"))) for row in matches])
            records.append(record)
            if not opened or matches:
                continue
            location = re.fullmatch(r"(.+):([1-9][0-9]*)(?:-([1-9][0-9]*))?", props["TARGET"])
            evidence = _provenance(receipts, path, target, page_hash, identity["repository_uuid"])
            if not location or not props.get("ACCEPTANCE") or not evidence:
                record.update(withheld="Missing TARGET, ACCEPTANCE or reviewed evidence provenance")
                continue
            filename, start, end = location[1], int(location[2]), int(location[3] or location[2])
            dream._path(filename)
            revisions = {e["commit"] for e in evidence if isinstance(e["commit"], str) and
                re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", e["commit"])}
            if len(revisions) != 1:
                record.update(withheld="Missing or ambiguous pinned project commit")
                continue
            pinned = revisions.pop()
            blob = source.entry(pinned, filename)
            if end < start or blob is None or end > len(source.blob(blob).splitlines()):
                record.update(withheld="TARGET is unavailable at the pinned project commit")
                continue
            if any(e["private"] for e in evidence) and not allow_private:
                record.update(withheld="Task cites session, external or missing evidence; use --allow-private after reviewing disclosure")
                continue
            link = f"https://github.com/{repository}/blob/{pinned}/{quote(filename, safe='/')}#L{start}-L{end}"
            end_of_text = min([heading["end"]] + [h["begin"] for h in page["headings"]
                if heading["begin"] < h["begin"] < heading["end"] and h["level"] > heading["level"]])
            text = raw[heading["begin"]:end_of_text].decode()
            if heading["logbook"]:
                text = text.replace(heading["logbook"], "")
            body = (text + f"\n\nTarget: [{filename}:{start}-{end}]({link})\n\n"
                f"Memory: `{relative}#{target}`, generation `{commit}`.\n\n"
                f"Acceptance: {props['ACCEPTANCE']}\n\n{marker}\n")
            pending.append(dict(title=heading["title"], body=body, labels=[LABEL], marker=marker,
                page=relative, target=target, target_url=link, private_evidence=any(e["private"] for e in evidence)))
    return dict(schema_version=1, operation="dream_issues", state="drafted", memory_commit=commit,
        repository=repository, drafts=pending, tasks=records)


def run(config, *, repository=None, file=False, list_only=False, allow_private=False, cwd=None):
    if file and list_only:
        raise DreamError("Use --file or --list, not both")
    root, name, item, _, _ = dream_explore.project(config, cwd)
    bound = replace(config, mirror_root=root.parent, mirrors={name: replace(item, mirror_name=root.name)}, targets={})
    source = dream.GitSource(bound, name, time.monotonic() + 60)
    branch = dream_memory.configured(bound, name)
    if not branch or not (commit := dream_memory.tip(source, branch)):
        raise DreamError("No dream memory branch exists for this project", 3)
    dream_memory.check_tree(source, commit)
    repository = repository_name(source, repository)
    value = drafts(source, commit, repository, existing_issues(repository), allow_private=allow_private)
    if list_only:
        value.update(state="listed")
        value.pop("drafts")
        return value
    if file and value["drafts"]:
        labels = json.loads(_gh("api", "--paginate", "--slurp", f"repos/{repository}/labels?per_page=100"))
        if not any(row["name"] == LABEL for page in labels for row in page):
            _gh("api", "--method", "POST", f"repos/{repository}/labels", "--input", "-",
                data=json.dumps(dict(name=LABEL, color="c5def5", description="Reviewed dream memory correction")))
        filed = []
        for draft in value["drafts"]:
            # Exact multiline body in a private temporary file; no shell interpolation.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="sucoder-dream-issue-", suffix=".md") as body:
                body.write(draft["body"])
                body.flush()
                url = _gh("issue", "create", "--repo", repository, "--title", draft["title"],
                    "--body-file", body.name, "--label", LABEL).strip()
            filed.append(dict(marker=draft["marker"], url=url))
        value.update(state="filed", filed=filed)
    return value
