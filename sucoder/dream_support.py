"""Prior receipt support is evidence history, not fresh source verification."""
from . import dream
from .dream import DreamError, canonical, digest


VERSION = 2
MAX_BYTES = 16384
SCAN_BYTES = 2 * 1024 * 1024
BRIEF = (
    " Prior_support records receipt-backed support for existing pages, previously admitted and not re-read. "
    "Keep those supported claims and their qualifications unless new evidence contradicts them; a narrower current "
    "evidence selection is not a reason to erase them or replace them with a claim of ignorance. "
    "Explain any substantive correction with the contradicting current evidence. Independent review must check "
    "preservation as well as correction. Missing prior support metadata is not proof that a claim lacked evidence. "
    "Date lessons drawn from sessions. Before stating a lesson about project state as current advice, verify it "
    "against committed project evidence from the launch commit; otherwise keep it explicitly historical or unresolved."
    " Record each proposed correction to a project file as an open TODO task with a stable CUSTOM_ID, "
    "a TARGET property of the form path:START-END at the pinned project commit, an ACCEPTANCE property stating "
    "what the correction must accomplish, and exact evidence references for that task. Corrections are tasks, "
    "not merely Talk prose. Keep open questions in Talk; do not invent a correction or require issue filing. "
    "When committed evidence demonstrates that a correction has been made, use the ordinary reviewed task transition "
    "to DONE with that commit as acceptance evidence; never infer completion from a changed target alone."
)


def check_receipt(value):
    """Malformed history is unavailable provenance, never a Python traceback."""
    if not isinstance(value, dict):
        raise DreamError("Memory receipt must be an object", 3)
    for key in ("outputs", "units", "references", "inherited_support"):
        rows = value.get(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise DreamError("Invalid memory receipt " + key, 3)
    for row in value.get("outputs", []):
        if not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise DreamError("Invalid memory receipt output binding", 3)
    units = [row.get("unit_id") for row in value.get("units", [])]
    if any(not isinstance(uid, str) or not uid for uid in units) or len(set(units)) != len(units):
        raise DreamError("Invalid memory receipt unit identities", 3)
    for row in value.get("references", []) + value.get("inherited_support", []):
        if not isinstance(row.get("path"), str) or not isinstance(row.get("target"), str):
            raise DreamError("Invalid memory receipt reference target", 3)
        citations = row.get("citations")
        if not isinstance(citations, list) or any(not isinstance(c, dict) for c in citations):
            raise DreamError("Invalid memory receipt citations", 3)
    if any(not isinstance(c.get("unit_id"), str) for row in value.get("references", []) for c in row["citations"]):
        raise DreamError("Invalid memory receipt citation identity", 3)
    if any(not isinstance(c.get("unit"), dict) for row in value.get("inherited_support", []) for c in row["citations"]):
        raise DreamError("Invalid inherited memory receipt source", 3)
    review = value.get("review", {})
    if (not isinstance(review, dict) or not isinstance(review.get("failures", []), list) or
            not isinstance(review.get("findings", []), list) or
            any(not isinstance(row, dict) for row in review.get("findings", []))):
        raise DreamError("Invalid memory receipt review", 3)


def collect(source, files, repository_uuid):
    blobs = {item["path"]: item["blob"] for item in files}
    found, read = [], 0
    for item in files:
        if not item["path"].startswith(".sucoder/wiki/receipts/") or not item["path"].endswith(".json"):
            continue
        size = int(source.git("cat-file", "-s", item["blob"]).strip())
        if size + read > SCAN_BYTES:
            continue
        raw = source.blob(item["blob"], SCAN_BYTES - read)
        read += len(raw)
        try:
            receipt = dream._json(raw)
            check_receipt(receipt)
        except DreamError:
            # Optional history cannot block planning or establish support.
            # Read failures above still propagate; export validates strictly.
            continue
        review = receipt.get("review", {})
        if (receipt.get("repository_uuid") != repository_uuid or not isinstance(review, dict) or
                review.get("verdict") != "accept" or review.get("failures") or
                any(row.get("severity") == "blocking" for row in review.get("findings", []))):
            continue
        outputs = {o.get("path"): o for o in receipt.get("outputs", []) if isinstance(o, dict)
            and o.get("rule") == "file-utf8-v1"}
        units = {u.get("unit_id"): u for u in receipt.get("units", []) if isinstance(u, dict)}
        matching = {}
        for path, output in outputs.items():
            if path not in blobs or not path.endswith(".org"):
                continue
            page = source.blob(blobs[path], dream.MAX_BLOB_BYTES)
            if digest(page) == output.get("sha256"):
                matching[path] = output["sha256"]
        for reference in receipt.get("references", []):
            if not isinstance(reference, dict) or reference.get("path") not in matching:
                continue
            citations = []
            for citation in reference.get("citations", []):
                unit = units.get(citation.get("unit_id"))
                if unit is None:
                    continue
                citations.append(dict(unit={key: unit[key] for key in ("kind", "source_id", "repository_uuid",
                    "repository", "commit", "blob", "path", "lines", "thread_id", "raw", "render", "sha256") if key in unit},
                    byte_start=citation.get("byte_start"), byte_end=citation.get("byte_end")))
            if citations:
                row = dict(path=reference["path"], target=reference.get("target", ""),
                    page_sha256=matching[reference["path"]], receipt=item["path"],
                    evidence_status=reference.get("evidence_status"), citations=citations,
                    status="previously_admitted_not_reread")
                if len(canonical(found + [row])) > MAX_BYTES:
                    return found
                found.append(row)
        for row in receipt.get("inherited_support", []):
            if (not isinstance(row, dict) or row.get("path") not in matching or
                    row.get("page_sha256") != matching[row["path"]] or row in found):
                continue
            if len(canonical(found + [row])) <= MAX_BYTES:
                found.append(row)
    return found


def briefs(original):
    return {role: brief + BRIEF for role, brief in original.items()}
