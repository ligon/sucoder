"""Emit the actual body-only protocol from existing semantic test fakes."""
import json

from sucoder import dream_submission


def source_text(source):
    return source["text"] if "text" in source else "".join(passage["text"] for passage in source["passages"])


def handle_reply(task, result):
    """Adapt old semantic fakes to supplied handles and explicit review checks."""
    if task["contract"] not in ("source-grounded-wiki-v7", "source-grounded-wiki-v8", "source-grounded-wiki-v9") or result is None:
        return
    result.setdefault("requests", [])
    reviewer = task["role"] == "reviewer"
    result.setdefault("task_checks" if reviewer else "task_actions", [])
    if reviewer:
        if task["contract"] in ("source-grounded-wiki-v8", "source-grounded-wiki-v9"):
            from sucoder import dream_review
            rows = [] if result.get("verdict") == "lookup" else [dict(unit_id=unit["unit_id"],
                **{name: dict(status="pass", reason="Fixture checks the supplied source and frozen prior.", evidence=[])
                   for name in dream_review.CATEGORIES}) for unit in task["input"]["candidate"]["review_gate"]["units"]]
            result.setdefault("page_checks", rows)
        return
    result.setdefault("decision", "change")
    result.setdefault("rationale", "Apply the source-grounded fixture correction.")
    passages = [(source["unit_id"], passage) for source in task["input"]["sources"] for passage in source.get("passages", [])]
    for owner in [*result["references"], *result["task_actions"]]:
        field = "citations" if "citations" in owner else "evidence"
        selected = []
        for citation in owner[field]:
            if "handle" in citation:
                selected.append(citation)
                continue
            quote = citation["quote"]
            matches = [dict(handle=passage["handle"]) for unit_id, passage in passages if unit_id == citation["unit_id"]
                and quote.strip() and (quote in passage["text"] or passage["text"].strip() and passage["text"] in quote)]
            selected.extend(matches or [citation])
        owner[field] = selected


def native_reply(task, result, usage, trace, request=None):
    handle_reply(task, result)
    if "submission_contract" not in task and task["contract"] not in ("source-grounded-wiki-v7", "source-grounded-wiki-v8", "source-grounded-wiki-v9"):
        return result, usage, trace
    if "submission_contract" in task:
        trace["submission"] = dream_submission.manifest(task, request)
    if result is None:
        return result, usage, trace
    body = ({key: value for key, value in result.items() if key not in dream_submission.FIELDS}
            if "submission_contract" in task else result)
    events = [json.loads(line) for line in trace["stdout"].splitlines() if line.strip()]
    for event in events:
        if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message":
            event["item"]["text"] = json.dumps(body)
        if event.get("type") == "result":
            event["structured_output"] = body
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == "StructuredOutput":
                    block["input"] = body
    trace["stdout"] = "\n".join(json.dumps(event) for event in events) + "\n"
    return result, usage, trace
