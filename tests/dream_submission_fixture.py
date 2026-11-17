"""Emit the actual body-only protocol from existing semantic test fakes."""
import json

from sucoder import dream_submission


def native_reply(task, result, usage, trace, request=None):
    if "submission_contract" not in task:
        return result, usage, trace
    trace["submission"] = dream_submission.manifest(task, request)
    if result is None:
        return result, usage, trace
    body = {key: value for key, value in result.items() if key not in dream_submission.FIELDS}
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
