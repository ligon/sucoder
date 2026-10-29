"""CLI adapter and isolated worker for the observation service.

Isolation bounds configuration reads/decoding as well as network children.
The relay worker owns its own deadline; it needs no signal from sudo's caller.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import sys
import warnings

import yaml

from typer.core import TyperGroup

from .config import Config, ConfigError, load_config
from .observation_process import Budget, ObservationFailure, capture
from .observations import (VERSION, MAX_BYTES, Result, WorkspaceBinding, context_data,
                           identity, list_sessions, now)


class ObservationGroup(TyperGroup):
    def parse_args(self, ctx, args):
        remaining = super().parse_args(ctx, args)
        # Click leaves the actual subcommand in protected_args and its options
        # in args. Save them before Group.invoke consumes those fields.
        command = list(getattr(ctx, "_protected_args", getattr(ctx, "protected_args", [])))
        tail = list(ctx.args)
        route = command + tail
        ctx.meta["observation"] = (("--json" in tail or "--help" in tail) and bool(route) and
            (route[0] in ("sessions", "context") or
             route[:2] in (["relay", "sessions"], ["relay", "capabilities"])))
        # Click reaches a subcommand's --help only after this group's callback
        # has loaded configuration, so flag it here for main() to skip that.
        # Tokens after "--" are operands, never the help option.
        options = route[:route.index("--")] if "--" in route else route
        ctx.meta["help_only"] = bool(command) and "--help" in options
        return remaining


def error_result(operation, budget, code):
    result = Result(operation, budget)
    if code in ("INVALID_ARGUMENT", "CONFIG_INVALID", "INTERNAL_ERROR", "INVALID_RESPONSE", "SCHEMA_UNSUPPORTED"):
        result.value["status"] = "error"
    else:
        result.value["data"] = (context_data(os.getcwd()) if operation == "session_context" else
            dict(scope="configured_remote_targets", selected_targets=[], sessions=[], records=[]))
    if code == "OUTPUT_LIMIT":
        result.source("aggregate:omitted", "aggregate")
        result.coverage("aggregate:omitted", "omitted_sources", "partial", reason=code)
    else:
        result.error(code)
    return result.finish()


def exit_code(value):
    if value["status"] == "complete":
        return 0
    if value["status"] != "error":
        return 3
    return 2 if any(e["code"] in ("CONFIG_INVALID", "INVALID_ARGUMENT") for e in value["errors"]) else 1


def validate_response(value, operation):
    """Validate our closed schema without adding a runtime JSON-schema dependency.

    Only schema keywords in the shipped contract are supported; tests compare
    this validator with Draft 2020-12 on fixtures and malformed responses.
    """
    if not isinstance(value, dict):
        raise ObservationFailure("INVALID_RESPONSE")
    if value.get("schema_version") != VERSION:
        raise ObservationFailure("SCHEMA_UNSUPPORTED")
    schema = json.loads(Path(__file__).with_name("observation-response.schema.json").read_text())

    def valid(v, s):
        if "$ref" in s:
            return valid(v, schema["$defs"][s["$ref"].split("/")[-1]])
        types = dict(object=dict, array=list, string=str, integer=int, boolean=bool, null=type(None))
        if "type" in s:
            allowed = s["type"] if isinstance(s["type"], list) else [s["type"]]
            if not any(type(v) is types[t] for t in allowed):
                return False
        if "const" in s and v != s["const"]:
            return False
        if "enum" in s and v not in s["enum"]:
            return False
        if isinstance(v, str) and not s.get("minLength", 0) <= len(v) <= s.get("maxLength", 4096):
            return False
        if type(v) is int and v < s.get("minimum", v):
            return False
        if isinstance(v, dict):
            if not set(s.get("required", [])).issubset(v):
                return False
            props = s.get("properties", {})
            if s.get("additionalProperties") is False and set(v) - set(props):
                return False
            if any(not valid(v[k], sub) for k, sub in props.items() if k in v):
                return False
        if isinstance(v, list):
            if len(v) > s.get("maxItems", len(v)) or any(not valid(x, s.get("items", {})) for x in v):
                return False
        if "oneOf" in s and sum(valid(v, sub) for sub in s["oneOf"]) != 1:
            return False
        if not all(valid(v, sub) for sub in s.get("allOf", [])):
            return False
        if "if" in s and not valid(v, s.get("then" if valid(v, s["if"]) else "else", {})):
            return False
        return True

    try:
        if not valid(value, schema) or value["operation"] != operation:
            raise ValueError()
        import uuid
        import datetime
        uuid.UUID(value["observation_id"])
        for key in ("started_at", "finished_at"):
            stamp = datetime.datetime.fromisoformat(value[key].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError()
        ids = [s["source_id"] for s in value["sources"]]
        if len(set(ids)) != len(ids):
            raise ValueError()
        references = [c["source_id"] for c in value["coverage"]] + [e["source_id"] for e in value["errors"]]
        data = value.get("data") or {}
        references += [r["source_id"] for r in data.get("records", [])]
        references += [r["key"]["source_id"] for r in data.get("sessions", [])]
        if any(s is not None and s not in ids for s in references):
            raise ValueError()
        if value["truncated"] and (value["status"] != "partial" or
                not any(e["code"] == "OUTPUT_LIMIT" for e in value["errors"])):
            raise ValueError()
        if value["status"] == "complete" and (value["errors"] or any(
                c["state"] not in ("complete", "skipped") for c in value["coverage"])):
            raise ValueError()
    except (ValueError, KeyError, TypeError, RecursionError):
        raise ObservationFailure("INVALID_RESPONSE") from None
    return value


def relay_sessions(config, options, budget, runner=capture):
    relay = config.session_relay
    sudo = ["sudo", "-n", "-u", relay.user]
    base = [*relay.command, "relay"]
    allowed = runner([*sudo, "-l", *base, "sessions", "--json"], budget)
    if allowed.returncode:
        raise ObservationFailure("AUTH_UNAVAILABLE")
    cap = runner([*sudo, *base, "capabilities", "--json"], budget)
    try:
        caps = json.loads(cap.stdout)
    except (ValueError, RecursionError):
        raise ObservationFailure("RELAY_UNSUPPORTED") from None
    if not cap.returncode and isinstance(caps, dict) and caps.get("schema_version") != VERSION:
        raise ObservationFailure("SCHEMA_UNSUPPORTED")
    if cap.returncode or caps != {"schema_version": VERSION, "operations": ["list_sessions"]}:
        raise ObservationFailure("RELAY_UNSUPPORTED")
    # The public CLI accepts integer seconds; round DOWN, never extend the
    # parent's collection deadline. Startup/cleanup also consume this budget.
    remaining = int(budget.remaining())
    if remaining < 1:
        raise ObservationFailure("TIMEOUT")
    args = ["sessions", "--json", "--timeout", str(remaining)]
    if options.get("target"):
        args += ["--target", options["target"]]
    if options.get("fast"):
        args += ["--fast"]
    if not options.get("login_nodes", True):
        args += ["--no-login-nodes"]
    reply = runner([*sudo, *base, *args], budget, limit=MAX_BYTES)
    try:
        value = validate_response(json.loads(reply.stdout), "list_sessions")
    except (ValueError, RecursionError):
        raise ObservationFailure("INVALID_RESPONSE") from None
    if reply.returncode != exit_code(value):
        raise ObservationFailure("INVALID_RESPONSE")
    # Keep discovery authority and source catalog from the relay, but identify
    # the process that was asked by the agent as observer.
    caller = identity()
    value["observer"]["host"] = caller["host"]
    value["observer"]["account"] = caller["account"]
    return value


def execute(options, budget):
    operation = options["operation"]
    try:
        path = Path(options["config"]).expanduser() if options.get("config") else Path("~/.sucoder/config.yaml").expanduser()
        if options.get("relay"):
            home = pwd.getpwuid(os.geteuid()).pw_dir
            if (os.path.realpath(os.path.expanduser("~")) != os.path.realpath(home) or
                path.resolve() != (Path(home) / ".sucoder/config.yaml").resolve()):
                raise ObservationFailure("CONFIG_INVALID")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if path.exists() or options.get("config") or options.get("relay"):
                config = load_config(path)
            else:
                config = Config(human_user=identity()["account"])
        budget.check()
        if operation == "session_context":
            return WorkspaceBinding(os.getcwd()).inspect(config, budget)
        if not options.get("relay") and config.session_relay and config.session_relay.user != identity()["account"]:
            return relay_sessions(config, options, budget)
        return list_sessions(config, budget, target=options.get("target"),
                             fast=options.get("fast", False), login_nodes=options.get("login_nodes", True))
    except ObservationFailure as exc:
        return error_result(operation, budget, exc.code)
    except (ConfigError, OSError, ValueError, yaml.YAMLError):
        return error_result(operation, budget, "CONFIG_INVALID")
    except Exception:
        return error_result(operation, budget, "INTERNAL_ERROR")


def dispatch(options, runner=capture):
    """One bounded child includes config parsing, record IO and remote probes."""
    budget = Budget(options["timeout"])
    started_at = now()
    child = dict(options, timeout=budget.remaining())
    try:
        reply = runner([sys.executable, "-c",
                        "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); "
                        "runpy.run_module('sucoder.observation_cli',run_name='__main__')",
                        str(Path(__file__).resolve().parent.parent), json.dumps(child)],
                       budget, limit=MAX_BYTES)
        value = validate_response(json.loads(reply.stdout), options["operation"])
        if reply.returncode != exit_code(value):
            raise ObservationFailure("INVALID_RESPONSE")
        import time
        value.update(started_at=started_at, finished_at=now(),
                     elapsed_ms=max(0, int((time.monotonic() - budget.started) * 1000)))
        return value
    except ObservationFailure as exc:
        return error_result(options["operation"], budget, exc.code)
    except (ValueError, RecursionError):
        return error_result(options["operation"], budget, "INVALID_RESPONSE")


if __name__ == "__main__":
    from . import observation_process
    observation_process.ISOLATED_WORKER = True
    options = json.loads(sys.argv[1])
    result = execute(options, Budget(options["timeout"]))
    print(json.dumps(result, ensure_ascii=True))
    sys.exit(exit_code(result))
