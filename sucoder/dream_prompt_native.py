"""Certified native instruction delivery, separate from the legacy transport."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import tempfile
import time

from . import dream, dream_native, dream_run
from .dream import DreamError, canonical, digest


PROFILE = "native-instructions-v1"
MAX_POLICY_BYTES = 32768
MAX_ARGV_ENV_BYTES = 65536
MAX_EXECUTABLE_BYTES = 512 * 1024 * 1024
MAX_CERTIFICATION_BYTES = 8 * 1024 * 1024
MAX_PROFILE_BYTES = 65536
MANAGERS = ("CODEX_MANAGED_BY_NPM", "CODEX_MANAGED_BY_BUN", "CODEX_MANAGED_BY_PNPM", "CODEX_MANAGED_BY_VITE_PLUS")


def fingerprint(path, deadline):
    path = Path(path).resolve(strict=True)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_EXECUTABLE_BYTES:
            raise DreamError("Native profile executable is not a bounded regular file", 3)
        state = hashlib.sha256()
        while True:
            if time.monotonic() >= deadline:
                raise DreamError("Native profile inspection deadline exhausted", 3)
            block = os.read(fd, 65536)
            if not block:
                break
            state.update(block)
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
            raise DreamError("Native profile executable changed during inspection", 3)
        return dict(path=str(path), sha256=state.hexdigest(), size=before.st_size,
            device=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns)
    finally:
        os.close(fd)


def resolve(harness, deadline):
    executable = shutil.which(harness)
    if harness not in ("codex", "claude") or executable is None:
        raise DreamError("Native cache profile requires an installed codex/claude", 3)
    entry = fingerprint(executable, deadline)
    native = entry
    launcher, environment = "direct-native", {}
    path = Path(entry["path"])
    if harness == "codex" and path.suffix == ".js":
        root = path.parent.parent
        wrapper = path.read_text()
        machine = platform.machine()
        arch, triple = (("arm64", "aarch64-unknown-linux-musl") if machine == "aarch64" else
                        ("x64", "x86_64-unknown-linux-musl") if machine == "x86_64" else (None, None))
        if (arch is None or root.name != "codex" or 'process.execve(binaryPath' not in wrapper or
                'CODEX_MANAGED_PACKAGE_ROOT: codexPackageRoot' not in wrapper):
            raise DreamError("Unsupported Codex launcher; stable layout requires explicit certification", 3)
        candidates = [root / "node_modules" / "@openai" / ("codex-linux-" + arch) / "vendor" / triple / "bin" / "codex",
                      root / "vendor" / triple / "bin" / "codex"]
        binary = next((item for item in candidates if item.is_file()), None)
        if binary is None:
            raise DreamError("Certified Codex vendor executable is unavailable", 3)
        native = fingerprint(binary, deadline)
        launcher = "direct-vendor-native"
        # Matches this package's wrapper in the isolated environment, which
        # inherits no npm/bun/pnpm/vite user-agent variables.
        environment = dict(CODEX_MANAGED_PACKAGE_ROOT=str(root), CODEX_MANAGED_BY_NPM="1")
    return dict(harness=harness, launcher=launcher, entry=entry, native=native, environment=environment,
                platform=dict(system=platform.system(), machine=platform.machine(), release=platform.release()))


def authentication(harness):
    key = "OPENAI_API_KEY" if harness == "codex" else "ANTHROPIC_API_KEY"
    if os.environ.get(key):
        return dict(route="environment-api-key", principal=None, account_continuity="unverified")
    base = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) if harness == "codex" else Path(
        os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    path = base / ("auth.json" if harness == "codex" else ".credentials.json")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return dict(route="unavailable", principal=None, account_continuity="unverified")
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise DreamError("Native authentication metadata is not a regular file", 3)
        raw = stream.read(1048577)
    if len(raw) > 1048576:
        raise DreamError("Native authentication metadata exceeds its allowance", 3)
    return credential_identity(raw, harness)


def credential_identity(raw, harness):
    """Observable identity only, never a credential/token fingerprint."""
    principal = None
    try:
        value = json.loads(raw)
        account = value.get("tokens", {}).get("account_id") if harness == "codex" else None
        if isinstance(account, str) and account:
            principal = digest(account.encode())  # account identifier, never a token/credential digest
    except (ValueError, AttributeError):
        pass
    return dict(route="native-credentials", principal=principal,
                account_continuity="observed" if principal else "unverified")


def validate_profile(profile):
    if not isinstance(profile, dict):
        raise DreamError("Native prompt profile must be an object", 3)
    from . import dream_submission
    dream_submission.validate(profile.get("submission_contract"))
    proof = profile.get("certification")
    capabilities = profile.get("capabilities")
    build = profile.get("build")
    auth = profile.get("authentication")
    if not all(isinstance(item, dict) for item in (proof, capabilities, build, auth)):
        raise DreamError("Native prompt profile has no complete certification", 3)
    if (not isinstance(profile, dict) or profile.get("contract") != PROFILE or
            profile.get("capabilities", {}).get("instruction_prefix") != "certified" or
            profile.get("capabilities", {}).get("history_prefix") != "unsupported" or
            profile.get("harness") not in ("codex", "claude") or
            not isinstance(profile.get("command_template"), list) or
            not all(isinstance(item, str) for item in profile["command_template"]) or
            profile.get("usage_mapping") not in ("codex-exec-inclusive-v1", "claude-print-exclusive-v1")):
        raise DreamError("Native prompt profile is not certified for instruction delivery", 3)
    variants = profile.get("variants")
    if (not isinstance(variants, list) or not variants or any(not isinstance(item, dict) for item in variants) or
            proof.get("contract") != "offline-native-cache-certification-v1" or
            proof.get("native_invocations") != 2 * len(variants) or
            any(not isinstance(proof.get(key), str) or len(proof[key]) != 64 or
                any(char not in "0123456789abcdef" for char in proof[key])
                for key in ("report_sha256", "prefix_sha256")) or
            build.get("harness") != profile["harness"] or not isinstance(build.get("environment"), dict) or
            auth.get("route") not in ("native-credentials", "environment-api-key") or
            not isinstance(profile.get("model"), str) or not profile["model"] or
            not isinstance(profile.get("version"), str) or not profile["version"] or
            profile["usage_mapping"] != ("codex-exec-inclusive-v1" if profile["harness"] == "codex" else "claude-print-exclusive-v1")):
        raise DreamError("Native prompt certification is incomplete or inconsistent", 3)
    for item in variants:
        if any(not isinstance(item.get(key), str) or len(item[key]) != 64 or
                any(char not in "0123456789abcdef" for char in item[key])
                for key in ("policy_sha256", "schema_sha256", "prefix_sha256")):
            raise DreamError("Native prompt certification has an incomplete policy/schema variant", 3)
    if profile.get("sha256") != digest(canonical({key: value for key, value in profile.items() if key != "sha256"})):
        raise DreamError("Frozen native prompt profile checksum differs", 3)


def draft_profile(build, model, version, auth, *, submission_contract="coordinator-submission-v1"):
    """Candidate profile for offline certification, never a published claim."""
    harness = build["harness"]
    template = dream_native.command(harness, "@native@", "@model@", {})
    if harness == "codex":
        template = template[:-1] + ["--disable", "code_mode_host", "--disable", "incremental_tools",
            "--disable", "prediction_preserve_parent_tools", template[-1]]
        template[template.index("--output-schema") + 1] = "@schema_path@"
        target = next(index for index, value in enumerate(template) if value.startswith("developer_instructions="))
        template[target] = "@policy_toml@"
    else:
        template[template.index("--json-schema") + 1] = "@schema_json@"
        template += ["--system-prompt", "@policy@"]
    value = dict(contract=PROFILE, harness=harness, model=model, version=version, build=build,
        authentication=auth, command_template=template, schema_family="native-role-schema-family-v1",
        capabilities=dict(instruction_prefix="unverified", user_block_boundaries="unverified",
            history_prefix="unsupported", cache_read_usage="verified", cache_write_usage="verified"),
        usage_mapping="codex-exec-inclusive-v1" if harness == "codex" else "claude-print-exclusive-v1")
    from . import dream_submission
    dream_submission.validate(submission_contract)
    if submission_contract is not None:
        value["submission_contract"] = submission_contract
    value["sha256"] = digest(canonical(value))
    return value


def variants(submission_contract="coordinator-submission-v1"):
    """Current production variants, including mandatory closeout schemas."""
    from . import dream_discover, dream_editor, dream_leads, dream_navigation, dream_prompt, dream_support
    values = []
    for mode, brief in (("project", dream_discover.BRIEF), ("experience", dream_discover.EXPERIENCE_BRIEF)):
        for leads in (False, True):
            policy = dream_prompt.GUARD + brief + dream_navigation.BRIEF + (dream_leads.BRIEF if leads else "")
            for finish in (False, True):
                schema = copy.deepcopy(dream_discover.SCHEMA_V3)
                if finish:
                    schema["properties"]["action"]["enum"] = ["finish"]
                values.append(dict(name="explorer-" + mode + ("-leads" if leads else "") + ("-finish" if finish else ""),
                    policy=policy, schema=schema))
    for editorial, briefs in ((1, dream_editor.BRIEFS), (2, dream_support.briefs(dream_editor.BRIEFS))):
        for role, brief in sorted(briefs.items()):
            values.append(dict(name=role + "-editorial-" + str(editorial), policy=dream_prompt.GUARD + brief,
                schema=dream_editor.schema(role)))
    if submission_contract is not None:
        from . import dream_submission
        dream_submission.validate(submission_contract)
        for value in values:
            value["policy"] = dream_submission.GUARD + value["policy"][len(dream_prompt.GUARD):]
            value["schema"] = dream_submission.schema(value["schema"])
    return values


def profile_key(build, model, route):
    return digest(canonical([PROFILE, build, model, route, variants()]))


def directory(config):
    path = (config.log_dir or Path("~/.sucoder/logs")).expanduser().resolve() / "dream-profiles"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def load_profile(config, harness, model, deadline):
    build, auth = resolve(harness, deadline), authentication(harness)
    key = profile_key(build, model, auth["route"])
    path = directory(config)
    fd = dream._open_private_directory(path)
    try:
        try:
            value = dream._json(dream_run._read(fd, key + ".json", MAX_PROFILE_BYTES))
        except FileNotFoundError:
            raise DreamError("No certified native cache profile for this build/model/authentication route; "
                "run dream certify-cache first, or explicitly select legacy layout", 3) from None
    finally:
        os.close(fd)
    if value.get("build") != build or value.get("model") != model or value.get("authentication", {}).get("route") != auth["route"]:
        raise DreamError("Native cache certificate does not match the selected build/model/authentication route", 3)
    validate_profile(value)
    from . import dream_submission
    if value.get("submission_contract") != dream_submission.CONTRACT:
        raise DreamError("Native cache profile uses an earlier submission contract; run dream certify-cache again", 3)
    expected = [(item["name"], digest(item["policy"].encode()), digest(canonical(item["schema"]))) for item in variants()]
    observed = [(item.get("name"), item["policy_sha256"], item["schema_sha256"]) for item in value["variants"]]
    if observed != expected:
        raise DreamError("Native cache profile does not cover the current submission policy/schema catalog", 3)
    fd = dream._open_private_directory(path)
    try:
        report = dream._json(dream_run._read(fd, "probe-" + value["certification"]["report_sha256"] + ".json", MAX_CERTIFICATION_BYTES))
    finally:
        os.close(fd)
    if (digest(canonical(report)) != value["certification"]["report_sha256"] or
            report.get("result") != "certified" or report.get("failures") or report.get("paid_inference") is not False or
            report.get("build") != build or report.get("model") != model or report.get("authentication_route") != auth["route"] or
            report.get("variants") != value["variants"] or report.get("prefix_sha256") != value["certification"]["prefix_sha256"]):
        raise DreamError("Native cache certificate has no matching successful offline report", 3)
    value = dict(value, authentication=auth)
    value.pop("sha256")
    value["sha256"] = digest(canonical(value))
    return value


def check_live(profile, deadline):
    validate_profile(profile)
    auth = authentication(profile["harness"])
    if auth != profile["authentication"]:
        raise DreamError("Native authentication route or observed principal changed", 3)
    for item in (profile["build"]["entry"], profile["build"]["native"]):
        info = os.stat(item["path"], follow_symlinks=False)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                item["device"], item["inode"], item["size"], item["mtime_ns"], item["ctime_ns"]):
            raise DreamError("Native profile build changed; preserve the job and recertify for a new job", 3)
    if time.monotonic() >= deadline:
        raise DreamError("Native prompt preflight deadline exhausted", 3)


def argv(profile, policy, schema, schema_path="@sealed-schema@"):
    slots = {"@native@": profile["build"]["native"]["path"], "@model@": profile["model"],
        "@schema_path@": schema_path, "@schema_json@": canonical(schema).decode(), "@policy@": policy,
        "@policy_toml@": "developer_instructions=" + json.dumps(policy, ensure_ascii=False)}
    return [slots.get(item, item) for item in profile["command_template"]]


def check_bounds(policy, arguments, environment):
    raw = policy.encode()
    if b"\0" in raw or len(raw) > MAX_POLICY_BYTES:
        raise DreamError("Native policy exceeds its 32 KiB bound or contains NUL", 3)
    if any("\0" in value for value in [*arguments, *environment.keys(), *environment.values()]):
        raise DreamError("Native argv/environment contains NUL", 3)
    count = sum(len(item.encode()) + 1 for item in arguments)
    count += sum(len(key.encode()) + len(value.encode()) + 2 for key, value in environment.items())
    overhead = 8192 + 8 * (len(arguments) + len(environment) + 2)
    if count > MAX_ARGV_ENV_BYTES or count + overhead > os.sysconf("SC_ARG_MAX"):
        raise DreamError("Native prompt argv/environment exceeds its launch allowance", 3)
    return count


def manifest(profile, policy, stdin, schema):
    validate_profile(profile)
    if not any(item["policy_sha256"] == digest(policy.encode()) and item["schema_sha256"] == digest(canonical(schema))
               for item in profile["variants"]):
        raise DreamError("Native instruction policy/schema variant has no offline certification", 3)
    arguments = argv(profile, policy, schema)
    # Actual environment/authentication is checked again before reservation and
    # before launch; never persist credential values or credential-file hashes.
    check_bounds(policy, arguments, {})
    return dict(contract="native-prompt-delivery-v1", profile_sha256=profile["sha256"],
        policy_sha256=digest(policy.encode()), policy_bytes=len(policy.encode()),
        stdin_sha256=digest(stdin), stdin_bytes=len(stdin), schema_sha256=digest(canonical(schema)),
        argv=arguments, environment_policy="authentication-only-native-v1", provider_request_sha256=None)


def sealed_schema(schema):
    fd = os.memfd_create("sucoder-dream-schema", os.MFD_ALLOW_SEALING)
    try:
        raw = canonical(schema)
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(raw)
            stream.flush()
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        return fd
    except BaseException:
        os.close(fd)
        raise


def environment(profile, scratch):
    """Honor the frozen route without staging competing credentials."""
    harness = profile["harness"]
    if profile["authentication"]["route"] == "environment-api-key":
        env = dict(PATH=os.environ.get("PATH", os.defpath), HOME=str(scratch), TMPDIR=str(scratch), LANG="C.UTF-8")
        home = Path(scratch) / (".codex" if harness == "codex" else ".claude")
        home.mkdir(mode=0o700)
        key = "OPENAI_API_KEY" if harness == "codex" else "ANTHROPIC_API_KEY"
        if not os.environ.get(key):
            raise DreamError("Frozen native API-key route is no longer available", 3)
        env[key] = os.environ[key]
        env["CODEX_HOME" if harness == "codex" else "CLAUDE_CONFIG_DIR"] = str(home)
        if harness == "codex":
            fd = os.open(home / "auth.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical({"OPENAI_API_KEY": env[key]}))
    else:
        env = dream_native.environment(scratch, harness)
        key = "OPENAI_API_KEY" if harness == "codex" else "ANTHROPIC_API_KEY"
        home = Path(env["CODEX_HOME" if harness == "codex" else "CLAUDE_CONFIG_DIR"])
        filename = "auth.json" if harness == "codex" else ".credentials.json"
        try:
            raw = (home / filename).read_bytes()
        except FileNotFoundError:
            raise DreamError("Frozen native credential route is no longer available", 3) from None
        if env.get(key) or credential_identity(raw, harness) != profile["authentication"]:
            raise DreamError("Staged native authentication differs from the frozen route or principal", 3)
    for key in MANAGERS:
        env.pop(key, None)
    env.update(profile["build"]["environment"])
    return env


def invoke(profile, task, schema, request, deadline):
    from . import dream_prompt, dream_submission
    dream_prompt.verify(request, task, profile, schema)
    output_schema = dream_submission.wire_schema(task)
    check_live(profile, deadline)
    policy, stdin = dream_prompt.render(request["plan"])
    assignment_sha256 = digest(canonical(task))
    submission = dream_submission.manifest(task, request)
    harness = profile["harness"]
    with tempfile.TemporaryDirectory(prefix="sucoder-dream-role-") as scratch:
        env = environment(profile, scratch)
        check_live(profile, deadline)
        boundary = dream_native.FilesystemBoundary(scratch)
        schema_fd = None
        try:
            boundary.allow(Path(profile["build"]["native"]["path"]), dream_native.READ)
            if harness == "codex":
                schema_fd = sealed_schema(output_schema)
            arguments = argv(profile, policy, output_schema, "/proc/self/fd/" + str(schema_fd))
            count = check_bounds(policy, arguments, env)
            started = time.monotonic()
            failure = None
            try:
                code, stdout, stderr = dream_native.capture(arguments, stdin, scratch, env, deadline, boundary,
                    extra_fds=() if schema_fd is None else (schema_fd,))
            except DreamError as exc:
                code, stdout, stderr = None, getattr(exc, "partial_stdout", b""), getattr(exc, "partial_stderr", b"")
                failure = str(exc)
            trace = dict(contract="native-prompt-trace-v1", harness=harness, model=profile["model"], version=profile["version"],
                input_sha256=assignment_sha256, stdout=stdout.decode(), stderr=stderr.decode(), returncode=code,
                assignment_sha256=assignment_sha256,
                boundary="linux-landlock-inline-v1", request_sha256=request["sha256"],
                delivery_manifest_sha256=digest(canonical(request["manifest"])), usage_mapping=profile["usage_mapping"],
                delivery=dict(policy_sha256=digest(policy.encode()), stdin_sha256=digest(stdin),
                    schema_sha256=digest(canonical(output_schema)), profile_sha256=profile["sha256"]),
                launch=dict(argv_bytes=count, schema_fd=schema_fd, elapsed_seconds=time.monotonic() - started))
            if submission is not None:
                trace["submission"] = submission
            if failure is not None:
                trace["validation_error"] = failure
            if code:
                return None, None, trace
            if code is None:
                return None, None, trace
            try:
                events = [dream._json(line) for line in stdout.splitlines() if line.strip()]
                kind = "turn.completed" if harness == "codex" else "result"
                if any(not isinstance(event, dict) for event in events) or sum(event.get("type") == kind for event in events) != 1:
                    raise DreamError("Native prompt trace has duplicate or missing final aggregates", 3)
                result, usage = dream_submission.parse(task, trace, request)
            except DreamError as exc:
                trace["validation_error"] = str(exc)
                return None, None, trace
            return result, usage, trace
        finally:
            boundary.close()
            if schema_fd is not None:
                os.close(schema_fd)


class Transport:
    def __init__(self, profile):
        validate_profile(profile)
        self.profile = copy.deepcopy(profile)
        self.version = profile["version"]

    def preflight(self, request, deadline):
        check_live(self.profile, deadline)
        from . import dream_prompt
        policy, _ = dream_prompt.render(request["plan"])
        # Use the same authentication-only environment without exposing it.
        with tempfile.TemporaryDirectory(prefix="sucoder-dream-preflight-") as scratch:
            env = environment(self.profile, scratch)
            from . import dream_submission
            check_bounds(policy, argv(self.profile, policy, dream_submission.wire_schema(request["assignment"])), env)

    def __call__(self, harness, model, task, schema, deadline, *, request):
        if harness != self.profile["harness"] or model != self.profile["model"]:
            raise DreamError("Frozen native prompt transport cannot change harness/model", 3)
        return invoke(self.profile, task, schema, request, deadline)
