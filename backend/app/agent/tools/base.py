"""Shared tool plumbing for the OpenAI Agents SDK.

Tools are plain functions decorated with @luma_tool. The SDK builds each tool's JSON schema from
the type hints and the docstring (summary = tool description, `Args:` = parameter descriptions).
The first parameter is always `ctx: RunContextWrapper[LumaContext]`; the database is `ctx.context.db`.
"""
from __future__ import annotations

import functools
import json
import logging
import re
import uuid
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pydantic
import relate
from agents import FunctionTool, RunContextWrapper, function_tool
from agents.exceptions import ModelBehaviorError

from backend.app.config import settings
from backend.app.db.guards import InvalidName

logger = logging.getLogger(__name__)


class ToolError(ValueError):
    """A problem the agent can fix by changing its call. The message is shown to the agent."""


_EXPECTED_ERRORS = (ToolError, InvalidName, ValueError, relate.Error, relate.RelateError)


_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "array": list, "object": dict, "null": type(None)}


def _describe_param(spec: dict) -> str:
    if "enum" in spec:
        return "one of " + "/".join(map(str, spec["enum"]))
    if "anyOf" in spec:
        return " or ".join(_describe_param(a) for a in spec["anyOf"])
    return spec.get("type", "any")


def _matches(spec: dict, value: Any) -> bool:
    """"""
    if "anyOf" in spec:
        return any(_matches(alt, value) for alt in spec["anyOf"])
    if "enum" in spec:
        return value in spec["enum"]
    expected = spec.get("type")
    if expected not in _JSON_TYPES:
        return True
    if isinstance(value, bool) and expected in ("integer", "number"):
        return False  # True/False are ints in Python, but not valid JSON numbers here
    return isinstance(value, _JSON_TYPES[expected])


def argument_hint(tool: FunctionTool | None, raw_arguments: str) -> str:
    """Explain what is wrong with a call using only the tool's schema (never the argument VALUES).

    The Agents SDK strips validation details by default (OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1) so that
    personal data can't leak into logs. That leaves the model unable to correct itself, so we rebuild a
    helpful message from names and types alone.
    """
    if tool is None:
        return "Check the parameter names and types and try again."
    schema = tool.params_json_schema
    props = schema.get("properties", {})
    try:
        given = json.loads(raw_arguments) if raw_arguments else {}
    except ValueError:
        return "The arguments were not valid JSON."
    if not isinstance(given, dict):
        return "The arguments must be a JSON object."
    missing = [k for k, v in props.items() if k not in given and "default" not in v]
    wrong = [f"{k} (expected {_describe_param(props[k])})"
             for k, v in given.items() if k in props and not _matches(props[k], v)]
    parts = []
    if missing:
        parts.append("missing required: " + ", ".join(missing))
    if wrong:
        parts.append("wrong type: " + ", ".join(wrong))
    expected = "; ".join(f"{k}: {_describe_param(v)}" for k, v in props.items())
    return ("; ".join(parts) + ". " if parts else "") + f"Parameters are: {expected}."


def luma_error(ctx: RunContextWrapper[Any], error: Exception, tool: FunctionTool | None = None) -> str:
    """Turn any tool failure into a JSON message the model can read and act on."""
    if isinstance(error, ModelBehaviorError):  # the model sent arguments that don't fit the schema
        name = tool.name if tool else "tool"
        cause = error.__cause__
        if isinstance(cause, pydantic.ValidationError):  # only present when the SDK is allowed to log tool data
            detail = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in cause.errors())
        else:
            detail = argument_hint(tool, getattr(ctx, "tool_arguments", "") or "")
        return json.dumps({"error": f"Invalid arguments for {name}: {detail}"})
    if isinstance(error, _EXPECTED_ERRORS):
        return json.dumps({"error": str(error)})
    logger.error("Unexpected tool failure", exc_info=error)
    return json.dumps({"error": f"Internal error ({type(error).__name__}). This is not your fault; tell the user it failed."})


def luma_tool(fn=None, *, strict: bool = True):
    """@function_tool preconfigured for Luma.

    - results are returned to the model as JSON (the SDK would otherwise send Python's str(dict))
    - failures become {"error": ...} messages instead of crashing the run
    - pass strict=False only for tools with free-form object arguments (strict mode forbids them)
    """
    def wrap(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return json.dumps(jsonable(f(*args, **kwargs)), ensure_ascii=False)

        holder: dict = {}  # lets the error handler see its own tool (created just below)
        tool = function_tool(wrapper, strict_mode=strict,
                             failure_error_function=lambda ctx, e: luma_error(ctx, e, holder.get("tool")))
        holder["tool"] = tool
        return tool
    return wrap(fn) if fn is not None else wrap


def collect_tools(*modules) -> list[FunctionTool]:
    """Every FunctionTool defined at the top level of the given modules (no list to maintain)."""
    return [v for m in modules for v in vars(m).values() if isinstance(v, FunctionTool)]


# ---- shared helpers ----------------------------------------------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def parse_time(value: str) -> datetime:
    """ISO-8601 -> aware datetime. Times without an offset use the configured zone."""
    try:
        dt = datetime.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        raise ToolError(f"Could not read time {value!r}. Use ISO 8601, e.g. 2026-11-24T15:30.")
    return dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(settings.timezone))


def jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_PRIMITIVES = (str, int, float, bool)


def clean_props(props: dict | None, reserved: frozenset[str]) -> dict:
    """Validate agent-supplied properties. Graph properties must be primitives or lists of them."""
    out: dict = {}
    for key, val in (props or {}).items():
        if not _KEY_RE.match(key):
            raise ToolError(f"Property name {key!r} must be snake_case (a-z, 0-9, _).")
        if key in reserved:
            raise ToolError(f"Property {key!r} is managed by Luma; leave it out.")
        if val is None:
            continue
        if isinstance(val, list) and all(isinstance(v, _PRIMITIVES) for v in val):
            out[key] = val
        elif isinstance(val, _PRIMITIVES):
            out[key] = val
        else:
            raise ToolError(f"Property {key!r} must be text, number, true/false or a list of those "
                            f"(no nested objects). Model structure as separate nodes and relationships.")
    return out