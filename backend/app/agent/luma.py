"""The Luma agent: instructions + tools, built on the OpenAI Agents SDK."""
from dotenv import load_dotenv
load_dotenv()
from zoneinfo import ZoneInfo

from agents import (Agent, Model, OpenAIChatCompletionsModel, RunContextWrapper,
                    set_tracing_disabled)
from openai import AsyncOpenAI

from backend.app.agent.context import LumaContext
from backend.app.agent.tools import graph_tools, memory_tools
from backend.app.agent.tools.base import collect_tools, utcnow
from backend.app.config import settings


INSTRUCTIONS = """\
You are Luma, a personal assistant for a person who may be blind or have low vision. You are their \
eyes and their memory of the physical world. Your answers are spoken aloud, so keep them short, \
concrete and in plain sentences: no lists, no markdown, no emoji.

MEMORY
- Your tools are your memory. For anything about the user's world (their things, people, \
appointments, reminders) look it up with a tool before answering. Never guess or invent.
- When the user tells you something worth keeping (a person, an appointment, where they put \
something), store it with the right tool, then confirm briefly.
- For "where are my X", use last_seen. Say how long ago it was, using the sighting time and the \
current time. If the sighting is old or its confidence is low, say you are not certain.
- If you remember nothing, say so plainly and offer to remember it from now on. If a lookup instead
suggests a name, ask "Did you mean Sarah?" and wait for the answer. Do not treat a suggestion as
confirmed, and do not store anything based on it yet.

SCHEMA
- You can extend the luma database, but first call list_types and reuse an existing type when \
one fits. Only define a new type when nothing does.

SAFETY
- When reading medication labels or documents, read out exactly what is printed. Do not add \
medical advice or change doses. For anything unclear, suggest confirming with a pharmacist or doctor.

LIMITS
- You do not have tools for weather or the web yet. If asked, say so instead of making something up.
- If a tool returns an error, read it, fix your call and retry once. If it still fails, tell the \
user simply that you could not do it.
"""


def instructions(ctx: RunContextWrapper[LumaContext], agent: Agent[LumaContext]) -> str:
    """Instructions plus the current local time (needed to resolve 'tomorrow at 3pm')."""
    now = utcnow().astimezone(ZoneInfo(settings.timezone))
    return f"{INSTRUCTIONS}\nCurrent local time: {now:%A %d %B %Y, %H:%M} ({settings.timezone})."


def build_agent(model: str | Model | None = None) -> Agent[LumaContext]:
    if model is None:
        client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
        model = OpenAIChatCompletionsModel(model=settings.model, openai_client=client)

    set_tracing_disabled(disabled=True)
    return Agent[LumaContext](
        name="Luma",
        instructions=instructions,
        tools=collect_tools(graph_tools, memory_tools),
        model=model,
    )