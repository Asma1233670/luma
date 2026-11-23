"""End-to-end wiring of the Luma agent with a scripted fake model (no OpenAI key or network needed).

Verifies the parts that unit tests on tools can't: the SDK Runner injects LumaContext into tools,
tool results reach the model as JSON, the dynamic instructions carry the time, and tools run from
the SDK's worker threads against the database.
"""
import asyncio
import json

from agents import ModelResponse, Runner, Usage
from agents.models.interface import Model
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from app.agent.context import LumaContext
from app.agent.luma import build_agent


def tool_call(name: str, args: dict, call_id: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(type="function_call", name=name, arguments=json.dumps(args),
                                    call_id=call_id, id=f"fc_{call_id}")


def message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1", role="assistant", status="completed", type="message",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])])


class ScriptedModel(Model):
    """Replays prepared responses and records what the agent sent it."""

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), []

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, *, previous_response_id=None, conversation_id=None, prompt=None):
        self.calls.append({"instructions": system_instructions, "input": input, "tools": tools})
        step = self.steps.pop(0)
        output = step(self.calls) if callable(step) else step
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *a, **k):
        raise NotImplementedError


def tool_outputs(call) -> list[str]:
    return [item["output"] for item in call["input"] if isinstance(item, dict) and item.get("type") == "function_call_output"]


def test_agent_remembers_then_recalls_through_the_sdk_runner(db):
    model = ScriptedModel(
        [tool_call("record_sighting", {"thing": "glasses", "place": "kitchen", "context": "on the table",
                                       "at": "2026-11-21T08:14:00+00:00"}, "c1")],
        [tool_call("last_seen", {"thing": "glasses"}, "c2")],
        lambda calls: [message("Your glasses were on the kitchen table.")],
    )
    result = asyncio.run(Runner.run(build_agent(model), "I left my glasses on the kitchen table, then ask later",
                                    context=LumaContext(db=db)))

    assert result.final_output == "Your glasses were on the kitchen table."
    # the tool really ran against the database, in the SDK's worker thread
    assert db.read("MATCH (s:Sighting)-[:IN]->(p:Place {name:'kitchen'}) RETURN count(s) AS c")[0]["c"] == 1
    # the model received the previous tool result as JSON text (not Python's repr)
    seen_by_model = json.loads(tool_outputs(model.calls[2])[-1])
    assert seen_by_model["found"] is True and seen_by_model["sightings"][0]["place"] == "kitchen"
    assert seen_by_model["sightings"][0]["at"] == "2026-11-21T08:14:00+00:00"


def test_agent_sees_tools_instructions_and_current_time(db):
    model = ScriptedModel([message("ok")])
    asyncio.run(Runner.run(build_agent(model), "hi", context=LumaContext(db=db)))
    sent = model.calls[0]
    assert {t.name for t in sent["tools"]} >= {"last_seen", "add_person", "create_node", "run_read_query"}
    assert "You are Luma" in sent["instructions"] and "Current local time:" in sent["instructions"]


def test_a_failing_tool_call_is_reported_to_the_model_not_raised(db):
    model = ScriptedModel(
        [tool_call("create_node", {"label": "Spaceship", "name": "x"}, "c1")],   # unregistered type
        lambda calls: [message("I could not do that.")],
    )
    result = asyncio.run(Runner.run(build_agent(model), "make a spaceship", context=LumaContext(db=db)))
    assert result.final_output == "I could not do that."
    assert "not registered" in json.loads(tool_outputs(model.calls[1])[-1])["error"]


def test_terminal_chat_loop(db):
    from app.cli import chat

    typed = iter(["hello", "", "exit"])
    said = []
    model = ScriptedModel([message("Hi, I'm Luma.")])
    asyncio.run(chat(build_agent(model), LumaContext(db=db), read=lambda prompt: next(typed), write=said.append))
    assert said == ["luma> Hi, I'm Luma."]          # the empty line was skipped, "exit" ended the loop