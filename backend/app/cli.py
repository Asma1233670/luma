"""Terminal chat with Luma. Run: python -m backend.app.cli"""
import asyncio
import os
import sys

from dotenv import load_dotenv

from agents import Runner

from backend.app.agent.context import LumaContext
from backend.app.agent.luma import build_agent
from backend.app.agent.tools.memory_tools import ensure_self
from backend.app.config import settings
from backend.app.db.bootstrap import bootstrap
from backend.app.db.client import LumaDB

load_dotenv()

async def chat(agent, ctx: LumaContext, read=input, write=print) -> None:
    history: list = []
    while True:
        try:
            text = (await asyncio.to_thread(read, "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        result = await Runner.run(agent, history + [{"role": "user", "content": text}], context=ctx)
        write(f"luma> {result.final_output}")
        history = result.to_input_list()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first (e.g. export OPENAI_API_KEY=...) or put it in .env.")
    db = LumaDB(settings.db_path)
    try:
        bootstrap(db, settings.embed_dim)
        ensure_self(db)
        asyncio.run(chat(build_agent(), LumaContext(db=db)))
    finally:
        db.close()


if __name__ == "__main__":
    main()