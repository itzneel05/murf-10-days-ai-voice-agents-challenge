import logging
import asyncio
import json
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
    llm,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

@dataclass
class TutorState:
    mode: str = "learn"
    concept_id: Optional[str] = None
    mastery: dict[str, int] = None

    def __post_init__(self):
        if self.mastery is None:
            self.mastery = {}

def _load_content() -> list[dict]:
    try:
        with open("../shared-data/day4_tutor_content.json", "r") as f:
            return json.load(f)
    except Exception:
        return []

def _content() -> dict[str, dict]:
    return {entry["id"]: entry for entry in _load_content() if "id" in entry}

def _concept_or_default(concept_id: str | None) -> dict:
    content = _content()
    if concept_id and concept_id in content:
        return content[concept_id]
    return next(iter(content.values()), {
        "id": "variables",
        "title": "Variables",
        "summary": "Variables store values so you can reuse them later.",
        "sample_question": "What is a variable and why is it useful?",
    })

def _match_concept_id(query: str | None) -> Optional[str]:
    if not query:
        return None
    q = query.strip().lower()
    content = _content()
    if q in content:
        return q
    for cid, c in content.items():
        title = str(c.get("title", "")).strip().lower()
        if title == q:
            return cid
    return None

def _ensure_concept(state: TutorState) -> dict:
    cid = state.concept_id
    if cid:
        concept = _concept_or_default(cid)
        state.concept_id = concept.get("id", "variables")
        return concept
    state.concept_id = "variables"
    return _concept_or_default(state.concept_id)

def _random_concept_id(exclude: Optional[str] = None) -> str:
    keys = list(_content().keys())
    if not keys:
        return "variables"
    if exclude in keys and len(keys) > 1:
        keys = [k for k in keys if k != exclude]
    return random.choice(keys)

def _quiz_options(concept_id: str | None) -> list[str]:
    concept = _concept_or_default(concept_id)
    opts = concept.get("options")
    if isinstance(opts, list) and len(opts) > 0:
        return opts
    cid = concept.get("id", "variables")
    if cid == "variables":
        return [
            "Option A A named storage for a value",
            "Option B A function that prints text",
            "Option C A loop that repeats steps",
        ]
    if cid == "loops":
        return [
            "Option A A way to repeat actions",
            "Option B A single-use constant",
            "Option C A comment for documentation",
        ]
    return [
        "Option A A named storage for a value",
        "Option B A function that prints text",
        "Option C A loop that repeats steps",
    ]

def _option_label_text(opt: str) -> tuple[str, str]:
    s = opt.strip()
    if len(s) >= 2 and s[1] == ')':
        label = s[0]
        text = s[2:].lstrip(') ').strip()
        return label, text
    if s.lower().startswith("option "):
        rest = s[7:].strip()
        if rest:
            return rest[0], rest[1:].strip()
    return "", s

async def _speak_options(sess, opts: list[str]) -> None:
    for opt in opts:
        label, text = _option_label_text(opt)
        if label:
            if text:
                sess.say(f"Option {label}: {text}")
            else:
                sess.say(f"Option {label}")
        else:
            sess.say(opt)

class BaseTutorAgent(Agent):
    def __init__(self, *, voice: str, mode_label: str, chat_ctx: llm.ChatContext | None = None) -> None:
        super().__init__(
            instructions=(
                "You are an Active Recall Coach. You operate in modes: 'learn', 'quiz', 'teach_back'. "
                "Use the provided tools to select concept, explain summaries, ask questions, and prompt teach-back. "
                "Be concise, friendly, and avoid complex formatting or emojis. Users may ask to switch modes at any time."
            ),
            tts=murf.TTS(
                voice=voice,
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True,
            ),
            chat_ctx=chat_ctx,
        )

    @function_tool
    async def select_concept(self, context: RunContext[TutorState], concept_id: str) -> str:
        matched = _match_concept_id(concept_id)
        if matched:
            context.userdata.concept_id = matched
        concept = _ensure_concept(context.userdata)
        return f"Selected concept: {concept['title']}."

    @function_tool
    async def switch_mode(self, context: RunContext[TutorState], mode: str) -> Agent:
        prev_ctx = self.chat_ctx
        context.userdata.mode = mode
        if mode == "learn":
            return LearnAgent(chat_ctx=prev_ctx)
        elif mode == "quiz":
            return QuizAgent(chat_ctx=prev_ctx)
        elif mode == "teach_back":
            return TeachBackAgent(chat_ctx=prev_ctx)
        else:
            return LearnAgent(chat_ctx=prev_ctx)

class LearnAgent(BaseTutorAgent):
    def __init__(self, chat_ctx: llm.ChatContext | None = None) -> None:
        super().__init__(voice="en-US-matthew", mode_label="learn", chat_ctx=chat_ctx)

    @function_tool
    async def explain(self, context: RunContext[TutorState]) -> str:
        concept = _ensure_concept(context.userdata)
        return f"{concept['title']}: {concept['summary']}"

    async def on_enter(self) -> None:
        sess = self.session
        concept = _ensure_concept(sess.userdata)
        sess.say(f"Learn mode: {concept['title']}.")
        sess.say(f"{concept['title']}: {concept['summary']}")
        sess.say("Say 'continue' to learn another topic.")

    @function_tool
    async def continue_learning(self, context: RunContext[TutorState]) -> str:
        current = context.userdata.concept_id
        next_id = _random_concept_id(exclude=current)
        context.userdata.concept_id = next_id
        concept = _concept_or_default(next_id)
        self.session.say(f"{concept['title']}: {concept['summary']}")
        return "OK"

class QuizAgent(BaseTutorAgent):
    def __init__(self, chat_ctx: llm.ChatContext | None = None) -> None:
        super().__init__(voice="en-US-alicia", mode_label="quiz", chat_ctx=chat_ctx)

    @function_tool
    async def ask(self, context: RunContext[TutorState]) -> str:
        concept = _ensure_concept(context.userdata)
        opts = _quiz_options(concept.get("id"))
        self.session.say(f"Question: {concept['sample_question']}")
        self.session.say("Choose one.")
        await _speak_options(self.session, opts)
        return "Please choose A, B, or C."

    async def on_enter(self) -> None:
        sess = self.session
        concept = _ensure_concept(sess.userdata)
        sess.say(
            f"Quiz mode. I'll ask about '{concept['title']}'. Say 'switch to learn' or 'switch to teach back' anytime.")
        opts = _quiz_options(concept.get("id"))
        sess.say(f"Question: {concept['sample_question']}")
        sess.say("Choose one.")
        await _speak_options(sess, opts)

class TeachBackAgent(BaseTutorAgent):
    def __init__(self, chat_ctx: llm.ChatContext | None = None) -> None:
        super().__init__(voice="en-US-ken", mode_label="teach_back", chat_ctx=chat_ctx)

    @function_tool
    async def prompt_teach_back(self, context: RunContext[TutorState]) -> str:
        concept = _ensure_concept(context.userdata)
        return (
            f"Teach back: Please explain '{concept['title']}' in your own words, "
            "covering its purpose and a simple example."
        )

    async def on_enter(self) -> None:
        sess = self.session
        concept = _ensure_concept(sess.userdata)
        sess.say(
            f"Teach-back mode. Please explain '{concept['title']}' in your own words. "
            "Say 'switch to learn' or 'switch to quiz' anytime.")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    tutor_state = TutorState()
    session = AgentSession[TutorState](
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        userdata=tutor_state,
    )
    # Metrics collection
    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    ctx.add_shutdown_callback(log_usage)
    await session.start(
        agent=RouterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await ctx.connect()

class RouterAgent(Agent):
    def __init__(self) -> None:
        instructions = (
            "You are the Teach-the-Tutor router. Greet the user and ask only for their preferred mode "
            "('learn', 'quiz', 'teach_back'). Do not ask for a topic. "
            "When the user says 'learn', call the tool 'start_tutoring' with mode='learn'. "
            "Similarly for 'quiz' or 'teach back'. Support switching any time."
        )
        super().__init__(instructions=instructions)

    @function_tool
    async def start_tutoring(self, context: RunContext[TutorState], mode: str, concept_id: Optional[str] = None) -> Agent:
        matched = _match_concept_id(concept_id)
        if mode == "learn":
            context.userdata.concept_id = matched if matched is not None else "variables"
        elif matched:
            context.userdata.concept_id = matched
        concept = _ensure_concept(context.userdata)
        context.userdata.mode = mode
        base = BaseTutorAgent(voice="en-US-matthew", mode_label="router")
        return await base.switch_mode(context, mode)

    async def on_enter(self) -> None:
        self.session.say(
            "Hi! I'm your Active Recall Coach. Which mode would you like: 'learn', 'quiz', or 'teach back'?" )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
