import logging
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
)
from livekit.plugins import deepgram, google, murf, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import random

logger = logging.getLogger("agent")
load_dotenv(".env.local")


@dataclass
class ImprovState:
    player_name: Optional[str]
    current_round: int
    max_rounds: int
    rounds: List[Dict[str, str]]
    phase: str

    def __init__(self) -> None:
        self.player_name = None
        self.current_round = 0
        self.max_rounds = 3
        self.rounds = []
        self.phase = "intro"


SCENARIOS: List[str] = [
    "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
    "You are a restaurant waiter who must calmly tell a customer that their order has escaped the kitchen.",
    "You are a customer trying to return an obviously cursed object to a very skeptical shop owner.",
    "You are a space mechanic convincing a nervous pilot that duct tape is enough for the next jump.",
    "You are a royal messenger who must announce very awkward news at a coronation without causing panic.",
]


class ImprovHost(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the host of a TV improv show called 'Improv Battle'. "
                "High-energy, witty, and clear about rules. Be realistic in reactions: sometimes amused, "
                "sometimes unimpressed, sometimes pleasantly surprised. Light teasing and honest critique are allowed, "
                "but always respectful and non-abusive. "
                "Structure: Introduce the show and basic rules. Run N rounds (default 3). For each round: set a scenario, "
                "tell the player to start improvising, listen, then react and move on. "
                "Use the provided tools to maintain game state: set player name, fetch next scenario, and complete scenes. "
                "Ask the player to say 'End scene' when they finish a round, or end automatically after a reasonable pause. "
                "After each scene, comment on what worked, what was weird or flat. Randomly choose a supportive, neutral, or mildly critical tone, "
                "while staying constructive and safe. Store your reaction via the tool when you finish reacting. "
                "When the final round is done, provide a short closing summary highlighting their style (character, absurdity, emotional range, etc.), "
                "mentioning specific moments that stood out, then thank the player and close the show. "
                "If the user says 'stop game' or 'end show', confirm and gracefully end the session."
            )
        )

    @function_tool
    async def set_player_name(self, context: RunContext[ImprovState], name: Optional[str] = None) -> str:
        context.userdata.player_name = name or context.userdata.player_name
        return context.userdata.player_name or ""

    @function_tool
    async def get_state(self, context: RunContext[ImprovState]) -> Dict[str, Any]:
        return {
            "player_name": context.userdata.player_name,
            "current_round": context.userdata.current_round,
            "max_rounds": context.userdata.max_rounds,
            "rounds": context.userdata.rounds,
            "phase": context.userdata.phase,
        }

    @function_tool
    async def next_scenario(self, context: RunContext[ImprovState]) -> Dict[str, Any]:
        if context.userdata.phase == "done":
            return {"done": True}
        if context.userdata.current_round >= context.userdata.max_rounds:
            context.userdata.phase = "done"
            return {"done": True}
        scenario = random.choice(SCENARIOS)
        context.userdata.phase = "awaiting_improv"
        return {
            "round_index": context.userdata.current_round + 1,
            "max_rounds": context.userdata.max_rounds,
            "scenario": scenario,
        }

    @function_tool
    async def complete_scene(
        self,
        context: RunContext[ImprovState],
        scenario: Optional[str] = None,
        host_reaction: Optional[str] = None,
        performance_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        idx = context.userdata.current_round + 1
        context.userdata.phase = "reacting"
        if host_reaction:
            context.userdata.rounds.append({
                "scenario": scenario or "",
                "host_reaction": host_reaction,
                "summary": performance_summary or "",
            })
        context.userdata.current_round += 1
        if context.userdata.current_round >= context.userdata.max_rounds:
            context.userdata.phase = "done"
        else:
            context.userdata.phase = "awaiting_improv"
        return {
            "current_round": context.userdata.current_round,
            "phase": context.userdata.phase,
        }

    @function_tool
    async def end_game(self, context: RunContext[ImprovState], reason: Optional[str] = None) -> Dict[str, Any]:
        context.userdata.phase = "done"
        return {"ended": True, "reason": reason or ""}

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    session = AgentSession[ImprovState](
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-Isha",
            style="Announcer",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        userdata=ImprovState(),
    )
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
        agent=ImprovHost(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
