# d:\MurfAIEvent\ten-days-of-voice-agents-2025\day-3-challenge\backend\src\test2.py
import logging
import os
import json
from datetime import datetime


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
    llm,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

WELLNESS_LOG_PATH = "wellness_log.json"

def load_history() -> list:
    if os.path.exists(WELLNESS_LOG_PATH):
        try:
            with open(WELLNESS_LOG_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def generate_system_prompt() -> str:
    history = load_history()
    base = (
        "You are a supportive, grounded Health & Wellness Voice Companion.\n"
        "Conduct a short daily check-in.\n\n"
        "Required steps:\n"
        "1) Ask about mood and energy.\n"
        "2) Ask for 1–3 practical objectives for today.\n"
        "3) Offer small, realistic, non-medical suggestions.\n"
        "4) Recap mood and objectives and confirm.\n"
        "5) Call the `log_checkin` tool with mood, energy, objectives.\n\n"
        "Guidelines:\n"
        "- Be friendly, concise, and avoid medical claims.\n"
        "- Suggestions should be small and actionable.\n"
        "- No complex formatting or emojis.\n"
    )
    if history:
        last = history[-1]
        prev_mood = last.get("mood")
        prev_energy = last.get("energy")
        if prev_mood:
            ref = f"\nReference only one prior detail: Last mood was '{prev_mood}'. Ask how it compares today."
        elif prev_energy:
            ref = f"\nReference only one prior detail: Last energy was '{prev_energy}'. Ask how it compares today."
        else:
            ref = ""
        return base + ref
    return base

class Assistant(Agent):
    def __init__(self, system_prompt: str) -> None:
        super().__init__(instructions=system_prompt)

    @llm.function_tool(description="Persist a wellness check-in entry.")
    def log_checkin(
        self,
        mood: str,
        energy: str,
        objectives: str,
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": mood,
            "objectives": objectives,
        }
        if energy and energy.strip():
            entry["energy"] = energy
        try:
            data = []
            if os.path.exists(WELLNESS_LOG_PATH):
                with open(WELLNESS_LOG_PATH, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
            data.append(entry)
            with open(WELLNESS_LOG_PATH, "w") as f:
                json.dump(data, f, indent=2)
            return "Check-in logged."
        except Exception as e:
            logger.error(f"Failed to log check-in: {e}")
            return "Failed to log."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    system_prompt = generate_system_prompt()
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-molly",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
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
        agent=Assistant(system_prompt=system_prompt),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
