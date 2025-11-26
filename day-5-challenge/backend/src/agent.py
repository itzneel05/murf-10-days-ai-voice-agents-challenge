import logging
import os
import json
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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

FAQ_PATH = "../shared-data/yellowai_faq.json"
LEADS_PATH = "../shared-data/leads.json"

def _load_faq() -> list[dict]:
    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _search_faq(query: str) -> Optional[dict]:
    q = query.lower().strip()
    best = None
    best_score = 0
    for entry in _load_faq():
        score = 0
        question = str(entry.get("question", "")).lower()
        answer = str(entry.get("answer", "")).lower()
        tags = [str(t).lower() for t in entry.get("tags", [])]
        for token in q.split():
            if token in question:
                score += 2
            if token in answer:
                score += 1
            if token in tags:
                score += 3
        if score > best_score:
            best = entry
            best_score = score
    return best

def _load_leads() -> list:
    try:
        if os.path.exists(LEADS_PATH):
            with open(LEADS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return []
    return []

def _upsert_lead(record: dict) -> None:
    leads = _load_leads()
    existing_index = None
    for i, r in enumerate(leads):
        if r.get("id") == record.get("id"):
            existing_index = i
            break
    if existing_index is None:
        leads.append(record)
    else:
        leads[existing_index] = record
    try:
        with open(LEADS_PATH, "w", encoding="utf-8") as f:
            json.dump(leads, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write leads: {e}")

@dataclass
class LeadState:
    id: str = ""
    name: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    use_case: Optional[str] = None
    team_size: Optional[str] = None
    timeline: Optional[str] = None

    def ensure_id(self):
        if not self.id:
            self.id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    def to_record(self, status: str = "in_progress") -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "company": self.company,
            "email": self.email,
            "role": self.role,
            "use_case": self.use_case,
            "team_size": self.team_size,
            "timeline": self.timeline,
            "status": status,
            "updated_at": datetime.now().isoformat(),
        }

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Yellow AI's friendly Sales Development Representative. "
                "Greet visitors warmly, ask what brought them here and what they're working on, "
                "and keep the conversation focused on understanding their needs. "
                "When asked about product, company, channels, integrations, or pricing, call the `answer_faq` tool with the user's question and answer only from the provided FAQ. "
                "If information is not in the FAQ, say you don't have that detail and offer to connect sales. "
                "Collect lead details naturally: name, company, email, role, use case, team size, and timeline. "
                "Whenever the user provides a field, call `record_lead_field` with the field and value. "
                "When the user indicates the conversation is done (phrases like 'that's all', 'I'm done', 'thanks'), call `complete_lead` to summarize and finalize. "
                "Be concise, helpful, and refuse harmful or inappropriate requests. Do not claim to know personal information about the user."
            )
        )

    @function_tool
    async def answer_faq(self, context: RunContext[LeadState], query: str) -> str:
        context.userdata.ensure_id()
        match = _search_faq(query)
        if match:
            return str(match.get("answer", ""))
        return "I don't have that detail in our FAQ. I can connect you to our sales team for specifics."

    @function_tool
    async def record_lead_field(self, context: RunContext[LeadState], field: str, value: str) -> str:
        context.userdata.ensure_id()
        f = field.strip().lower()
        v = value.strip()
        if f == "name":
            context.userdata.name = v
        elif f == "company":
            context.userdata.company = v
        elif f == "email":
            context.userdata.email = v
        elif f == "role":
            context.userdata.role = v
        elif f == "use case" or f == "use_case":
            context.userdata.use_case = v
        elif f == "team size" or f == "team_size":
            context.userdata.team_size = v
        elif f == "timeline":
            context.userdata.timeline = v
        rec = context.userdata.to_record(status="in_progress")
        _upsert_lead(rec)
        return "Got it."

    @function_tool
    async def complete_lead(self, context: RunContext[LeadState]) -> str:
        context.userdata.ensure_id()
        rec = context.userdata.to_record(status="completed")
        _upsert_lead(rec)
        name = context.userdata.name or "a prospective customer"
        company = context.userdata.company or "their company"
        use_case = context.userdata.use_case or "a potential use case"
        timeline = context.userdata.timeline or "an upcoming timeline"
        return f"Summary: {name} from {company} is interested in {use_case}. Timeline: {timeline}."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    lead_state = LeadState()
    session = AgentSession[LeadState](
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
        userdata=lead_state,
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
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
