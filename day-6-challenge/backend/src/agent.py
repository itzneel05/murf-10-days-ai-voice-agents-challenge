import logging
import os
import sqlite3
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
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    metrics,
    tokenize,
)
from livekit.plugins import deepgram, google, murf, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

DB_PATH = os.path.join(os.path.dirname(__file__), "fraud_cases.db")

def _db_connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def _load_case(username: str) -> Optional[dict]:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, customer_name, security_id, masked_card, amount, merchant, location, timestamp, security_question, security_answer, status, outcome_note FROM fraud_cases WHERE username = ? ORDER BY id DESC LIMIT 1",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "customer_name": row[1],
            "security_id": row[2],
            "masked_card": row[3],
            "amount": row[4],
            "merchant": row[5],
            "location": row[6],
            "timestamp": row[7],
            "security_question": row[8],
            "security_answer": row[9],
            "status": row[10],
            "outcome_note": row[11],
        }
    finally:
        conn.close()

def _update_status(case_id: int, status: str, note: str) -> None:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE fraud_cases SET status = ?, outcome_note = ? WHERE id = ?",
            (status, note, case_id),
        )
        conn.commit()
        logger.info(f"Fraud case {case_id} updated: {status} - {note}")
    finally:
        conn.close()

@dataclass
class FraudCaseState:
    username: Optional[str] = None
    case: Optional[dict] = None
    verified: bool = False
    final_status: Optional[str] = None

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a calm, professional fraud detection representative named Alex for Easy Bank. "
                "At the start of the call, clearly introduce Easy Bank and yourself, explain you are contacting the user about a suspicious card transaction, and ask for their username to locate the case. "
                "After the user provides a username, call the `load_fraud_case` tool with it. If no case is found, politely explain and ask for a different username or end the call. "
                "Use only non-sensitive verification. Call `get_security_question` to retrieve the security question from the loaded case and ask it verbatim. Do not ask for full card numbers, PINs, passwords, or credentials. When the user answers, call `verify_answer` with the user's answer. If verification fails, apologize, say you cannot proceed, call `finalize_verification_failed`, and end the call. "
                "If verification passes, use only database values to describe the transaction by calling `read_transaction_details`: include merchant, location, timestamp, amount, and only the masked card's last four digits. Then ask if they made this transaction (yes or no). Based on the answer, call `finalize_case` with true for yes or false for no. "
                "When finalizing, the status must be one of confirmed_safe or confirmed_fraud, with a concise outcome note. End the call by confirming the action taken. "
                "Be concise, reassuring, and refuse harmful or inappropriate requests. Do not claim to know personal information about the user beyond what is in the case data."
            )
        )

    @function_tool
    async def get_security_question(self, context: RunContext[FraudCaseState]) -> str:
        if not context.userdata.case:
            return "No case loaded."
        q = str(context.userdata.case.get("security_question", "")).strip()
        if not q:
            return "No security question available."
        return q

    @function_tool
    async def load_fraud_case(self, context: RunContext[FraudCaseState], username: str) -> str:
        context.userdata.username = username.strip()
        case = _load_case(context.userdata.username)
        context.userdata.case = case
        if not case:
            return "No fraud case found for that username."
        return "Case loaded. Call `get_security_question` and ask it."

    @function_tool
    async def verify_answer(self, context: RunContext[FraudCaseState], answer: str) -> str:
        if not context.userdata.case:
            return "No case loaded."
        provided = (answer or "").strip().lower()
        expected = str(context.userdata.case.get("security_answer", "")).strip().lower()
        ok = provided == expected and expected != ""
        context.userdata.verified = ok
        if ok:
            return "Verification passed. Read transaction details and ask if it was made."
        return "Verification failed."

    @function_tool
    async def read_transaction_details(self, context: RunContext[FraudCaseState]) -> str:
        if not context.userdata.case:
            return "No case loaded."
        c = context.userdata.case
        digits = "".join(d for d in str(c.get("masked_card", "")) if d.isdigit())
        last4 = digits[-4:] if digits else ""
        return f"Suspicious transaction: {c['merchant']} at {c['location']} around {c['timestamp']} for ${c['amount']:.2f} on card ending {last4}."

    @function_tool
    async def finalize_case(self, context: RunContext[FraudCaseState], is_legit: bool) -> str:
        if not context.userdata.case:
            return "No case loaded."
        case_id = int(context.userdata.case["id"])
        if is_legit:
            status = "confirmed_safe"
            note = "Customer confirmed transaction as legitimate."
        else:
            status = "confirmed_fraud"
            note = "Customer denied transaction; card blocked and dispute initiated."
        _update_status(case_id, status, note)
        context.userdata.final_status = status
        return f"Status updated: {status}."

    @function_tool
    async def finalize_verification_failed(self, context: RunContext[FraudCaseState]) -> str:
        if not context.userdata.case:
            return "No case loaded."
        case_id = int(context.userdata.case["id"])
        status = "verification_failed"
        note = "Verification failed; unable to proceed."
        _update_status(case_id, status, note)
        context.userdata.final_status = status
        return f"Status updated: {status}."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    state = FraudCaseState()
    session = AgentSession[FraudCaseState](
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-Isha",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        userdata=state,
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
