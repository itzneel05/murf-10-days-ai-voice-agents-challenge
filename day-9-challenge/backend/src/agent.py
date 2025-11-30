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
from pathlib import Path
import json
from datetime import datetime
_CATALOG_PATH = Path(__file__).parent / "catalog.json"

def _load_products() -> List[Dict[str, Any]]:
    try:
        with _CATALOG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

PRODUCTS: List[Dict[str, Any]] = _load_products()

logger = logging.getLogger("agent")
load_dotenv(".env.local")

@dataclass
class ShopState:
    recent_products: List[Dict[str, Any]]
    last_order: Optional[Dict[str, Any]]

    def __init__(self) -> None:
        self.recent_products = []
        self.last_order = None

ORDERS_PATH = Path(__file__).parent.parent / "orders.json"

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice shopping assistant."
                "Understand the user's buying intent, browse products, and place simple orders. "
                "Use tools to list products with filters and to create orders. "
                "When browsing, summarize a few relevant items with name and price. "
                "Confirm details before ordering and be friendly and safe."
            )
        )

    @function_tool
    async def list_products(
        self,
        context: RunContext[ShopState],
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        f = filters or {}
        out: List[Dict[str, Any]] = []
        for p in PRODUCTS:
            if "category" in f and p.get("category") != f["category"]:
                continue
            if "max_price" in f and p.get("price", 0) > int(f["max_price"]):
                continue
            if "color" in f and p.get("color") != f["color"]:
                continue
            if "size" in f:
                sizes = p.get("sizes")
                if sizes and f["size"] not in sizes:
                    continue
                if sizes is None and f["size"]:
                    continue
            if "name_contains" in f:
                term = str(f["name_contains"]).lower()
                if term not in p.get("name", "").lower():
                    continue
            out.append(p)
        context.userdata.recent_products = out
        return out

    @function_tool
    async def create_order(
        self,
        context: RunContext[ShopState],
        line_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        total = 0
        currency = "INR"
        for li in line_items:
            pid = li.get("product_id")
            qty = int(li.get("quantity", 1))
            prod = None
            if pid:
                prod = next((p for p in PRODUCTS if p["id"] == pid), None)
            elif "recent_index" in li:
                try:
                    idx = int(li.get("recent_index")) - 1
                    prod = context.userdata.recent_products[idx] if 0 <= idx < len(context.userdata.recent_products) else None
                except Exception:
                    prod = None
            elif "product_name_contains" in li:
                term = str(li.get("product_name_contains")).lower()
                prod = next((p for p in PRODUCTS if term in p.get("name", "").lower()), None)
            if not prod:
                continue
            price = int(prod["price"]) * qty
            total += price
            currency = prod.get("currency", currency)
            items.append({
                "product_id": prod["id"],
                "name": prod.get("name"),
                "quantity": qty,
                "unit_price": int(prod["price"]),
                "line_total": price,
                "color": li.get("color") or prod.get("color"),
                "size": li.get("size"),
            })
        oid = f"order-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        order = {
            "id": oid,
            "items": items,
            "total": total,
            "currency": currency,
            "created_at": datetime.now().isoformat(),
        }
        context.userdata.last_order = order
        if not items:
            return order
        try:
            data: List[Dict[str, Any]] = []
            try:
                with ORDERS_PATH.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
            data.append(order)
            with ORDERS_PATH.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        return order

    @function_tool
    async def get_last_order(self, context: RunContext[ShopState]) -> Optional[Dict[str, Any]]:
        return context.userdata.last_order

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    session = AgentSession[ShopState](
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
        userdata=ShopState(),
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
