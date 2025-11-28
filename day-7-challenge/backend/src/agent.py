import logging
import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict

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

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")
ORDER_PATH = os.path.join(os.path.dirname(__file__), "current_order.json")

@dataclass
class CartItem:
    name: str
    price: float
    quantity: int
    notes: Optional[str] = None
    category: Optional[str] = None

@dataclass
class OrderState:
    catalog: List[Dict] = field(default_factory=list)
    cart: List[CartItem] = field(default_factory=list)
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None

def _load_catalog() -> List[Dict]:
    if not os.path.exists(CATALOG_PATH):
        return []
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _find_item(catalog: List[Dict], item_name: str) -> Optional[Dict]:
    q = item_name.strip().lower()
    for it in catalog:
        if str(it.get("name", "")).strip().lower() == q:
            return it
    return None

def _cart_total(cart: List[CartItem]) -> float:
    return sum(i.price * i.quantity for i in cart)

def _qty_word(n: int) -> str:
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
    }
    return words.get(max(1, n), str(n))

def _fmt_currency(amount: float) -> str:
    return f"₹{amount:.2f}"

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly ordering assistant for SafeBazaar. "
                "Greet the user, explain you can help order groceries, snacks, and simple prepared foods. "
                "Ask for clarifications when needed such as size, brand, and quantity. "
                "Use tools to add, remove, update, and list cart items, and to add recipe ingredients. You can list catalog items by category (including Prepared Food) or by tags like vegetarian, vegan, gluten-free, or spicy. "
                "Confirm cart changes out loud after each tool call so the user knows what happened. "
                "When the user says they are done, call `place_order` to finalize, summarize the cart and total, and save the order. "
                "Refuse harmful or inappropriate requests and do not claim to know private user information."
            )
        )

    @function_tool
    async def list_catalog(self, context: RunContext[OrderState], category: Optional[str] = None, tag: Optional[str] = None) -> str:
        items = context.userdata.catalog
        if category:
            c = category.strip().lower()
            items = [i for i in items if str(i.get("category", "")).lower() == c]
        if tag:
            t = tag.strip().lower()
            items = [i for i in items if t in [str(x).lower() for x in i.get("tags", [])]]
        if not items:
            return "No items found."
        lines = [f"{i['name']} ({_fmt_currency(float(i['price']))})" for i in items[:25]]
        return ", ".join(lines)

    @function_tool
    async def add_item(self, context: RunContext[OrderState], item_name: str, quantity: int = 1, notes: Optional[str] = None) -> str:
        if quantity <= 0:
            return "Quantity must be at least 1."
        item = _find_item(context.userdata.catalog, item_name)
        if not item:
            return "Item not found in catalog."
        for ci in context.userdata.cart:
            if ci.name.lower() == item["name"].lower() and (ci.notes or "") == (notes or ""):
                ci.quantity += quantity
                total = _cart_total(context.userdata.cart)
                return f"Updated {ci.name} to {ci.quantity}. Cart total {_fmt_currency(total)}."
        context.userdata.cart.append(
            CartItem(
                name=item["name"],
                price=float(item["price"]),
                quantity=quantity,
                notes=notes,
                category=item.get("category"),
            )
        )
        total = _cart_total(context.userdata.cart)
        return f"Added {item['name']} x{quantity}. Cart total {_fmt_currency(total)}."

    @function_tool
    async def remove_item(self, context: RunContext[OrderState], item_name: str) -> str:
        before = len(context.userdata.cart)
        context.userdata.cart = [c for c in context.userdata.cart if c.name.lower() != item_name.strip().lower()]
        after = len(context.userdata.cart)
        if before == after:
            return "Item not found in cart."
        total = _cart_total(context.userdata.cart)
        return f"Removed {item_name}. Cart total {_fmt_currency(total)}."

    @function_tool
    async def update_quantity(self, context: RunContext[OrderState], item_name: str, quantity: int) -> str:
        if quantity <= 0:
            return "Quantity must be at least 1."
        for ci in context.userdata.cart:
            if ci.name.lower() == item_name.strip().lower():
                ci.quantity = quantity
                total = _cart_total(context.userdata.cart)
                return f"Set {ci.name} to {quantity}. Cart total {_fmt_currency(total)}."
        return "Item not found in cart."

    @function_tool
    async def list_cart(self, context: RunContext[OrderState]) -> str:
        if not context.userdata.cart:
            return "Your cart is empty."
        parts = []
        for ci in context.userdata.cart:
            d = f"{_qty_word(ci.quantity)} {ci.name}"
            if ci.notes:
                d += f" ({ci.notes})"
            d += f" @ {_fmt_currency(ci.price)}"
            parts.append(d)
        total = _cart_total(context.userdata.cart)
        return f"Items: {', '.join(parts)}. Total {_fmt_currency(total)}."

    @function_tool
    async def add_recipe_items(self, context: RunContext[OrderState], recipe: str, servings: int = 1) -> str:
        rec = recipe.strip().lower()
        mapping = {
            "masala chai": ["Tea Leaves", "Milk", "Sugar", "Cardamom Pods"],
            "dal": ["Toor Dal", "Turmeric Powder", "Cumin Seeds", "Ghee"],
            "paneer curry": ["Paneer", "Tomato Puree", "Onion", "Garam Masala", "Ginger Garlic Paste"],
            "biryani": ["Basmati Rice", "Biryani Masala", "Mixed Vegetables"],
            "roti": ["Atta (Wheat Flour)", "Ghee"],
            "poha": ["Poha (Flattened Rice)", "Peanuts", "Mustard Seeds", "Green Chilies", "Onion"],
        }
        items = mapping.get(rec)
        if not items:
            return "Recipe not found."
        added = []
        for name in items:
            item = _find_item(context.userdata.catalog, name)
            if item:
                qty = max(1, servings)
                exists = False
                for ci in context.userdata.cart:
                    if ci.name.lower() == item["name"].lower():
                        ci.quantity += qty
                        exists = True
                        break
                if not exists:
                    context.userdata.cart.append(
                        CartItem(
                            name=item["name"],
                            price=float(item["price"]),
                            quantity=qty,
                            category=item.get("category"),
                        )
                    )
                added.append(f"{item['name']} x{qty}")
        total = _cart_total(context.userdata.cart)
        return f"Added {', '.join(added)}. Cart total {_fmt_currency(total)}."

    @function_tool
    async def place_order(self, context: RunContext[OrderState], customer_name: Optional[str] = None, customer_address: Optional[str] = None) -> str:
        if not context.userdata.cart:
            return "Your cart is empty."
        context.userdata.customer_name = customer_name or context.userdata.customer_name
        context.userdata.customer_address = customer_address or context.userdata.customer_address
        items = [
            {
                "name": ci.name,
                "quantity": ci.quantity,
                "unit_price": ci.price,
                "line_total": round(ci.price * ci.quantity, 2),
                "notes": ci.notes,
                "category": ci.category,
            }
            for ci in context.userdata.cart
        ]
        total = round(_cart_total(context.userdata.cart), 2)
        order = {
            "customer_name": context.userdata.customer_name,
            "customer_address": context.userdata.customer_address,
            "items": items,
            "total": total,
            "currency": "INR",
        }
        with open(ORDER_PATH, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2)
        return f"Order placed. {len(items)} items, total {_fmt_currency(total)}. Saved."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    state = OrderState(catalog=_load_catalog())
    session = AgentSession[OrderState](
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
