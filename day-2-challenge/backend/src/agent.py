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
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from dataclasses import dataclass
import json
from datetime import datetime
from typing import Optional

logger = logging.getLogger("agent")
load_dotenv(".env.local")

@dataclass
class OrderState:
    drinkType: Optional[str] = None
    size: Optional[str] = None
    milk: Optional[str] = None
    extras: list[str] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = []

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and enthusiastic coffee shop barista at Cozy cafe. 
            
            Your main responsibilities are:
            - Greet customers warmly when they arrive
            - Take their coffee orders (espresso, latte, cappuccino, americano, cold brew, etc.)
            - Ask about size preferences (small, medium, large)
            - Offer customizations (milk type: whole, oat, almond, soy; extra shots, flavor syrups)
            - Suggest popular items or daily specials
            - Confirm orders clearly before finalizing
            - Be conversational and friendly, like a real barista
            
            Once all details are collected (name, drinkType, size, milk, extras), call save_order to complete the order.
            Avoid complex formatting, emojis, or special characters in your responses.
            
            Be helpful, upbeat, and make customers feel welcome!"""
        )

    @function_tool
    async def set_name(self, context: RunContext[OrderState], name: str) -> str:
        """Set the customer's name."""
        context.userdata.name = name
        return f"Hi {name}! Welcome to Cozy Cafe—what drink can I craft for you today?"

    @function_tool
    async def set_drink_type(self, context: RunContext[OrderState], drink_type: str) -> str:
        """Set the drink type (e.g., latte, espresso)."""
        context.userdata.drinkType = drink_type
        return f"Nice pick—a {drink_type}! What size: small, medium, or large?"

    @function_tool
    async def set_size(self, context: RunContext[OrderState], size: str) -> str:
        """Set the size (small, medium, large)."""
        context.userdata.size = size
        return f"Gotcha, {size} size. Milk preference: whole, oat, almond, soy or none?"

    @function_tool
    async def set_milk(self, context: RunContext[OrderState], milk: str) -> str:
        """Set the milk type (whole, oat, almond, soy or none)."""
        context.userdata.milk = milk
        return "Sweet! Any extras like syrup or whipped cream? Say 'none' if not."

    @function_tool
    async def add_extra(self, context: RunContext[OrderState], extra: str) -> str:
        """Add an extra to the order (call multiple times if needed)."""
        context.userdata.extras.append(extra)
        return f"Added {extra}! Anything else, or good to go?"

    @function_tool
    async def save_order(self, context: RunContext[OrderState]) -> str:
        """Save the order once all fields are set. Saves to JSON and returns summary."""
        state = context.userdata
        missing = [f for f, v in [("name", state.name), ("drinkType", state.drinkType), ("size", state.size), ("milk", state.milk)] if v is None]
        if missing:
            return f"Almost there! Still need: {', '.join(missing)}. Let's finish up."

        # Save to JSON
        timestamp = datetime.now().isoformat()
        order_data = {
            "name": state.name,
            "drinkType": state.drinkType,
            "size": state.size,
            "milk": state.milk,
            "extras": state.extras,
            "timestamp": timestamp
        }
        filename = f"order_{state.name.replace(' ', '_')}_{timestamp.split('T')[0]}.json"
        with open(filename, "w") as f:
            json.dump(order_data, f, indent=2)
        logger.info(f"Order saved to {filename}")

        # Neat text summary
        extras_str = f" + {', '.join(state.extras)}" if state.extras else ""
        summary = f"Order Summary for {state.name}:\n- {state.size.title()} {state.drinkType} with {state.milk} milk{extras_str}\nReady in a flash at Aroma Heaven!"

        return summary

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    # Initialize order state
    order_state = OrderState()
    # Set up a voice AI pipeline
    session = AgentSession[OrderState](
        # Speech-to-text (STT)
        stt=deepgram.STT(model="nova-3"),
        # Large Language Model (LLM)
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # Text-to-speech (TTS)
        tts=murf.TTS(
            voice="en-US-molly",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        # VAD and turn detection
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # Preemptive generation
        preemptive_generation=True,
        # Pass the order state
        userdata=order_state,
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
    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    # Join the room and connect to the user
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))