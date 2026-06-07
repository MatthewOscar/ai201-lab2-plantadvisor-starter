import json
from groq import Groq, BadRequestError
from config import GROQ_API_KEY, LLM_MODEL, MAX_TOOL_ROUNDS
from tools import lookup_plant, get_seasonal_conditions

_client = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────
# Tool definitions
#
# These are the schemas that tell the LLM what tools are available and how to
# call them. The LLM reads these descriptions and decides when (and how) to use
# each tool. They're already complete — your job is to implement the tool
# functions in tools.py and the agent loop below.
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_plant",
            "description": (
                "Look up care information for a specific houseplant by name. "
                "Returns detailed watering, light, humidity, and temperature requirements. "
                "Use this whenever the user asks about a specific plant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_name": {
                        "type": "string",
                        "description": "The plant name to look up. Can be a common name, scientific name, or nickname (e.g., 'pothos', 'devil's ivy', 'Monstera deliciosa').",
                    }
                },
                "required": ["plant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_conditions",
            "description": (
                "Get seasonal care adjustments for houseplants. "
                "Returns guidance on watering, fertilizing, light, and pests for the current or specified season. "
                "Use this when a user asks a season-specific question, or to complement plant care advice with seasonal context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {
                        "type": "string",
                        "description": "The season to get care conditions for. If omitted, the current season is detected automatically.",
                        "enum": ["spring", "summer", "fall", "winter"],
                    }
                },
                "required": [],
            },
        },
    },
]

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly plant care advisor. "
    "Help users care for their houseplants by looking up specific plant information "
    "and current seasonal conditions using your available tools.\n\n"
    "Always use your tools to look up plant-specific information before answering — "
    "don't rely on your general knowledge alone.\n\n"
    "When a tool returns found: False, that plant is not in your database. Do not "
    "invent specific care numbers such as watering schedules, temperatures, or light "
    "levels as if they were real data. Instead, clearly tell the user the plant is "
    "not in your database, offer general guidance for that type of plant based on "
    "what they describe, and point them to a reliable source (for example a "
    "horticultural society or a reputable plant care site) for exact figures.\n\n"
    "Keep your advice practical and specific. Cite the source of your information "
    "when you have it (e.g., 'According to the care data for your monstera...')."
)

# Returned when a turn produces no usable text (empty content, or MAX_TOOL_ROUNDS
# reached). Keeps run_agent's contract of never returning an empty string.
_FALLBACK = "Sorry, I ran into a problem answering that. Could you try rephrasing your question?"

# ──────────────────────────────────────────────
# Tool dispatch
#
# This is already complete. It routes tool calls from the LLM to the actual
# Python functions in tools.py, and returns results as JSON strings (which is
# what the Groq API expects for tool results).
# ──────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """Route a tool call to the correct function and return the result as a JSON string."""
    print(f"  → Tool call: {tool_name}({tool_args})")
    if tool_name == "lookup_plant":
        result = lookup_plant(tool_args["plant_name"])
    elif tool_name == "get_seasonal_conditions":
        result = get_seasonal_conditions(tool_args.get("season"))
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    print(f"  ← Result: {json.dumps(result)[:120]}{'...' if len(json.dumps(result)) > 120 else ''}")
    return json.dumps(result)


# ──────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────

def _complete(messages: list, use_tools: bool = True, retries: int = 2):
    """
    Call the LLM, retrying the intermittent Groq 'tool_use_failed' error.

    llama-3.3-70b occasionally emits malformed function-call syntax that Groq
    rejects with a 400 (code 'tool_use_failed'). It is non-deterministic, so a
    bounded retry usually recovers on the next attempt. Other errors propagate.
    """
    kwargs = {"model": LLM_MODEL, "messages": messages}
    if use_tools:
        kwargs["tools"] = TOOL_DEFINITIONS
        kwargs["tool_choice"] = "auto"

    for attempt in range(retries + 1):
        try:
            return _client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if "tool_use_failed" in str(e) and attempt < retries:
                print(f"  ⚠ tool_use_failed from the model, retrying ({attempt + 1}/{retries})")
                continue
            raise


def run_agent(user_message: str, history: list) -> str:
    """
    Run the plant care agent for one user turn and return its response.

    TODO — Milestone 2:

    The agent loop follows a specific pattern that you'll implement here. Read
    specs/agent-loop-spec.md carefully before writing any code — understand the
    full loop before implementing any part of it.

    The loop works like this:
      1. Build a messages list: system prompt + conversation history + new user message
      2. Call the LLM with messages and TOOL_DEFINITIONS
      3. If the response contains tool_calls:
           a. Append the assistant message (with tool_calls) to messages
           b. For each tool call: execute via dispatch_tool(), append the result
           c. Call the LLM again with the updated messages
           d. Repeat until no more tool_calls (or MAX_TOOL_ROUNDS is reached)
      4. Return the final text response

    Key details to get right:
      - The assistant message must be appended BEFORE tool results
      - Tool result messages use role="tool" with a tool_call_id field
      - Append the assistant's message object directly (not just its content)
      - The history format from Gradio: list of [user_message, assistant_message] pairs

    Before writing code, complete specs/agent-loop-spec.md.
    """
    # 1. Build the messages list: system prompt, prior history, then the new turn.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history:
        # Gradio 6 passes OpenAI-style dicts: {"role": ..., "content": ...}.
        # Tolerate the legacy [user, assistant] tuple format too.
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})
        else:
            user_msg, assistant_msg = turn
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": user_message})

    # 2. Tool-calling loop, capped at MAX_TOOL_ROUNDS. The whole exchange is wrapped
    #    so any API failure degrades to a readable fallback instead of crashing the
    #    chat (the contract requires a non-empty return).
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = _complete(messages, use_tools=True)
            assistant_message = response.choices[0].message

            # Exit (a): no tool calls means the LLM produced its final answer.
            if not assistant_message.tool_calls:
                return assistant_message.content or _FALLBACK

            # The assistant message must be appended BEFORE its tool results so the
            # API can match each tool_call_id to the call that requested it.
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                # llama-3.3 sometimes emits "null" or "" for a no-argument call,
                # which json.loads turns into None. dispatch_tool expects a dict.
                tool_args = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(tool_args, dict):
                    tool_args = {}
                tool_result = dispatch_tool(tool_name, tool_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        # 3. Exit (b): MAX_TOOL_ROUNDS reached. Force a final text answer (no tools).
        final = _complete(messages, use_tools=False)
        return final.choices[0].message.content or _FALLBACK
    except Exception as e:
        # Logged to the terminal so failures stay visible while debugging.
        print(f"  ⚠ Agent error: {type(e).__name__}: {e}")
        return _FALLBACK
