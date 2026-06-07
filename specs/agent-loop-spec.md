# Spec: `run_agent()`

**File:** `agent.py`
**Status:** Partially pre-filled — complete the two blank fields before implementing

---

## Purpose

Orchestrate a single conversational turn for the Plant Advisor agent. Given a user message and the conversation history, call the LLM with available tools, execute any tool calls the LLM requests, and return the final text response.

This is the core of what makes Plant Advisor an *agent* rather than a simple chatbot: the ability to decide which tools to call, use their results to inform its response, and loop until it has everything it needs.

---

## Input / Output Contract

**Inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_message` | `str` | The user's current message |
| `history` | `list` | Gradio conversation history — list of `[user_msg, assistant_msg]` pairs |

**Output:** `str`

The agent's final text response for this turn. Should never be empty — if something goes wrong, return a user-readable fallback message.

---

## Design Decisions

*Read `specs/system-design.md` (especially the "How the Groq Tool Calling API Works" section) before reviewing these. Complete the two blank fields before writing any code.*

---

### Messages list structure

The messages list must start with the system prompt, then replay the conversation
history, then add the new user message. Gradio history is a list of `[user, assistant]`
pairs — convert each pair to two API-format dicts:

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for user_msg, assistant_msg in history:
    messages.append({"role": "user", "content": user_msg})
    if assistant_msg:
        messages.append({"role": "assistant", "content": assistant_msg})

messages.append({"role": "user", "content": user_message})
```

---

### Initial LLM call

Pass the model, the messages list, the tool definitions, and `tool_choice="auto"`
so the LLM can decide whether to call a tool or respond directly:

```python
response = client.chat.completions.create(
    model=LLM_MODEL,
    messages=messages,
    tools=TOOL_DEFINITIONS,
    tool_choice="auto",
)
```

---

### Detecting tool calls in the response

The response object has a `choices` list. Index 0 gives the assistant message.
Check its `tool_calls` attribute — if it's truthy, the LLM wants to call tools:

```python
assistant_message = response.choices[0].message

if not assistant_message.tool_calls:
    # No tool calls — LLM has a final answer
    ...
```

---

### Appending the assistant message

When there are tool calls, append the full assistant message object to `messages`
**before** appending any tool results. The API requires this ordering — a tool
result message must immediately follow the assistant message that requested it:

```python
messages.append(assistant_message)  # must come first
```

---

### Executing and appending tool results

For each tool call, extract the name and arguments, call `dispatch_tool()`, and
append the result as a `"tool"` role message. The `tool_call_id` links this result
back to the specific tool call that requested it:

```python
for tool_call in assistant_message.tool_calls:
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    tool_result = dispatch_tool(tool_name, tool_args)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result,
    })
```

---

### Loop termination conditions

*The loop should stop when: (a) the LLM returns a response with no tool calls, OR (b) the MAX_TOOL_ROUNDS limit is reached. Describe how you will detect each condition and what you will return in each case.*

```
Cap the loop at MAX_TOOL_ROUNDS iterations (from config.py).

(a) No tool calls: after each LLM call, read response.choices[0].message. If its
.tool_calls is empty or None, the LLM has a final answer. Return .content, with a
static fallback string if .content is empty so the function never returns "".

(b) MAX_TOOL_ROUNDS reached: if every iteration still requested tools, leave the
loop and make one more LLM call with no tools attached, which forces a text answer
(the model cannot request more tools). Return that .content, again with a fallback
if empty. This guarantees both termination and a non-empty return.
```

---

### Extracting the final text response

*Once the loop exits because there are no more tool calls, how do you extract the text content from the response object? What field holds the string you should return?*

```
The text lives at response.choices[0].message.content (a str). Get the assistant
message with response.choices[0].message, then read .content. Because content can
be None or "" when a turn was only tool calls, return a fallback string in that case
to honor the "never empty" contract.
```

---

## Implementation Notes

*Fill this in after implementing and testing.*

**Trace of a working agent turn (what tools were called and in what order):**

```
Query: "How should I care for my calathea?"
Round 1 tool call: lookup_plant({"plant_name": "calathea"}) -> found: True
Round 2 tool call: none (the model answered directly after the lookup)
Final response: detailed calathea care (filtered/consistent moisture, high humidity,
                low-to-medium indirect light, monthly feeding), citing the care data.

Note: for a generic "how do I care for X" question the model calls only
lookup_plant. For a season-specific question like "how should I water my monstera
this time of year?" it calls both lookup_plant and get_seasonal_conditions. The
agent picks the tools per question rather than always calling both.
```

**What happens when you ask about a plant that isn't in the database?**

```
For "string of pearls" (not in the database), the agent calls lookup_plant, gets
found: False with the not-found message, then tells the user the plant is not in
the database and falls back to general care guidance. The Milestone 1 not-found
message is what steers it toward that graceful fallback instead of a dead end.
```

**One thing about the tool call API that surprised you:**

```
Two things, both about robustness rather than the happy path:
1. For a no-argument call (get_seasonal_conditions), llama-3.3 sometimes sends the
   arguments as the JSON value null, not "{}". json.loads then returns None and the
   dispatcher's tool_args.get(...) raises AttributeError. The loop has to coerce the
   parsed arguments to a dict before dispatch.
2. The model intermittently emits malformed function-call text that Groq rejects
   with a 400 'tool_use_failed'. It is non-deterministic, so a bounded retry on the
   create() call is needed for the agent to be reliable.
```
