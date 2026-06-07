# Spec: Tool Functions

**File:** `tools.py`
**Status:** `get_seasonal_conditions` — Pre-implemented, read through. `lookup_plant` — complete spec fields before implementing.

---

## Purpose

These two functions are the tools the agent can call. They retrieve structured data from the local plant database and seasonal data files and return it to the agent loop, which passes it to the LLM as context for generating a response.

---

## Function 1: `lookup_plant()`

### Input / Output Contract

**Inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `plant_name` | `str` | The plant name as entered by the user or chosen by the LLM — may be any casing, common name, scientific name, or alias |

**Output:** `dict`

When the plant is **found**, return:
```python
{"found": True, "plant": <the full plant dict from _plant_db>}
```

When the plant is **not found**, return:
```python
{"found": False, "name": <normalized input>, "message": <helpful string>}
```

---

### Design Decisions

*Complete the two blank fields below before writing code. The others are pre-filled for you.*

---

#### Input normalization

Strip leading/trailing whitespace and convert to lowercase before any comparison.

```python
normalized = plant_name.strip().lower()
```

---

#### Search order

Search in this order: direct key → display name → aliases. Keys are the fastest
lookup (O(1) dict access), so check those first. Display names are the next most
likely match for clean user input. Aliases are the broadest net, so they go last.

```
1. Direct key match: normalized in _plant_db
2. Display name match: plant["display_name"].lower() == normalized
3. Alias match: normalized in [alias.lower() for alias in plant["aliases"]]
```

---

#### Alias matching approach

*Aliases are stored as a list of strings. How will you check if the normalized input matches any alias in the list? Write your approach in pseudocode or plain English.*

```
For each plant, build a lowercased copy of its aliases list and test membership:
    normalized in [a.lower() for a in plant["aliases"]]
This stays case-insensitive and reads clearly. With 15 plants a linear scan is fine.

If the database grew to thousands of plants, I would precompute one flat dict at
module load that maps every key, display name, and alias (all lowercased) to its
slug. Each lookup then becomes a single O(1) dict access instead of scanning every
plant's alias list.
```

---

#### Not-found message

*When a plant isn't found, the agent will read your message and use it to decide what to tell the user. Write the exact string you'll return — make it useful to the agent, not just to a human reading logs.*

```
No plant matching '{name}' was found in the care database (which covers 15 common
houseplants). Tell the user this specific plant is not in your database, then offer
general care guidance based on the details or symptoms they describe.
```

The message acknowledges the miss, states the database scope, and tells the agent
to fall back to general guidance, matching the system prompt's graceful-degradation
intent. `{name}` is the normalized input, and the count is built with
`len(_plant_db)` so it stays correct if plants are added.

---

#### Implementation Notes

*Fill this in after implementing and running the app.*

**Test: does `"devil's ivy"` return the pothos entry?**
```
Yes. Resolves via the alias pass and returns the full Pothos dict with found: True.
```

**Test: does `"SNAKE PLANT"` return the snake plant entry?**
```
Yes. After strip().lower() it matches the "Snake Plant" display name. "  POTHOS "
(extra whitespace) and "sansevieria" (alias) also resolve correctly.
```

**One edge case you discovered while implementing:**
```
scientific_name is not part of the search order (direct key, display name, aliases),
even though the tool definition's description advertises scientific names like
"Monstera deliciosa". So lookup_plant("Monstera deliciosa") returns found: False
unless that string also appears in the aliases list. Following the spec's stated
search order here; adding a scientific_name check would be a one-line follow-up.
```

---

## Function 2: `get_seasonal_conditions()`

### Input / Output Contract

**Inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `season` | `str \| None` | One of `"spring"`, `"summer"`, `"fall"`, `"winter"`, or `None` to auto-detect |

**Output:** `dict`

The full season dict from `_season_data`, plus one additional field:

| Added field | Type | Value |
|-------------|------|-------|
| `"detected_season"` | `bool` | `True` if auto-detected from the month; `False` if season was passed as an argument |

---

### Design Decisions

*This function is pre-implemented — read through these fields and the code before working on `lookup_plant`.*

---

#### Auto-detection logic

When `season` is `None`, get the current calendar month with `datetime.now().month`
and look it up in the `_MONTH_TO_SEASON` dict, which maps month numbers to season strings.

```python
current_month = datetime.now().month
season_key = _MONTH_TO_SEASON[current_month]
```

---

#### Season validation

If the caller passes an invalid season string (e.g., `"monsoon"`), the function
falls back to auto-detection — same as if `None` were passed. The `VALID_SEASONS`
set acts as the gate:

```python
VALID_SEASONS = {"spring", "summer", "fall", "winter"}
if season and season.lower() in VALID_SEASONS:
    ...  # use provided season
else:
    ...  # auto-detect
```

---

#### Return structure

The full season dict from `_season_data`, plus a `detected_season` boolean. Example for spring:

```python
{
    "season": "spring",
    "watering": "Increase watering frequency as plants break dormancy ...",
    "fertilizing": "Resume feeding with a balanced fertilizer ...",
    "light": "Days are lengthening — move plants closer to windows ...",
    "pests": "Watch for spider mites and aphids as temperatures rise ...",
    "detected_season": True   # True = auto-detected; False = caller specified
}
```

---

#### Implementation Notes

*Fill this in after testing.*

**Test: does calling with `season=None` return the correct season for the current month?**
```
Current month: June
Expected season: summer
Returned season: summer (detected_season: True)
```

**Test: does calling with `season="winter"` return winter data regardless of the current month?**
```
Yes. Returns the Winter dict with detected_season: False even though it is June.
```

---

## Function 3: `get_plant_list()`

*Added as an optional challenge: a third tool so the agent can answer "what plants
do you know about?" and difficulty-based questions like "what's a good beginner
plant?".*

### Input / Output Contract

**Inputs:** none.

**Output:** `dict`

```python
{
    "count": <int>,                       # number of plants in the database
    "plants": [                           # sorted alphabetically by name
        {"name": <display_name>, "difficulty": <"easy" | "moderate" | "hard">},
        ...
    ]
}
```

### Design Decisions

- **Reuses `_plant_db`.** No new data source. Reads only `display_name` and
  `difficulty` from each entry, so the payload stays small (the LLM does not need
  the full care record just to list or recommend).
- **Sorted by name** for a stable, readable list.
- **No parameters.** The schema in `TOOL_DEFINITIONS` declares an empty
  `properties` object, matching a tool the LLM calls with no arguments.
- **When the LLM calls it** is driven by the tool description: browsing the
  database, "what do you know about", or a difficulty-based recommendation. For
  recommendations the LLM reads the `difficulty` field (for example, suggesting an
  `easy` plant for a beginner).

### Implementation Notes

**Test: does "what plants do you know about?" trigger get_plant_list?**
```
Yes. The agent calls get_plant_list (no args) and lists the 15 plants.
```

**Test: does "what's a good beginner plant?" use the difficulty field?**
```
Yes. The agent calls get_plant_list and recommends easy-difficulty plants
(pothos, snake plant, ZZ plant, etc.).
```
