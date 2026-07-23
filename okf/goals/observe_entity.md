---
type: Goal Kind
title: observe_entity
description: Hold a target distance from a specific, currently-known human or obstacle entity.
tags: [goal]
---

# Schema

```json
{"kind": "observe_entity", "target": {"entity_id": "<id>", "desired_distance": <meters>}}
```

# Valid entity_id values

Must match `^(human|obstacle)-\d+$` — e.g. `human-0` through `human-4`, `obstacle-0` through
`obstacle-6`. Any other value (a place, an abstract topic, anything not currently tracked in
working memory) is not a valid target and must be rejected, not guessed.

# Valid desired_distance range

Between 0.1 and 20.0 meters. Values outside this range are rejected.

# What this goal means

The agent will try to hold approximately `desired_distance` from the named entity, using its own
learned world model to imagine which action gets it there — subject to
[the safety envelope](/safety/envelope.md), which can never be overridden by a goal.
