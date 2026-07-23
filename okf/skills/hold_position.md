---
type: Robot Skill
title: hold_position
description: Stay in place, no directed heading change. Safe under nearly all conditions.
tags: [skill, actuate.motion]
---

# What it does

Stay in place, no directed heading change. Safe under nearly all conditions.

# Default arguments

`{'velocity': 0.0}`

# Permissions required

actuate.motion

# Safety

Every proposal for this skill, regardless of source, is validated by
[the safety envelope](/safety/envelope.md) before it can execute — this document
describes the skill, it does not grant it any exemption.
