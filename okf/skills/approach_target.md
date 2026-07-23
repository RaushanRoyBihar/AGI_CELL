---
type: Robot Skill
title: approach_target
description: Turn toward and move at reduced velocity toward a named goal-target entity.
tags: [skill, actuate.motion, sensor.focus]
---

# What it does

Turn toward and move at reduced velocity toward a named goal-target entity.

# Default arguments

`{'velocity': 0.4}`

# Permissions required

actuate.motion, sensor.focus

# Safety

Every proposal for this skill, regardless of source, is validated by
[the safety envelope](/safety/envelope.md) before it can execute — this document
describes the skill, it does not grant it any exemption.
