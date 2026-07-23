---
type: Robot Skill
title: avoid_obstacle
description: Turn away from the nearest detected obstacle and move at reduced velocity.
tags: [skill, actuate.motion]
---

# What it does

Turn away from the nearest detected obstacle and move at reduced velocity.

# Default arguments

`{'velocity': 0.5}`

# Permissions required

actuate.motion

# Safety

Every proposal for this skill, regardless of source, is validated by
[the safety envelope](/safety/envelope.md) before it can execute — this document
describes the skill, it does not grant it any exemption.
