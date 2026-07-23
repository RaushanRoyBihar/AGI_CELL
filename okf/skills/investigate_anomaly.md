---
type: Robot Skill
title: investigate_anomaly
description: Move at low velocity when the world model's surprise score exceeds threshold.
tags: [skill, actuate.motion, sensor.focus]
---

# What it does

Move at low velocity when the world model's surprise score exceeds threshold.

# Default arguments

`{'velocity': 0.2}`

# Permissions required

actuate.motion, sensor.focus

# Safety

Every proposal for this skill, regardless of source, is validated by
[the safety envelope](/safety/envelope.md) before it can execute — this document
describes the skill, it does not grant it any exemption.
