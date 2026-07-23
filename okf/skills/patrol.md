---
type: Robot Skill
title: patrol
description: Move forward at default velocity with gentle heading drift; the default action when nothing salient is in view.
tags: [skill, actuate.motion]
---

# What it does

Move forward at default velocity with gentle heading drift; the default action when nothing salient is in view.

# Default arguments

`{'velocity': 1.0}`

# Permissions required

actuate.motion

# Safety

Every proposal for this skill, regardless of source, is validated by
[the safety envelope](/safety/envelope.md) before it can execute — this document
describes the skill, it does not grant it any exemption.
