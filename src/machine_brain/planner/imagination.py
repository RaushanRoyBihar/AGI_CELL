"""Imagination-based planner: the "basic AGI" upgrade over the purely
reactive `Planner`. Instead of matching hand-written rules against the
current state, this planner asks the action-conditioned dynamics model
"what would happen if I did each of my candidate actions", scores each
imagined outcome against safety margins and the active goal, and proposes
the best-scoring one — using competence (procedural success rate) as a
tie-break, tying learning directly back into planning.

This is still bounded and still doesn't get the final word: SutraFlow and
the safety governor validate whatever this planner proposes exactly the
same as they validate the reactive planner's proposals. Imagination
changes *which* action gets proposed, never whether a proposal can bypass
the guards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from machine_brain.contracts import ActionProposal, Goal, WorldEntity
from machine_brain.procedural.skills import SkillRegistry
from machine_brain.world_model.dynamics import ActionConditionedDynamics

# (skill_id, args) menu of actions the planner is allowed to imagine and
# choose between. emergency_stop is deliberately excluded — it is reached
# only via SutraFlow's apavada exemption path, never proposed by planning.
CANDIDATE_ACTIONS: tuple[tuple[str, dict], ...] = (
    ("patrol", {"velocity": 1.0}),
    ("avoid_obstacle", {"velocity": 0.5}),
    ("yield_to_human", {"velocity": 0.0}),
    ("hold_position", {"velocity": 0.0}),
    ("approach_target", {"velocity": 0.4}),
    ("investigate_anomaly", {"velocity": 0.2}),
)

KNOWN_SKILLS: tuple[str, ...] = tuple(skill_id for skill_id, _ in CANDIDATE_ACTIONS)
ACTION_DIM = len(KNOWN_SKILLS) + 1  # one-hot over skills + normalized velocity


def encode_action(skill_id: str, velocity: float) -> np.ndarray:
    onehot = np.zeros(len(KNOWN_SKILLS))
    if skill_id in KNOWN_SKILLS:
        onehot[KNOWN_SKILLS.index(skill_id)] = 1.0
    return np.concatenate([onehot, [velocity / 2.0]])


@dataclass
class ImaginationConfig:
    human_safety_margin: float = 1.0
    obstacle_safety_margin: float = 0.5
    risk_weight: float = 2.0
    goal_weight: float = 1.5
    velocity_preference_weight: float = 0.05
    competence_weight: float = 0.12
    min_train_steps: int = 20  # don't trust imagination until the dynamics model has seen this many real transitions
    horizon: int = 4
    discount: float = 0.85

    def __post_init__(self) -> None:
        # horizon < 1 would leave `imagine_candidates`'s rollout loop never
        # executing, leaving `first_predicted` at None and crashing later
        # wherever a candidate's predicted_next_state is read (caught by
        # mypy flagging the implicit Optional, not by a runtime report —
        # fail loudly here instead of deep inside a rollout).
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1 to imagine at least one step, got {self.horizon}")
    # Single-step imagination only sees the immediate next state, so a
    # small, noisy step toward a goal looks about as good as no progress
    # at all — nothing distinguishes "this genuinely helps over time" from
    # "this barely moved the needle." Rolling the *same* candidate action
    # forward `horizon` steps (open-loop "shooting", the standard cheap
    # MPC simplification — not a full tree search) and summing discounted
    # per-step utility gives sustained progress a real advantage over
    # one-shot noise. The dynamics model still only ever trains on real,
    # single-step transitions (see cognitive_loop.py) — only the *planning*
    # evaluation is multi-step; the model itself stays honest.
    exploration_epsilon: float = 0.15
    exploration_seed: int = 0
    # Pure greedy utility-maximization creates a lock-in loop: whichever
    # action wins early accumulates competence (it gets executed and
    # succeeds more), which raises its own utility further, while the
    # dynamics model never collects (state, action, next_state) triples
    # for the alternatives it keeps skipping — so it never learns to
    # predict them accurately either, entrenching the bias. Epsilon-greedy
    # exploration is what breaks that loop: occasionally propose a
    # non-maximal candidate so the model (and the competence stats) stay
    # informed about every action, not just the incumbent. Explored
    # proposals still go through SutraFlow + the safety governor exactly
    # like any other — exploration is safe by construction, not despite it.


@dataclass(frozen=True)
class ImaginedCandidate:
    skill_id: str
    args: dict
    predicted_next_state: np.ndarray
    risk: float
    goal_score: float
    competence: float
    utility: float


class ImaginationPlanner:
    def __init__(self, dynamics: ActionConditionedDynamics, config: ImaginationConfig | None = None) -> None:
        self.dynamics = dynamics
        self.config = config or ImaginationConfig()
        self._rng = np.random.default_rng(self.config.exploration_seed)

    def ready(self) -> bool:
        """Bounded trust: imagination is not used for real decisions until
        the dynamics model has learned from enough real transitions.
        Before that, the caller should fall back to the reactive planner —
        an untrained model imagining consequences is worse than no model."""
        return self.dynamics.train_steps >= self.config.min_train_steps

    def _step_scores(self, predicted: np.ndarray, args: dict, goal: Goal | None) -> tuple[float, float]:
        cfg = self.config
        risk = (max(0.0, cfg.human_safety_margin - predicted[0]) * cfg.risk_weight
                + max(0.0, cfg.obstacle_safety_margin - predicted[1]) * cfg.risk_weight)

        if goal is not None and goal.kind == "observe_entity":
            desired = goal.target.get("desired_distance", 1.0)
            goal_score = -abs(predicted[4] - desired) * cfg.goal_weight
        else:
            goal_score = args.get("velocity", 0.0) * cfg.velocity_preference_weight
        return risk, goal_score

    def imagine_candidates(self, state: np.ndarray, goal: Goal | None,
                              skill_registry: SkillRegistry) -> list[ImaginedCandidate]:
        cfg = self.config
        candidates = []
        for skill_id, args in CANDIDATE_ACTIONS:
            action_vec = encode_action(skill_id, args.get("velocity", 0.0))

            rollout_state = state
            discount = 1.0
            total_utility = 0.0
            first_predicted: np.ndarray | None = None
            last_risk = last_goal_score = 0.0
            for _ in range(cfg.horizon):
                predicted = self.dynamics.predict(rollout_state, action_vec)
                if first_predicted is None:
                    first_predicted = predicted
                last_risk, last_goal_score = self._step_scores(predicted, args, goal)
                total_utility += discount * (last_goal_score - last_risk)
                discount *= cfg.discount
                rollout_state = predicted

            # ImaginationConfig.__post_init__ guarantees horizon >= 1, so the
            # loop above always runs at least once — this assertion makes
            # that invariant explicit and checked here, at the one place a
            # violation would otherwise surface as a confusing None deep in
            # a rollout, rather than trusting the constructor validation
            # silently held all the way to this point.
            assert first_predicted is not None, "rollout loop must execute at least once (horizon >= 1)"

            competence = skill_registry.success_rate(skill_id, 1)
            competence = competence if competence is not None else 0.5  # neutral prior, no data yet

            utility = total_utility + cfg.competence_weight * competence
            candidates.append(ImaginedCandidate(skill_id, args, first_predicted, last_risk, last_goal_score,
                                                    competence, utility))
        return candidates

    def propose(self, state: np.ndarray, goal: Goal | None, skill_registry: SkillRegistry,
                 retrieved_episode_ids: tuple[str, ...] = ()) -> tuple[ActionProposal, list[ImaginedCandidate]]:
        candidates = self.imagine_candidates(state, goal, skill_registry)
        greedy_best = max(candidates, key=lambda c: c.utility)

        explored = self._rng.random() < self.config.exploration_epsilon
        best = candidates[self._rng.integers(len(candidates))] if explored else greedy_best

        # Confidence reflects how safe/well-understood the chosen action's
        # predicted consequence is (risk, prior competence) — NOT whether
        # it happened to be the utility-maximizing pick. Deliberately
        # dampening an exploratory pick's confidence would make it more
        # likely to be held by SutraFlow's low-confidence gate, which
        # would starve exploration of exactly the executed-outcome data it
        # exists to collect — a self-defeating interaction found and fixed
        # during testing, not a hypothetical concern.
        confidence = 0.5 + 0.4 * best.competence - 0.15 * min(best.risk, 2.0)
        confidence = float(np.clip(confidence, 0.05, 0.95))

        goal_desc = f"goal={goal.kind}({goal.target})" if goal is not None else "goal=none (default patrol_safely)"
        mode = "EXPLORING" if explored else "exploiting"
        justification = (
            f"imagined {len(candidates)} candidates over a {self.config.horizon}-step rollout ({mode}, "
            f"greedy pick was '{greedy_best.skill_id}'); chose '{best.skill_id}' "
            f"(next-step human_dist={best.predicted_next_state[0]:.2f}, "
            f"obstacle_dist={best.predicted_next_state[1]:.2f}, end-of-rollout risk={best.risk:.2f}, "
            f"goal_score={best.goal_score:.2f}, competence={best.competence:.2f}); {goal_desc}"
        )
        proposal = ActionProposal.make(
            skill_id=best.skill_id, args=best.args, justification=justification,
            predicted_confidence=confidence, source_episode_ids=retrieved_episode_ids,
        )
        return proposal, candidates
