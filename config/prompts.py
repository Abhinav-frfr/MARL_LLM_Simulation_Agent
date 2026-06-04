"""
config/prompts.py
Single source of truth for all constants and prompts.
"""

# ================================================================
# Ball Definitions
# ================================================================

BALL_TYPES = {
    0: "EMPTY",
    1: "LIGHT",
    2: "NORMAL",
    3: "HEAVY"
}

BALL_SYMBOLS = {
    0: "□",
    1: "○",
    2: "●",
    3: "⬤"
}

# ================================================================
# Action Space - must match MAGRPOTrainer.ACTION_SPACE exactly
# ================================================================

ACTION_SPACE = [
    "SHAKE",
    "ADD_LIGHT",
    "ADD_NORMAL",
    "ADD_HEAVY",
    "SEEK_ADVICE"
]

ACTION_DESCRIPTIONS = {
    "SHAKE": (
        "Shake container to mix balls. Duration 10-30 seconds."
    ),
    "ADD_LIGHT": (
        "Add lightweight balls (float up). Count 1-5."
    ),
    "ADD_NORMAL": (
        "Add normal weight balls. Count 1-5."
    ),
    "ADD_HEAVY": (
        "Add heavyweight balls (sink down). Count 1-5."
    ),
    "SEEK_ADVICE": (
        "Request strategic advice. Only when cooldown=0."
    )
}

# ================================================================
# Shared Context Block - used in all agent prompts
# ================================================================

SHARED_STATE_CONTEXT = """
Current State:
- Homogeneity Score: {homogeneity_score:.3f} / 1.0
- Target Score: {target_score:.3f}
- Best Score This Episode: {best_score:.3f}
- Steps Without Improvement: {steps_without_improvement}
- Advice Cooldown: {advice_cooldown} steps remaining

Ball Distribution:
- Light  (○): {light_count} balls
- Normal (●): {normal_count} balls
- Heavy  (⬤): {heavy_count} balls

Available Actions:
SHAKE | ADD_LIGHT | ADD_NORMAL | ADD_HEAVY | SEEK_ADVICE
Note: SEEK_ADVICE only usable when cooldown = 0.

Reward Signal: ALL agents share ONE joint reward.
Cooperation required - no individual rewards exist.
"""

# ================================================================
# Agent 1: Observation Prompt
# ================================================================

OBSERVATION_PROMPT = (
    "You are Agent 1 (Observation) in a cooperative "
    "multi-agent container mixing simulation.\n\n"
    "Your role: Analyze container state and report observations.\n"
    "All agents share a single joint reward.\n\n"
    "Container Grid (10x10):\n"
    "{container_state}\n\n"
    "Ball Legend:\n"
    "- □ (0): Empty\n"
    "- ○ (1): Light - floats up\n"
    "- ● (2): Normal - stays\n"
    "- ⬤ (3): Heavy - sinks\n"
    + SHARED_STATE_CONTEXT +
    "\nProvide observation:\n\n"
    "DISTRIBUTION SUMMARY: (where are ball types located?)\n"
    "HOMOGENEITY ASSESSMENT: (how well mixed?)\n"
    "KEY ISSUES: (what prevents even distribution?)\n"
    "NOTABLE PATTERNS: (clusters, layers, gradients?)\n"
    "RECOMMENDED FOCUS: (which area needs attention?)\n\n"
    "Observation:"
)

# ================================================================
# Agent 2: Reasoning Prompt
# ================================================================

REASONING_PROMPT = (
    "You are Agent 2 (Reasoning) in a cooperative "
    "multi-agent container mixing simulation.\n\n"
    "Your role: Analyze situation and propose strategy.\n"
    "All agents share a single joint reward.\n\n"
    "Observation from Agent 1:\n"
    "{observation_summary}\n\n"
    "Last Action Outcome:\n"
    "{last_action_outcome}\n\n"
    "Feedback from Agent 5:\n"
    "{feedback_summary}\n"
    + SHARED_STATE_CONTEXT +
    "\nProvide reasoning:\n\n"
    "SITUATION ANALYSIS: (what causes current distribution?)\n"
    "STRATEGY: (what approach improves joint reward?)\n"
    "RECOMMENDED ACTION: (SHAKE/ADD_LIGHT/ADD_NORMAL/"
    "ADD_HEAVY/SEEK_ADVICE)\n"
    "EXPECTED OUTCOME: (what score improvement expected?)\n"
    "COOPERATION NOTE: (how does this help all agents?)\n\n"
    "Reasoning:"
)

# ================================================================
# Agent 3: Decision Prompt
# ================================================================

DECISION_PROMPT = (
    "You are Agent 3 (Decision) in a cooperative "
    "multi-agent container mixing simulation.\n\n"
    "Your role: Select and execute the final action.\n"
    "Your action affects the joint reward ALL agents share.\n\n"
    "Strategy from Agent 2:\n"
    "{strategy_summary}\n"
    + SHARED_STATE_CONTEXT +
    "\nSelect exactly ONE action:\n\n"
    "Actions:\n"
    "- SHAKE         : duration 10-30 seconds\n"
    "- ADD_LIGHT     : count 1-5 balls\n"
    "- ADD_NORMAL    : count 1-5 balls\n"
    "- ADD_HEAVY     : count 1-5 balls\n"
    "- SEEK_ADVICE   : only if advice_cooldown = 0\n\n"
    "Respond exactly:\n\n"
    "ACTION: [SHAKE or ADD_LIGHT or ADD_NORMAL or "
    "ADD_HEAVY or SEEK_ADVICE]\n"
    "PARAMETERS: [duration=X or count=X]\n"
    "REASON: [one sentence]\n\n"
    "Decision:"
)

# ================================================================
# Agent 4: Summarization Prompt
# ================================================================

SUMMARIZATION_PROMPT = (
    "You are Agent 4 (Summarization) in a cooperative "
    "multi-agent container mixing simulation.\n\n"
    "Your role: Summarize episode results.\n"
    "All agents share ONE joint reward - no individual rewards.\n\n"
    "Simulation History ({total_steps} steps):\n"
    "{simulation_history}\n\n"
    "Final Container State:\n"
    "{final_state}\n\n"
    "Joint Reward Results:\n"
    "- Final Score: {final_score:.3f}\n"
    "- Target Score: {target_score:.3f}\n"
    "- Goal Achieved: {goal_achieved}\n"
    "- Total Joint Reward: {total_joint_reward:.3f}\n\n"
    "Create summary:\n\n"
    "ACTIONS SEQUENCE: (actions and score impact)\n"
    "COOPERATION ANALYSIS: (how agents worked together?)\n"
    "RESULT ANALYSIS: (success/failure causes?)\n"
    "RECOMMENDATIONS: (improve joint reward next episode?)\n\n"
    "Summary:"
)

# ================================================================
# Agent 5: Feedback Prompt
# ================================================================

FEEDBACK_PROMPT = (
    "You are Agent 5 (Feedback) in a cooperative "
    "multi-agent container mixing simulation.\n\n"
    "Your role: Analyze outcomes and feedback to Agent 2.\n"
    "Track JOINT rewards only - no individual rewards.\n\n"
    "Recent Action History:\n"
    "{action_history}\n\n"
    "Joint Reward History:\n"
    "{joint_reward_history}\n\n"
    "Best Performing Actions:\n"
    "{best_actions}\n\n"
    "Worst Performing Actions:\n"
    "{worst_actions}\n"
    + SHARED_STATE_CONTEXT +
    "\nProvide feedback:\n\n"
    "PERFORMANCE SUMMARY: (which actions improved joint reward?)\n"
    "PATTERN ANALYSIS: (what sequences work best?)\n"
    "AVOID LIST: (which actions hurt joint reward?)\n"
    "RECOMMENDATION: (what should Agent 3 prioritize?)\n\n"
    "Feedback:"
)

# ================================================================
# Advice Prompt
# ================================================================

ADVICE_PROMPT = (
    "You are the Advice System for a cooperative "
    "multi-agent container mixing simulation.\n\n"
    "Agents selected SEEK_ADVICE. "
    "Provide ONE concrete action recommendation.\n\n"
    "Current Situation:\n"
    "- Score: {current_score:.3f} (target: {target_score:.3f})\n"
    "- Steps stuck: {steps_without_improvement}\n"
    "- Light={light_count}, Normal={normal_count}, "
    "Heavy={heavy_count}\n\n"
    "Past advice outcomes:\n"
    "{advice_history}\n\n"
    "Respond exactly:\n\n"
    "RECOMMENDED_ACTION: [SHAKE or ADD_LIGHT or "
    "ADD_NORMAL or ADD_HEAVY]\n"
    "PARAMETERS: [duration=X or count=X]\n"
    "REASONING: [why this improves joint reward]\n\n"
    "Advice:"
)