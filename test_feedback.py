# test_feedback.py
from simulation.container_simulation import ContainerSimulation
from agents.feedback_agent import FeedbackAgent

def test_feedback():
    """Test the feedback system"""
    
    print("\n" + "="*60)
    print("TESTING FEEDBACK SYSTEM")
    print("="*60)
    
    # Create feedback agent
    feedback = FeedbackAgent()
    
    # Simulate some actions
    test_actions = [
        ("SHAKE", {"duration": 15}, 0.2, 0.35),
        ("SHAKE", {"duration": 20}, 0.35, 0.42),
        ("ADD_HEAVY", {"count": 5}, 0.42, 0.38),
        ("SHAKE", {"duration": 15}, 0.38, 0.51),
        ("ADD_LIGHT", {"count": 5}, 0.51, 0.48),
        ("SHAKE", {"duration": 10}, 0.48, 0.55),
    ]
    
    for action, params, before, after in test_actions:
        reward = feedback.record_action(action, params, before, after, "Test action")
        print(f"{action}: {before:.2f} → {after:.2f} | Reward: {reward:.2f}")
    
    # Check avoidance
    print("\n" + "-"*40)
    print("Action Avoidance Check:")
    
    for action in ["SHAKE", "ADD_HEAVY", "ADD_LIGHT"]:
        should_avoid, reason = feedback.should_avoid_action(action, 0.5)
        status = "🚫 AVOID" if should_avoid else "✅ ALLOW"
        print(f"  {action}: {status} - {reason}")
    
    # Show statistics
    stats = feedback.get_statistics()
    print("\n" + "-"*40)
    print("Statistics:")
    print(f"  Best action: {stats['best_action']}")
    print(f"  Worst action: {stats['worst_action']}")
    print(f"  Success rate: {stats['successful_actions']/max(1,stats['total_actions'])*100:.1f}%")

if __name__ == "__main__":
    test_feedback()