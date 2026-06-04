import sys
import os
import time
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import SimulationMetrics
from evaluation.visualizer import SimulationVisualizer
from main import LLMSimulationController


def run_single_evaluation():
    
    print("\n" + "="*70)
    print("🔬 SINGLE SIMULATION EVALUATION")
    print("="*70)
    
    metrics = SimulationMetrics()
    visualizer = SimulationVisualizer()
    
    controller = LLMSimulationController(
        max_steps=20,
        target_score=0.7,
        model="llama3.1:8b",
        use_feedback=True
    )
    
    print("\n🔄 Running simulation...")
    start_time = time.time()
    
    final_score, summary = controller.run()
    
    run_time = time.time() - start_time
    
    for step in controller.history:
        
        score_before = step.get('score_before', 0)
        score_after = step.get('score_after', 0)
        delta = step.get('delta', 0) 
        
        step_data = {
            'step': step.get('step', 0),
            'action': step.get('action', 'unknown'),
            'parameters': step.get('parameters', {}),
            'score_before': score_before,
            'score_after': score_after,
            'delta': delta,
            'response_time': step.get('response_time', 1.0),
            'reward': step.get('reward', 0)
        }
        
        if delta > 0.05:
            step_data['outcome'] = 'strong_success'
        elif delta > 0:
            step_data['outcome'] = 'weak_success'
        elif delta > -0.05:
            step_data['outcome'] = 'neutral_or_noise'
        elif delta > -0.15:
            step_data['outcome'] = 'moderate_failure'
        else:
            step_data['outcome'] = 'severe_failure'
        
        metrics.update(step_data)
    
    print(f"\n📈 Collected {len(controller.history)} steps of data")
    
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n📈 BASIC STATISTICS:")
    print(f"   Total Steps: {len(controller.history)}")
    print(f"   Run Time: {run_time:.2f} seconds")
    print(f"   Final Score: {final_score:.3f}" if final_score else "   Final Score: N/A")
    print(f"   Best Score: {controller.best_score:.3f}")
    print(f"   Target Score: {controller.target_score}")
    print(f"   Goal Achieved: {'✅ YES' if final_score and final_score >= controller.target_score else '❌ NO'}")
    
    if len(controller.history) > 0:
        report = metrics.get_comprehensive_report()
        
        metrics.print_report()
        
        print("\n📊 Generating visualizations...")
        
        scores = [step.get('score_after', 0) for step in controller.history]
        deltas = [step.get('delta', 0) for step in controller.history]
        
        action_stats = report['action_performance'].get('action_stats', {})
        
        try:
            score_plot = visualizer.plot_score_progression(scores, "Simulation Score Progression")
            print(f"   ✅ Score progression plot saved")
        except Exception as e:
            print(f"   ⚠️ Could not create score plot: {e}")
        
        try:
            learning_plot = visualizer.plot_learning_curve(scores, deltas)
            print(f"   ✅ Learning curve plot saved")
        except Exception as e:
            print(f"   ⚠️ Could not create learning curve plot: {e}")
        
        if action_stats:
            try:
                action_plot = visualizer.plot_action_performance(action_stats)
                print(f"   ✅ Action performance plot saved")
            except Exception as e:
                print(f"   ⚠️ Could not create action plot: {e}")
        
        # Create dashboard
        try:
            dashboard = visualizer.create_evaluation_dashboard(report, scores, deltas, action_stats)
            print(f"   ✅ Evaluation dashboard saved")
        except Exception as e:
            print(f"   ⚠️ Could not create dashboard: {e}")
    else:
        print("\nNo steps recorded in history")
        report = {}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/evaluation/single_run_{timestamp}.json"
    
    os.makedirs("results/evaluation", exist_ok=True)
    
    results = {
        "timestamp": timestamp,
        "run_time": run_time,
        "total_steps": len(controller.history),
        "final_score": final_score,
        "best_score": controller.best_score,
        "target_score": controller.target_score,
        "goal_achieved": final_score and final_score >= controller.target_score,
        "history": controller.history,
        "metrics_report": report
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Plots saved to: results/evaluation/")
    
    return report, controller


def run_multi_run_evaluation(num_runs: int = 3):
   
    print("\n" + "="*70)
    print(f"🔬 MULTI-RUN EVALUATION ({num_runs} runs)")
    print("="*70)
    
    all_results = []
    
    for run_id in range(num_runs):
        print(f"\n--- Run {run_id + 1}/{num_runs} ---")
        
        controller = LLMSimulationController(
            max_steps=20,
            target_score=0.7,
            model="llama3.1:8b",
            use_feedback=True
        )
        
        start_time = time.time()
        final_score, summary = controller.run()
        run_time = time.time() - start_time
        
        initial_score = controller.history[0]['score_before'] if controller.history else 0
        
        result = {
            "run_id": run_id + 1,
            "final_score": final_score if final_score else 0,
            "best_score": controller.best_score,
            "total_steps": len(controller.history),
            "run_time": round(run_time, 2),
            "success": final_score and final_score >= controller.target_score,
            "total_improvement": (final_score - initial_score) if final_score else 0
        }
        
        all_results.append(result)
        
        print(f"   Final Score: {result['final_score']:.3f}")
        print(f"   Best Score: {result['best_score']:.3f}")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        print(f"   Time: {result['run_time']:.1f}s")
    
    successful_runs = sum(1 for r in all_results if r['success'])
    success_rate = (successful_runs / num_runs) * 100 if num_runs > 0 else 0
    
    avg_final_score = sum(r['final_score'] for r in all_results) / num_runs if num_runs > 0 else 0
    avg_best_score = sum(r['best_score'] for r in all_results) / num_runs if num_runs > 0 else 0
    avg_run_time = sum(r['run_time'] for r in all_results) / num_runs if num_runs > 0 else 0
    
    print("\n" + "="*70)
    print("📊 MULTI-RUN EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\n📈 SUCCESS STATISTICS:")
    print(f"   Successful Runs: {successful_runs}/{num_runs}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    print(f"\n🎯 SCORE STATISTICS:")
    print(f"   Avg Final Score: {avg_final_score:.3f}")
    print(f"   Avg Best Score: {avg_best_score:.3f}")
    print(f"   Max Final Score: {max(r['final_score'] for r in all_results):.3f}" if all_results else "N/A")
    print(f"   Min Final Score: {min(r['final_score'] for r in all_results):.3f}" if all_results else "N/A")
    
    print(f"\n⏱️ TIME STATISTICS:")
    print(f"   Avg Run Time: {avg_run_time:.1f}s")
    print(f"   Total Run Time: {sum(r['run_time'] for r in all_results):.1f}s")
    
    print(f"\n🏆 PERFORMANCE RATING:")
    if success_rate >= 80:
        print("   🌟 EXCELLENT - System consistently achieves goals")
    elif success_rate >= 60:
        print("   👍 GOOD - System usually achieves goals")
    elif success_rate >= 40:
        print("   📈 FAIR - System sometimes achieves goals")
    else:
        print("   🔧 NEEDS IMPROVEMENT - System rarely achieves goals")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/evaluation/multi_run_{timestamp}.json"
    
    os.makedirs("results/evaluation", exist_ok=True)
    
    summary = {
        "timestamp": timestamp,
        "num_runs": num_runs,
        "successful_runs": successful_runs,
        "success_rate": success_rate,
        "avg_final_score": avg_final_score,
        "avg_best_score": avg_best_score,
        "avg_run_time": avg_run_time,
        "individual_results": all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return summary


def compare_without_feedback():
 
    print("\n" + "="*70)
    print("🔬 COMPARISON: WITH FEEDBACK vs WITHOUT FEEDBACK")
    print("="*70)
    
    results = {}
    
    for use_feedback in [True, False]:
        print(f"\n--- Testing {'WITH' if use_feedback else 'WITHOUT'} Feedback ---")
        
        controller = LLMSimulationController(
            max_steps=20,
            target_score=0.7,
            model="llama3.1:8b",
            use_feedback=use_feedback
        )
        
        start_time = time.time()
        final_score, summary = controller.run()
        run_time = time.time() - start_time
        
        results[use_feedback] = {
            "final_score": final_score if final_score else 0,
            "best_score": controller.best_score,
            "total_steps": len(controller.history),
            "run_time": round(run_time, 2),
            "success": final_score and final_score >= controller.target_score
        }
        
        print(f"   Final Score: {results[use_feedback]['final_score']:.3f}")
        print(f"   Best Score: {results[use_feedback]['best_score']:.3f}")
        print(f"   Success: {'✅' if results[use_feedback]['success'] else '❌'}")
    
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<20} {'With Feedback':<18} {'Without Feedback':<18}")
    print("-" * 56)
    print(f"{'Final Score':<20} {results[True]['final_score']:<18.3f} {results[False]['final_score']:<18.3f}")
    print(f"{'Best Score':<20} {results[True]['best_score']:<18.3f} {results[False]['best_score']:<18.3f}")
    print(f"{'Success':<20} {'✅' if results[True]['success'] else '❌':<18} {'✅' if results[False]['success'] else '❌':<18}")
    print(f"{'Run Time (s)':<20} {results[True]['run_time']:<18.1f} {results[False]['run_time']:<18.1f}")
    
    # Calculate improvement
    improvement = results[True]['final_score'] - results[False]['final_score']
    print(f"\n📈 Feedback Improvement: {improvement:+.3f}")
    
    return results


def quick_evaluate():

    print("\n" + "="*70)
    print("⚡ QUICK EVALUATION")
    print("="*70)
    
    controller = LLMSimulationController(
        max_steps=15,
        target_score=0.7,
        model="llama3.1:8b",
        use_feedback=True
    )
    
    start_time = time.time()
    final_score, summary = controller.run()
    run_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("📊 QUICK EVALUATION RESULTS")
    print("="*70)
    print(f"\n   Final Score: {final_score:.3f}" if final_score else "   Final Score: N/A")
    print(f"   Best Score: {controller.best_score:.3f}")
    print(f"   Total Steps: {len(controller.history)}")
    print(f"   Run Time: {run_time:.2f}s")
    print(f"   Success: {'✅ YES' if final_score and final_score >= controller.target_score else '❌ NO'}")
    print("="*70)


def main():
    """Main evaluation menu"""
    
    print("\n" + "="*70)
    print("🤖 LLM SIMULATION AGENT - EVALUATION SUITE")
    print("="*70)
    
    print("\nSelect evaluation type:")
    print("1. Single Run Evaluation (detailed metrics + plots)")
    print("2. Multi-Run Evaluation (statistical analysis)")
    print("3. Compare With/Without Feedback")
    print("4. Quick Evaluation (no plots, fast)")
    print("5. Run All Evaluations")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    os.makedirs("results/evaluation", exist_ok=True)
    
    if choice == "1":
        run_single_evaluation()
    
    elif choice == "2":
        try:
            num_runs = int(input("Number of runs (default 3): ").strip() or "3")
            run_multi_run_evaluation(num_runs)
        except ValueError:
            print("Invalid input. Using default 3 runs.")
            run_multi_run_evaluation(3)
    
    elif choice == "3":
        compare_without_feedback()
    
    elif choice == "4":
        quick_evaluate()
    
    elif choice == "5":
        print("\n🔄 Running all evaluations...")
        quick_evaluate()
        print("\n" + "="*70)
        run_single_evaluation()
        print("\n" + "="*70)
        run_multi_run_evaluation(2)
        print("\n" + "="*70)
        compare_without_feedback()
    
    else:
        print("Invalid choice. Running quick evaluation by default.")
        quick_evaluate()
    
    print("\n✅ Evaluation complete! Check results/evaluation/ for outputs.\n")


if __name__ == "__main__":
    main()