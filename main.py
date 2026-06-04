import os
import time
import json
import random
from datetime import datetime
from colorama import init, Fore, Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from langchain_ollama import ChatOllama

from simulation.container_simulation import ContainerSimulation
from agents.observation_agent import ObservationAgent
from agents.reasoning_agent import ReasoningAgent
from agents.decision_agent import DecisionAgent
from agents.summarization_agent import SummarizationAgent
from agents.feedback_agent import FeedbackAgent

init(autoreset=True)
console = Console()

class LLMSimulationController:
    """This is the main controller for LLM-driven simulation with RL feedback system"""
    
    def __init__(self, max_steps: int = 20, target_score: float = 0.7, 
                 model: str = "llama3.1:8b", use_feedback: bool = True):
        self.simulation = ContainerSimulation()
        self.model = model
        self.use_feedback = use_feedback
        
        self.observation_agent = ObservationAgent(model)
        self.reasoning_agent = ReasoningAgent(model)
        self.decision_agent = DecisionAgent(model)
        self.summarization_agent = SummarizationAgent(model)
        
        self.feedback_agent = None
        if use_feedback:
            self.feedback_agent = FeedbackAgent(model)
            self.reasoning_agent.set_feedback_agent(self.feedback_agent)
            self.decision_agent.set_feedback_agent(self.feedback_agent)
            console.print("[green]Feedback learning system ENABLED[/green]")
        else:
            console.print("[yellow]Feedback learning system DISABLED[/yellow]")
        
        self.max_steps = max_steps
        self.target_score = target_score
        self.history = []
        
        self.best_score = 0.0
        self.steps_without_improvement = 0
        
        console.print(Panel.fit(
            f"[bold cyan]LLM Multi-Agent System for Simulation Control[/bold cyan]\n"
            f"Model: {model} (via LangChain ChatOllama)\n"
            f"Target Score: {target_score}\n"
            f"Feedback Learning: {'Enabled' if use_feedback else 'Disabled'}",
            border_style="cyan"
        ))
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            test_llm = ChatOllama(model=self.model, base_url="http://localhost:11434")
            test_llm.invoke("Test connection")
            return True
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return False
    
    def run(self, target_goal: str = "Achieve an evenly distributed mixture of balls"):
        
        if self.use_feedback and self.feedback_agent:
            stats = self.feedback_agent.get_statistics()
            if stats['total_actions'] > 0:
                console.print(f"[dim] Loaded {stats['total_actions']} past actions from memory[/dim]")
      
        console.print(f"\n[bold]Checking Ollama connection...[/bold]")
        if not self.check_ollama_connection():
            console.print("[red]❌ Cannot connect to Ollama. Make sure it's running:[/red]")
            console.print("[yellow]   ollama serve[/yellow]")
            console.print(f"[yellow]   ollama pull {self.model}[/yellow]")
            return None, None
        
        console.print(f"[green]✅ Connected to Ollama with model: {self.model}[/green]")
        
        console.print(f"\n[bold yellow]🎯 Target Goal:[/bold yellow] {target_goal}")
        console.print(f"[bold yellow]📊 Target Homogeneity Score:[/bold yellow] {self.target_score}")
        console.print(f"[bold yellow]🔄 Max Steps:[/bold yellow] {self.max_steps}")
        console.print(f"[bold yellow]🤖 Model:[/bold yellow] {self.model}")
        console.print(f"[bold yellow]📚 Feedback Learning:[/bold yellow] {'Enabled' if self.use_feedback else 'Disabled'}\n")
        
        self.simulation.add_multiple_balls(1, 8)   # Light
        self.simulation.add_multiple_balls(2, 8)   # Normal
        self.simulation.add_multiple_balls(3, 8)   # Heavy
        
        self.best_score = 0.0
        self.steps_without_improvement = 0
        
        for step in range(self.max_steps):
            console.print(f"\n[bold green]--- Step {step + 1}/{self.max_steps} ---[/bold green]")
            
            state = self.simulation.get_state()
            current_score = self.simulation.calculate_homogeneity()
            
            self._display_state(state, current_score)
            
            if current_score >= self.target_score:
                console.print(f"\n[bold green]✅ GOAL ACHIEVED![/bold green] Score: {current_score:.3f}")
                break
            
            if current_score > self.best_score:
                self.best_score = current_score
                self.steps_without_improvement = 0
                console.print(f"[green]🎉 New best score! {self.best_score:.3f}[/green]")
            else:
                self.steps_without_improvement += 1
            
            if self.steps_without_improvement >= 8:
                console.print(f"\n[yellow] No improvement for {self.steps_without_improvement} steps. Stopping early.[/yellow]")
                break
            
            # 1. Observation Agent
            console.print(f"\n[bold blue]👁️ Observation Agent:[/bold blue]")
            observation = self.observation_agent.process(state, current_score)
            console.print(f"   {observation.get('distribution_summary', 'N/A')[:]}...")
            
            # 2. Reasoning Agent (with feedback)
            console.print(f"\n[bold magenta]🧠 Reasoning Agent:[/bold magenta]")
            reasoning = self.reasoning_agent.process(
                observation, state, current_score, target_goal
            )
            console.print(f"   Strategy: {reasoning.get('proposed_strategy', 'N/A')[:]}...")
            
            # 3. Decision Agent (with feedback)
            console.print(f"\n[bold yellow]🎯 Decision Agent:[/bold yellow]")
            action, parameters, reason = self.decision_agent.process(
                reasoning, state, current_score, self.target_score,self.simulation
            )
            console.print(f"   Action: {action}")
            console.print(f"   Parameters: {parameters}")
            
            # 4. Execute action
            success, message = self.simulation.execute_action(action, parameters)
            new_score = self.simulation.calculate_homogeneity()
            delta=new_score-current_score

            # 5. Record feedback
            if self.use_feedback and self.feedback_agent:
                reward = self.feedback_agent.record_action(
                    action, parameters, current_score, new_score, reason
                )
                console.print(f"   [dim]Reward: {reward:.2f}[/dim]")
            
            self.history.append({
                "step": step + 1,
                "action": action,
                "parameters": parameters,
                "reason": reason,
                "score_before": current_score,
                "score_after": new_score,
                "delta": delta,
                "success": success
            })
            
            if delta>0:
                console.print(f"   [green] {message} | ↑{delta:+.3f}[/green]")
                console.print(f"   [dim] Learning: {action} was effective![/dim]")
            elif delta < 0:
                console.print(f"   [red] {message} | ↓{delta:+.3f}[/red]")
                console.print(f"   [dim] Learning: {action} decreased score. Agent will learn from this.[/dim]")
            else:
                console.print(f"   [yellow] {message} | No change[/yellow]")
                
            time.sleep(1.5)
        
        # 5. Summarization Agent
        console.print(f"\n[bold purple]📝 Summarization Agent:[/bold purple]")
        final_state = self.simulation.get_state()
        final_score = self.simulation.calculate_homogeneity()
        
        summary = self.summarization_agent.process(
            self.history, final_state, final_score, target_goal
        )
        
        self._display_summary(summary, final_score)
        
        # Display feedback statistics if enabled
        if self.use_feedback and self.feedback_agent:
            self._display_feedback_stats()
        
        # Save results
        self._save_results(final_score, summary)
        
        return final_score, summary
    
    def _display_state(self, state: list, score: float):
        from config.prompts import BALL_SYMBOLS
        
        table = Table(title="Container State", show_header=False)
        table.add_column("", style="cyan")
        
        for i, row in enumerate(state):
            row_str = ""
            for cell in row:
                row_str += f"{BALL_SYMBOLS.get(cell, '?')} "
            table.add_row(f"Row {i+1:2}", row_str)
        
        console.print(table)
        
        if score >= 0.7:
            console.print(f"[bold green]Homogeneity Score: {score:.3f} [/bold green]")
        elif score >= 0.5:
            console.print(f"[bold yellow]Homogeneity Score: {score:.3f} [/bold yellow]")
        else:
            console.print(f"[bold red]Homogeneity Score: {score:.3f} [/bold red]")
        
        console.print(f"[dim]Best score so far: {self.best_score:.3f}[/dim]")
        console.print(f"[dim]Steps without improvement: {self.steps_without_improvement}[/dim]")
    
    def _display_summary(self, summary: dict, final_score: float):
        console.print("\n" + "=" * 60)
        console.print("[bold cyan] SIMULATION SUMMARY[/bold cyan]")
        console.print("=" * 60)
        
        console.print(f"\n[bold]Actions Sequence:[/bold]")
        console.print(f"   {summary.get('actions_sequence', 'N/A')[:200]}")
        
        console.print(f"\n[bold]Result Analysis:[/bold]")
        console.print(f"   {summary.get('result_analysis', 'N/A')[:200]}")
        
        console.print(f"\n[bold]Recommendations:[/bold]")
        console.print(f"   {summary.get('recommendations', 'N/A')[:200]}")
        
        console.print(f"\n[bold]Final Score:[/bold] {final_score:.3f}")
        console.print(f"[bold]Best Score:[/bold] {self.best_score:.3f}")
        console.print(f"[bold]Total Steps:[/bold] {len(self.history)}")
        
        if self.history:
            initial_score = self.history[0]['score_before']
            improvement = final_score - initial_score
            console.print(f"[bold]Total Improvement:[/bold] {improvement:+.3f}")
    
    def _display_feedback_stats(self):
        if not self.feedback_agent:
            return
        
        stats = self.feedback_agent.get_statistics()
        
        console.print("\n" + "=" * 60)
        console.print("[bold cyan] FEEDBACK LEARNING STATISTICS[/bold cyan]")
        console.print("=" * 60)
        
        console.print(f"\n[bold]Action Statistics:[/bold]")
        console.print(f"   Total Actions: {stats['total_actions']}")
        if stats['total_actions'] > 0:
            console.print(f"   Successful: {stats['successful_actions']} ({stats['successful_actions']/stats['total_actions']*100:.1f}%)")
            console.print(f"   Failed: {stats['failed_actions']} ({stats['failed_actions']/stats['total_actions']*100:.1f}%)")
        
        console.print(f"\n[bold]Best Action:[/bold] {stats.get('best_action', 'N/A')}")
        console.print(f"[bold]Worst Action:[/bold] {stats.get('worst_action', 'N/A')}")
        
        if stats['action_performance']:
            console.print(f"\n[bold]Action Performance:[/bold]")
            for action, perf in sorted(stats['action_performance'].items(), key=lambda x: x[1], reverse=True):
                color = "green" if perf > 0 else "red"
                console.print(f"   {action}: [{color}]{perf:.2f}[/{color}]")
    
    def _save_results(self, final_score: float, summary: dict):
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "final_score": final_score,
            "best_score": self.best_score,
            "target_score": self.target_score,
            "total_steps": len(self.history),
            "feedback_enabled": self.use_feedback,
            "history": self.history,
            "summary": summary
        }
        
        if self.use_feedback and self.feedback_agent:
            results["feedback_stats"] = self.feedback_agent.get_statistics()
        
        filename = f"results/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[green] Results saved to: {filename}[/green]")

def main():
    
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║     LLM Multi-Agent System for Simulation Control            ║[/bold cyan]")
    console.print("[bold cyan]║     With Reinforcement Learning Feedback System              ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]")
    
    SELECTED_MODEL = "llama3.1:8b"  # or "llama3.2:3b", "llama3.1:8b"
    USE_FEEDBACK = True
    MAX_STEPS = 20
    TARGET_SCORE = 0.7
    
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"   Model: {SELECTED_MODEL}")
    console.print(f"   Max Steps: {MAX_STEPS}")
    console.print(f"   Target Score: {TARGET_SCORE}")
    console.print(f"   Feedback Learning: {'Enabled' if USE_FEEDBACK else 'Disabled'}")
    
    controller = LLMSimulationController(
        max_steps=MAX_STEPS,
        target_score=TARGET_SCORE,
        model=SELECTED_MODEL,
        use_feedback=USE_FEEDBACK
    )
    
    try:
        final_score, summary = controller.run(
            target_goal="Achieve an evenly distributed mixture of balls in the container"
        )
        
        console.print("\n" + "=" * 60)
        if final_score and final_score >= controller.target_score:
            console.print("[bold green] SUCCESS! Goal achieved![/bold green]")
        elif final_score and final_score >= 0.5:
            console.print(f"[bold yellow] Partial success. Score: {final_score:.3f}[/bold yellow]")
        else:
            console.print(f"[bold red] Model struggled. Consider using a larger model.[/bold red]")
            console.print("[dim]Try: ollama pull llama3.2:3b or ollama pull llama3.1:8b[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n[bold yellow] Simulation interrupted by user[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red] Error: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()