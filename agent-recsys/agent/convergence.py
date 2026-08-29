"""
Stopping rule + resource counters, matching the problem statement exactly:
  - converged: validation primary hasn't improved by more than epsilon over
    the last N consecutive iterations
  - OR hit the iteration cap
  - OR hit the wall-clock ceiling
whichever comes first. Also the single place token/wall-clock totals are
accumulated, so the Feasibility deliverable doesn't need reconstruction
after the fact.
"""
import time


class ConvergenceMonitor:
    def __init__(self, epsilon: float = 0.002, n_iterations: int = 3,
                 max_iterations: int = 50, max_wall_clock_hours: float = 6.0):
        self.epsilon = epsilon
        self.n_iterations = n_iterations
        self.max_iterations = max_iterations
        self.max_wall_clock_hours = max_wall_clock_hours

        self.history = []  # list of {"iteration": int, "score": float}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.start_time = time.time()

    def record(self, iteration: int, valid_primary: float,
               input_tokens: int = 0, output_tokens: int = 0):
        self.history.append({"iteration": iteration, "score": valid_primary})
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def _stalled(self) -> bool:
        if len(self.history) < self.n_iterations + 1:
            return False
        # "validation primary hasn't improved by more than epsilon over the
        # last N iterations": every one of the last N iterations must fail
        # to beat the running best (from strictly before that window) by
        # more than epsilon.
        window = self.history[-self.n_iterations:]
        running_best = max(h["score"] for h in self.history[: -self.n_iterations])
        return all(h["score"] <= running_best + self.epsilon for h in window)

    def _iterations_exhausted(self, current_iteration: int) -> bool:
        return current_iteration + 1 >= self.max_iterations

    def _wall_clock_exhausted(self) -> bool:
        return (time.time() - self.start_time) / 3600 >= self.max_wall_clock_hours

    def should_stop(self, current_iteration: int) -> tuple[bool, str]:
        if self._wall_clock_exhausted():
            return True, "wall_clock_budget_exhausted"
        if self._iterations_exhausted(current_iteration):
            return True, "iteration_cap_reached"
        if self._stalled():
            return True, "converged_epsilon_n"
        return False, ""

    def best_iteration(self) -> int:
        return max(self.history, key=lambda h: h["score"])["iteration"]

    def resource_summary(self) -> dict:
        return {
            "llm_input_tokens": self.total_input_tokens,
            "llm_output_tokens": self.total_output_tokens,
            "total_llm_tokens": self.total_input_tokens + self.total_output_tokens,
            "wall_clock_hours": round((time.time() - self.start_time) / 3600, 4),
            "iterations_run": len(self.history),
        }
