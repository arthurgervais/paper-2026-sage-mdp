"""
Signal monitors for SAGE.

Monitors track utility, cost, and risk signals from the agent system
and provide observations to the controller.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
import time
from collections import deque


@dataclass
class Signal:
    """A single measurement from a monitor."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalMonitor(ABC):
    """Base class for signal monitors."""
    
    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        
    @abstractmethod
    def measure(self) -> Signal:
        """Take a measurement and return a signal."""
        pass
    
    def record(self, signal: Signal):
        """Record a signal to history."""
        self.history.append(signal)
    
    def get_mean(self) -> float:
        """Get mean of recent measurements."""
        if not self.history:
            return 0.0
        return sum(s.value for s in self.history) / len(self.history)
    
    def get_std(self) -> float:
        """Get standard deviation of recent measurements."""
        if len(self.history) < 2:
            return 0.0
        mean = self.get_mean()
        variance = sum((s.value - mean) ** 2 for s in self.history) / len(self.history)
        return variance ** 0.5
    
    def get_latest(self) -> Optional[Signal]:
        """Get most recent signal."""
        if not self.history:
            return None
        return self.history[-1]


class UtilitySignal(SignalMonitor):
    """
    Monitors utility signals from the agent system.
    
    Utility is typically measured as:
    - Coverage delta (new code/paths covered)
    - Vulnerabilities found
    - Exploits validated
    - Patches generated
    """
    
    def __init__(self, 
                 name: str = "utility",
                 measure_fn: Optional[Callable[[], float]] = None,
                 window_size: int = 100):
        super().__init__(name, window_size)
        self._measure_fn = measure_fn or (lambda: 0.0)
        
    def measure(self) -> Signal:
        value = self._measure_fn()
        signal = Signal(name=self.name, value=value)
        self.record(signal)
        return signal
    
    def record_value(self, value: float, **metadata) -> Signal:
        """Manually record a utility value."""
        signal = Signal(name=self.name, value=value, metadata=metadata)
        self.record(signal)
        return signal


class CostSignal(SignalMonitor):
    """
    Monitors cost signals from the agent system.
    
    Cost includes:
    - API costs (tokens, compute)
    - Time/latency
    - Resource usage
    """
    
    def __init__(self,
                 name: str = "cost",
                 measure_fn: Optional[Callable[[], float]] = None,
                 window_size: int = 100):
        super().__init__(name, window_size)
        self._measure_fn = measure_fn or (lambda: 0.0)
        self.cumulative_cost = 0.0
        
    def measure(self) -> Signal:
        value = self._measure_fn()
        signal = Signal(name=self.name, value=value)
        self.record(signal)
        self.cumulative_cost += value
        return signal
    
    def record_value(self, value: float, **metadata) -> Signal:
        """Manually record a cost value."""
        signal = Signal(name=self.name, value=value, metadata=metadata)
        self.record(signal)
        self.cumulative_cost += value
        return signal
    
    def get_cumulative(self) -> float:
        """Get total cumulative cost."""
        return self.cumulative_cost
    
    def reset_cumulative(self):
        """Reset cumulative cost tracker."""
        self.cumulative_cost = 0.0


class RiskSignal(SignalMonitor):
    """
    Monitors risk signals from the agent system.
    
    Risk indicators include:
    - Prompt injection detection alerts
    - Output validation failures
    - Anomaly detection flags
    - Guard triggering rate
    """
    
    def __init__(self,
                 name: str = "risk",
                 measure_fn: Optional[Callable[[], float]] = None,
                 window_size: int = 100,
                 alert_threshold: float = 0.5):
        super().__init__(name, window_size)
        self._measure_fn = measure_fn or (lambda: 0.0)
        self.alert_threshold = alert_threshold
        self.alert_count = 0
        
    def measure(self) -> Signal:
        value = self._measure_fn()
        signal = Signal(name=self.name, value=value)
        self.record(signal)
        
        if value > self.alert_threshold:
            self.alert_count += 1
            
        return signal
    
    def record_value(self, value: float, **metadata) -> Signal:
        """Manually record a risk value."""
        signal = Signal(name=self.name, value=value, metadata=metadata)
        self.record(signal)
        
        if value > self.alert_threshold:
            self.alert_count += 1
            
        return signal
    
    def get_alert_rate(self) -> float:
        """Get rate of high-risk alerts."""
        if not self.history:
            return 0.0
        return self.alert_count / len(self.history)
    
    def is_alert(self) -> bool:
        """Check if current risk level is above threshold."""
        latest = self.get_latest()
        if latest is None:
            return False
        return latest.value > self.alert_threshold


class CompositeMonitor:
    """
    Combines multiple monitors into a single observation vector.
    """
    
    def __init__(self):
        self.monitors: Dict[str, SignalMonitor] = {}
        
    def add_monitor(self, name: str, monitor: SignalMonitor):
        """Add a monitor."""
        self.monitors[name] = monitor
        
    def remove_monitor(self, name: str):
        """Remove a monitor."""
        if name in self.monitors:
            del self.monitors[name]
    
    def measure_all(self) -> Dict[str, Signal]:
        """Take measurements from all monitors."""
        return {name: mon.measure() for name, mon in self.monitors.items()}
    
    def get_observation_vector(self) -> List[float]:
        """Get a flat observation vector from all monitors."""
        obs = []
        for monitor in self.monitors.values():
            obs.append(monitor.get_mean())
            obs.append(monitor.get_std())
        return obs
    
    def get_observation_dict(self) -> Dict[str, float]:
        """Get observation as a dictionary."""
        obs = {}
        for name, monitor in self.monitors.items():
            obs[f"{name}_mean"] = monitor.get_mean()
            obs[f"{name}_std"] = monitor.get_std()
            latest = monitor.get_latest()
            if latest:
                obs[f"{name}_latest"] = latest.value
        return obs


# Pre-built monitors for common security signals

class CoverageDeltaMonitor(UtilitySignal):
    """Monitors code coverage improvement."""
    
    def __init__(self, coverage_fn: Callable[[], float]):
        super().__init__(name="coverage_delta")
        self._last_coverage = 0.0
        self._coverage_fn = coverage_fn
        
    def measure(self) -> Signal:
        current = self._coverage_fn()
        delta = max(0, current - self._last_coverage)
        self._last_coverage = current
        return self.record_value(delta)


class ValidationRateMonitor(UtilitySignal):
    """Monitors exploit validation success rate."""
    
    def __init__(self):
        super().__init__(name="validation_rate")
        self.attempts = 0
        self.successes = 0
        
    def record_attempt(self, success: bool):
        self.attempts += 1
        if success:
            self.successes += 1
        rate = self.successes / self.attempts if self.attempts > 0 else 0
        self.record_value(rate, attempts=self.attempts, successes=self.successes)


class FalsePositiveRateMonitor(RiskSignal):
    """Monitors false positive rate of vulnerability reports."""
    
    def __init__(self, window_size: int = 50):
        super().__init__(name="false_positive_rate", window_size=window_size)
        self.total_reports = 0
        self.false_positives = 0
        
    def record_report(self, is_false_positive: bool):
        self.total_reports += 1
        if is_false_positive:
            self.false_positives += 1
        rate = self.false_positives / self.total_reports if self.total_reports > 0 else 0
        self.record_value(rate, total=self.total_reports, fps=self.false_positives)


class PromptInjectionMonitor(RiskSignal):
    """Monitors for prompt injection attacks."""
    
    def __init__(self, detector_fn: Callable[[str], float], window_size: int = 100):
        super().__init__(
            name="prompt_injection",
            window_size=window_size,
            alert_threshold=0.7
        )
        self._detector_fn = detector_fn
        
    def check(self, text: str) -> Signal:
        """Check text for prompt injection risk."""
        risk = self._detector_fn(text)
        return self.record_value(risk, text_length=len(text))


class TokenCostMonitor(CostSignal):
    """Monitors API token costs."""
    
    # Approximate costs per 1K tokens
    COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__(name="token_cost")
        self.model = model
        self.costs = self.COSTS.get(model, {"input": 0.01, "output": 0.03})
        
    def record_usage(self, input_tokens: int, output_tokens: int) -> Signal:
        """Record token usage and compute cost."""
        cost = (
            (input_tokens / 1000) * self.costs["input"] +
            (output_tokens / 1000) * self.costs["output"]
        )
        return self.record_value(
            cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model
        )

