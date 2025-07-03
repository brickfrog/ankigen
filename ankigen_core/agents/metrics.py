# Agent performance metrics collection and analysis

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

from ankigen_core.logging import logger


@dataclass
class AgentExecution:
    """Single agent execution record"""

    agent_name: str
    start_time: datetime
    end_time: datetime
    success: bool
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Execution duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_name": self.agent_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": self.duration,
            "success": self.success,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class AgentStats:
    """Aggregated statistics for an agent"""

    agent_name: str
    total_executions: int = 0
    successful_executions: int = 0
    total_duration: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    error_count: int = 0
    last_execution: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100

    @property
    def average_duration(self) -> float:
        """Average execution duration in seconds"""
        if self.total_executions == 0:
            return 0.0
        return self.total_duration / self.total_executions

    @property
    def average_cost(self) -> float:
        """Average cost per execution"""
        if self.total_executions == 0:
            return 0.0
        return self.total_cost / self.total_executions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_name": self.agent_name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": self.success_rate,
            "total_duration": self.total_duration,
            "average_duration": self.average_duration,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "average_cost": self.average_cost,
            "error_count": self.error_count,
            "last_execution": self.last_execution.isoformat()
            if self.last_execution
            else None,
        }


class AgentMetrics:
    """Agent performance metrics collector and analyzer"""

    def __init__(self, persistence_dir: Optional[str] = None):
        self.persistence_dir = (
            Path(persistence_dir) if persistence_dir else Path("metrics/agents")
        )
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

        self.executions: List[AgentExecution] = []
        self.agent_stats: Dict[str, AgentStats] = {}
        self._load_persisted_metrics()

    def record_execution(
        self,
        agent_name: str,
        start_time: datetime,
        end_time: datetime,
        success: bool,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a single agent execution"""
        execution = AgentExecution(
            agent_name=agent_name,
            start_time=start_time,
            end_time=end_time,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            error_message=error_message,
            metadata=metadata or {},
        )

        self.executions.append(execution)
        self._update_agent_stats(execution)

        # Persist immediately for crash resilience
        self._persist_execution(execution)

        logger.debug(
            f"Recorded execution for {agent_name}: {execution.duration:.2f}s, success={success}"
        )

    def _update_agent_stats(self, execution: AgentExecution):
        """Update aggregated statistics for an agent"""
        agent_name = execution.agent_name

        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = AgentStats(agent_name=agent_name)

        stats = self.agent_stats[agent_name]
        stats.total_executions += 1
        stats.total_duration += execution.duration
        stats.last_execution = execution.end_time

        if execution.success:
            stats.successful_executions += 1
        else:
            stats.error_count += 1

        if execution.input_tokens:
            stats.total_input_tokens += execution.input_tokens

        if execution.output_tokens:
            stats.total_output_tokens += execution.output_tokens

        if execution.cost:
            stats.total_cost += execution.cost

    def get_agent_stats(self, agent_name: str) -> Optional[AgentStats]:
        """Get statistics for a specific agent"""
        return self.agent_stats.get(agent_name)

    def get_all_agent_stats(self) -> Dict[str, AgentStats]:
        """Get statistics for all agents"""
        return self.agent_stats.copy()

    def get_executions(
        self,
        agent_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success_only: Optional[bool] = None,
    ) -> List[AgentExecution]:
        """Get filtered execution records"""
        filtered = self.executions

        if agent_name:
            filtered = [e for e in filtered if e.agent_name == agent_name]

        if start_time:
            filtered = [e for e in filtered if e.start_time >= start_time]

        if end_time:
            filtered = [e for e in filtered if e.end_time <= end_time]

        if success_only is not None:
            filtered = [e for e in filtered if e.success == success_only]

        return filtered

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a performance report for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_executions = self.get_executions(start_time=cutoff_time)

        if not recent_executions:
            return {
                "period": f"Last {hours} hours",
                "total_executions": 0,
                "agents": {},
            }

        # Group by agent
        agent_executions = {}
        for execution in recent_executions:
            if execution.agent_name not in agent_executions:
                agent_executions[execution.agent_name] = []
            agent_executions[execution.agent_name].append(execution)

        # Calculate metrics per agent
        agent_reports = {}
        total_executions = 0
        total_successful = 0
        total_duration = 0.0
        total_cost = 0.0

        for agent_name, executions in agent_executions.items():
            successful = len([e for e in executions if e.success])
            total_dur = sum(e.duration for e in executions)
            total_cost_agent = sum(e.cost or 0 for e in executions)

            agent_reports[agent_name] = {
                "executions": len(executions),
                "successful": successful,
                "success_rate": (successful / len(executions)) * 100,
                "average_duration": total_dur / len(executions),
                "total_cost": total_cost_agent,
                "average_cost": total_cost_agent / len(executions)
                if total_cost_agent > 0
                else 0,
            }

            total_executions += len(executions)
            total_successful += successful
            total_duration += total_dur
            total_cost += total_cost_agent

        return {
            "period": f"Last {hours} hours",
            "total_executions": total_executions,
            "total_successful": total_successful,
            "overall_success_rate": (total_successful / total_executions) * 100
            if total_executions > 0
            else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / total_executions
            if total_executions > 0
            else 0,
            "total_cost": total_cost,
            "average_cost": total_cost / total_executions
            if total_cost > 0 and total_executions > 0
            else 0,
            "agents": agent_reports,
        }

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality-focused metrics for card generation"""
        # Get recent judge decisions
        judge_executions = [
            e for e in self.executions if "judge" in e.agent_name.lower() and e.success
        ]

        if not judge_executions:
            return {"message": "No judge data available"}

        # Analyze judge decisions from metadata
        total_cards_judged = 0
        total_accepted = 0
        total_rejected = 0
        total_needs_revision = 0

        judge_stats = {}

        for execution in judge_executions:
            metadata = execution.metadata
            agent_name = execution.agent_name

            if agent_name not in judge_stats:
                judge_stats[agent_name] = {
                    "total_cards": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "needs_revision": 0,
                }

            # Extract decisions from metadata (format depends on implementation)
            cards_judged = metadata.get("cards_judged", 1)
            accepted = metadata.get("accepted", 0)
            rejected = metadata.get("rejected", 0)
            needs_revision = metadata.get("needs_revision", 0)

            judge_stats[agent_name]["total_cards"] += cards_judged
            judge_stats[agent_name]["accepted"] += accepted
            judge_stats[agent_name]["rejected"] += rejected
            judge_stats[agent_name]["needs_revision"] += needs_revision

            total_cards_judged += cards_judged
            total_accepted += accepted
            total_rejected += rejected
            total_needs_revision += needs_revision

        # Calculate rates
        acceptance_rate = (
            (total_accepted / total_cards_judged) * 100 if total_cards_judged > 0 else 0
        )
        rejection_rate = (
            (total_rejected / total_cards_judged) * 100 if total_cards_judged > 0 else 0
        )
        revision_rate = (
            (total_needs_revision / total_cards_judged) * 100
            if total_cards_judged > 0
            else 0
        )

        return {
            "total_cards_judged": total_cards_judged,
            "acceptance_rate": acceptance_rate,
            "rejection_rate": rejection_rate,
            "revision_rate": revision_rate,
            "judge_breakdown": judge_stats,
        }

    def _persist_execution(self, execution: AgentExecution):
        """Persist a single execution to disk"""
        try:
            today = execution.start_time.strftime("%Y-%m-%d")
            file_path = self.persistence_dir / f"executions_{today}.jsonl"

            with open(file_path, "a") as f:
                f.write(json.dumps(execution.to_dict()) + "\n")

        except Exception as e:
            logger.error(f"Failed to persist execution: {e}")

    def _load_persisted_metrics(self):
        """Load persisted metrics from disk"""
        try:
            # Load executions from the last 7 days
            for i in range(7):
                date = datetime.now() - timedelta(days=i)
                date_str = date.strftime("%Y-%m-%d")
                file_path = self.persistence_dir / f"executions_{date_str}.jsonl"

                if file_path.exists():
                    with open(file_path, "r") as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                execution = AgentExecution(
                                    agent_name=data["agent_name"],
                                    start_time=datetime.fromisoformat(
                                        data["start_time"]
                                    ),
                                    end_time=datetime.fromisoformat(data["end_time"]),
                                    success=data["success"],
                                    input_tokens=data.get("input_tokens"),
                                    output_tokens=data.get("output_tokens"),
                                    cost=data.get("cost"),
                                    error_message=data.get("error_message"),
                                    metadata=data.get("metadata", {}),
                                )
                                self.executions.append(execution)
                                self._update_agent_stats(execution)
                            except Exception as e:
                                logger.warning(f"Failed to parse execution record: {e}")

            logger.info(f"Loaded {len(self.executions)} persisted execution records")

        except Exception as e:
            logger.error(f"Failed to load persisted metrics: {e}")

    def cleanup_old_data(self, days: int = 30):
        """Clean up execution data older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)

        # Remove from memory
        self.executions = [e for e in self.executions if e.start_time >= cutoff_time]

        # Rebuild stats from remaining executions
        self.agent_stats.clear()
        for execution in self.executions:
            self._update_agent_stats(execution)

        # Remove old files
        try:
            for file_path in self.persistence_dir.glob("executions_*.jsonl"):
                try:
                    date_str = file_path.stem.split("_")[1]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff_time:
                        file_path.unlink()
                        logger.info(f"Removed old metrics file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to process metrics file {file_path}: {e}")

        except Exception as e:
            logger.error(f"Failed to cleanup old metrics data: {e}")


# Global metrics instance
_global_metrics: Optional[AgentMetrics] = None


def get_metrics() -> AgentMetrics:
    """Get the global agent metrics instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = AgentMetrics()
    return _global_metrics


def record_agent_execution(
    agent_name: str, start_time: datetime, end_time: datetime, success: bool, **kwargs
):
    """Convenience function to record an agent execution"""
    metrics = get_metrics()
    metrics.record_execution(
        agent_name=agent_name,
        start_time=start_time,
        end_time=end_time,
        success=success,
        **kwargs,
    )
