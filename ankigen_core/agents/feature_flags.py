# Feature flags for gradual agent migration rollout

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ankigen_core.logging import logger


class AgentMode(Enum):
    """Agent system operation modes"""
    LEGACY = "legacy"  # Use original LLM interface
    AGENT_ONLY = "agent_only"  # Use agents for everything
    HYBRID = "hybrid"  # Mix agents and legacy based on flags
    A_B_TEST = "a_b_test"  # Random selection for A/B testing


@dataclass
class AgentFeatureFlags:
    """Feature flags for controlling agent system rollout"""
    
    # Main mode controls
    mode: AgentMode = AgentMode.LEGACY
    
    # Generation agents
    enable_subject_expert_agent: bool = False
    enable_pedagogical_agent: bool = False
    enable_content_structuring_agent: bool = False
    enable_generation_coordinator: bool = False
    
    # Judge agents
    enable_content_accuracy_judge: bool = False
    enable_pedagogical_judge: bool = False
    enable_clarity_judge: bool = False
    enable_technical_judge: bool = False
    enable_completeness_judge: bool = False
    enable_judge_coordinator: bool = False
    
    # Enhancement agents
    enable_revision_agent: bool = False
    enable_enhancement_agent: bool = False
    
    # Workflow features
    enable_multi_agent_generation: bool = False
    enable_parallel_judging: bool = False
    enable_agent_handoffs: bool = False
    enable_agent_tracing: bool = True
    
    # A/B testing
    ab_test_ratio: float = 0.5  # Percentage for A group
    ab_test_user_hash: Optional[str] = None
    
    # Performance
    agent_timeout: float = 30.0
    max_agent_retries: int = 3
    enable_agent_caching: bool = True
    
    # Quality thresholds
    min_judge_consensus: float = 0.6  # Minimum agreement between judges
    max_revision_iterations: int = 3
    
    @classmethod
    def from_env(cls) -> "AgentFeatureFlags":
        """Load feature flags from environment variables"""
        return cls(
            mode=AgentMode(os.getenv("ANKIGEN_AGENT_MODE", "legacy")),
            
            # Generation agents
            enable_subject_expert_agent=_env_bool("ANKIGEN_ENABLE_SUBJECT_EXPERT"),
            enable_pedagogical_agent=_env_bool("ANKIGEN_ENABLE_PEDAGOGICAL_AGENT"),
            enable_content_structuring_agent=_env_bool("ANKIGEN_ENABLE_CONTENT_STRUCTURING"),
            enable_generation_coordinator=_env_bool("ANKIGEN_ENABLE_GENERATION_COORDINATOR"),
            
            # Judge agents
            enable_content_accuracy_judge=_env_bool("ANKIGEN_ENABLE_CONTENT_JUDGE"),
            enable_pedagogical_judge=_env_bool("ANKIGEN_ENABLE_PEDAGOGICAL_JUDGE"),
            enable_clarity_judge=_env_bool("ANKIGEN_ENABLE_CLARITY_JUDGE"),
            enable_technical_judge=_env_bool("ANKIGEN_ENABLE_TECHNICAL_JUDGE"),
            enable_completeness_judge=_env_bool("ANKIGEN_ENABLE_COMPLETENESS_JUDGE"),
            enable_judge_coordinator=_env_bool("ANKIGEN_ENABLE_JUDGE_COORDINATOR"),
            
            # Enhancement agents
            enable_revision_agent=_env_bool("ANKIGEN_ENABLE_REVISION_AGENT"),
            enable_enhancement_agent=_env_bool("ANKIGEN_ENABLE_ENHANCEMENT_AGENT"),
            
            # Workflow features
            enable_multi_agent_generation=_env_bool("ANKIGEN_ENABLE_MULTI_AGENT_GEN"),
            enable_parallel_judging=_env_bool("ANKIGEN_ENABLE_PARALLEL_JUDGING"),
            enable_agent_handoffs=_env_bool("ANKIGEN_ENABLE_AGENT_HANDOFFS"),
            enable_agent_tracing=_env_bool("ANKIGEN_ENABLE_AGENT_TRACING", default=True),
            
            # A/B testing
            ab_test_ratio=float(os.getenv("ANKIGEN_AB_TEST_RATIO", "0.5")),
            ab_test_user_hash=os.getenv("ANKIGEN_AB_TEST_USER_HASH"),
            
            # Performance
            agent_timeout=float(os.getenv("ANKIGEN_AGENT_TIMEOUT", "30.0")),
            max_agent_retries=int(os.getenv("ANKIGEN_MAX_AGENT_RETRIES", "3")),
            enable_agent_caching=_env_bool("ANKIGEN_ENABLE_AGENT_CACHING", default=True),
            
            # Quality thresholds
            min_judge_consensus=float(os.getenv("ANKIGEN_MIN_JUDGE_CONSENSUS", "0.6")),
            max_revision_iterations=int(os.getenv("ANKIGEN_MAX_REVISION_ITERATIONS", "3")),
        )
    
    def should_use_agents(self) -> bool:
        """Determine if agents should be used based on current mode"""
        if self.mode == AgentMode.LEGACY:
            return False
        elif self.mode == AgentMode.AGENT_ONLY:
            return True
        elif self.mode == AgentMode.HYBRID:
            # Use agents if any agent features are enabled
            return (
                self.enable_subject_expert_agent or
                self.enable_pedagogical_agent or
                self.enable_content_structuring_agent or
                any([
                    self.enable_content_accuracy_judge,
                    self.enable_pedagogical_judge,
                    self.enable_clarity_judge,
                    self.enable_technical_judge,
                    self.enable_completeness_judge,
                ])
            )
        elif self.mode == AgentMode.A_B_TEST:
            # Use hash-based or random selection for A/B testing
            if self.ab_test_user_hash:
                # Use consistent hash-based selection
                import hashlib
                hash_value = int(hashlib.md5(self.ab_test_user_hash.encode()).hexdigest(), 16)
                return (hash_value % 100) < (self.ab_test_ratio * 100)
            else:
                # Use random selection (note: not session-consistent)
                import random
                return random.random() < self.ab_test_ratio
        
        return False
    
    def get_enabled_agents(self) -> Dict[str, bool]:
        """Get a dictionary of all enabled agents"""
        return {
            "subject_expert": self.enable_subject_expert_agent,
            "pedagogical": self.enable_pedagogical_agent,
            "content_structuring": self.enable_content_structuring_agent,
            "generation_coordinator": self.enable_generation_coordinator,
            "content_accuracy_judge": self.enable_content_accuracy_judge,
            "pedagogical_judge": self.enable_pedagogical_judge,
            "clarity_judge": self.enable_clarity_judge,
            "technical_judge": self.enable_technical_judge,
            "completeness_judge": self.enable_completeness_judge,
            "judge_coordinator": self.enable_judge_coordinator,
            "revision_agent": self.enable_revision_agent,
            "enhancement_agent": self.enable_enhancement_agent,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging"""
        return {
            "mode": self.mode.value,
            "enabled_agents": self.get_enabled_agents(),
            "workflow_features": {
                "multi_agent_generation": self.enable_multi_agent_generation,
                "parallel_judging": self.enable_parallel_judging,
                "agent_handoffs": self.enable_agent_handoffs,
                "agent_tracing": self.enable_agent_tracing,
            },
            "ab_test_ratio": self.ab_test_ratio,
            "performance_config": {
                "timeout": self.agent_timeout,
                "max_retries": self.max_agent_retries,
                "caching": self.enable_agent_caching,
            },
            "quality_thresholds": {
                "min_judge_consensus": self.min_judge_consensus,
                "max_revision_iterations": self.max_revision_iterations,
            }
        }


def _env_bool(env_var: str, default: bool = False) -> bool:
    """Helper to parse boolean environment variables"""
    value = os.getenv(env_var, str(default)).lower()
    return value in ("true", "1", "yes", "on", "enabled")


# Global instance - can be overridden in tests or specific deployments
_global_flags: Optional[AgentFeatureFlags] = None


def get_feature_flags() -> AgentFeatureFlags:
    """Get the global feature flags instance"""
    global _global_flags
    if _global_flags is None:
        _global_flags = AgentFeatureFlags.from_env()
        logger.info(f"Loaded agent feature flags: {_global_flags.mode.value}")
        logger.debug(f"Feature flags config: {_global_flags.to_dict()}")
    return _global_flags


def set_feature_flags(flags: AgentFeatureFlags):
    """Set global feature flags (for testing or runtime reconfiguration)"""
    global _global_flags
    _global_flags = flags
    logger.info(f"Updated agent feature flags: {flags.mode.value}")


def reset_feature_flags():
    """Reset feature flags (reload from environment)"""
    global _global_flags
    _global_flags = None