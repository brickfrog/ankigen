# AnkiGen Agent System

A sophisticated multi-agent system for generating high-quality flashcards using specialized AI agents.

## Overview

The AnkiGen Agent System replaces the traditional single-LLM approach with a pipeline of specialized agents:

- **Generator Agents**: Create cards with domain expertise
- **Judge Agents**: Assess quality using multiple criteria  
- **Enhancement Agents**: Improve and enrich card content
- **Coordinators**: Orchestrate workflows and handoffs

## Quick Start

### 1. Installation

```bash
pip install pyyaml
```

Or use the project's dependency management:

```bash
uv pip install -e .
```

### 2. Environment Configuration

Create a `.env` file or set environment variables:

```bash
# Basic agent mode
export ANKIGEN_AGENT_MODE=hybrid

# Enable specific agents
export ANKIGEN_ENABLE_SUBJECT_EXPERT=true
export ANKIGEN_ENABLE_CONTENT_JUDGE=true
export ANKIGEN_ENABLE_CLARITY_JUDGE=true

# Performance settings
export ANKIGEN_AGENT_TIMEOUT=30.0
export ANKIGEN_MIN_JUDGE_CONSENSUS=0.6
```

### 3. Usage

```python
from ankigen_core.agents.integration import AgentOrchestrator
from ankigen_core.llm_interface import OpenAIClientManager

# Initialize
client_manager = OpenAIClientManager()
orchestrator = AgentOrchestrator(client_manager)
await orchestrator.initialize("your-openai-api-key")

# Generate cards with agents
cards, metadata = await orchestrator.generate_cards_with_agents(
    topic="Python Functions",
    subject="programming",
    num_cards=5,
    difficulty="intermediate"
)
```

## Agent Types

### Generation Agents

#### SubjectExpertAgent
- **Purpose**: Domain-specific card generation
- **Specializes**: Technical accuracy, terminology, real-world applications
- **Configuration**: `ANKIGEN_ENABLE_SUBJECT_EXPERT=true`

#### PedagogicalAgent  
- **Purpose**: Educational effectiveness review
- **Specializes**: Bloom's taxonomy, cognitive load, learning objectives
- **Configuration**: `ANKIGEN_ENABLE_PEDAGOGICAL_AGENT=true`

#### ContentStructuringAgent
- **Purpose**: Consistent formatting and organization
- **Specializes**: Metadata enrichment, standardization
- **Configuration**: `ANKIGEN_ENABLE_CONTENT_STRUCTURING=true`

#### GenerationCoordinator
- **Purpose**: Orchestrates multi-agent generation workflows
- **Configuration**: `ANKIGEN_ENABLE_GENERATION_COORDINATOR=true`

### Judge Agents

#### ContentAccuracyJudge
- **Evaluates**: Factual correctness, terminology, misconceptions
- **Model**: GPT-4o (high accuracy needed)
- **Configuration**: `ANKIGEN_ENABLE_CONTENT_JUDGE=true`

#### PedagogicalJudge
- **Evaluates**: Educational effectiveness, cognitive levels
- **Model**: GPT-4o
- **Configuration**: `ANKIGEN_ENABLE_PEDAGOGICAL_JUDGE=true`

#### ClarityJudge
- **Evaluates**: Communication clarity, readability
- **Model**: GPT-4o-mini (cost-effective)
- **Configuration**: `ANKIGEN_ENABLE_CLARITY_JUDGE=true`

#### TechnicalJudge
- **Evaluates**: Code syntax, best practices (technical content only)
- **Model**: GPT-4o
- **Configuration**: `ANKIGEN_ENABLE_TECHNICAL_JUDGE=true`

#### CompletenessJudge
- **Evaluates**: Required fields, metadata, quality standards
- **Model**: GPT-4o-mini
- **Configuration**: `ANKIGEN_ENABLE_COMPLETENESS_JUDGE=true`

### Enhancement Agents

#### RevisionAgent
- **Purpose**: Improves rejected cards based on judge feedback
- **Configuration**: `ANKIGEN_ENABLE_REVISION_AGENT=true`

#### EnhancementAgent
- **Purpose**: Adds missing content and enriches metadata
- **Configuration**: `ANKIGEN_ENABLE_ENHANCEMENT_AGENT=true`

## Operating Modes

### Legacy Mode
```bash
export ANKIGEN_AGENT_MODE=legacy
```
Uses the original single-LLM approach.

### Agent-Only Mode
```bash
export ANKIGEN_AGENT_MODE=agent_only
```
Forces use of agent system for all generation.

### Hybrid Mode
```bash
export ANKIGEN_AGENT_MODE=hybrid
```
Uses agents when enabled via feature flags, falls back to legacy otherwise.

### A/B Testing Mode
```bash
export ANKIGEN_AGENT_MODE=a_b_test
export ANKIGEN_AB_TEST_RATIO=0.5
```
Randomly assigns users to agent vs legacy generation for comparison.

## Configuration

### Agent Configuration Files

Agents can be configured via YAML files in `config/agents/`:

```yaml
# config/agents/defaults/generators.yaml
agents:
  subject_expert:
    instructions: "You are a world-class expert in {subject}..."
    model: "gpt-4o"
    temperature: 0.7
    timeout: 45.0
    custom_prompts:
      math: "Focus on problem-solving strategies"
      science: "Emphasize experimental design"
```

### Environment Variables

#### Agent Control
- `ANKIGEN_AGENT_MODE`: Operating mode (legacy/agent_only/hybrid/a_b_test)
- `ANKIGEN_ENABLE_*`: Enable specific agents (true/false)

#### Performance
- `ANKIGEN_AGENT_TIMEOUT`: Agent execution timeout (seconds)
- `ANKIGEN_MAX_AGENT_RETRIES`: Maximum retry attempts
- `ANKIGEN_ENABLE_AGENT_CACHING`: Enable response caching

#### Quality Control
- `ANKIGEN_MIN_JUDGE_CONSENSUS`: Minimum agreement between judges (0.0-1.0)
- `ANKIGEN_MAX_REVISION_ITERATIONS`: Maximum revision attempts

## Monitoring & Metrics

### Built-in Metrics
The system automatically tracks:
- Agent execution times and success rates
- Quality approval/rejection rates
- Token usage and costs
- Judge consensus scores

### Performance Dashboard
```python
orchestrator = AgentOrchestrator(client_manager)
metrics = orchestrator.get_performance_metrics()

print(f"24h Performance: {metrics['agent_performance']}")
print(f"Quality Metrics: {metrics['quality_metrics']}")
```

### Tracing
OpenAI Agents SDK provides built-in tracing UI for debugging workflows.

## Quality Pipeline

### Phase 1: Generation
1. Route to appropriate subject expert
2. Generate initial cards
3. Optional pedagogical review
4. Optional content structuring

### Phase 2: Quality Assessment
1. Route cards to relevant judges
2. Parallel evaluation by multiple specialists
3. Calculate consensus scores
4. Approve/reject based on thresholds

### Phase 3: Improvement
1. Revise rejected cards using judge feedback
2. Re-evaluate revised cards
3. Enhance approved cards with additional content

## Cost Optimization

### Model Selection
- **Generation**: GPT-4o for accuracy
- **Simple Judges**: GPT-4o-mini for cost efficiency
- **Critical Judges**: GPT-4o for quality

### Caching Strategy
- Response caching at agent level
- Shared cache across similar requests
- Configurable cache TTL

### Parallel Processing
- Judge agents run in parallel
- Batch processing for multiple cards
- Async execution throughout

## Migration Strategy

### Gradual Rollout
1. Start with single judge agent
2. Enable A/B testing
3. Gradually enable more agents
4. Monitor quality improvements

### Rollback Plan
- Keep legacy system as fallback
- Feature flags for quick disable
- Performance comparison dashboards

### Success Metrics
- 20%+ improvement in card quality scores
- Reduced manual editing needs
- Better user satisfaction ratings
- Maintained or improved generation speed

## Troubleshooting

### Common Issues

#### Agents Not Initializing
- Check OpenAI API key validity
- Verify agent mode configuration
- Check feature flag settings

#### Poor Quality Results
- Adjust judge consensus thresholds
- Enable more specialized judges
- Review agent configuration prompts

#### Performance Issues
- Enable caching
- Use parallel processing
- Optimize model selection

### Debug Mode
```bash
export ANKIGEN_ENABLE_AGENT_TRACING=true
```

Enables detailed logging and tracing UI for workflow debugging.

## Examples

### Basic Usage
```python
# Simple generation with agents
cards, metadata = await orchestrator.generate_cards_with_agents(
    topic="Machine Learning",
    subject="data_science",
    num_cards=10
)
```

### Advanced Configuration
```python
# Custom enhancement targets
cards = await enhancement_agent.enhance_card_batch(
    cards=cards,
    enhancement_targets=["prerequisites", "learning_outcomes", "examples"]
)
```

### Quality Pipeline
```python
# Manual quality assessment
judge_results = await judge_coordinator.coordinate_judgment(
    cards=cards,
    enable_parallel=True,
    min_consensus=0.8
)
```

## Contributing

### Adding New Agents
1. Inherit from `BaseAgentWrapper`
2. Add configuration in YAML files
3. Update feature flags
4. Add to coordinator workflows

### Testing
```bash
python -m pytest tests/unit/test_agents/
python -m pytest tests/integration/test_agent_workflows.py
```

## Support

For issues and questions:
- Check the troubleshooting guide
- Review agent tracing logs  
- Monitor performance metrics
- Enable debug mode for detailed logging