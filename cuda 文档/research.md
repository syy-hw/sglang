# Research Context

Mode: Exploration, investigation, learning
Focus: Understanding before acting

## Behavior
- Read widely before concluding
- Ask clarifying questions
- Document findings as you go
- Don't write code until understanding is clear

## Research Process
1. Understand the question
2. Explore relevant code/docs
3. Form hypothesis
4. Verify with evidence
5. Summarize findings

## Tools to favor
- Read for understanding code
- Grep, Glob for finding patterns
- WebSearch, WebFetch for external docs
- Task with Explore agent for codebase questions

## Output
Findings first, recommendations second

## Context Conservation Principle (CRITICAL)

**Main agent context is a scarce resource. If the main agent's context overflows, all global state is lost. If a subagent's context overflows, only that subagent's work is lost and can be retried.**

### When to Delegate

The main agent SHOULD NOT do bulk source code reading when:
- The task requires reading 5+ source files, OR
- The main agent estimates the code to read would exceed 20% of context

In these cases, delegate all source code reading, tracing, and searching to subagents.

### Orchestration Rules

1. **All subagents run in parallel** — no serial dependencies between subagents. Subagent B must never wait for Subagent A to finish before starting.

2. **Main agent responsibilities only**:
   - Read existing artifacts (docs, specs, output files — not raw source code)
   - Plan task decomposition and dispatch subagents
   - Assemble subagent results into final output
   - Verify results

3. **Human-in-the-loop review** — When subagents complete:
   - Main agent MUST present each subagent's process and results to the user
   - Main agent evaluates quality itself (completeness, accuracy, format)
   - User reviews and provides feedback
   - If either the user or the main agent has feedback, the subagent iterates on its previous work (not starts over) until both are satisfied

### Subagent Failure Handling

- **Rate limit error**: Wait 60 seconds, re-launch with identical prompt
- **Context overflow**: Split into 2 smaller subagents, each covering half the scope
- **Incomplete or wrong output**: Launch a corrective subagent to fix the gap

### When NOT to Delegate

- Single-file edits with clear scope
- Simple bug fixes with known location
- Configuration changes
- Tasks where you already know exactly which files to modify
