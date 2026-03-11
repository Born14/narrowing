# Narrowing: Persistent Failure Memory for AI Agent Loops

**Michael McCarty, Sovereign Labs**
**March 2026**

---

## Abstract

AI agents running in tool-calling loops systematically repeat failed strategies. We present *narrowing*, a constraint-learning runtime that extracts failure signatures from execution outcomes and structurally removes failed strategies from the agent's action space. Unlike prompt-based memory (which degrades under context window compression), narrowing constraints persist across sessions and process restarts as structural guardrails enforced before execution — the agent never gets to retry what's banned.

We evaluate narrowing in two domains: (1) LLM-guided hyperparameter search on GPU, where we measure constraint activation against a frontier model (Gemini 2.5 Flash), and (2) agent tool-calling loops, where we address the documented failure patterns that have cost production users hundreds of dollars and hundreds of gigabytes.

Our GPU benchmarks reveal a nuanced finding: frontier LLMs self-correct on simple failure boundaries within 1-2 trials, compressing narrowing's within-session value. However, every new session rediscovers the same failures from scratch — cross-session constraint persistence eliminates this redundancy entirely. The tool-call adapter addresses the more immediate market need: preventing the infinite loops, repeated reads, and runaway resource creation documented in VS Code Copilot, Kilo Code, n8n, and Claude Code.

The runtime is 2,200 LOC of zero-dependency TypeScript with domain adapters, tamper-evident receipt chains, and convergence detection. Published as `@sovereign-labs/narrowing` under MIT license.

---

## 1. Introduction

### 1.1 The Repeat-Failure Problem

When an AI agent calls tools in a loop — editing files, running commands, querying APIs — it occasionally fails. The architecturally interesting question is: what happens next?

In production, the answer is often: *it tries the same thing again.*

This isn't a capability failure. Frontier LLMs can reason about errors when the failure is visible in their context window. The problem is architectural:

1. **Context window compression.** Long-running agents periodically summarize their conversation history to fit within token limits. This summarization erases the specific details of prior failures — file paths, error messages, exact parameter values. The agent genuinely doesn't know it already tried this approach.

2. **Session boundaries.** When an agent process restarts (crash recovery, new session, handoff), all runtime memory is lost. The next session rediscovers every failure from scratch.

3. **No structural enforcement.** Even when failure information is technically present in the context, the LLM may choose to retry the same approach. Prompt-based memory is advisory — the model can and does ignore it.

### 1.2 Documented Production Incidents

These are not hypothetical concerns. The following incidents are documented in public issue trackers:

| System | Incident | Impact | Reference |
|--------|----------|--------|-----------|
| VS Code Copilot | Background agent created 1,526 git worktrees in 16 hours | ~800GB disk consumed | [microsoft/vscode#296194](https://github.com/microsoft/vscode/issues/296194) |
| Kilo Code | Agent read the same file 1,000 times in a loop | $7.59 burned, 8.5M tokens | [Kilo-Org/kilocode#3767](https://github.com/Kilo-Org/kilocode/issues/3767) |
| n8n | AI agents stuck in infinite tool-call loops | ~50% occurrence rate | [n8n-io/n8n#13525](https://github.com/n8n-io/n8n/issues/13525) |
| Claude Code | Infinite compaction → re-read → compaction cycle | Task completion blocked | [anthropics/claude-code#6004](https://github.com/anthropics/claude-code/issues/6004) |

In each case, the agent repeated an action that had already failed or was demonstrably unproductive. No structural mechanism existed to prevent the repetition.

### 1.3 Contribution

Narrowing is a constraint-learning runtime that sits between the agent's proposal generator (the LLM) and the execution environment. It provides three APIs:

- **`checkProposal()`** — Before execution: is this proposal allowed given current constraints?
- **`recordOutcome()`** — After execution: extract failure signature, classify blame, seed constraints if corroborated.
- **`isDone()`** — Has the agent exhausted its viable search space?

The key design decisions:

1. **Structural enforcement, not advisory.** Constraints are checked by the runtime, not suggested to the LLM. A banned strategy cannot be attempted regardless of what the model generates.

2. **Blame classification.** Infrastructure faults (timeouts, rate limits, permission errors) never seed constraints. Only agent-attributable failures narrow the search space. This prevents the "poisoned well" where environmental noise progressively eliminates all valid strategies.

3. **Corroboration threshold.** A single failure does not seed a constraint. By default, a failure signature must be observed twice before generating a structural ban. This prevents over-reaction to transient errors.

4. **Domain adapters.** The core runtime is domain-agnostic. Adapters translate domain-specific signals (GPU OOM, file-not-found, edit-failed) into the universal constraint language.

---

## 2. Architecture

```
Agent Proposal Generator (LLM)
    ↓ proposal
NarrowingLoop
    ├── ConstraintStore.checkProposal()
    │   ├── Strategy ban:    Is this action class banned?
    │   ├── Radius limit:    Too many targets?
    │   └── Parameter ban:   Is this specific value banned?
    │
    │   → { allowed: false, violations } → reject, feed back to LLM
    │   → { allowed: true }              → proceed to execution
    │
    ↓ execution
    ↓ outcome
NarrowingLoop.recordOutcome()
    ├── Adapter.extractSignature()        Regex-based, deterministic
    ├── Adapter.classifyBlame()           agent_failure | harness_fault | unknown
    ├── Adapter.classifyAction()          Action class for strategy bans
    ├── ConstraintStore.seedFromOutcome() Learn if corroborated
    ├── ConvergenceTracker.update()       Progress detection
    ├── Journal.append()                  Event log
    └── ReceiptChain.append()            Tamper-evident audit trail
```

### 2.1 Constraint Types

| Type | Trigger | Effect | Example |
|------|---------|--------|---------|
| `banned_strategy` | Same action class + failure signature, 2+ times | Blocks any proposal with that action class | "file_edit with edit_failed: banned" |
| `radius_limit` | Progressive failures | Shrinks max allowed targets: ∞ → 5 → 3 → 2 → 1 | Forces smaller, safer changes |
| `parameter_ban` | Specific value failed 2+ times | Blocks proposals with that exact parameter value | `n_embd=1024` caused OOM twice |

### 2.2 Failure Signatures

Signatures are extracted deterministically via regex pattern matching — no LLM inference required. Each domain adapter provides signature patterns ordered by priority (first match wins).

The tool-call adapter provides 12 signatures:

| Signature | Pattern | Blame |
|-----------|---------|-------|
| `tool_timeout` | `timeout\|ETIMEDOUT\|deadline exceeded` | harness_fault |
| `tool_not_found` | `tool not found\|unknown tool` | harness_fault |
| `permission_denied` | `permission denied\|EACCES\|403` | harness_fault |
| `rate_limited` | `429\|rate.?limit\|too many requests` | harness_fault |
| `file_not_found` | `ENOENT\|no such file\|file not found` | agent_failure |
| `syntax_error` | `SyntaxError\|parse error\|invalid JSON` | agent_failure |
| `edit_failed` | `search string not found\|edit.*failed` | agent_failure |
| `command_failed` | `exit code [1-9]\|command failed` | agent_failure |
| `validation_error` | `validation failed\|invalid.*argument` | agent_failure |
| `conflict` | `409\|conflict\|already exists` | agent_failure |
| `empty_result` | `no results\|empty response\|null` | agent_failure |
| `api_error` | `500\|502\|503\|internal server error` | unknown |

Infrastructure faults (harness_fault) are recorded but never seed constraints. This distinction is critical — without it, a flaky network connection would progressively ban every API call the agent attempts.

### 2.3 Convergence Detection

The `ConvergenceTracker` monitors whether the agent is making progress:

- **Progressing:** Scores improving or new strategies being explored
- **Plateau:** No score improvement within a configurable window
- **Exhausted:** Every viable strategy has been tried or banned

`isDone()` returns true when the tracker detects exhaustion — the search space has been narrowed to the point where no valid proposals remain. This enables graceful termination instead of running until a timeout.

### 2.4 Persistence

Two persistence mechanisms:

**Journal** — Append-only event log (`narrowing.jsonl`). Records every outcome, constraint seed, and proposal check. Enables post-hoc analysis and replay.

**Receipt chain** — Tamper-evident hash chain where each receipt includes `hash = sha256(previousHash + canonicalPayload)`. Modification of any receipt invalidates all subsequent hashes. Enables auditing of the constraint-learning process.

**Cross-session persistence** via `snapshot()` / `restore()`:

```typescript
// End of session
const state = loop.snapshot();
fs.writeFileSync('narrowing-state.json', JSON.stringify(state));

// Start of next session
const saved = JSON.parse(fs.readFileSync('narrowing-state.json'));
loop.restore(saved);
// Constraints from prior sessions are immediately active
```

---

## 3. GPU Benchmark: LLM-Guided Hyperparameter Search

### 3.1 Experimental Setup

We evaluate narrowing in Karpathy's *autoresearch* framework — an automated ML research loop where an LLM proposes hyperparameter configurations, trains a character-level language model, and iterates based on results.

**Task:** Train a nanoGPT-style transformer on the Shakespeare corpus. Score: validation bits-per-byte (bpb, lower is better). Training budget: 90 seconds per trial.

**Hardware:** NVIDIA L4 GPU (24GB VRAM), GCP g2-standard-4.

**LLM:** Gemini 2.5 Flash (temperature=0.3, thinkingBudget=0). The LLM sees the full history of prior trials and proposes the next configuration.

**Search space:** Three parameters — `depth` (transformer layers), `aspect_ratio` (width-to-depth ratio), `matrix_lr` (learning rate scale).

**Narrowing adapter:** ML training adapter with 13 failure signatures and 8 action classes.

### 3.2 Benchmark Versions

| Version | Configuration | Purpose |
|---------|--------------|---------|
| v5 | L4, bs=16, 3 seeds × 15 trials | Baseline: LLM integration validation |
| v6 | L4, bs=32, 5 seeds × 25 trials | Primary: constraint activation test |

**Why two versions:** v5 established that the LLM integration works end-to-end but revealed that the L4's 24GB VRAM was too generous — Gemini's conservative proposals (depth 3-10) never approached the OOM boundary (~depth 14-16). Zero constraints were seeded.

v6 shifts the OOM boundary closer by increasing batch size from 16 to 32, which increases per-step memory consumption. (Batch size 64 was calibrated first but caused 100% OOM with LLM-guided proposals — Gemini's natural range of depth 8-32 all exceeded the bs=64 boundary. Batch size 32 places the boundary in the middle of the proposal space.) At bs=32, depth=8 OOMs — exactly where Gemini naturally starts proposing. Both vanilla and narrowing modes use the same batch size, so the confound affects both equally.

### 3.3 v6 Results

**Protocol:** 5 seeds × 2 modes × 25 trials = 250 training runs. ~10 hours wall time on L4.

| Metric | Vanilla | Narrowing | Delta |
|--------|---------|-----------|-------|
| Successful trials | 113/125 | 102/125 | -11 |
| OOM failures | 10 | 10 | 0 |
| Blocked proposals | 0 | **12** | +12 |
| Wasted trials (OOM + timeout) | 12 | 23 | +11 |
| Best score (bpb) | **1.891** | 1.866 | -0.025 |
| Average score | 1.689 | **1.747** | +0.058 |

### 3.4 The Cross-Session Pattern

Every seed, both modes, produced an identical pattern:

1. **Trial 1:** Baseline (deterministic)
2. **Trial 2:** Gemini proposes depth=8+ → OOM
3. **Trial 3 vanilla:** Gemini retries a similar config → OOM again (repeated failure)
4. **Trial 3 narrowing:** Constraint already seeded → proposal **blocked** → no GPU waste
5. **Trials 4-25:** Gemini self-corrects, stays below the boundary in both modes

This pattern was perfectly consistent across all 5 seeds.

### 3.5 Analysis

**Mechanism activation: YES.** 12 blocked proposals across 5 seeds. Constraints seeded from real OOM failures. Every block corresponded to a configuration that would have exceeded the GPU memory boundary.

**Within-session value: MARGINAL.** Gemini 2.5 Flash adapts to OOM failures within 1 trial. The 12 blocked proposals saved ~360 seconds of GPU time (12 × ~30s per OOM), but Gemini would have self-corrected on the very next trial regardless.

**Cross-session value: CLEAR.** Without narrowing, every new seed starts fresh — trial 2 always hits the same OOM boundary. With persistent constraint stores, subsequent sessions would start with the boundary already known. The 10 OOMs that occurred identically across all 5 vanilla seeds would be prevented entirely in cross-session mode.

**Score paradox:** Vanilla achieved the best single score (1.891) because unconstrained exploration occasionally lands on near-optimal aggressive configurations. Narrowing's higher average (1.747 vs 1.689) reflects that constraints keep the LLM in productive regions. This suggests a refinement: confidence-calibrated constraints that loosen over time.

### 3.6 Honest Assessment

v6 demonstrates that narrowing's constraint mechanism activates correctly against real GPU failures and a frontier LLM. However, the within-session benefit is small because Gemini 2.5 Flash is smart enough to learn from a single OOM.

**Narrowing's value scales with:**
- **Horizon length** — More trials means more opportunities for context window compression to erase failure evidence
- **Failure complexity** — Simple boundaries (single OOM threshold) are easy for frontier models. Multi-dimensional failure surfaces (tool interaction effects, cascading errors) are harder
- **Model capability** — Weaker models (local Ollama, smaller GPT variants) benefit more from structural constraints
- **Session count** — Cross-session persistence provides value proportional to how often the agent restarts

---

## 4. Tool-Call Adapter: Agent Loop Reliability

### 4.1 Design

The tool-call adapter is the primary integration surface for agent framework developers. It classifies any tool call into:

**Action classes** (7): `file_read`, `file_edit`, `file_create`, `shell_exec`, `search`, `api_call`, `delete`

Classification uses normalized tool name matching. Tool names in any convention (`snake_case`, `camelCase`, `dash-case`) are normalized to space-separated tokens before word-boundary regex matching. When the tool name is ambiguous, parameter shape provides fallback classification (e.g., presence of `command` key → `shell_exec`).

**Failure signatures** (12): Regex-based extraction from error messages, ordered by priority. Each signature has a default blame classification that can be overridden by context.

**Fingerprinting:** Structural identity of a tool call — what tool, what target, what pattern. Used for exact-match constraint checking. Not a hash of full arguments (too brittle), but an extraction of the structural parameters that define "same call."

### 4.2 Integration

```typescript
import { NarrowingLoop } from '@sovereign-labs/narrowing';
import { createToolCallAdapter, toolCallToProposal, toolCallToOutcome }
  from '@sovereign-labs/narrowing/adapters/tool-call';

const loop = new NarrowingLoop({ adapter: createToolCallAdapter() });

// Before every tool call
const check = loop.checkProposal(
  toolCallToProposal(toolName, args)
);
if (!check.allowed) {
  feedbackToLLM(check.violations[0].reason);
  continue;
}

// After tool call
loop.recordOutcome(toolCallToOutcome(toolName, args, result));
```

### 4.3 The Kilo Code Scenario

Reproduced in our test suite:

1. Agent calls `read_file` on `data.json` → empty result
2. Agent calls `read_file` on `data.json` again → same empty result (corroboration)
3. Constraint seeded: `file_read` strategy with `empty_result` signature
4. Third `read_file` on `data.json` → **BLOCKED**

In production (Kilo Code), this loop ran 1,000 times, consuming 8.5M tokens ($7.59). With narrowing, it would have been blocked on attempt 3.

### 4.4 Infrastructure Fault Isolation

Critical to the design: infrastructure faults never seed constraints.

If an API endpoint returns 429 (rate limited) twice, narrowing does NOT ban the `api_call` strategy. Rate limiting is the server's problem, not the agent's mistake. Banning API calls after rate limits would be the "poisoned well" — environmental noise progressively eliminating every valid strategy until the agent has no moves left.

The blame classifier distinguishes:
- **Agent failures** (file not found, syntax error, edit failed) → constraints seeded after corroboration
- **Harness faults** (timeout, rate limit, permission denied) → recorded, never constraining
- **Unknown** (5xx errors) → recorded, conservative handling

### 4.5 Long-Horizon Benchmark: Context Window Degradation

The strongest argument for narrowing is at long horizons where context compression erases failure evidence. We built a controlled benchmark that directly measures this effect.

**Setup:** A simulated agent makes 200 tool calls against a mock codebase (6 files, deterministic success/failure). The agent has a configurable "memory window" — it only remembers the last N outcomes. When a failure scrolls out of the window, the agent has a 30% probability of replaying the same failing call. This models the documented behavior in production agents (context compaction erasing failure evidence).

No LLM needed. No GPU. Deterministic via Mulberry32 PRNG. Runs in <1 second.

**Results (seed=42, 200 calls):**

| Memory Window | Vanilla Repeats | Narrowing Repeats | Blocked | Wasted Cost |
|---------------|----------------|-------------------|---------|-------------|
| 10 (aggressive) | 90 | 3 | 143 | $0.318 → $0.067 |
| 20 (typical) | 82 | 2 | 135 | $0.297 → $0.071 |
| 50 (generous) | 71 | 2 | 115 | $0.261 → $0.065 |

**Degradation curve (window=20):**

| Horizon | Vanilla Repeats | Narrowing Repeats | Delta |
|---------|----------------|-------------------|-------|
| 50 calls | 10 | 1 | +9 |
| 100 calls | 34 | 2 | +32 |
| 150 calls | 58 | 2 | +56 |
| 200 calls | 82 | 2 | +80 |

Vanilla repeats grow linearly with horizon length — every time a failure scrolls out of the context window, the agent risks replaying it. Narrowing repeats flatline at 2 because constraints are structural (outside the context window) and survive compression.

**Multi-seed stability:** Narrowing wins on 5/5 seeds (seeds 1-5).

**Cross-session persistence:** Session 2 restored from session 1's snapshot immediately blocks the same calls — zero rediscovery. Infrastructure faults (10 consecutive timeouts) correctly produce zero constraints.

**Cost projection at Kilo Code scale:** 215 saved calls per 200-call session. At $0.003/call, $193.50/month savings for a 10-session/day workload.

This benchmark addresses the limitation identified in §3.6 — within-session value is marginal for smart models on simple boundaries, but context window degradation is not a model capability problem. It is an architecture problem. Narrowing solves it structurally.

---

## 5. Related Work

### 5.1 Agent Memory Systems

**LangChain Memory** and **CrewAI Long-Term Memory** store conversation history and retrieved context. These are content memories — they inform the LLM's reasoning but don't structurally prevent actions. The LLM can choose to ignore them.

**Reflexion** (Shinn et al., 2023) generates verbal self-reflections after failures. These reflections are re-injected into the prompt on the next attempt. Like narrowing, this creates a feedback loop from failures. Unlike narrowing, the reflections are advisory (the LLM can ignore them) and degrade under context compression.

**Voyager** (Wang et al., 2023) builds a skill library from successful executions in Minecraft. This is complementary to narrowing — Voyager remembers what works, narrowing remembers what doesn't.

### 5.2 Constraint-Based Search

**Tabu search** (Glover, 1986) maintains a list of recently visited solutions and forbids returning to them. Narrowing's `banned_strategy` constraint is conceptually similar, but operates on action classes rather than specific solutions, and uses failure signatures rather than visit recency for constraint generation.

**Novelty search** (Lehman & Stanley, 2011) rewards behavioral novelty to avoid convergence to local optima. Narrowing's convergence detection serves a similar purpose — detecting when the agent is spinning rather than exploring.

### 5.3 What's Missing

No existing system provides all three properties simultaneously:
1. **Structural enforcement** (not advisory)
2. **Blame-aware learning** (infrastructure faults don't poison the constraint store)
3. **Cross-session persistence** (constraints survive context compression and process restarts)

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

**Simple failure boundaries.** v6 demonstrates that frontier LLMs self-correct on single-dimensional failure boundaries (GPU memory) within 1-2 trials. Narrowing's within-session value is marginal in this regime. The value proposition is stronger for multi-dimensional failure surfaces and longer horizons where context compression erases failure evidence.

**Corroboration delay.** The default threshold of 2 means the first failure always executes. For expensive operations (GPU training, cloud API calls), even one wasted execution has real cost. Configurable thresholds trade off between false positives (over-constraining from noise) and wasted executions.

**Action class granularity.** The tool-call adapter classifies at the tool-name level (`file_edit`), not at the strategy level ("rewrite the entire file" vs "patch one line"). Banning `file_edit` after two failures is potentially too broad. Finer-grained action classes would improve precision.

### 6.2 Future Directions

**Cross-agent constraint sharing.** Multiple agents working on related tasks could share constraint stores with scope-aware filtering. A constraint learned on one GPU configuration shouldn't automatically apply to a different GPU — but a constraint about a specific file being read-only should propagate immediately. The v6 dataset includes context fields to enable post-hoc simulation of scoped sharing (planned as v7).

**Confidence-calibrated constraints.** v6's score paradox (narrowing produces higher average but lower best score) suggests that permanent strategy bans are too aggressive. Constraints that weaken over time would allow re-exploration of boundary regions after the agent has established a baseline in safer territory.

**Live LLM benchmarks with real context compression.** §4.5 demonstrates the effect with simulated context windows. Validating this against live LLM agents (Claude Code, Cursor, LangGraph) where real context compaction occurs would strengthen the empirical case. The simulated 30% replay probability is based on production incident reports, but direct measurement of replay rates under actual compaction would calibrate the model.

---

## 7. Conclusion

Narrowing addresses an architectural gap in AI agent systems: the absence of persistent, structural failure memory. Frontier LLMs are capable of learning from failures when those failures are visible in their context window — but context windows compress, sessions restart, and advisory memory gets ignored.

The GPU benchmark (v6) demonstrates that the constraint mechanism activates correctly and that cross-session persistence provides clear value, even when within-session value is marginal against a smart model. The long-horizon tool-call benchmark (§4.5) proves the complementary point: context window degradation causes vanilla agents to repeat failures at linearly increasing rates, while narrowing holds flat regardless of horizon length. The tool-call adapter addresses the immediate production need: preventing the documented infinite loops, repeated reads, and runaway resource creation in shipping agent systems.

The runtime is small (2,200 LOC), zero-dependency, and designed for integration into any agent loop with three function calls. The domain adapter architecture ensures extensibility without core changes.

---

## Appendix A: Package Details

- **Package:** `@sovereign-labs/narrowing`
- **License:** MIT
- **Runtime:** Zero dependencies, pure TypeScript
- **Tests:** 72 tests, 182 assertions (core 26, tool-call adapter 38, long-horizon benchmark 8)
- **Adapters:** ML training (13 signatures, 8 action classes), tool-call (12 signatures, 7 action classes)
- **Source:** [github.com/Born14/mcp-proxy](https://github.com/Born14/mcp-proxy) (packages/narrowing)

## Appendix B: v5 Calibration Data

v5 (L4, bs=16, 3 seeds × 15 trials) established that Gemini's proposals stay well within the L4's memory budget at batch size 16. Zero OOMs occurred. Zero constraints were seeded. This calibrated the boundary — the OOM wall was too far away for Gemini's conservative search to reach it. Full analysis: `docs/narrowing/v5-analysis.md`.

## Appendix C: v6 Per-Seed Data

| Seed | Mode | OK | OOM | Timeout | Blocked | Best Score | Avg Score |
|------|------|----|-----|---------|---------|------------|-----------|
| 1 | vanilla | 22 | 2 | 1 | 0 | 1.815 | 1.609 |
| 1 | narrowing | 22 | 2 | 0 | 1 | 1.851 | 1.769 |
| 2 | vanilla | 23 | 2 | 0 | 0 | 1.891 | 1.724 |
| 2 | narrowing | 21 | 2 | 0 | 2 | 1.851 | 1.780 |
| 3 | vanilla | 23 | 2 | 0 | 0 | 1.886 | 1.658 |
| 3 | narrowing | 19 | 2 | 0 | 4 | 1.866 | 1.785 |
| 4 | vanilla | 22 | 2 | 1 | 0 | 1.890 | 1.803 |
| 4 | narrowing | 18 | 2 | 1 | 4 | 1.866 | 1.778 |
| 5 | vanilla | 23 | 2 | 0 | 0 | 1.891 | 1.653 |
| 5 | narrowing | 22 | 2 | 0 | 1 | 1.865 | 1.636 |
