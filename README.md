# @sovereign-labs/narrowing

**Persistent failure memory for AI agents.** The memory that survives when your agent's context window doesn't.

```
npm install @sovereign-labs/narrowing
```

## The Problem

AI agents running in tool-calling loops repeat the same failures. This isn't a capability gap — it's an architecture gap.

| Incident | What happened | Root cause |
|----------|--------------|------------|
| [VS Code Copilot — 800GB](https://github.com/microsoft/vscode/issues/296194) | Created 1,526 git worktrees in 16 hours | No memory that worktree creation was failing |
| [Kilo Code — $8 burn](https://github.com/Kilo-Org/kilocode/issues/3767) | Read the same file 1,000 times, 8.5M tokens | No constraint on repeated identical reads |
| [n8n — 50% loop rate](https://github.com/n8n-io/n8n/issues/13525) | Agents stuck calling the same tool forever | No detection of action-class repetition |
| [Claude Code — compaction loop](https://github.com/anthropics/claude-code/issues/6004) | Infinite compaction → re-read → compaction cycle | Context window erases failure evidence |

Every agent framework has this problem. LangChain, CrewAI, AutoGen, Claude Code, Cursor, n8n — any system where an LLM calls tools in a loop.

**Why it persists:** LLMs process context windows, not execution history. After enough turns, context compression erases failure evidence. The agent genuinely doesn't know it already tried this.

## The Fix

Three lines of integration. One import.

```typescript
import { NarrowingLoop } from '@sovereign-labs/narrowing';
import { createToolCallAdapter, toolCallToProposal, toolCallToOutcome } from '@sovereign-labs/narrowing/adapters/tool-call';

const loop = new NarrowingLoop({ adapter: createToolCallAdapter() });

// In your agent loop — before every tool call:
const check = loop.checkProposal(
  toolCallToProposal('edit_file', { file: 'server.js', old_string: 'foo', new_string: 'bar' })
);

if (!check.allowed) {
  // Feed back to LLM: "This exact approach already failed. Try something different."
  // check.violations tells you WHY it's blocked.
  agent.feedbackToLLM(check.violations[0].reason);
  continue;
}

// After tool call completes:
loop.recordOutcome(toolCallToOutcome('edit_file', args, {
  success: false,
  error: 'search string not found in file',
  durationMs: 45,
}));
```

That's it. The loop learns from failures and structurally prevents the agent from repeating them.

## How It Works

### Three APIs

| API | When | What it does |
|-----|------|-------------|
| `checkProposal()` | Before execution | Returns `{ allowed, violations }`. Blocks proposals that match known-failed strategies. |
| `recordOutcome()` | After execution | Extracts failure signature, classifies blame, seeds constraints if corroborated. |
| `isDone()` | End of loop | Convergence detection — has the agent exhausted its search space? |

### What happens on failure

```
Tool call fails → Extract failure signature (regex, deterministic)
                → Classify blame (agent mistake or infrastructure fault?)
                → If agent's fault: track corroboration count
                → After 2 occurrences: seed constraint
                → Next proposal matching that constraint: BLOCKED
```

### Three constraint types

| Type | What it bans | Example |
|------|-------------|---------|
| `banned_strategy` | An approach that failed 2+ times | "file_edit strategy failed with edit_failed signature" |
| `radius_limit` | Progressive cap on change count | ∞ → 5 → 3 → 2 → 1 files per attempt |
| `parameter_ban` | Specific values proven broken | `n_embd=1024` caused OOM twice |

### Infrastructure faults never seed constraints

Timeouts, rate limits, permission errors — these aren't the agent's fault. Narrowing classifies blame before learning:

- **Agent failure** (syntax error, file not found, edit failed) → learns, seeds constraints
- **Harness fault** (timeout, rate limit, permission denied) → records, does NOT constrain

This prevents the "poisoned well" — where infrastructure noise narrows the search space until the agent has no valid moves left.

## Domain Adapters

Narrowing is domain-agnostic. Adapters translate domain-specific signals into the universal constraint language.

### Tool-Call Adapter (any agent framework)

```typescript
import { createToolCallAdapter } from '@sovereign-labs/narrowing/adapters/tool-call';
```

12 failure signatures: `tool_timeout`, `tool_not_found`, `permission_denied`, `rate_limited`, `file_not_found`, `syntax_error`, `edit_failed`, `command_failed`, `validation_error`, `conflict`, `empty_result`, `api_error`

7 action classes: `file_read`, `file_edit`, `file_create`, `shell_exec`, `search`, `api_call`, `delete`

Works with any tool name convention: `snake_case`, `camelCase`, `dash-case`, `dot.notation`.

### ML Training Adapter (autoresearch / hyperparameter search)

```typescript
import { createMLTrainingAdapter } from '@sovereign-labs/narrowing/adapters/ml-training';
```

13 failure signatures including `oom_gpu`, `training_divergence`, `gradient_explosion`, `tensor_shape_error`.

8 action classes based on parameter deltas: `scale_up_width`, `scale_up_depth`, `scale_down`, `lr_increase`, `lr_decrease`, `batch_size_increase`, `architecture_swap`, `optimizer_change`.

### Writing Your Own Adapter

```typescript
import type { DomainAdapter } from '@sovereign-labs/narrowing/types';

const myAdapter: DomainAdapter = {
  name: 'my-domain',
  extractSignature(error: string): string | undefined { /* regex matching */ },
  classifyBlame(error: string): 'agent_failure' | 'harness_fault' | 'unknown' { /* ... */ },
  classifyAction(params, targets): string | undefined { /* action class */ },
  extractParameters(raw): Record<string, unknown> { /* domain-relevant params */ },
  signaturePatterns: [ /* { pattern, signature, typicallyHarness, description } */ ],
};
```

## Persistence

Constraints survive process restarts. One config field:

```typescript
const loop = new NarrowingLoop({
  adapter: createToolCallAdapter(),
  statePath: './.narrowing/state.json',  // Auto-persist constraints to disk
});
```

That's it. On every `recordOutcome()`, the loop writes all constraints, convergence state, and attempt counter to disk. On construction, it loads existing state if the file exists. Each new process gets a fresh `sessionId` but inherits all constraints from prior sessions.

- Missing file → starts fresh (no error)
- Corrupt file → starts fresh (no error)
- Write failure → non-fatal (constraints still live in memory)
- Parent directories created automatically

**Why this matters:** Within-run memory is table stakes. Cross-session structural constraints — failure knowledge that persists across context window resets, process restarts, and agent handoffs — is what prevents the $8 burn from happening on day 2.

### Receipts & Journal

Every decision is also recorded in a tamper-evident hash chain (optional):

```typescript
const loop = new NarrowingLoop({
  adapter: createToolCallAdapter(),
  statePath: './.narrowing/state.json',  // Auto-persist constraints
  receipts: true,                         // Enable hash-chained audit trail
  journalPath: './.narrowing/journal.jsonl',  // Append-only event log
  receiptPath: './.narrowing/receipts.jsonl', // Tamper-evident receipt chain
});
```

### Manual Persistence (advanced)

For full control over when state is saved/loaded:

```typescript
// Save state
const state = loop.snapshot();
fs.writeFileSync('narrowing-state.json', JSON.stringify(state));

// Restore on next run
const saved = JSON.parse(fs.readFileSync('narrowing-state.json', 'utf-8'));
loop.restore(saved);
```

## Convergence Detection

The loop tracks whether the agent is making progress or spinning:

```typescript
const state = loop.getConvergence();
// { status: 'progressing' | 'plateau' | 'exhausted', totalAttempts, ... }

if (loop.isDone()) {
  // Search space exhausted — every viable strategy has been tried or banned
}
```

## Configuration

```typescript
const loop = new NarrowingLoop({
  adapter: createToolCallAdapter(),

  // Auto-persist constraints across process restarts
  statePath: './.narrowing/state.json',  // Default: undefined (no auto-persist)

  // How many times must a failure repeat before seeding a constraint?
  corroborationThreshold: 2,    // Default: 2

  // Progressive radius shrinking curve
  radiusCurve: [Infinity, 5, 3, 2, 1],  // Default

  // How long do constraints live?
  constraintTtlMs: 3600000,    // Default: 1 hour

  // Max active constraints before escalation
  maxConstraintDepth: 5,        // Default: 5

  // Score optimization direction (for scored domains like ML training)
  direction: 'minimize',        // or 'maximize'
});
```

## Architecture

```
Agent Loop
    ↓ proposal
NarrowingLoop.checkProposal()
    ├── ConstraintStore.checkProposal()  ← Are any constraints violated?
    │   ├── Strategy ban check           ← Is this action class banned?
    │   ├── Radius limit check           ← Too many targets?
    │   └── Parameter ban check          ← Is this specific value banned?
    ↓ { allowed: true }
Agent executes tool call
    ↓ outcome
NarrowingLoop.recordOutcome()
    ├── Adapter.extractSignature()       ← What went wrong? (regex)
    ├── Adapter.classifyBlame()          ← Agent's fault or infrastructure?
    ├── Adapter.classifyAction()         ← What strategy was this?
    ├── ConstraintStore.seedFromOutcome() ← Learn from failure
    ├── ConvergenceTracker.update()      ← Are we making progress?
    ├── Journal.record()                 ← Append to event log
    └── ReceiptChain.append()            ← Hash-chained audit trail
```

## Research

For the full empirical evaluation — GPU benchmarks against Gemini 2.5 Flash, honest assessment of where narrowing helps and where it doesn't, and related work analysis — see the [research paper](../../docs/narrowing/paper.md).

Key findings: frontier LLMs self-correct on simple failure boundaries within 1-2 trials (marginal within-session value), but every new session rediscovers the same failures from scratch (clear cross-session value). Narrowing's value scales with horizon length, failure complexity, and session count.

## Package Info

- **Runtime:** Zero dependencies. Pure TypeScript.
- **Size:** ~2,200 LOC across 8 source files
- **Tests:** 80 tests, 201 assertions
- **License:** MIT
- **Requires:** Bun or Node.js 18+

```
@sovereign-labs/narrowing
├── src/
│   ├── loop.ts           # NarrowingLoop — the 3-API orchestrator
│   ├── constraints.ts    # ConstraintStore — seed, check, gc
│   ├── convergence.ts    # ConvergenceTracker — progress detection
│   ├── signatures.ts     # Universal failure patterns
│   ├── blame.ts          # Blame classification engine
│   ├── journal.ts        # Append-only event log
│   ├── receipts.ts       # Tamper-evident hash chain
│   ├── types.ts          # All interfaces
│   └── adapters/
│       ├── ml-training.ts  # ML hyperparameter search
│       └── tool-call.ts    # Universal agent tool loops
└── tests/
    ├── narrowing-physics.test.ts  # Core loop + persistence tests (34 tests)
    ├── tool-call.test.ts          # Tool-call adapter tests (38 tests)
    └── long-horizon.test.ts       # Context degradation benchmark (8 tests)
```
