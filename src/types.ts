/**
 * @sovereign-labs/narrowing — Type System
 *
 * The type contract for constraint-learning runtimes.
 * Every type here is domain-agnostic. Domain specifics live in adapters.
 *
 * Core insight: Agents that structurally learn what NOT to try again.
 * The search space monotonically shrinks through execution-conditioned
 * constraint accumulation.
 */

// =============================================================================
// OUTCOMES — What happened when the agent tried something
// =============================================================================

/**
 * The result of a single execution attempt.
 *
 * Not opinion. Physics. Every field is observable fact.
 */
export interface Outcome {
  /** Unique attempt identifier */
  id: string;

  /** When this attempt ran */
  timestamp: number;

  /** Duration in milliseconds */
  durationMs: number;

  /** The numeric score produced (val_bpb, test accuracy, deploy success, etc.) */
  score: number | null;

  /** Is higher better ('maximize') or lower better ('minimize')? */
  direction: ScoreDirection;

  /** Did the attempt succeed or fail? */
  status: 'success' | 'failure' | 'error';

  /** If failed: what category of failure? (deterministic regex, not LLM) */
  failureSignature?: string;

  /** Who's at fault — infrastructure or agent? */
  failureKind?: FailureKind;

  /** What strategy class was attempted? (strategy-level, not instance-level) */
  actionClass?: string;

  /** The parameters/config that were tried */
  parameters: Record<string, unknown>;

  /** What files/targets were modified */
  targets: string[];

  /** Arbitrary domain metadata */
  metadata?: Record<string, unknown>;

  /** Raw error string (for signature extraction) */
  error?: string;
}

export type ScoreDirection = 'minimize' | 'maximize';

// =============================================================================
// FAILURE CLASSIFICATION — Who's at fault?
// =============================================================================

/**
 * FailureKind — prevents "poisoned well" by distinguishing infrastructure
 * errors from agent mistakes. Infrastructure faults don't seed constraints.
 *
 * - harness_fault: Infrastructure broke (OOM, NCCL timeout, SSH failure)
 * - agent_failure: Agent's proposal was genuinely wrong (syntax, bad config)
 * - unknown: Can't tell — no special treatment
 */
export type FailureKind = 'harness_fault' | 'agent_failure' | 'unknown';

// =============================================================================
// CONSTRAINTS — What the agent is NOT allowed to try
// =============================================================================

/**
 * A structural constraint derived from execution failures.
 *
 * Constraints are hard guardrails, not advice. They are enforced by the
 * runtime before execution — the agent never gets to try what's banned.
 *
 * Three types:
 * - banned_strategy: An approach that failed 2+ times (e.g., "scale_up_width")
 * - radius_limit: Progressive cap on how many things can change per attempt
 * - parameter_ban: Specific parameter value that's proven broken
 */
export interface Constraint {
  /** Unique constraint identifier */
  id: string;

  /** What kind of constraint */
  type: 'banned_strategy' | 'radius_limit' | 'parameter_ban';

  /** The failure signature that spawned this constraint */
  signature: string;

  /** The action class being banned (for strategy bans) */
  actionClass?: string;

  /** Why this constraint exists (human-readable) */
  reason: string;

  /** Has this been corroborated (seen 2+ times)? */
  corroborated: boolean;

  /** Number of failure occurrences that led to this constraint */
  occurrences: number;

  /** When this constraint was created */
  introducedAt: number;

  /** Auto-expiry timestamp (optional — some constraints are permanent) */
  expiresAt?: number;

  /** Scope: session-level (cleared when loop ends) or persistent */
  scope: 'session' | 'persistent';

  /** For radius limits: maximum allowed change count */
  maxChanges?: number;

  /** For parameter bans: which parameter and what value(s) are banned */
  bannedParameter?: string;
  bannedValues?: unknown[];

  /** The targets (files, params) this constraint applies to */
  appliesTo?: string[];
}

// =============================================================================
// PROPOSALS — What the agent wants to try next
// =============================================================================

/**
 * A proposal is the agent's next intended action, checked BEFORE execution.
 * This is the star API — checkProposal() makes the runtime active, not passive.
 */
export interface Proposal {
  /** What parameters/config the agent wants to use */
  parameters: Record<string, unknown>;

  /** What files/targets will be modified */
  targets: string[];

  /** What strategy class this represents (optional — auto-classified if omitted) */
  actionClass?: string;

  /** Arbitrary domain metadata */
  metadata?: Record<string, unknown>;
}

/**
 * The result of checking a proposal against active constraints.
 */
export interface ProposalCheck {
  /** Is the proposal allowed? */
  allowed: boolean;

  /** If blocked: which constraints blocked it */
  violations: ConstraintViolation[];

  /** Active constraint count at time of check */
  activeConstraints: number;

  /** Current radius limit (if any) */
  radiusLimit?: number;

  /** Search space reduction percentage (0-100) */
  searchSpaceReduction: number;
}

export interface ConstraintViolation {
  /** The constraint that was violated */
  constraint: Constraint;

  /** Why this proposal violates it */
  reason: string;

  /** What type of ban triggered */
  banType: 'strategy' | 'radius' | 'parameter';
}

// =============================================================================
// CONVERGENCE — Is the agent making progress?
// =============================================================================

/**
 * Convergence state tracks whether the search is making progress,
 * has stalled, or is exhausted.
 */
export interface ConvergenceState {
  /** Overall assessment */
  status: 'progressing' | 'plateau' | 'exhausted' | 'constrained_out';

  /** Total attempts so far */
  totalAttempts: number;

  /** Best score achieved */
  bestScore: number | null;

  /** Which attempt achieved the best score */
  bestAttempt: number;

  /** Consecutive attempts without improvement */
  noImprovementStreak: number;

  /** How much of the original search space is still available (0.0-1.0) */
  searchSpaceRemaining: number;

  /** Number of active constraints */
  activeConstraintCount: number;

  /** Number of banned strategy classes */
  bannedStrategyCount: number;
}

// =============================================================================
// DOMAIN ADAPTER — Pluggable domain translation
// =============================================================================

/**
 * The adapter is how narrowing speaks your domain's language.
 *
 * Each domain (ML training, code generation, CI/CD, etc.) implements this
 * interface to translate between domain-specific execution details and the
 * generic narrowing primitives.
 */
export interface DomainAdapter {
  /** Human-readable name for this domain */
  name: string;

  /**
   * Extract a failure signature from an error string.
   * Deterministic regex matching — not LLM inference.
   * Return undefined for unrecognized errors.
   */
  extractSignature(error: string): string | undefined;

  /**
   * Classify a failure as infrastructure vs agent error.
   * Infrastructure faults don't seed constraints — prevents "poisoned well."
   */
  classifyBlame(error: string, context?: Record<string, unknown>): FailureKind;

  /**
   * Classify the strategy class of a proposal or failed attempt.
   * Strategy-level, not instance-level. "scale_up_width" not "n_embd=1536".
   */
  classifyAction(params: Record<string, unknown>, targets: string[]): string | undefined;

  /**
   * Extract the parameters that matter for constraint checking.
   * Filters out irrelevant metadata, normalizes values.
   */
  extractParameters(raw: Record<string, unknown>): Record<string, unknown>;

  /**
   * The failure signature patterns this adapter recognizes.
   * Used for documentation and testing — the runtime uses extractSignature().
   */
  signaturePatterns: SignaturePattern[];
}

export interface SignaturePattern {
  /** Regex pattern that matches this failure */
  pattern: RegExp;

  /** Canonical signature name */
  signature: string;

  /** Is this typically an infrastructure fault? */
  typicallyHarness: boolean;

  /** Human-readable description */
  description: string;
}

// =============================================================================
// CONFIGURATION
// =============================================================================

export interface NarrowingConfig {
  /** Domain-specific adapter */
  adapter: DomainAdapter;

  /** Score optimization direction */
  direction: ScoreDirection;

  /** Require N failures before seeding a constraint (default: 2) */
  corroborationThreshold: number;

  /**
   * Progressive radius curve: how maxChanges shrinks on failure.
   * Default: [Infinity, 5, 3, 2, 1]
   * Index = number of failures that triggered radius shrink
   */
  radiusCurve: number[];

  /** Constraint TTL in ms (default: 1 hour = 3_600_000) */
  constraintTtlMs: number;

  /** Max constraints before declaring exhaustion (default: 5) */
  maxConstraintDepth: number;

  /** How many attempts with no improvement before plateau (default: 10) */
  plateauWindow: number;

  /** Score improvement threshold below which we consider plateau (default: 0.001) */
  plateauTolerance: number;

  /** Path to persist the journal (default: .narrowing/journal.jsonl) */
  journalPath: string;

  /** Enable tamper-evident receipt chain (default: true) */
  receipts: boolean;

  /** Path to persist receipts (default: .narrowing/receipts.jsonl) */
  receiptPath: string;

  /**
   * Path to auto-persist constraint state as JSON.
   * When set:
   * - On construction: loads existing state if file exists
   * - On every recordOutcome(): saves state to disk
   * This is how constraints survive across process restarts.
   * Default: undefined (no auto-persist — use snapshot()/restore() manually)
   */
  statePath?: string;
}

// =============================================================================
// NARROWING RESULT — What the runtime returns after each attempt
// =============================================================================

/**
 * The full result of recording an outcome and updating constraints.
 * Returned by NarrowingLoop.recordOutcome().
 */
export interface NarrowingResult {
  /** The outcome that was recorded */
  outcome: Outcome;

  /** Any new constraints seeded from this outcome */
  newConstraints: Constraint[];

  /** All currently active constraints */
  activeConstraints: Constraint[];

  /** Current radius limit */
  radiusLimit: number;

  /** Convergence state after this outcome */
  convergence: ConvergenceState;

  /** Receipt hash (if receipts enabled) */
  receiptHash?: string;
}

// =============================================================================
// JOURNAL — Structured execution log
// =============================================================================

/**
 * A journal entry records everything that happened for one attempt.
 */
export interface JournalEntry {
  /** Entry type */
  type: 'outcome' | 'constraint_seeded' | 'proposal_blocked' | 'convergence_update';

  /** Timestamp */
  timestamp: number;

  /** Attempt number (1-based) */
  attempt: number;

  /** The data payload */
  data: Outcome | Constraint | ProposalCheck | ConvergenceState;
}
