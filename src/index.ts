/**
 * @sovereign-labs/narrowing
 *
 * Constraint-learning runtime for iterative agent loops.
 * Agents that structurally learn what NOT to try again.
 *
 * The search space monotonically shrinks through execution-conditioned
 * constraint accumulation. Infrastructure faults are isolated from
 * learning signal. Every decision is receipted in a tamper-evident chain.
 *
 * Usage:
 *   import { NarrowingLoop } from '@sovereign-labs/narrowing';
 *   import { createMLTrainingAdapter } from '@sovereign-labs/narrowing/adapters/ml-training';
 *
 *   const loop = new NarrowingLoop({
 *     adapter: createMLTrainingAdapter(),
 *     direction: 'minimize',
 *   });
 *
 *   while (!loop.isDone()) {
 *     const proposal = agent.next();
 *     const check = loop.checkProposal(proposal);
 *     if (!check.allowed) { agent.reject(check.violations); continue; }
 *     const result = agent.execute(proposal);
 *     loop.recordOutcome(result);
 *   }
 */

// Core
export { NarrowingLoop } from './loop.js';
export { ConstraintStore } from './constraints.js';
export { ConvergenceTracker } from './convergence.js';

// Primitives
export { extractSignature, getAllPatterns, UNIVERSAL_PATTERNS } from './signatures.js';
export { classifyBlame } from './blame.js';

// Persistence
export { Journal } from './journal.js';
export { ReceiptChain, sha256, stableStringify } from './receipts.js';

// Types
export type {
  // Core
  Outcome,
  Constraint,
  Proposal,
  ProposalCheck,
  ConstraintViolation,
  NarrowingResult,
  NarrowingConfig,
  ConvergenceState,
  ScoreDirection,
  FailureKind,

  // Adapter
  DomainAdapter,
  SignaturePattern,

  // Journal
  JournalEntry,
} from './types.js';
