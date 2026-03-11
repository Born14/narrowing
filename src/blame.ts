/**
 * Blame Classification — Who's at fault?
 *
 * The critical gate between "the agent tried something wrong" and
 * "the infrastructure broke." Infrastructure faults NEVER seed constraints.
 * Without this, the learning system poisons itself.
 *
 * The adapter provides domain-specific blame rules. This module provides
 * the classification engine and universal infrastructure patterns.
 *
 * Extracted from: src/lib/services/memory.ts (classifyFailureKind)
 */

import type { DomainAdapter, FailureKind } from './types.js';

// =============================================================================
// UNIVERSAL INFRASTRUCTURE PATTERNS
// =============================================================================

/**
 * Errors that are ALWAYS infrastructure, regardless of domain.
 * The agent's code didn't cause these — the environment did.
 */
const UNIVERSAL_HARNESS_PATTERNS: RegExp[] = [
  /out of memory|OOM|ENOMEM|killed.*memory|oom.kill/i,
  /segmentation fault|SIGSEGV|core dumped/i,
  /EADDRINUSE|port.*in use|address.*in use/i,
  /permission denied|EACCES/i,
  /disk.*full|no space|ENOSPC/i,
  /ECONNREFUSED|connection refused/i,
];

/**
 * Errors that are ALWAYS the agent's fault, regardless of domain.
 */
const UNIVERSAL_AGENT_PATTERNS: RegExp[] = [
  /SyntaxError|Unexpected token|Parse error|Unterminated string/i,
  /Cannot find module|MODULE_NOT_FOUND|ModuleNotFoundError|ImportError/i,
  /NameError|ReferenceError|TypeError.*undefined/i,
  /IndentationError|TabError/i,
];

// =============================================================================
// CLASSIFICATION ENGINE
// =============================================================================

/**
 * Classify a failure as infrastructure vs agent error.
 *
 * Resolution order:
 * 1. Adapter-specific classification (domain knowledge)
 * 2. Universal infrastructure patterns
 * 3. Universal agent patterns
 * 4. 'unknown' (can't tell)
 *
 * Pure function — deterministic, no side effects.
 */
export function classifyBlame(
  error: string,
  adapter?: DomainAdapter,
  context?: Record<string, unknown>,
): FailureKind {
  if (!error) return 'unknown';

  // Adapter gets first crack — it has domain knowledge
  if (adapter) {
    const adapterBlame = adapter.classifyBlame(error, context);
    if (adapterBlame !== 'unknown') return adapterBlame;
  }

  // Universal infrastructure patterns
  for (const pattern of UNIVERSAL_HARNESS_PATTERNS) {
    if (pattern.test(error)) return 'harness_fault';
  }

  // Universal agent patterns
  for (const pattern of UNIVERSAL_AGENT_PATTERNS) {
    if (pattern.test(error)) return 'agent_failure';
  }

  return 'unknown';
}
