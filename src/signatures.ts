/**
 * Signature Extraction — Deterministic failure classification.
 *
 * Every failure gets a canonical signature via regex matching.
 * The adapter provides domain-specific patterns. This module provides
 * the matching engine and a base set of universal patterns.
 *
 * Priority order: first match wins. Adapter patterns checked first,
 * then universal fallbacks.
 *
 * Extracted from: src/lib/services/memory.ts (extractSignature, SIGNATURE_PATTERNS)
 */

import type { DomainAdapter, SignaturePattern } from './types.js';

// =============================================================================
// UNIVERSAL PATTERNS — Infrastructure failures any domain can hit
// =============================================================================

export const UNIVERSAL_PATTERNS: SignaturePattern[] = [
  {
    pattern: /out of memory|OOM|ENOMEM|killed.*memory|oom.kill/i,
    signature: 'oom_killed',
    typicallyHarness: true,
    description: 'Process killed by OS memory pressure',
  },
  {
    pattern: /segmentation fault|SIGSEGV|core dumped/i,
    signature: 'segfault',
    typicallyHarness: true,
    description: 'Segmentation fault (memory access violation)',
  },
  {
    pattern: /timeout|exceeded.*time|timed?\s*out/i,
    signature: 'timeout',
    typicallyHarness: false,
    description: 'Operation exceeded time limit',
  },
  {
    pattern: /EADDRINUSE|port.*in use|address.*in use/i,
    signature: 'port_conflict',
    typicallyHarness: true,
    description: 'Network port already in use',
  },
  {
    pattern: /ECONNREFUSED|connection refused/i,
    signature: 'connection_refused',
    typicallyHarness: true,
    description: 'Connection refused by remote host',
  },
  {
    pattern: /SyntaxError|Unexpected token|Parse error|Unterminated string/i,
    signature: 'syntax_error',
    typicallyHarness: false,
    description: 'Code syntax error',
  },
  {
    pattern: /Cannot find module|MODULE_NOT_FOUND|ModuleNotFoundError|ImportError/i,
    signature: 'missing_module',
    typicallyHarness: false,
    description: 'Required module/package not found',
  },
  {
    pattern: /permission denied|EACCES/i,
    signature: 'permission_denied',
    typicallyHarness: true,
    description: 'Insufficient permissions',
  },
  {
    pattern: /disk.*full|no space|ENOSPC/i,
    signature: 'disk_full',
    typicallyHarness: true,
    description: 'Disk space exhausted',
  },
];

// =============================================================================
// EXTRACTION ENGINE
// =============================================================================

/**
 * Extract a failure signature from an error string.
 *
 * Resolution order:
 * 1. Adapter-specific patterns (first match wins)
 * 2. Universal patterns (infrastructure failures)
 * 3. undefined (unrecognized)
 *
 * Pure function — deterministic, no side effects.
 */
export function extractSignature(
  error: string,
  adapter?: DomainAdapter,
): string | undefined {
  if (!error) return undefined;

  // Adapter patterns first — domain-specific takes priority
  if (adapter) {
    const adapterSig = adapter.extractSignature(error);
    if (adapterSig) return adapterSig;
  }

  // Universal patterns
  for (const { pattern, signature } of UNIVERSAL_PATTERNS) {
    if (pattern.test(error)) {
      return signature;
    }
  }

  return undefined;
}

/**
 * Get all registered patterns (adapter + universal).
 * Useful for documentation, testing, and introspection.
 */
export function getAllPatterns(adapter?: DomainAdapter): SignaturePattern[] {
  const adapterPatterns = adapter?.signaturePatterns ?? [];
  return [...adapterPatterns, ...UNIVERSAL_PATTERNS];
}
