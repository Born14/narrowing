/**
 * Tamper-Evident Receipt Chain
 *
 * Every narrowing decision produces a receipt — an immutable record
 * with a cryptographic hash chain. Any modification breaks the chain
 * downstream.
 *
 * Format: JSONL. Each receipt: hash = sha256(previousHash + payload).
 * First receipt uses previousHash = 'genesis'.
 *
 * Reuses the pattern from @sovereign-labs/mcp-proxy.
 */

import { createHash } from 'crypto';
import { existsSync, mkdirSync, appendFileSync, readFileSync } from 'fs';
import { dirname } from 'path';

// =============================================================================
// TYPES
// =============================================================================

export interface Receipt {
  /** SHA-256 hash of this receipt */
  hash: string;

  /** Hash of the previous receipt (or 'genesis') */
  previousHash: string;

  /** When this receipt was created */
  timestamp: number;

  /** What happened */
  type: 'outcome_recorded' | 'constraint_seeded' | 'proposal_blocked' | 'proposal_allowed' | 'convergence_escalation';

  /** The payload — deterministically serialized */
  payload: Record<string, unknown>;

  /** Attempt number */
  attempt: number;
}

// =============================================================================
// DETERMINISTIC SERIALIZATION
// =============================================================================

/**
 * Stable JSON stringification with sorted keys.
 * Deterministic — same input always produces same output.
 */
export function stableStringify(obj: unknown): string {
  if (obj === null || obj === undefined) return 'null';
  if (typeof obj !== 'object') return JSON.stringify(obj);
  if (Array.isArray(obj)) {
    return '[' + obj.map(stableStringify).join(',') + ']';
  }
  const sorted = Object.keys(obj as Record<string, unknown>).sort();
  const pairs = sorted.map(k =>
    `${JSON.stringify(k)}:${stableStringify((obj as Record<string, unknown>)[k])}`
  );
  return '{' + pairs.join(',') + '}';
}

/**
 * SHA-256 hash of a string.
 */
export function sha256(input: string): string {
  return createHash('sha256').update(input).digest('hex');
}

// =============================================================================
// RECEIPT CHAIN
// =============================================================================

export class ReceiptChain {
  private readonly path: string;
  private lastHash = 'genesis';
  private count = 0;

  constructor(path: string) {
    this.path = path;
    this.ensureDir();
    this.loadLastHash();
  }

  /**
   * Append a receipt to the chain.
   * Returns the receipt with its computed hash.
   */
  append(type: Receipt['type'], payload: Record<string, unknown>, attempt: number): Receipt {
    const canonical = stableStringify(payload);
    const hash = sha256(this.lastHash + canonical);

    const receipt: Receipt = {
      hash,
      previousHash: this.lastHash,
      timestamp: Date.now(),
      type,
      payload,
      attempt,
    };

    appendFileSync(this.path, JSON.stringify(receipt) + '\n');
    this.lastHash = hash;
    this.count++;

    return receipt;
  }

  /**
   * Verify the entire chain for tampering.
   * Returns { valid: true } or { valid: false, brokenAt: index }.
   */
  verify(): { valid: boolean; brokenAt?: number; receiptCount: number } {
    const receipts = this.readAll();

    let expectedPrev = 'genesis';
    for (let i = 0; i < receipts.length; i++) {
      const r = receipts[i];

      // Check previous hash linkage
      if (r.previousHash !== expectedPrev) {
        return { valid: false, brokenAt: i, receiptCount: receipts.length };
      }

      // Recompute hash
      const canonical = stableStringify(r.payload);
      const expectedHash = sha256(expectedPrev + canonical);

      if (r.hash !== expectedHash) {
        return { valid: false, brokenAt: i, receiptCount: receipts.length };
      }

      expectedPrev = r.hash;
    }

    return { valid: true, receiptCount: receipts.length };
  }

  /** Read all receipts */
  readAll(): Receipt[] {
    if (!existsSync(this.path)) return [];

    const content = readFileSync(this.path, 'utf-8').trim();
    if (!content) return [];

    return content.split('\n')
      .filter(line => line.trim())
      .map(line => {
        try {
          return JSON.parse(line) as Receipt;
        } catch {
          return null;
        }
      })
      .filter((r): r is Receipt => r !== null);
  }

  /** Get the last hash (for external chaining) */
  getLastHash(): string {
    return this.lastHash;
  }

  /** Get the receipt count */
  getCount(): number {
    return this.count;
  }

  // =========================================================================
  // INTERNAL
  // =========================================================================

  private loadLastHash(): void {
    const receipts = this.readAll();
    if (receipts.length > 0) {
      this.lastHash = receipts[receipts.length - 1].hash;
      this.count = receipts.length;
    }
  }

  private ensureDir(): void {
    const dir = dirname(this.path);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
  }
}
