/**
 * Execution Journal — Structured, append-only execution log.
 *
 * Every attempt, constraint, and convergence update is recorded.
 * The journal is the forensic record — it answers "why did the agent
 * end up here?" long after the session ends.
 *
 * Format: JSONL (one JSON object per line). Append-only. Crash-safe
 * via sync writes.
 */

import { existsSync, mkdirSync, appendFileSync, readFileSync } from 'fs';
import { dirname } from 'path';
import type { JournalEntry, Outcome, Constraint, ProposalCheck, ConvergenceState } from './types.js';

// =============================================================================
// JOURNAL
// =============================================================================

export class Journal {
  private readonly path: string;
  private attempt = 0;

  constructor(path: string) {
    this.path = path;
    this.ensureDir();
  }

  /** Record an outcome */
  recordOutcome(outcome: Outcome): void {
    this.attempt++;
    this.append({
      type: 'outcome',
      timestamp: Date.now(),
      attempt: this.attempt,
      data: outcome,
    });
  }

  /** Record a constraint being seeded */
  recordConstraint(constraint: Constraint): void {
    this.append({
      type: 'constraint_seeded',
      timestamp: Date.now(),
      attempt: this.attempt,
      data: constraint,
    });
  }

  /** Record a proposal being blocked */
  recordBlocked(check: ProposalCheck): void {
    this.append({
      type: 'proposal_blocked',
      timestamp: Date.now(),
      attempt: this.attempt,
      data: check,
    });
  }

  /** Record convergence state update */
  recordConvergence(state: ConvergenceState): void {
    this.append({
      type: 'convergence_update',
      timestamp: Date.now(),
      attempt: this.attempt,
      data: state,
    });
  }

  /** Read all journal entries */
  readAll(): JournalEntry[] {
    if (!existsSync(this.path)) return [];

    const content = readFileSync(this.path, 'utf-8').trim();
    if (!content) return [];

    return content.split('\n')
      .filter(line => line.trim())
      .map(line => {
        try {
          return JSON.parse(line) as JournalEntry;
        } catch {
          return null;
        }
      })
      .filter((e): e is JournalEntry => e !== null);
  }

  /** Get current attempt number */
  getAttempt(): number {
    return this.attempt;
  }

  /** Load attempt counter from existing journal */
  resume(): void {
    const entries = this.readAll();
    const outcomes = entries.filter(e => e.type === 'outcome');
    this.attempt = outcomes.length;
  }

  // =========================================================================
  // INTERNAL
  // =========================================================================

  private append(entry: JournalEntry): void {
    appendFileSync(this.path, JSON.stringify(entry) + '\n');
  }

  private ensureDir(): void {
    const dir = dirname(this.path);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
  }
}
