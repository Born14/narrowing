/**
 * Convergence Tracking — Is the search making progress?
 *
 * Three terminal states:
 * - plateau: No score improvement for N consecutive attempts
 * - exhausted: Budget exceeded or max constraints reached
 * - constrained_out: All known strategies are banned
 *
 * The runtime uses convergence state to decide when to escalate to human.
 *
 * Extracted from: packages/kernel/src/kernel/convergence.ts
 */

import type { ConvergenceState, Outcome, NarrowingConfig, ScoreDirection } from './types.js';
import type { ConstraintStore } from './constraints.js';

// =============================================================================
// CONVERGENCE TRACKER
// =============================================================================

export class ConvergenceTracker {
  private state: ConvergenceState;
  private readonly config: Pick<NarrowingConfig, 'plateauWindow' | 'plateauTolerance' | 'direction' | 'maxConstraintDepth'>;

  constructor(config: Pick<NarrowingConfig, 'plateauWindow' | 'plateauTolerance' | 'direction' | 'maxConstraintDepth'>) {
    this.config = config;
    this.state = {
      status: 'progressing',
      totalAttempts: 0,
      bestScore: null,
      bestAttempt: 0,
      noImprovementStreak: 0,
      searchSpaceRemaining: 1.0,
      activeConstraintCount: 0,
      bannedStrategyCount: 0,
    };
  }

  /**
   * Update convergence state after an outcome.
   */
  update(outcome: Outcome, constraints: ConstraintStore): ConvergenceState {
    this.state.totalAttempts++;

    const active = constraints.getActive();
    this.state.activeConstraintCount = active.length;
    this.state.bannedStrategyCount = active.filter(c => c.type === 'banned_strategy').length;

    // Update search space remaining
    const totalStrategies = Math.max(
      this.state.bannedStrategyCount + 1, // at least 1 unbanned
      active.length + 1
    );
    this.state.searchSpaceRemaining = Math.max(
      0,
      1 - (this.state.bannedStrategyCount / totalStrategies)
    );

    // Score tracking
    if (outcome.score !== null) {
      const improved = this.isImprovement(outcome.score);

      if (improved) {
        this.state.bestScore = outcome.score;
        this.state.bestAttempt = this.state.totalAttempts;
        this.state.noImprovementStreak = 0;
      } else {
        this.state.noImprovementStreak++;
      }
    } else {
      // No score (error/crash) — counts as no improvement
      this.state.noImprovementStreak++;
    }

    // Classify status
    this.state.status = this.classifyStatus();

    return { ...this.state };
  }

  /**
   * Get current convergence state (immutable copy).
   */
  getState(): ConvergenceState {
    return { ...this.state };
  }

  /**
   * Check if the convergence process should escalate.
   */
  shouldEscalate(): { escalate: boolean; reason: string } {
    switch (this.state.status) {
      case 'plateau':
        return {
          escalate: true,
          reason: `No improvement for ${this.state.noImprovementStreak} consecutive attempts (plateau window: ${this.config.plateauWindow})`,
        };
      case 'exhausted':
        return {
          escalate: true,
          reason: `${this.state.activeConstraintCount} active constraints — constraint depth exceeded (max: ${this.config.maxConstraintDepth})`,
        };
      case 'constrained_out':
        return {
          escalate: true,
          reason: 'All known strategies are banned — search space exhausted',
        };
      case 'progressing':
        return { escalate: false, reason: 'Search is progressing' };
      default:
        return { escalate: false, reason: 'Unknown status' };
    }
  }

  /**
   * Load state from persisted data.
   */
  load(state: ConvergenceState): void {
    this.state = { ...state };
  }

  // =========================================================================
  // INTERNAL
  // =========================================================================

  private isImprovement(score: number): boolean {
    if (this.state.bestScore === null) return true;

    const delta = this.config.direction === 'minimize'
      ? this.state.bestScore - score
      : score - this.state.bestScore;

    return delta > this.config.plateauTolerance;
  }

  private classifyStatus(): ConvergenceState['status'] {
    // Constrained out: search space below 10%
    if (this.state.searchSpaceRemaining < 0.1) {
      return 'constrained_out';
    }

    // Exhausted: too many constraints
    if (this.state.activeConstraintCount >= this.config.maxConstraintDepth) {
      return 'exhausted';
    }

    // Plateau: no improvement streak exceeds window
    if (this.state.noImprovementStreak >= this.config.plateauWindow) {
      return 'plateau';
    }

    return 'progressing';
  }
}
