/**
 * Constraint Engine — The core of narrowing.
 *
 * Constraints are structural guardrails derived from execution failures.
 * They are enforced by the runtime BEFORE execution — the agent never
 * gets to try what's banned.
 *
 * Three constraint types:
 * - banned_strategy: An approach that failed 2+ times
 * - radius_limit: Progressive cap on change count
 * - parameter_ban: Specific values proven broken
 *
 * The star API: checkProposal() — makes the runtime active, not passive.
 *
 * Extracted from: src/lib/services/memory.ts (seedConstraintFromFailure,
 * checkPlanConstraints, buildRadiusLimit, buildStrategyBan)
 */

import type {
  Constraint,
  Proposal,
  ProposalCheck,
  ConstraintViolation,
  Outcome,
  FailureKind,
  NarrowingConfig,
} from './types.js';

// =============================================================================
// CONSTRAINT STORE — In-memory with persistence hooks
// =============================================================================

export class ConstraintStore {
  private constraints: Constraint[] = [];
  private corroborationCounts = new Map<string, number>();
  private readonly config: Pick<NarrowingConfig, 'corroborationThreshold' | 'radiusCurve' | 'constraintTtlMs' | 'maxConstraintDepth'>;

  constructor(config: Pick<NarrowingConfig, 'corroborationThreshold' | 'radiusCurve' | 'constraintTtlMs' | 'maxConstraintDepth'>) {
    this.config = config;
  }

  // =========================================================================
  // CHECK PROPOSAL — The star API
  // =========================================================================

  /**
   * Check a proposal against all active constraints.
   *
   * This is the primary API — call before every execution attempt.
   * Returns whether the proposal is allowed, and if not, why.
   */
  checkProposal(proposal: Proposal): ProposalCheck {
    const active = this.getActive();
    const violations: ConstraintViolation[] = [];

    for (const constraint of active) {
      const violation = this.checkOne(constraint, proposal);
      if (violation) violations.push(violation);
    }

    const totalStrategies = this.countTotalStrategies();
    const bannedStrategies = active.filter(c => c.type === 'banned_strategy').length;
    const searchSpaceReduction = totalStrategies > 0
      ? Math.round((bannedStrategies / totalStrategies) * 100)
      : 0;

    return {
      allowed: violations.length === 0,
      violations,
      activeConstraints: active.length,
      radiusLimit: this.getCurrentRadiusLimit(),
      searchSpaceReduction,
    };
  }

  // =========================================================================
  // SEED — Learn from failures
  // =========================================================================

  /**
   * Attempt to seed a constraint from a failure outcome.
   *
   * Rules:
   * 1. Infrastructure faults NEVER seed constraints
   * 2. Requires corroboration (2+ occurrences) by default
   * 3. Max constraint depth prevents suffocation
   * 4. Dedup: don't create identical constraints
   *
   * Returns the new constraint, or null if none was seeded.
   */
  seedFromOutcome(outcome: Outcome, sessionId?: string): Constraint | null {
    // Rule 1: Infrastructure faults don't seed
    if (outcome.failureKind === 'harness_fault') return null;

    // Only seed from failures
    if (outcome.status !== 'failure' && outcome.status !== 'error') return null;

    // Max depth check
    const sessionConstraints = sessionId
      ? this.constraints.filter(c => c.scope === 'session')
      : this.constraints;
    if (sessionConstraints.length >= this.config.maxConstraintDepth) return null;

    const signature = outcome.failureSignature || 'unknown';

    // Track corroboration
    const key = `${signature}:${outcome.actionClass || 'none'}`;
    const count = (this.corroborationCounts.get(key) || 0) + 1;
    this.corroborationCounts.set(key, count);

    let constraint: Constraint | null = null;

    // Strategy ban: same action class failed enough times
    if (outcome.actionClass && count >= this.config.corroborationThreshold) {
      constraint = this.buildStrategyBan(outcome, count);
    }

    // Radius limit: progressive shrinking after 2+ attempts
    if (!constraint && count >= 2) {
      constraint = this.buildRadiusLimit(outcome);
    }

    // Parameter ban: specific values that failed
    if (!constraint && count >= this.config.corroborationThreshold) {
      constraint = this.buildParameterBan(outcome, count);
    }

    if (!constraint) return null;

    // Dedup
    if (this.isDuplicate(constraint)) return null;

    // Set session scope if session provided
    if (sessionId) {
      constraint.scope = 'session';
    }

    this.constraints.push(constraint);
    return constraint;
  }

  // =========================================================================
  // LIFECYCLE
  // =========================================================================

  /** Get all non-expired constraints */
  getActive(): Constraint[] {
    const now = Date.now();
    return this.constraints.filter(c =>
      !c.expiresAt || c.expiresAt > now
    );
  }

  /** Get current radius limit (minimum of all active radius constraints) */
  getCurrentRadiusLimit(): number {
    const radiusConstraints = this.getActive().filter(c => c.type === 'radius_limit');
    if (radiusConstraints.length === 0) return Infinity;
    return Math.min(...radiusConstraints.map(c => c.maxChanges ?? Infinity));
  }

  /** Remove all session-scoped constraints */
  clearSession(): void {
    this.constraints = this.constraints.filter(c => c.scope !== 'session');
  }

  /** Remove expired constraints (garbage collection) */
  gc(): number {
    const now = Date.now();
    const before = this.constraints.length;
    this.constraints = this.constraints.filter(c =>
      !c.expiresAt || c.expiresAt > now
    );
    return before - this.constraints.length;
  }

  /** Get all constraints (including expired, for persistence) */
  getAll(): Constraint[] {
    return [...this.constraints];
  }

  /** Load constraints from persisted state */
  load(constraints: Constraint[]): void {
    this.constraints = [...constraints];
    // Rebuild corroboration counts
    this.corroborationCounts.clear();
    for (const c of constraints) {
      const key = `${c.signature}:${c.actionClass || 'none'}`;
      this.corroborationCounts.set(key, c.occurrences);
    }
  }

  /** Get count of all known strategy classes (for search space math) */
  private countTotalStrategies(): number {
    // Collect unique action classes seen in all outcomes tracked via corroboration
    const strategies = new Set<string>();
    for (const [key] of this.corroborationCounts) {
      const actionClass = key.split(':')[1];
      if (actionClass !== 'none') strategies.add(actionClass);
    }
    // Minimum 1 to avoid division by zero
    return Math.max(strategies.size, 1);
  }

  // =========================================================================
  // CONSTRAINT BUILDERS
  // =========================================================================

  private buildStrategyBan(outcome: Outcome, occurrences: number): Constraint {
    return {
      id: `c_strategy_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      type: 'banned_strategy',
      signature: outcome.failureSignature || 'unknown',
      actionClass: outcome.actionClass,
      reason: `Strategy "${outcome.actionClass}" failed ${occurrences} times with signature "${outcome.failureSignature}"`,
      corroborated: occurrences >= this.config.corroborationThreshold,
      occurrences,
      introducedAt: Date.now(),
      expiresAt: this.config.constraintTtlMs > 0
        ? Date.now() + this.config.constraintTtlMs
        : undefined,
      scope: 'session',
      appliesTo: outcome.targets,
    };
  }

  private buildRadiusLimit(outcome: Outcome): Constraint {
    const currentRadius = this.getCurrentRadiusLimit();
    const curve = this.config.radiusCurve;

    // Find current position in the curve
    let curveIndex = curve.findIndex(r => r <= currentRadius);
    if (curveIndex === -1) curveIndex = 0;

    // Move to next tighter radius
    const nextIndex = Math.min(curveIndex + 1, curve.length - 1);
    const newRadius = curve[nextIndex];

    // Don't create if radius wouldn't change
    if (newRadius >= currentRadius) return null as unknown as Constraint;

    return {
      id: `c_radius_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      type: 'radius_limit',
      signature: outcome.failureSignature || 'unknown',
      reason: `Radius shrunk ${currentRadius === Infinity ? '∞' : currentRadius} → ${newRadius} after ${outcome.failureSignature || 'failure'}`,
      corroborated: true,
      occurrences: 1,
      introducedAt: Date.now(),
      expiresAt: this.config.constraintTtlMs > 0
        ? Date.now() + this.config.constraintTtlMs
        : undefined,
      scope: 'session',
      maxChanges: newRadius,
    };
  }

  private buildParameterBan(outcome: Outcome, occurrences: number): Constraint | null {
    // Find parameters that are plausible ban candidates
    // Only ban if we can identify a specific parameter
    const params = outcome.parameters;
    const keys = Object.keys(params);
    if (keys.length === 0) return null;

    // Ban the parameter that changed (heuristic: first key with non-default value)
    const paramKey = keys[0];
    const paramValue = params[paramKey];

    return {
      id: `c_param_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      type: 'parameter_ban',
      signature: outcome.failureSignature || 'unknown',
      reason: `Parameter "${paramKey}=${paramValue}" failed ${occurrences} times`,
      corroborated: occurrences >= this.config.corroborationThreshold,
      occurrences,
      introducedAt: Date.now(),
      expiresAt: this.config.constraintTtlMs > 0
        ? Date.now() + this.config.constraintTtlMs
        : undefined,
      scope: 'session',
      bannedParameter: paramKey,
      bannedValues: [paramValue],
    };
  }

  // =========================================================================
  // CHECKING
  // =========================================================================

  private checkOne(constraint: Constraint, proposal: Proposal): ConstraintViolation | null {
    switch (constraint.type) {
      case 'banned_strategy':
        return this.checkStrategyBan(constraint, proposal);
      case 'radius_limit':
        return this.checkRadiusLimit(constraint, proposal);
      case 'parameter_ban':
        return this.checkParameterBan(constraint, proposal);
      default:
        return null;
    }
  }

  private checkStrategyBan(constraint: Constraint, proposal: Proposal): ConstraintViolation | null {
    if (!constraint.actionClass) return null;

    // Check if proposal's action class matches the banned one
    const proposalClass = proposal.actionClass;
    if (proposalClass && proposalClass === constraint.actionClass) {
      return {
        constraint,
        reason: `Strategy "${constraint.actionClass}" is banned: ${constraint.reason}`,
        banType: 'strategy',
      };
    }

    return null;
  }

  private checkRadiusLimit(constraint: Constraint, proposal: Proposal): ConstraintViolation | null {
    if (constraint.maxChanges === undefined) return null;

    const changeCount = proposal.targets.length;
    if (changeCount > constraint.maxChanges) {
      return {
        constraint,
        reason: `Proposal touches ${changeCount} targets but radius limit is ${constraint.maxChanges}`,
        banType: 'radius',
      };
    }

    return null;
  }

  private checkParameterBan(constraint: Constraint, proposal: Proposal): ConstraintViolation | null {
    if (!constraint.bannedParameter || !constraint.bannedValues) return null;

    const value = proposal.parameters[constraint.bannedParameter];
    if (value === undefined) return null;

    if (constraint.bannedValues.includes(value)) {
      return {
        constraint,
        reason: `Parameter "${constraint.bannedParameter}=${value}" is banned: ${constraint.reason}`,
        banType: 'parameter',
      };
    }

    return null;
  }

  // =========================================================================
  // DEDUP
  // =========================================================================

  private isDuplicate(candidate: Constraint): boolean {
    return this.constraints.some(c =>
      c.type === candidate.type &&
      c.signature === candidate.signature &&
      c.actionClass === candidate.actionClass &&
      c.bannedParameter === candidate.bannedParameter
    );
  }
}
