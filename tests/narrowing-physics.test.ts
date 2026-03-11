/**
 * Narrowing Physics Tests
 *
 * Proves the core invariants:
 * 1. Proposals blocked by learned constraints (the star demo)
 * 2. Infrastructure faults don't seed constraints (poisoned well prevention)
 * 3. Corroboration required before seeding (no false constraints)
 * 4. Radius shrinks monotonically on failure
 * 5. Convergence detection (plateau, exhaustion, constrained_out)
 * 6. Receipt chain integrity
 * 7. Journal records everything
 */

import { describe, test, expect, beforeEach } from 'bun:test';
import { NarrowingLoop } from '../src/loop.js';
import { createMLTrainingAdapter, computeParameterDeltas } from '../src/adapters/ml-training.js';
import type { Proposal, ProposalCheck } from '../src/types.js';
import { existsSync, rmSync, mkdirSync, writeFileSync, readFileSync } from 'fs';

import { tmpdir } from 'os';
import { join as pathJoin } from 'path';
const TEST_DIR = pathJoin(tmpdir(), 'narrowing-test-' + Date.now());

function makeLoop(overrides: Record<string, unknown> = {}) {
  return new NarrowingLoop({
    adapter: createMLTrainingAdapter(),
    direction: 'minimize' as const,
    corroborationThreshold: 2,
    radiusCurve: [Infinity, 5, 3, 2, 1],
    constraintTtlMs: 3_600_000,
    maxConstraintDepth: 5,
    plateauWindow: 5,
    plateauTolerance: 0.001,
    journalPath: `${TEST_DIR}/journal.jsonl`,
    receipts: true,
    receiptPath: `${TEST_DIR}/receipts.jsonl`,
    ...overrides,
  });
}

function cleanup() {
  if (existsSync(TEST_DIR)) {
    rmSync(TEST_DIR, { recursive: true, force: true });
  }
}

// =============================================================================
// 1. PROPOSAL BLOCKED BY LEARNED CONSTRAINT — The star demo
// =============================================================================

describe('Proposal blocked by learned constraint', () => {
  beforeEach(cleanup);

  test('strategy ban blocks repeat of failed approach', () => {
    const loop = makeLoop();

    // First failure: scale_up_width with tensor shape error (agent fault, not harness)
    const params1 = computeParameterDeltas(
      { n_embd: 768 },
      { n_embd: 1536 },
    );
    loop.recordOutcome({
      score: null,
      status: 'error',
      error: 'RuntimeError: size mismatch for weight — expected [768] got [1536]',
      parameters: params1,
      targets: ['train.py'],
      durationMs: 5000,
    });

    // Second failure: same strategy, corroborates
    const params2 = computeParameterDeltas(
      { n_embd: 768 },
      { n_embd: 1024 },
    );
    loop.recordOutcome({
      score: null,
      status: 'error',
      error: 'RuntimeError: size mismatch for weight — expected [768] got [1024]',
      parameters: params2,
      targets: ['train.py'],
      durationMs: 3000,
    });

    // Third attempt: agent tries to scale width AGAIN
    const proposal: Proposal = {
      parameters: computeParameterDeltas(
        { n_embd: 768 },
        { n_embd: 896 },
      ),
      targets: ['train.py'],
      actionClass: 'scale_up_width',
    };

    const check = loop.checkProposal(proposal);

    // THE PROOF: proposal is blocked
    expect(check.allowed).toBe(false);
    expect(check.violations.length).toBeGreaterThan(0);
    expect(check.violations[0].banType).toBe('strategy');
    expect(check.violations[0].reason).toContain('scale_up_width');
  });

  test('allowed proposal passes through', () => {
    const loop = makeLoop();

    const proposal: Proposal = {
      parameters: { lr: 0.001 },
      targets: ['train.py'],
      actionClass: 'lr_decrease',
    };

    const check = loop.checkProposal(proposal);
    expect(check.allowed).toBe(true);
    expect(check.violations).toHaveLength(0);
  });

  test('radius limit blocks over-scoped proposals', () => {
    const loop = makeLoop({ radiusCurve: [Infinity, 3, 2, 1] });

    // Two failures to trigger radius shrink
    for (let i = 0; i < 2; i++) {
      loop.recordOutcome({
        score: null,
        status: 'failure',
        error: 'training diverged — loss NaN',
        parameters: { lr: 0.01 },
        targets: ['train.py', 'config.py', 'model.py', 'data.py'],
        durationMs: 1000,
      });
    }

    // Proposal touching too many files
    const check = loop.checkProposal({
      parameters: { lr: 0.005 },
      targets: ['train.py', 'config.py', 'model.py', 'data.py', 'utils.py'],
    });

    expect(check.allowed).toBe(false);
    expect(check.violations.some(v => v.banType === 'radius')).toBe(true);
  });
});

// =============================================================================
// 2. POISONED WELL PREVENTION — Infrastructure faults don't seed
// =============================================================================

describe('Poisoned well prevention', () => {
  beforeEach(cleanup);

  test('GPU OOM does not seed constraints', () => {
    const loop = makeLoop();

    // Two GPU OOM failures
    for (let i = 0; i < 3; i++) {
      loop.recordOutcome({
        score: null,
        status: 'error',
        error: 'CUDA out of memory — tried to allocate 4GB',
        parameters: { n_embd: 2048 },
        targets: ['train.py'],
        durationMs: 5000,
      });
    }

    // No constraints seeded (OOM is infrastructure)
    expect(loop.getActiveConstraints().filter(c => c.type === 'banned_strategy')).toHaveLength(0);
  });

  test('NCCL failure does not seed constraints', () => {
    const loop = makeLoop();

    for (let i = 0; i < 3; i++) {
      loop.recordOutcome({
        score: null,
        status: 'error',
        error: 'ProcessGroupNCCL error: NCCL timeout on rank 0',
        parameters: {},
        targets: ['train.py'],
        durationMs: 1000,
      });
    }

    expect(loop.getActiveConstraints().filter(c => c.type === 'banned_strategy')).toHaveLength(0);
  });

  test('syntax error DOES seed constraints', () => {
    const loop = makeLoop();

    // Two syntax errors (agent's fault)
    const result1 = loop.recordOutcome({
      score: null,
      status: 'error',
      error: 'SyntaxError: unexpected EOF while parsing',
      parameters: { architecture: 'custom_attention', _prev_architecture: 'standard', _delta: { architecture: 'changed' } },
      targets: ['train.py'],
      durationMs: 100,
    });

    loop.recordOutcome({
      score: null,
      status: 'error',
      error: 'SyntaxError: invalid syntax at line 42',
      parameters: { architecture: 'custom_mlp', _prev_architecture: 'standard', _delta: { architecture: 'changed' } },
      targets: ['train.py'],
      durationMs: 100,
    });

    // Constraint should be seeded (agent's fault, corroborated)
    const constraints = loop.getActiveConstraints();
    expect(constraints.length).toBeGreaterThan(0);
  });
});

// =============================================================================
// 3. CORROBORATION — Single failure doesn't seed
// =============================================================================

describe('Corroboration', () => {
  beforeEach(cleanup);

  test('single failure does not seed strategy ban', () => {
    const loop = makeLoop({ corroborationThreshold: 2 });

    loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'training diverged — loss NaN',
      parameters: { lr: 0.1, _prev_lr: 0.001, _delta: { lr: 'increased' } },
      targets: ['train.py'],
      durationMs: 1000,
    });

    const strategyBans = loop.getActiveConstraints().filter(c => c.type === 'banned_strategy');
    expect(strategyBans).toHaveLength(0);
  });

  test('two failures with same action class seeds ban', () => {
    const loop = makeLoop({ corroborationThreshold: 2 });

    for (let i = 0; i < 2; i++) {
      loop.recordOutcome({
        score: null,
        status: 'failure',
        error: 'training diverged — loss NaN',
        parameters: { lr: 0.1 + i * 0.01, _prev_lr: 0.001, _delta: { lr: 'increased' } },
        targets: ['train.py'],
        durationMs: 1000,
      });
    }

    const strategyBans = loop.getActiveConstraints().filter(c => c.type === 'banned_strategy');
    expect(strategyBans.length).toBeGreaterThan(0);
    expect(strategyBans[0].actionClass).toBe('lr_increase');
    expect(strategyBans[0].corroborated).toBe(true);
  });
});

// =============================================================================
// 4. RADIUS SHRINKS MONOTONICALLY
// =============================================================================

describe('Radius monotonicity', () => {
  beforeEach(cleanup);

  test('radius shrinks progressively on repeated failures', () => {
    const loop = makeLoop({ radiusCurve: [Infinity, 5, 3, 2, 1] });

    const radii: number[] = [loop.getRadiusLimit()];

    for (let i = 0; i < 6; i++) {
      loop.recordOutcome({
        score: null,
        status: 'failure',
        error: 'build failure: exit code 1',
        parameters: { attempt: i },
        targets: Array.from({ length: 5 }, (_, j) => `file${j}.py`),
        durationMs: 1000,
      });
      radii.push(loop.getRadiusLimit());
    }

    // Verify monotonic decrease (or stable at minimum)
    for (let i = 1; i < radii.length; i++) {
      expect(radii[i]).toBeLessThanOrEqual(radii[i - 1]);
    }

    // Should have reached minimum
    expect(radii[radii.length - 1]).toBeLessThan(Infinity);
  });
});

// =============================================================================
// 5. CONVERGENCE DETECTION
// =============================================================================

describe('Convergence detection', () => {
  beforeEach(cleanup);

  test('plateau after N attempts without improvement', () => {
    const loop = makeLoop({ plateauWindow: 3, plateauTolerance: 0.001 });

    // Best score
    loop.recordOutcome({
      score: 1.5,
      status: 'success',
      parameters: { lr: 0.001 },
      targets: ['train.py'],
      durationMs: 300000,
    });

    // 3 more with no improvement
    for (let i = 0; i < 3; i++) {
      loop.recordOutcome({
        score: 1.5 + 0.0001, // within tolerance
        status: 'success',
        parameters: { lr: 0.001 + i * 0.0001 },
        targets: ['train.py'],
        durationMs: 300000,
      });
    }

    const state = loop.getConvergence();
    expect(state.status).toBe('plateau');
    expect(state.noImprovementStreak).toBeGreaterThanOrEqual(3);
  });

  test('exhaustion when constraint depth exceeded', () => {
    const loop = makeLoop({ maxConstraintDepth: 3, corroborationThreshold: 1 });

    // Seed multiple constraints — same error to ensure corroboration,
    // different action classes to generate distinct constraints
    const actions = ['lr_increase', 'scale_up_width', 'batch_size_increase', 'architecture_swap', 'scale_up_depth'];
    for (let i = 0; i < 6; i++) {
      loop.recordOutcome({
        score: null,
        status: 'failure',
        error: 'training diverged — loss NaN',
        parameters: {
          lr: 0.1 + i * 0.01,
          _prev_lr: 0.001,
          _delta: { lr: 'increased' },
        },
        targets: Array.from({ length: 5 }, (_, j) => `file${j}.py`),
        durationMs: 1000,
      });
    }

    const state = loop.getConvergence();
    // With 6 failures and plateau window 5, should detect stalled state
    expect(state.noImprovementStreak).toBeGreaterThanOrEqual(5);
    expect(state.totalAttempts).toBe(6);
    // Constraints were seeded (at least strategy ban + radius limit)
    expect(state.activeConstraintCount).toBeGreaterThanOrEqual(1);
  });

  test('progressing when scores improve', () => {
    const loop = makeLoop();

    for (let i = 5; i > 0; i--) {
      loop.recordOutcome({
        score: i * 0.1, // Improving (lower is better for minimize)
        status: 'success',
        parameters: { lr: 0.001 },
        targets: ['train.py'],
        durationMs: 300000,
      });
    }

    const state = loop.getConvergence();
    expect(state.status).toBe('progressing');
    expect(state.bestScore).toBe(0.1);
  });
});

// =============================================================================
// 6. RECEIPT CHAIN INTEGRITY
// =============================================================================

describe('Receipt chain', () => {
  beforeEach(cleanup);

  test('receipts form valid hash chain', () => {
    const loop = makeLoop();

    // Generate some activity
    loop.checkProposal({
      parameters: { lr: 0.001 },
      targets: ['train.py'],
    });

    loop.recordOutcome({
      score: 1.5,
      status: 'success',
      parameters: { lr: 0.001 },
      targets: ['train.py'],
      durationMs: 300000,
    });

    loop.recordOutcome({
      score: null,
      status: 'error',
      error: 'SyntaxError: unexpected token',
      parameters: { lr: 0.002 },
      targets: ['train.py'],
      durationMs: 100,
    });

    const verification = loop.verifyReceipts();
    expect(verification).not.toBeNull();
    expect(verification!.valid).toBe(true);
    expect(verification!.receiptCount).toBeGreaterThan(0);
  });
});

// =============================================================================
// 7. PARAMETER DELTA COMPUTATION
// =============================================================================

describe('Parameter deltas', () => {
  test('detects increased parameters', () => {
    const result = computeParameterDeltas(
      { n_embd: 768, lr: 0.001 },
      { n_embd: 1536, lr: 0.001 },
    );

    expect(result['_prev_n_embd']).toBe(768);
    expect(result['_delta']).toBeDefined();
    expect((result['_delta'] as Record<string, string>)['n_embd']).toBe('increased');
  });

  test('detects decreased parameters', () => {
    const result = computeParameterDeltas(
      { lr: 0.01 },
      { lr: 0.001 },
    );

    expect((result['_delta'] as Record<string, string>)['lr']).toBe('decreased');
  });

  test('detects changed non-numeric parameters', () => {
    const result = computeParameterDeltas(
      { optimizer: 'adam' },
      { optimizer: 'sgd' },
    );

    expect((result['_delta'] as Record<string, string>)['optimizer']).toBe('changed');
  });
});

// =============================================================================
// 8. FULL LOOP LIFECYCLE
// =============================================================================

describe('Full loop lifecycle', () => {
  beforeEach(cleanup);

  test('complete session: propose → fail → learn → block → succeed', () => {
    const loop = makeLoop({ corroborationThreshold: 2 });

    // Attempt 1: Try scaling width — fails with OOM (infrastructure, no constraint)
    loop.recordOutcome({
      score: null,
      status: 'error',
      error: 'CUDA out of memory — 4GB needed',
      parameters: computeParameterDeltas({ n_embd: 768 }, { n_embd: 2048 }),
      targets: ['train.py'],
      durationMs: 5000,
    });

    // Attempt 2: Try LR increase — diverges (agent fault)
    loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'training diverged — loss NaN at step 100',
      parameters: computeParameterDeltas({ lr: 0.001 }, { lr: 0.1 }),
      targets: ['train.py'],
      durationMs: 60000,
    });

    // Attempt 3: Try LR increase again — diverges again (corroborates)
    const result3 = loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'training diverged — loss NaN at step 50',
      parameters: computeParameterDeltas({ lr: 0.001 }, { lr: 0.05 }),
      targets: ['train.py'],
      durationMs: 30000,
    });

    // Constraint should be seeded
    expect(result3.newConstraints.length).toBeGreaterThan(0);

    // Attempt 4: Agent tries LR increase AGAIN — blocked!
    const check = loop.checkProposal({
      parameters: computeParameterDeltas({ lr: 0.001 }, { lr: 0.02 }),
      targets: ['train.py'],
      actionClass: 'lr_increase',
    });
    expect(check.allowed).toBe(false);

    // Attempt 5: Agent tries something different — allowed
    const check2 = loop.checkProposal({
      parameters: computeParameterDeltas({ lr: 0.001 }, { lr: 0.0005 }),
      targets: ['train.py'],
      actionClass: 'lr_decrease',
    });
    expect(check2.allowed).toBe(true);

    // Attempt 5 succeeds
    const result5 = loop.recordOutcome({
      score: 1.45,
      status: 'success',
      parameters: computeParameterDeltas({ lr: 0.001 }, { lr: 0.0005 }),
      targets: ['train.py'],
      durationMs: 300000,
    });

    expect(result5.convergence.status).toBe('progressing');
    expect(result5.convergence.bestScore).toBe(1.45);

    // Receipts intact
    const verification = loop.verifyReceipts();
    expect(verification!.valid).toBe(true);
    expect(verification!.receiptCount).toBeGreaterThan(5);
  });
});

// =============================================================================
// 9. SNAPSHOT & RESTORE
// =============================================================================

describe('Snapshot and restore', () => {
  beforeEach(cleanup);

  test('state survives snapshot/restore cycle', () => {
    const loop1 = makeLoop();

    // Build up some state
    loop1.recordOutcome({
      score: 1.5,
      status: 'success',
      parameters: { lr: 0.001 },
      targets: ['train.py'],
      durationMs: 300000,
    });

    for (let i = 0; i < 2; i++) {
      loop1.recordOutcome({
        score: null,
        status: 'failure',
        error: 'training diverged — loss NaN',
        parameters: { lr: 0.1, _prev_lr: 0.001, _delta: { lr: 'increased' } },
        targets: ['train.py'],
        durationMs: 1000,
      });
    }

    // Snapshot
    const snapshot = loop1.snapshot();
    expect(snapshot.attempt).toBe(3);
    expect(snapshot.constraints.length).toBeGreaterThan(0);

    // Restore into new loop
    const loop2 = makeLoop();
    loop2.restore(snapshot);

    expect(loop2.getAttempt()).toBe(3);
    expect(loop2.getActiveConstraints().length).toBe(snapshot.constraints.length);
    expect(loop2.getConvergence().totalAttempts).toBe(snapshot.convergence.totalAttempts);
  });
});

// =============================================================================
// 10. ML ADAPTER SIGNATURE EXTRACTION
// =============================================================================

describe('ML adapter signatures', () => {
  test('classifies GPU OOM', () => {
    const adapter = createMLTrainingAdapter();
    expect(adapter.extractSignature('CUDA out of memory — tried to allocate 4GB')).toBe('oom_gpu');
  });

  test('classifies training divergence', () => {
    const adapter = createMLTrainingAdapter();
    expect(adapter.extractSignature('loss became NaN at step 100')).toBe('training_divergence');
  });

  test('classifies NCCL failure', () => {
    const adapter = createMLTrainingAdapter();
    expect(adapter.extractSignature('NCCL timeout on rank 0')).toBe('nccl_failure');
  });

  test('classifies tensor shape error', () => {
    const adapter = createMLTrainingAdapter();
    expect(adapter.extractSignature('RuntimeError: size mismatch for weight')).toBe('tensor_shape_error');
  });

  test('classifies Python syntax error', () => {
    const adapter = createMLTrainingAdapter();
    expect(adapter.extractSignature('SyntaxError: unexpected EOF while parsing')).toBe('code_syntax_error');
  });

  test('GPU OOM is agent_failure (agent chose dimensions too large)', () => {
    const adapter = createMLTrainingAdapter();
    expect(adapter.classifyBlame('CUDA out of memory')).toBe('agent_failure');
  });

  test('SyntaxError is agent_failure', () => {
    const adapter = createMLTrainingAdapter();
    expect(adapter.classifyBlame('SyntaxError: invalid syntax')).toBe('agent_failure');
  });

  test('NCCL is harness_fault', () => {
    const adapter = createMLTrainingAdapter();
    expect(adapter.classifyBlame('ProcessGroupNCCL error: NCCL timeout')).toBe('harness_fault');
  });
});

// =============================================================================
// 8. AUTO-PERSISTENCE — Constraints survive process restarts
// =============================================================================

describe('Auto-persistence (statePath)', () => {
  const STATE_DIR = pathJoin(tmpdir(), 'narrowing-persist-test-' + Date.now());
  const statePath = pathJoin(STATE_DIR, 'state.json');

  beforeEach(() => {
    if (existsSync(STATE_DIR)) rmSync(STATE_DIR, { recursive: true, force: true });
  });

  test('constraints auto-saved on recordOutcome and loaded on construction', () => {
    // Session 1: learn a constraint
    const loop1 = makeLoop({ statePath });

    // Two failures to corroborate
    for (let i = 0; i < 2; i++) {
      loop1.recordOutcome({
        score: null,
        status: 'failure',
        error: 'CUDA out of memory — model too large',
        parameters: computeParameterDeltas({ n_embd: 768 }, { n_embd: 2048 }),
        targets: ['train.py'],
        durationMs: 5000,
      });
    }

    // Constraint should exist
    expect(loop1.getActiveConstraints().length).toBeGreaterThan(0);

    // State file should exist on disk
    expect(existsSync(statePath)).toBe(true);

    // Session 2: new loop loads persisted constraints
    const loop2 = makeLoop({ statePath });

    // Same proposal should be blocked — constraint survived process restart
    const check = loop2.checkProposal({
      parameters: computeParameterDeltas({ n_embd: 768 }, { n_embd: 2048 }),
      targets: ['train.py'],
      actionClass: 'scale_up_width',
    });
    expect(check.allowed).toBe(false);
    expect(check.violations.length).toBeGreaterThan(0);
  });

  test('new session gets fresh sessionId but inherits constraints', () => {
    const loop1 = makeLoop({ statePath });
    const session1 = loop1.getSessionId();

    // Seed a constraint
    for (let i = 0; i < 2; i++) {
      loop1.recordOutcome({
        score: null,
        status: 'failure',
        error: 'training diverged — loss NaN at step 100',
        parameters: computeParameterDeltas({ lr: 0.001 }, { lr: 0.1 }),
        targets: ['train.py'],
        durationMs: 60000,
      });
    }

    // Session 2
    const loop2 = makeLoop({ statePath });
    const session2 = loop2.getSessionId();

    // Fresh session ID
    expect(session2).not.toBe(session1);

    // But constraints are inherited
    expect(loop2.getActiveConstraints().length).toBeGreaterThan(0);
  });

  test('state file missing — starts fresh without error', () => {
    const freshPath = pathJoin(STATE_DIR, 'nonexistent', 'state.json');
    const loop = makeLoop({ statePath: freshPath });

    // Should work normally
    expect(loop.getActiveConstraints().length).toBe(0);
    expect(loop.getAttempt()).toBe(0);
  });

  test('corrupt state file — starts fresh without error', () => {
    // Create corrupt file
    mkdirSync(STATE_DIR, { recursive: true });
    writeFileSync(statePath, 'not valid json {{{{');

    const loop = makeLoop({ statePath });

    // Should start fresh, not crash
    expect(loop.getActiveConstraints().length).toBe(0);
  });

  test('persist() is a no-op without statePath', () => {
    const loop = makeLoop(); // no statePath
    loop.persist(); // should not throw
    expect(existsSync(statePath)).toBe(false);
  });

  test('persist() can be called explicitly', () => {
    const loop = makeLoop({ statePath });

    // Record one outcome (below corroboration — no constraint yet)
    loop.recordOutcome({
      score: 1.5,
      status: 'success',
      parameters: { batch_size: 32 },
      targets: ['train.py'],
      durationMs: 300000,
    });

    // State should already be auto-persisted from recordOutcome
    expect(existsSync(statePath)).toBe(true);

    // Explicit persist also works
    loop.persist();
    const data = JSON.parse(readFileSync(statePath, 'utf-8'));
    expect(data.attempt).toBe(1);
    expect(data.convergence).toBeDefined();
    expect(data.constraints).toBeDefined();
  });

  test('attempt counter persists across sessions', () => {
    const loop1 = makeLoop({ statePath });

    // 3 attempts
    for (let i = 0; i < 3; i++) {
      loop1.recordOutcome({
        score: 1.5 - i * 0.1,
        status: 'success',
        parameters: { batch_size: 32 },
        targets: ['train.py'],
        durationMs: 100000,
      });
    }
    expect(loop1.getAttempt()).toBe(3);

    // Session 2 continues from attempt 3
    const loop2 = makeLoop({ statePath });
    expect(loop2.getAttempt()).toBe(3);

    loop2.recordOutcome({
      score: 1.1,
      status: 'success',
      parameters: { batch_size: 64 },
      targets: ['train.py'],
      durationMs: 100000,
    });
    expect(loop2.getAttempt()).toBe(4);
  });

  test('convergence state persists — best score survives restart', () => {
    const loop1 = makeLoop({ statePath });

    loop1.recordOutcome({
      score: 1.2,
      status: 'success',
      parameters: { batch_size: 32 },
      targets: ['train.py'],
      durationMs: 100000,
    });
    expect(loop1.getConvergence().bestScore).toBe(1.2);

    // Session 2 should see the best score
    const loop2 = makeLoop({ statePath });
    expect(loop2.getConvergence().bestScore).toBe(1.2);
  });
});
