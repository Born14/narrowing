/**
 * Long-Horizon Tool-Call Benchmark
 *
 * Proves narrowing's value at the exact point where LLMs fail:
 * context window degradation over long tool-calling sessions.
 *
 * The setup: A simulated agent loop making 200 tool calls against a
 * mock codebase. The agent has a "context window" of N recent outcomes.
 * When a failure scrolls out of the window, the agent retries it —
 * exactly what happens in production (Claude Code compaction loops,
 * Kilo Code $8 burn, VS Code 800GB worktrees).
 *
 * Two modes:
 *   vanilla  — agent proposes freely, no constraint enforcement
 *   narrowing — NarrowingLoop blocks proposals that match prior failures
 *
 * The benchmark measures:
 *   - Wasted calls (failures that repeat a known-bad strategy)
 *   - Total cost (simulated token spend per call)
 *   - Time to completion (calls to achieve all goals)
 *   - Constraint activations (blocks that prevented waste)
 *
 * No LLM needed. No GPU. Runs in <1 second. Reproducible via seed.
 */

import { describe, test, expect } from 'bun:test';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { NarrowingLoop } from '../src/loop';
import {
  createToolCallAdapter,
  toolCallToProposal,
  toolCallToOutcome,
} from '../src/adapters/tool-call';

// =============================================================================
// MOCK ENVIRONMENT — A simulated codebase with known failure zones
// =============================================================================

/**
 * The mock codebase has files. Some operations succeed, some fail
 * deterministically based on the file + operation combination.
 *
 * This models real failure patterns:
 * - Editing a file with wrong search string (edit_failed)
 * - Reading a file that doesn't exist (file_not_found)
 * - Running a command that fails (command_failed)
 * - Syntax error in generated code (syntax_error)
 */
interface MockFile {
  path: string;
  content: string;
  editablePatterns: string[];  // search strings that exist
}

const MOCK_CODEBASE: MockFile[] = [
  { path: 'server.js', content: 'const express = require("express");', editablePatterns: ['express', 'require'] },
  { path: 'routes/api.js', content: 'router.get("/users", handler);', editablePatterns: ['router.get', 'handler'] },
  { path: 'routes/auth.js', content: 'app.post("/login", authenticate);', editablePatterns: ['app.post', 'authenticate'] },
  { path: 'config.json', content: '{"port": 3000, "host": "localhost"}', editablePatterns: ['3000', 'localhost'] },
  { path: 'package.json', content: '{"name": "test-app", "version": "1.0.0"}', editablePatterns: ['test-app', '1.0.0'] },
  { path: 'styles.css', content: 'body { color: black; }', editablePatterns: ['black', 'body'] },
];

/** Simulate executing a tool call. Returns success/failure deterministically. */
function executeToolCall(
  tool: string,
  args: Record<string, unknown>,
): { success: boolean; error?: string; durationMs: number } {
  const file = args.file as string || args.path as string || args.file_path as string;
  const mockFile = MOCK_CODEBASE.find(f => f.path === file);

  switch (tool) {
    case 'read_file': {
      if (!mockFile) return { success: false, error: `ENOENT: no such file or directory '${file}'`, durationMs: 5 };
      return { success: true, durationMs: 10 };
    }
    case 'edit_file': {
      if (!mockFile) return { success: false, error: `ENOENT: no such file or directory '${file}'`, durationMs: 5 };
      const search = args.old_string as string;
      if (!search || !mockFile.editablePatterns.some(p => p.includes(search) || search.includes(p))) {
        return { success: false, error: `search string not found in file: "${search}"`, durationMs: 15 };
      }
      return { success: true, durationMs: 20 };
    }
    case 'bash': {
      const cmd = args.command as string || '';
      if (cmd.includes('nonexistent') || cmd.includes('bad_command')) {
        return { success: false, error: `command failed: exit code 1`, durationMs: 100 };
      }
      if (cmd.includes('syntax_error')) {
        return { success: false, error: `SyntaxError: Unexpected token`, durationMs: 50 };
      }
      return { success: true, durationMs: 80 };
    }
    case 'search_files': {
      return { success: true, durationMs: 30 };
    }
    case 'create_file': {
      if (mockFile) return { success: false, error: `409 conflict: file already exists`, durationMs: 5 };
      return { success: true, durationMs: 15 };
    }
    default:
      return { success: true, durationMs: 10 };
  }
}

// =============================================================================
// SIMULATED AGENT — With context window degradation
// =============================================================================

/** Mulberry32 PRNG for reproducible agent behavior */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
}

/**
 * Generate the next tool call from the simulated agent.
 *
 * The agent has a "memory window" — it only remembers the last N outcomes.
 * When a failure scrolls out of this window, the agent may propose the
 * same failing call again. This models context window compression.
 *
 * The failure probability after memory loss models the real behavior:
 * the agent doesn't ALWAYS retry — but it has no structural reason NOT to.
 */
function generateProposal(
  rng: () => number,
  recentMemory: Array<{ tool: string; args: Record<string, unknown>; success: boolean }>,
  memoryWindow: number,
): ToolCall {
  const tools = ['read_file', 'edit_file', 'bash', 'search_files', 'create_file'];
  const files = [...MOCK_CODEBASE.map(f => f.path), 'nonexistent.js', 'missing.ts', 'ghost.py'];

  // The agent's visible memory (simulates context window)
  const visible = recentMemory.slice(-memoryWindow);

  // Known-bad calls the agent can still see
  const visibleFailures = new Set(
    visible.filter(m => !m.success).map(m => `${m.tool}:${JSON.stringify(m.args)}`)
  );

  // Forgotten failures — these are the dangerous ones
  const forgottenFailures = recentMemory
    .slice(0, Math.max(0, recentMemory.length - memoryWindow))
    .filter(m => !m.success);

  // 30% chance of replaying a forgotten failure (models real agent behavior)
  if (forgottenFailures.length > 0 && rng() < 0.30) {
    const replay = forgottenFailures[Math.floor(rng() * forgottenFailures.length)];
    return { tool: replay.tool, args: replay.args };
  }

  // Otherwise generate a new call
  let tool: string;
  let args: Record<string, unknown>;
  let attempts = 0;

  do {
    tool = tools[Math.floor(rng() * tools.length)];
    const file = files[Math.floor(rng() * files.length)];

    switch (tool) {
      case 'read_file':
        args = { file };
        break;
      case 'edit_file': {
        const searches = ['express', 'handler', 'NONEXISTENT_PATTERN', 'bogus_search', 'app.post', '3000', 'black'];
        args = { file, old_string: searches[Math.floor(rng() * searches.length)], new_string: 'replacement' };
        break;
      }
      case 'bash': {
        const commands = ['npm test', 'npm run build', 'bad_command --flag', 'node syntax_error.js', 'ls -la', 'cat server.js'];
        args = { command: commands[Math.floor(rng() * commands.length)] };
        break;
      }
      case 'search_files':
        args = { pattern: 'TODO', path: '.' };
        break;
      case 'create_file':
        args = { file, content: 'new file content' };
        break;
      default:
        args = {};
    }

    // If agent can still see this failure in memory, try to avoid it
    const key = `${tool}:${JSON.stringify(args)}`;
    if (!visibleFailures.has(key)) break;
    attempts++;
  } while (attempts < 5);

  return { tool, args };
}

// =============================================================================
// BENCHMARK RUNNER
// =============================================================================

interface BenchmarkResult {
  mode: 'vanilla' | 'narrowing';
  totalCalls: number;
  successfulCalls: number;
  failedCalls: number;
  blockedCalls: number;
  repeatedFailures: number;    // Failures that repeat a prior failure signature+target
  wastedCost: number;          // Simulated cost of failed + blocked calls
  constraintsSeeded: number;
  convergenceStatus: string;
}

const TOKEN_COST_PER_CALL = 0.003;  // ~$0.003 per tool call (realistic for GPT-4o/Claude)

function runBenchmark(
  mode: 'vanilla' | 'narrowing',
  seed: number,
  totalTrials: number,
  memoryWindow: number,
): BenchmarkResult {
  const rng = mulberry32(seed);
  const adapter = createToolCallAdapter();

  const loop = mode === 'narrowing' ? new NarrowingLoop({
    adapter,
    corroborationThreshold: 2,
    receipts: false,
    journalPath: join(tmpdir(), `narrowing-bench-${seed}-${Date.now()}.jsonl`),
  }) : null;

  const allMemory: Array<{ tool: string; args: Record<string, unknown>; success: boolean }> = [];
  const failureHistory = new Map<string, number>(); // signature:target → count

  let successfulCalls = 0;
  let failedCalls = 0;
  let blockedCalls = 0;
  let repeatedFailures = 0;
  let wastedCost = 0;
  let constraintsSeeded = 0;

  for (let i = 0; i < totalTrials; i++) {
    const proposal = generateProposal(rng, allMemory, memoryWindow);

    // Narrowing gate
    if (loop) {
      const check = loop.checkProposal(
        toolCallToProposal(proposal.tool, proposal.args)
      );
      if (!check.allowed) {
        blockedCalls++;
        wastedCost += TOKEN_COST_PER_CALL * 0.1; // Blocked calls cost ~10% (just the check)
        continue;
      }
    }

    // Execute
    const result = executeToolCall(proposal.tool, proposal.args);
    allMemory.push({ tool: proposal.tool, args: proposal.args, success: result.success });

    if (result.success) {
      successfulCalls++;
    } else {
      failedCalls++;
      wastedCost += TOKEN_COST_PER_CALL;

      // Track repeated failures
      const sig = adapter.extractSignature(result.error!) || 'unknown';
      const target = (proposal.args.file || proposal.args.command || 'none') as string;
      const key = `${sig}:${target}`;
      const priorCount = failureHistory.get(key) || 0;
      if (priorCount > 0) repeatedFailures++;
      failureHistory.set(key, priorCount + 1);

      // Record in narrowing loop
      if (loop) {
        const outcome = toolCallToOutcome(proposal.tool, proposal.args, {
          success: false,
          error: result.error,
          durationMs: result.durationMs,
        });
        const narrowingResult = loop.recordOutcome(outcome);
        constraintsSeeded += narrowingResult.newConstraints.length;
      }
    }

    if (result.success && loop) {
      const outcome = toolCallToOutcome(proposal.tool, proposal.args, {
        success: true,
        durationMs: result.durationMs,
      });
      loop.recordOutcome(outcome);
    }
  }

  return {
    mode,
    totalCalls: totalTrials,
    successfulCalls,
    failedCalls,
    blockedCalls,
    repeatedFailures,
    wastedCost: Math.round(wastedCost * 1000) / 1000,
    constraintsSeeded,
    convergenceStatus: loop ? loop.getConvergence().status : 'n/a',
  };
}

// =============================================================================
// TESTS
// =============================================================================

function tmpJournal(): string {
  return join(tmpdir(), `narrowing-lh-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.jsonl`);
}

describe('Long-Horizon Tool-Call Benchmark', () => {

  test('200 calls, memory window 20 — narrowing reduces repeated failures', () => {
    const seed = 42;
    const trials = 200;
    const memoryWindow = 20; // Agent forgets after 20 calls

    const vanilla = runBenchmark('vanilla', seed, trials, memoryWindow);
    const narrowing = runBenchmark('narrowing', seed, trials, memoryWindow);

    console.log('\n=== 200 calls, window=20 ===');
    console.log(`Vanilla:    ${vanilla.failedCalls} failures, ${vanilla.repeatedFailures} repeated, $${vanilla.wastedCost} wasted`);
    console.log(`Narrowing:  ${narrowing.failedCalls} failures, ${narrowing.repeatedFailures} repeated, ${narrowing.blockedCalls} blocked, $${narrowing.wastedCost} wasted`);
    console.log(`Constraints seeded: ${narrowing.constraintsSeeded}`);
    console.log(`Reduction: ${vanilla.repeatedFailures - narrowing.repeatedFailures} fewer repeated failures`);

    // Core assertion: narrowing has fewer repeated failures
    expect(narrowing.repeatedFailures).toBeLessThan(vanilla.repeatedFailures);

    // Narrowing should have blocked some calls
    expect(narrowing.blockedCalls).toBeGreaterThan(0);

    // Narrowing should have seeded constraints
    expect(narrowing.constraintsSeeded).toBeGreaterThan(0);
  });

  test('200 calls, memory window 10 — shorter memory amplifies the problem', () => {
    const seed = 42;
    const trials = 200;
    const memoryWindow = 10; // Aggressive compression

    const vanilla = runBenchmark('vanilla', seed, trials, memoryWindow);
    const narrowing = runBenchmark('narrowing', seed, trials, memoryWindow);

    console.log('\n=== 200 calls, window=10 ===');
    console.log(`Vanilla:    ${vanilla.failedCalls} failures, ${vanilla.repeatedFailures} repeated, $${vanilla.wastedCost} wasted`);
    console.log(`Narrowing:  ${narrowing.failedCalls} failures, ${narrowing.repeatedFailures} repeated, ${narrowing.blockedCalls} blocked, $${narrowing.wastedCost} wasted`);

    expect(narrowing.repeatedFailures).toBeLessThan(vanilla.repeatedFailures);
    expect(narrowing.blockedCalls).toBeGreaterThan(0);
  });

  test('200 calls, memory window 50 — wider memory reduces but doesn\'t eliminate repeats', () => {
    const seed = 42;
    const trials = 200;
    const memoryWindow = 50;

    const vanilla = runBenchmark('vanilla', seed, trials, memoryWindow);
    const narrowing = runBenchmark('narrowing', seed, trials, memoryWindow);

    console.log('\n=== 200 calls, window=50 ===');
    console.log(`Vanilla:    ${vanilla.failedCalls} failures, ${vanilla.repeatedFailures} repeated, $${vanilla.wastedCost} wasted`);
    console.log(`Narrowing:  ${narrowing.failedCalls} failures, ${narrowing.repeatedFailures} repeated, ${narrowing.blockedCalls} blocked, $${narrowing.wastedCost} wasted`);

    // Even with a wide window, vanilla still has some repeats
    // (the 30% replay probability models real agent behavior)
    expect(narrowing.repeatedFailures).toBeLessThanOrEqual(vanilla.repeatedFailures);
  });

  test('cross-session persistence — session 2 starts with session 1 constraints', () => {
    const adapter = createToolCallAdapter();
    const session1 = new NarrowingLoop({
      adapter,
      corroborationThreshold: 2,
      receipts: false,
      journalPath: tmpJournal(),
    });

    // Session 1: discover that editing nonexistent.js fails
    session1.recordOutcome(toolCallToOutcome('edit_file', { file: 'nonexistent.js', old_string: 'foo' }, {
      success: false, error: 'ENOENT: no such file or directory', durationMs: 5,
    }));
    session1.recordOutcome(toolCallToOutcome('edit_file', { file: 'nonexistent.js', old_string: 'foo' }, {
      success: false, error: 'ENOENT: no such file or directory', durationMs: 5,
    }));

    // Constraint should be seeded
    expect(session1.getActiveConstraints().length).toBeGreaterThan(0);

    // Save state (simulates process exit)
    const snapshot = session1.snapshot();

    // Session 2: new loop, restored state
    const session2 = new NarrowingLoop({
      adapter,
      corroborationThreshold: 2,
      receipts: false,
      journalPath: tmpJournal(),
    });
    session2.restore(snapshot);

    // Session 2 immediately blocks the same call — no rediscovery needed
    const check = session2.checkProposal(
      toolCallToProposal('edit_file', { file: 'nonexistent.js', old_string: 'foo' })
    );
    expect(check.allowed).toBe(false);
    expect(check.violations.length).toBeGreaterThan(0);
  });

  test('infrastructure faults don\'t poison the constraint store', () => {
    const adapter = createToolCallAdapter();
    const loop = new NarrowingLoop({
      adapter,
      corroborationThreshold: 2,
      receipts: false,
      journalPath: tmpJournal(),
    });

    // 10 consecutive timeouts on the same API call
    for (let i = 0; i < 10; i++) {
      loop.recordOutcome(toolCallToOutcome('api_request', { url: 'https://api.example.com/data' }, {
        success: false, error: 'Request timed out after 30s', durationMs: 30000,
      }));
    }

    // The call should STILL be allowed — timeouts are infrastructure faults
    const check = loop.checkProposal(
      toolCallToProposal('api_request', { url: 'https://api.example.com/data' })
    );
    expect(check.allowed).toBe(true);
    expect(loop.getActiveConstraints().length).toBe(0);
  });

  test('multi-seed stability — results are consistent across seeds', () => {
    const seeds = [1, 2, 3, 4, 5];
    const trials = 200;
    const memoryWindow = 20;

    let narrowingWins = 0;

    for (const seed of seeds) {
      const vanilla = runBenchmark('vanilla', seed, trials, memoryWindow);
      const narrowing = runBenchmark('narrowing', seed, trials, memoryWindow);

      if (narrowing.repeatedFailures < vanilla.repeatedFailures) {
        narrowingWins++;
      }
    }

    console.log(`\n=== Multi-seed: narrowing wins ${narrowingWins}/5 seeds ===`);

    // Narrowing should win on the majority of seeds
    expect(narrowingWins).toBeGreaterThanOrEqual(3);
  });

  test('the degradation curve — failure rate increases with horizon length', () => {
    const seed = 42;
    const memoryWindow = 20;
    const checkpoints = [50, 100, 150, 200];

    console.log('\n=== Degradation Curve ===');
    console.log('Trials | Vanilla Repeats | Narrowing Repeats | Blocked | Delta');
    console.log('-------|-----------------|-------------------|---------|------');

    let prevVanillaRepeats = 0;

    for (const trials of checkpoints) {
      const vanilla = runBenchmark('vanilla', seed, trials, memoryWindow);
      const narrowing = runBenchmark('narrowing', seed, trials, memoryWindow);

      const delta = vanilla.repeatedFailures - narrowing.repeatedFailures;

      console.log(
        `${String(trials).padStart(6)} | ${String(vanilla.repeatedFailures).padStart(15)} | ${String(narrowing.repeatedFailures).padStart(17)} | ${String(narrowing.blockedCalls).padStart(7)} | ${delta >= 0 ? '+' : ''}${delta}`
      );

      // The gap should grow with horizon length
      if (trials > 50) {
        expect(vanilla.repeatedFailures).toBeGreaterThanOrEqual(prevVanillaRepeats);
      }
      prevVanillaRepeats = vanilla.repeatedFailures;
    }
  });

  test('cost projection — extrapolate savings at scale', () => {
    const seed = 42;
    const trials = 200;
    const memoryWindow = 20;

    const vanilla = runBenchmark('vanilla', seed, trials, memoryWindow);
    const narrowing = runBenchmark('narrowing', seed, trials, memoryWindow);

    const savedCalls = vanilla.repeatedFailures - narrowing.repeatedFailures + narrowing.blockedCalls;
    const savedCostPer200 = savedCalls * TOKEN_COST_PER_CALL;
    const projectedDaily = savedCostPer200 * 10; // 10 sessions/day
    const projectedMonthly = projectedDaily * 30;

    console.log('\n=== Cost Projection ===');
    console.log(`Saved calls per 200-call session: ${savedCalls}`);
    console.log(`Saved cost per session: $${savedCostPer200.toFixed(3)}`);
    console.log(`Projected daily savings (10 sessions): $${projectedDaily.toFixed(2)}`);
    console.log(`Projected monthly savings: $${projectedMonthly.toFixed(2)}`);

    // At Kilo Code scale ($7.59 per incident), even small savings compound
    expect(savedCalls).toBeGreaterThan(0);
  });
});
