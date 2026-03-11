/**
 * Tool-Call Adapter Tests
 *
 * Proves the universal adapter correctly classifies tool calls,
 * extracts failure signatures, assigns blame, and enables narrowing
 * to prevent agents from repeating failed tool invocations.
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

/** Create a temp journal path for integration tests */
function tmpJournal(): string {
  return join(tmpdir(), `narrowing-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.jsonl`);
}

// =============================================================================
// SIGNATURE EXTRACTION
// =============================================================================

describe('Tool-Call Adapter — Signature Extraction', () => {
  const adapter = createToolCallAdapter();

  test('extracts tool_timeout from timeout errors', () => {
    expect(adapter.extractSignature('Request timed out after 30s')).toBe('tool_timeout');
    expect(adapter.extractSignature('ETIMEDOUT: connection timed out')).toBe('tool_timeout');
    expect(adapter.extractSignature('deadline exceeded')).toBe('tool_timeout');
  });

  test('extracts tool_not_found from missing tool errors', () => {
    expect(adapter.extractSignature('tool not found: edit_filee')).toBe('tool_not_found');
    expect(adapter.extractSignature('Unknown tool: foobar')).toBe('tool_not_found');
    expect(adapter.extractSignature('Tool "xyz" does not exist')).toBe('tool_not_found');
  });

  test('extracts permission_denied from access errors', () => {
    expect(adapter.extractSignature('Error: EACCES: permission denied')).toBe('permission_denied');
    expect(adapter.extractSignature('403 Forbidden')).toBe('permission_denied');
  });

  test('extracts rate_limited from throttle errors', () => {
    expect(adapter.extractSignature('429 Too Many Requests')).toBe('rate_limited');
    expect(adapter.extractSignature('Rate limit exceeded, retry after 60s')).toBe('rate_limited');
  });

  test('extracts file_not_found from ENOENT errors', () => {
    expect(adapter.extractSignature('ENOENT: no such file or directory')).toBe('file_not_found');
    expect(adapter.extractSignature('File not found: config.json')).toBe('file_not_found');
    expect(adapter.extractSignature('Path /tmp/foo does not exist')).toBe('file_not_found');
  });

  test('extracts syntax_error from parse errors', () => {
    expect(adapter.extractSignature('SyntaxError: Unexpected token }')).toBe('syntax_error');
    expect(adapter.extractSignature('JSON.parse: invalid JSON')).toBe('syntax_error');
  });

  test('extracts edit_failed from search/replace errors', () => {
    expect(adapter.extractSignature('search string not found in file')).toBe('edit_failed');
    expect(adapter.extractSignature('Edit application failed: no match')).toBe('edit_failed');
  });

  test('extracts command_failed from non-zero exit', () => {
    expect(adapter.extractSignature('Process exited with exit code 1')).toBe('command_failed');
    expect(adapter.extractSignature('Command failed: non-zero exit')).toBe('command_failed');
  });

  test('extracts validation_error from 400 errors', () => {
    expect(adapter.extractSignature('400 Bad Request: validation failed')).toBe('validation_error');
    expect(adapter.extractSignature('Invalid argument: "count" must be positive')).toBe('validation_error');
  });

  test('extracts conflict from 409 errors', () => {
    expect(adapter.extractSignature('409 Conflict: resource already exists')).toBe('conflict');
    expect(adapter.extractSignature('Duplicate entry for key "email"')).toBe('conflict');
  });

  test('extracts empty_result from null responses', () => {
    expect(adapter.extractSignature('No results found for query')).toBe('empty_result');
    expect(adapter.extractSignature('Response was null')).toBe('empty_result');
  });

  test('extracts api_error from server errors', () => {
    expect(adapter.extractSignature('500 Internal Server Error')).toBe('api_error');
    expect(adapter.extractSignature('502 Bad Gateway')).toBe('api_error');
    expect(adapter.extractSignature('503 Service Unavailable')).toBe('api_error');
  });

  test('returns undefined for unrecognized errors', () => {
    expect(adapter.extractSignature('something completely unknown happened')).toBeUndefined();
  });
});

// =============================================================================
// BLAME CLASSIFICATION
// =============================================================================

describe('Tool-Call Adapter — Blame Classification', () => {
  const adapter = createToolCallAdapter();

  test('classifies infrastructure faults correctly', () => {
    expect(adapter.classifyBlame('Request timed out')).toBe('harness_fault');
    expect(adapter.classifyBlame('Unknown tool: xyz')).toBe('harness_fault');
    expect(adapter.classifyBlame('EACCES: permission denied')).toBe('harness_fault');
    expect(adapter.classifyBlame('429 rate limit exceeded')).toBe('harness_fault');
  });

  test('classifies agent failures correctly', () => {
    expect(adapter.classifyBlame('ENOENT: no such file')).toBe('agent_failure');
    expect(adapter.classifyBlame('SyntaxError: Unexpected token')).toBe('agent_failure');
    expect(adapter.classifyBlame('search string not found')).toBe('agent_failure');
    expect(adapter.classifyBlame('exit code 1')).toBe('agent_failure');
    expect(adapter.classifyBlame('validation failed')).toBe('agent_failure');
    expect(adapter.classifyBlame('409 conflict')).toBe('agent_failure');
    expect(adapter.classifyBlame('no results found')).toBe('agent_failure');
  });

  test('classifies api_error as unknown (ambiguous)', () => {
    expect(adapter.classifyBlame('500 Internal Server Error')).toBe('unknown');
    expect(adapter.classifyBlame('502 Bad Gateway')).toBe('unknown');
  });

  test('classifies unrecognized errors as unknown', () => {
    expect(adapter.classifyBlame('something weird happened')).toBe('unknown');
  });
});

// =============================================================================
// ACTION CLASSIFICATION
// =============================================================================

describe('Tool-Call Adapter — Action Classification', () => {
  const adapter = createToolCallAdapter();

  test('classifies file operations by tool name', () => {
    expect(adapter.classifyAction({ tool: 'read_file' }, [])).toBe('file_read');
    expect(adapter.classifyAction({ tool: 'cat' }, [])).toBe('file_read');
    expect(adapter.classifyAction({ tool: 'view_file' }, [])).toBe('file_read');

    expect(adapter.classifyAction({ tool: 'edit_file' }, [])).toBe('file_edit');
    expect(adapter.classifyAction({ tool: 'write_file' }, [])).toBe('file_edit');
    expect(adapter.classifyAction({ tool: 'patch' }, [])).toBe('file_edit');
    expect(adapter.classifyAction({ tool: 'replace_in_file' }, [])).toBe('file_edit');

    expect(adapter.classifyAction({ tool: 'create_file' }, [])).toBe('file_create');
    expect(adapter.classifyAction({ tool: 'touch' }, [])).toBe('file_create');
    expect(adapter.classifyAction({ tool: 'mkdir' }, [])).toBe('file_create');
  });

  test('classifies shell execution by tool name', () => {
    expect(adapter.classifyAction({ tool: 'execute_command' }, [])).toBe('shell_exec');
    expect(adapter.classifyAction({ tool: 'run_bash' }, [])).toBe('shell_exec');
    expect(adapter.classifyAction({ tool: 'shell' }, [])).toBe('shell_exec');
    expect(adapter.classifyAction({ tool: 'bash' }, [])).toBe('shell_exec');
  });

  test('classifies search operations by tool name', () => {
    expect(adapter.classifyAction({ tool: 'search_files' }, [])).toBe('search');
    expect(adapter.classifyAction({ tool: 'grep' }, [])).toBe('search');
    expect(adapter.classifyAction({ tool: 'find' }, [])).toBe('search');
    expect(adapter.classifyAction({ tool: 'glob' }, [])).toBe('search');
  });

  test('classifies API calls by tool name', () => {
    expect(adapter.classifyAction({ tool: 'api_request' }, [])).toBe('api_call');
    expect(adapter.classifyAction({ tool: 'http_fetch' }, [])).toBe('api_call');
    expect(adapter.classifyAction({ tool: 'curl' }, [])).toBe('api_call');
  });

  test('classifies delete operations by tool name', () => {
    expect(adapter.classifyAction({ tool: 'delete_file' }, [])).toBe('delete');
    expect(adapter.classifyAction({ tool: 'remove' }, [])).toBe('delete');
    expect(adapter.classifyAction({ tool: 'rm' }, [])).toBe('delete');
  });

  test('falls back to parameter shape when tool name is unclear', () => {
    expect(adapter.classifyAction({ tool: 'do_thing', command: 'ls -la' }, [])).toBe('shell_exec');
    expect(adapter.classifyAction({ tool: 'do_thing', old_string: 'foo' }, [])).toBe('file_edit');
    expect(adapter.classifyAction({ tool: 'do_thing', url: 'https://api.example.com' }, [])).toBe('api_call');
    expect(adapter.classifyAction({ tool: 'do_thing', query: 'search term' }, [])).toBe('search');
    expect(adapter.classifyAction({ tool: 'do_thing', content: 'hello', file: 'test.txt' }, [])).toBe('file_create');
  });

  test('returns undefined for unclassifiable tool calls', () => {
    expect(adapter.classifyAction({ tool: 'mysterious_thing' }, [])).toBeUndefined();
    expect(adapter.classifyAction({}, [])).toBeUndefined();
  });
});

// =============================================================================
// PARAMETER EXTRACTION
// =============================================================================

describe('Tool-Call Adapter — Parameter Extraction', () => {
  const adapter = createToolCallAdapter();

  test('extracts tool name and target', () => {
    const extracted = adapter.extractParameters({
      tool: 'edit_file',
      file: 'server.js',
      old_string: 'foo',
      new_string: 'bar',
      irrelevant: 'data',
    });

    expect(extracted.tool).toBe('edit_file');
    expect(extracted.target).toBe('server.js');
    expect(extracted._fingerprint).toContain('t:edit_file');
    expect(extracted._fingerprint).toContain('@server.js');
  });

  test('extracts target from various key names', () => {
    expect(adapter.extractParameters({ file_path: '/tmp/x.ts' }).target).toBe('/tmp/x.ts');
    expect(adapter.extractParameters({ path: '/api/users' }).target).toBe('/api/users');
    expect(adapter.extractParameters({ url: 'https://example.com' }).target).toBe('https://example.com');
    expect(adapter.extractParameters({ endpoint: '/health' }).target).toBe('/health');
  });

  test('fingerprint includes pattern for edit operations', () => {
    const extracted = adapter.extractParameters({
      tool: 'edit_file',
      file: 'server.js',
      old_string: 'const x = 1;',
    });

    expect(extracted._fingerprint).toContain('p:const x = 1;');
  });

  test('fingerprint truncates long patterns', () => {
    const longPattern = 'a'.repeat(200);
    const extracted = adapter.extractParameters({
      tool: 'edit_file',
      old_string: longPattern,
    });

    // Fingerprint should contain truncated version
    const fp = extracted._fingerprint as string;
    expect(fp.length).toBeLessThan(200);
  });
});

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

describe('Tool-Call Adapter — Helpers', () => {
  test('toolCallToProposal builds correct proposal shape', () => {
    const proposal = toolCallToProposal('edit_file', {
      file: 'server.js',
      old_string: 'foo',
      new_string: 'bar',
    });

    expect(proposal.parameters.tool).toBe('edit_file');
    expect(proposal.parameters.file).toBe('server.js');
    expect(proposal.targets).toEqual(['server.js']);
  });

  test('toolCallToProposal extracts target from various keys', () => {
    expect(toolCallToProposal('read', { file_path: '/tmp/x' }).targets).toEqual(['/tmp/x']);
    expect(toolCallToProposal('fetch', { url: 'https://a.com' }).targets).toEqual(['https://a.com']);
    expect(toolCallToProposal('check', { endpoint: '/health' }).targets).toEqual(['/health']);
  });

  test('toolCallToProposal handles missing target gracefully', () => {
    const proposal = toolCallToProposal('think', { thought: 'hmm' });
    expect(proposal.targets).toEqual([]);
  });

  test('toolCallToOutcome builds correct outcome shape', () => {
    const outcome = toolCallToOutcome('edit_file', { file: 'server.js' }, {
      success: false,
      error: 'search string not found',
      durationMs: 45,
    });

    expect(outcome.status).toBe('failure');
    expect(outcome.error).toBe('search string not found');
    expect(outcome.parameters.tool).toBe('edit_file');
    expect(outcome.targets).toEqual(['server.js']);
    expect(outcome.score).toBeNull();
  });

  test('toolCallToOutcome handles success', () => {
    const outcome = toolCallToOutcome('read_file', { file: 'index.ts' }, {
      success: true,
      durationMs: 12,
    });

    expect(outcome.status).toBe('success');
    expect(outcome.error).toBeUndefined();
  });
});

// =============================================================================
// INTEGRATION — Full loop with tool-call adapter
// =============================================================================

describe('Tool-Call Adapter — Full Loop Integration', () => {
  test('blocks repeated edit failures after corroboration', () => {
    const loop = new NarrowingLoop({
      adapter: createToolCallAdapter(),
      corroborationThreshold: 2,
      receipts: false,
      journalPath: tmpJournal(),
    });

    // First edit fails
    loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'search string not found in file',
      parameters: { tool: 'edit_file', file: 'server.js', old_string: 'foo' },
      targets: ['server.js'],
      durationMs: 30,
    });

    // Second edit fails with same signature — constraint seeded
    const result = loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'search string not found in file',
      parameters: { tool: 'edit_file', file: 'server.js', old_string: 'foo' },
      targets: ['server.js'],
      durationMs: 25,
    });

    expect(result.newConstraints.length).toBeGreaterThan(0);

    // Third attempt with same action class — blocked
    const check = loop.checkProposal(
      toolCallToProposal('edit_file', { file: 'server.js', old_string: 'foo' }),
    );

    expect(check.allowed).toBe(false);
    expect(check.violations.length).toBeGreaterThan(0);
    expect(check.violations[0].banType).toBe('strategy');
  });

  test('does not block on infrastructure faults', () => {
    const loop = new NarrowingLoop({
      adapter: createToolCallAdapter(),
      corroborationThreshold: 2,
      receipts: false,
      journalPath: tmpJournal(),
    });

    // Timeout — harness fault, should not seed constraint
    loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'Request timed out after 30s',
      parameters: { tool: 'api_request', url: 'https://api.example.com' },
      targets: ['https://api.example.com'],
      durationMs: 30000,
    });

    loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'Request timed out after 30s',
      parameters: { tool: 'api_request', url: 'https://api.example.com' },
      targets: ['https://api.example.com'],
      durationMs: 30000,
    });

    // Should still be allowed — timeouts are infrastructure, not agent mistakes
    const check = loop.checkProposal(
      toolCallToProposal('api_request', { url: 'https://api.example.com' }),
    );

    expect(check.allowed).toBe(true);
  });

  test('the Kilo Code scenario — blocks repeated file reads', () => {
    const loop = new NarrowingLoop({
      adapter: createToolCallAdapter(),
      corroborationThreshold: 2,
      receipts: false,
      journalPath: tmpJournal(),
    });

    // Agent reads a file, gets empty result
    loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'No results found for query',
      parameters: { tool: 'read_file', file: 'data.json' },
      targets: ['data.json'],
      durationMs: 5,
    });

    // Reads again — same failure
    loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'No results found for query',
      parameters: { tool: 'read_file', file: 'data.json' },
      targets: ['data.json'],
      durationMs: 5,
    });

    // Third read — BLOCKED. In vanilla, this would repeat 1,000 times.
    const check = loop.checkProposal(
      toolCallToProposal('read_file', { file: 'data.json' }),
    );

    expect(check.allowed).toBe(false);
    // Total cost saved: $7.59 in tokens that would have been burned
  });

  test('allows different tool calls after blocking one', () => {
    const loop = new NarrowingLoop({
      adapter: createToolCallAdapter(),
      corroborationThreshold: 2,
      receipts: false,
      journalPath: tmpJournal(),
    });

    // edit_file fails twice — constraint seeded for file_edit strategy
    loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'search string not found',
      parameters: { tool: 'edit_file', file: 'server.js', old_string: 'missing' },
      targets: ['server.js'],
      durationMs: 10,
    });
    loop.recordOutcome({
      score: null,
      status: 'failure',
      error: 'search string not found',
      parameters: { tool: 'edit_file', file: 'server.js', old_string: 'missing' },
      targets: ['server.js'],
      durationMs: 10,
    });

    // edit_file blocked
    const editCheck = loop.checkProposal(
      toolCallToProposal('edit_file', { file: 'server.js', old_string: 'missing' }),
    );
    expect(editCheck.allowed).toBe(false);

    // But read_file on the same file is fine — different action class
    const readCheck = loop.checkProposal(
      toolCallToProposal('read_file', { file: 'server.js' }),
    );
    expect(readCheck.allowed).toBe(true);

    // And shell_exec is fine — different strategy entirely
    const shellCheck = loop.checkProposal(
      toolCallToProposal('bash', { command: 'cat server.js' }),
    );
    expect(shellCheck.allowed).toBe(true);
  });

  test('convergence tracks tool call attempts', () => {
    const loop = new NarrowingLoop({
      adapter: createToolCallAdapter(),
      receipts: false,
      journalPath: tmpJournal(),
    });

    // Record some successes
    for (let i = 0; i < 3; i++) {
      loop.recordOutcome({
        score: null,
        status: 'success',
        parameters: { tool: 'read_file', file: `file${i}.ts` },
        targets: [`file${i}.ts`],
        durationMs: 10,
      });
    }

    const state = loop.getConvergence();
    expect(state.totalAttempts).toBe(3);
    expect(state.status).toBe('progressing');
  });
});
