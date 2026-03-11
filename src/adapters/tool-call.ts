/**
 * Tool-Call Adapter — Domain translation for agent tool loops.
 *
 * The universal adapter. Every agent framework (LangChain, CrewAI, AutoGen,
 * Claude Code, Cursor, n8n) calls tools in a loop. This adapter classifies
 * tool calls, extracts failure signatures, and enables narrowing to prevent
 * agents from repeating the same failed tool invocation.
 *
 * The Kilo Code $8 burn: agent read the same file 1,000 times.
 * The VS Code 800GB incident: agent created 1,526 worktrees in a day.
 * The n8n 50% loop rate: agents stuck calling the same tool infinitely.
 *
 * This adapter would have blocked the second call.
 *
 * 12 failure signatures, 7 action classes, source-sensitive blame.
 */

import type { DomainAdapter, FailureKind, SignaturePattern } from '../types.js';

// =============================================================================
// FAILURE SIGNATURES — What went wrong with the tool call
// =============================================================================

const TOOL_CALL_SIGNATURE_PATTERNS: SignaturePattern[] = [
  // Infrastructure faults — not the agent's fault
  {
    pattern: /timeout|ETIMEDOUT|deadline exceeded|timed?\s*out/i,
    signature: 'tool_timeout',
    typicallyHarness: true,
    description: 'Tool call timed out',
  },
  {
    pattern: /tool not found|unknown tool|no such tool|not available|tool.*does not exist/i,
    signature: 'tool_not_found',
    typicallyHarness: true,
    description: 'Tool does not exist or is not registered',
  },
  {
    pattern: /permission denied|EACCES|forbidden|403/i,
    signature: 'permission_denied',
    typicallyHarness: true,
    description: 'Insufficient permissions for tool operation',
  },
  {
    pattern: /429|rate.?limit|too many requests|throttl/i,
    signature: 'rate_limited',
    typicallyHarness: true,
    description: 'Rate limited by external service',
  },

  // Agent faults — the agent proposed something wrong
  {
    pattern: /ENOENT|no such file|file not found|path.*not found|does not exist/i,
    signature: 'file_not_found',
    typicallyHarness: false,
    description: 'Target file or path does not exist',
  },
  {
    pattern: /SyntaxError|parse error|invalid JSON|Unexpected token|JSON\.parse/i,
    signature: 'syntax_error',
    typicallyHarness: false,
    description: 'Syntax or parse error in tool input/output',
  },
  {
    pattern: /search string not found|edit.*application failed|old_string.*not found|no match/i,
    signature: 'edit_failed',
    typicallyHarness: false,
    description: 'Edit/replace operation could not find target string',
  },
  {
    pattern: /exit code [1-9]|non.?zero exit|command failed|exited with/i,
    signature: 'command_failed',
    typicallyHarness: false,
    description: 'Shell command exited with non-zero code',
  },
  {
    pattern: /validation failed|invalid.*argument|invalid.*param|400|bad request/i,
    signature: 'validation_error',
    typicallyHarness: false,
    description: 'Input validation failed',
  },
  {
    pattern: /409|conflict|already exists|duplicate/i,
    signature: 'conflict',
    typicallyHarness: false,
    description: 'Resource conflict or duplicate',
  },
  {
    pattern: /no results|empty response|null|undefined|not found in response/i,
    signature: 'empty_result',
    typicallyHarness: false,
    description: 'Tool returned empty or null result',
  },

  // Ambiguous — could be either
  {
    pattern: /500|502|503|internal server error|service unavailable|bad gateway/i,
    signature: 'api_error',
    typicallyHarness: false,
    description: 'Server-side API error',
  },
];

// =============================================================================
// ACTION CLASSES — What kind of tool call was this
// =============================================================================

/**
 * Keyword sets for classifying tool names into action classes.
 *
 * Tool names use diverse separators: snake_case (read_file), dash-case (api-request),
 * camelCase (editFile), dot notation (file.read). We normalize to space-separated
 * before matching, so \b works naturally.
 */
const ACTION_CLASS_KEYWORDS: Array<{
  keywords: RegExp;
  actionClass: string;
}> = [
  {
    keywords: /\b(delete|remove|rm|drop|destroy|unlink|purge|clean)\b/i,
    actionClass: 'delete',
  },
  {
    keywords: /\b(create|new|touch|mkdir|init|scaffold|generate)\b/i,
    actionClass: 'file_create',
  },
  {
    keywords: /\b(edit|write|patch|replace|update|modify|set|put|save|overwrite|insert|append)\b/i,
    actionClass: 'file_edit',
  },
  {
    keywords: /\b(read|cat|view|head|tail|show|display|open|load)\b/i,
    actionClass: 'file_read',
  },
  {
    keywords: /\b(exec|run|bash|shell|command|spawn|terminal|sh|cmd|eval|execute)\b/i,
    actionClass: 'shell_exec',
  },
  {
    keywords: /\b(search|grep|find|glob|rg|ripgrep|locate|query|lookup)\b/i,
    actionClass: 'search',
  },
  {
    keywords: /\b(api|request|fetch|http|curl|post|call|invoke|webhook|endpoint)\b/i,
    actionClass: 'api_call',
  },
];

/** Normalize tool name separators to spaces for word boundary matching */
function normalizeToolName(name: string): string {
  // snake_case and dot.case → space
  let normalized = name.replace(/[_.\-]/g, ' ');
  // camelCase → space (insertBefore uppercase followed by lowercase)
  normalized = normalized.replace(/([a-z])([A-Z])/g, '$1 $2');
  return normalized;
}

// =============================================================================
// FINGERPRINTING — Structural identity of a tool call
// =============================================================================

/**
 * Create a short fingerprint of tool call arguments.
 * Used for exact-match constraint checking — "same tool, same args = same call."
 *
 * Not a hash of the full args (too brittle). Instead, extracts the structural
 * identity: what file, what pattern, what command.
 */
function fingerprint(params: Record<string, unknown>): string {
  const parts: string[] = [];

  // Tool name
  if (params.tool) parts.push(`t:${params.tool}`);

  // Target (file, endpoint, resource)
  const target = params.target || params.file || params.path ||
    params.file_path || params.url || params.endpoint;
  if (target) parts.push(`@${target}`);

  // Pattern (search/replace fingerprint, command, query)
  const pattern = params.pattern || params.old_string || params.search ||
    params.command || params.query;
  if (typeof pattern === 'string') {
    // Truncate long patterns to first 80 chars for matching
    parts.push(`p:${pattern.slice(0, 80)}`);
  }

  // Method (for API calls)
  if (params.method) parts.push(`m:${params.method}`);

  return parts.join('|') || 'unknown';
}

// =============================================================================
// ADAPTER IMPLEMENTATION
// =============================================================================

/**
 * Create a Tool-Call domain adapter.
 *
 * Usage:
 *   import { createToolCallAdapter } from '@sovereign-labs/narrowing/adapters/tool-call';
 *   const adapter = createToolCallAdapter();
 *   const loop = new NarrowingLoop({ adapter });
 *
 *   // In your agent loop:
 *   const check = loop.checkProposal({
 *     parameters: { tool: 'edit_file', file: 'server.js', old_string: 'foo', new_string: 'bar' },
 *     targets: ['server.js'],
 *   });
 *   if (!check.allowed) {
 *     // Feed back to LLM: "This approach already failed. Try something different."
 *   }
 */
export function createToolCallAdapter(): DomainAdapter {
  return {
    name: 'tool-call',

    extractSignature(error: string): string | undefined {
      for (const { pattern, signature } of TOOL_CALL_SIGNATURE_PATTERNS) {
        if (pattern.test(error)) return signature;
      }
      return undefined;
    },

    classifyBlame(error: string, _context?: Record<string, unknown>): FailureKind {
      // Check each signature pattern for its typical blame
      for (const { pattern, typicallyHarness } of TOOL_CALL_SIGNATURE_PATTERNS) {
        if (pattern.test(error)) {
          if (typicallyHarness) return 'harness_fault';

          // api_error (500/502/503) is ambiguous
          if (/500|502|503|internal server error|service unavailable|bad gateway/i.test(error)) {
            return 'unknown';
          }

          return 'agent_failure';
        }
      }
      return 'unknown';
    },

    classifyAction(params: Record<string, unknown>, _targets: string[]): string | undefined {
      // Primary: classify by tool name (normalized for word boundary matching)
      const rawToolName = String(params.tool || params.tool_name || params.name || '');
      if (rawToolName) {
        const normalized = normalizeToolName(rawToolName);
        for (const { keywords, actionClass } of ACTION_CLASS_KEYWORDS) {
          if (keywords.test(normalized)) return actionClass;
        }
      }

      // Fallback: classify by parameter shape
      if (params.command || params.cmd) return 'shell_exec';
      if (params.old_string || params.search || params.patch) return 'file_edit';
      if (params.content && (params.file || params.file_path || params.path)) return 'file_create';
      if (params.url || params.endpoint) return 'api_call';
      if (params.query || params.pattern || params.glob) return 'search';

      return undefined;
    },

    extractParameters(raw: Record<string, unknown>): Record<string, unknown> {
      const extracted: Record<string, unknown> = {};

      // Tool identity
      const tool = raw.tool || raw.tool_name || raw.name;
      if (tool) extracted.tool = tool;

      // Target
      const target = raw.target || raw.file || raw.path ||
        raw.file_path || raw.url || raw.endpoint;
      if (target) extracted.target = target;

      // Fingerprint for constraint matching
      extracted._fingerprint = fingerprint(raw);

      // Action class hint (if provided)
      if (raw.actionClass) extracted.actionClass = raw.actionClass;

      // Preserve delta markers for action classification
      if (raw['_delta']) extracted['_delta'] = raw['_delta'];
      for (const key of Object.keys(raw)) {
        if (key.startsWith('_prev_')) extracted[key] = raw[key];
      }

      return extracted;
    },

    signaturePatterns: TOOL_CALL_SIGNATURE_PATTERNS,
  };
}

/**
 * Helper: build a proposal from a tool call.
 *
 * Convenience function so users don't have to manually construct
 * the Proposal shape from their framework's tool call format.
 *
 * Usage:
 *   const proposal = toolCallToProposal('edit_file', {
 *     file: 'server.js',
 *     old_string: 'foo',
 *     new_string: 'bar',
 *   });
 *   const check = loop.checkProposal(proposal);
 */
export function toolCallToProposal(
  toolName: string,
  args: Record<string, unknown>,
): { parameters: Record<string, unknown>; targets: string[] } {
  // Extract targets from args
  const targets: string[] = [];
  const targetKeys = ['file', 'file_path', 'path', 'target', 'url', 'endpoint'];
  for (const key of targetKeys) {
    const val = args[key];
    if (typeof val === 'string') {
      targets.push(val);
      break;
    }
  }

  return {
    parameters: { tool: toolName, ...args },
    targets,
  };
}

/**
 * Helper: build an outcome from a tool call result.
 *
 * Convenience function for recording tool call results back into the loop.
 *
 * Usage:
 *   const outcome = toolCallToOutcome('edit_file', args, {
 *     success: false,
 *     error: 'search string not found in file',
 *     durationMs: 45,
 *   });
 *   loop.recordOutcome(outcome);
 */
export function toolCallToOutcome(
  toolName: string,
  args: Record<string, unknown>,
  result: {
    success: boolean;
    error?: string;
    durationMs: number;
    metadata?: Record<string, unknown>;
  },
): {
  score: null;
  status: 'success' | 'failure';
  error?: string;
  parameters: Record<string, unknown>;
  targets: string[];
  durationMs: number;
  metadata?: Record<string, unknown>;
} {
  const { targets, parameters } = toolCallToProposal(toolName, args);

  return {
    score: null,
    status: result.success ? 'success' : 'failure',
    error: result.error,
    parameters,
    targets,
    durationMs: result.durationMs,
    metadata: result.metadata,
  };
}
