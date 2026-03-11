/**
 * ML Training Adapter — Domain translation for ML training loops.
 *
 * Designed for Karpathy's autoresearch pattern:
 * - Fixed evaluation metric (val_bpb — bits per byte)
 * - Fixed time budget (5 min)
 * - Agent modifies train.py hyperparameters and architecture
 * - Score direction: minimize (lower bpb = better)
 *
 * 13 failure signatures, 8 action classes, source-sensitive blame.
 */

import type { DomainAdapter, FailureKind, SignaturePattern } from '../types.js';

// =============================================================================
// FAILURE SIGNATURES — What went wrong
// =============================================================================

const ML_SIGNATURE_PATTERNS: SignaturePattern[] = [
  // GPU/Memory failures
  {
    pattern: /CUDA out of memory|torch\.cuda\.OutOfMemoryError|RuntimeError:.*out of memory/i,
    signature: 'oom_gpu',
    typicallyHarness: true,
    description: 'GPU out of memory — model too large for available VRAM',
  },
  {
    pattern: /loss.*nan|loss.*inf|gradient.*nan|gradient.*inf|nan.*loss/i,
    signature: 'training_divergence',
    typicallyHarness: false,
    description: 'Training diverged — loss became NaN/Inf',
  },
  {
    pattern: /gradient.*explod|gradient.*overflow|gradient.*clip.*fail/i,
    signature: 'gradient_explosion',
    typicallyHarness: false,
    description: 'Gradient explosion — values too large',
  },
  // Multi-GPU / distributed
  {
    pattern: /NCCL.*error|NCCL.*timeout|ProcessGroupNCCL|all_reduce/i,
    signature: 'nccl_failure',
    typicallyHarness: true,
    description: 'NCCL communication failure (multi-GPU)',
  },
  // Tensor/shape errors
  {
    pattern: /RuntimeError.*size mismatch|shape.*mismatch|tensor.*shape|dimension.*mismatch/i,
    signature: 'tensor_shape_error',
    typicallyHarness: false,
    description: 'Tensor shape mismatch — incompatible dimensions',
  },
  // CUDA/cuBLAS errors
  {
    pattern: /CUBLAS_STATUS|cublas.*error|cublasLt/i,
    signature: 'cublas_error',
    typicallyHarness: true,
    description: 'cuBLAS library error (GPU compute)',
  },
  // Compilation errors (torch.compile, triton)
  {
    pattern: /torch\.compile|triton.*error|compilation.*failed|CompilationError/i,
    signature: 'compile_failure',
    typicallyHarness: false,
    description: 'torch.compile or Triton compilation failed',
  },
  // Python code errors
  {
    pattern: /SyntaxError|IndentationError|TabError/i,
    signature: 'code_syntax_error',
    typicallyHarness: false,
    description: 'Python syntax error in modified code',
  },
  {
    pattern: /ImportError|ModuleNotFoundError|No module named/i,
    signature: 'missing_import',
    typicallyHarness: false,
    description: 'Missing Python import/module',
  },
  {
    pattern: /RuntimeError|ValueError|TypeError|AttributeError|KeyError|NameError/i,
    signature: 'runtime_crash',
    typicallyHarness: false,
    description: 'Python runtime error',
  },
  // OS-level
  {
    pattern: /Killed|signal 9|SIGKILL|oom.kill/i,
    signature: 'oom_killed',
    typicallyHarness: true,
    description: 'Process killed by OS (likely OOM)',
  },
  // Timeout (model too large to train in time budget)
  {
    pattern: /timed?\s*out|timeout|exceeded.*time/i,
    signature: 'timeout',
    typicallyHarness: false,
    description: 'Training exceeded time budget — model too large or slow',
  },
  // Score regression (not an error, but a signal)
  {
    pattern: /score.*worse|score.*regress|bpb.*increased|loss.*increased/i,
    signature: 'score_regression',
    typicallyHarness: false,
    description: 'Score regressed from previous best',
  },
];

// =============================================================================
// ACTION CLASSES — What strategy was attempted
// =============================================================================

/**
 * ML Training strategy taxonomy.
 *
 * These classify the APPROACH, not the specific values.
 * "scale_up_width" bans making the model wider — not a specific n_embd value.
 */
const ACTION_CLASSIFIERS: Array<{
  test: (params: Record<string, unknown>) => boolean;
  actionClass: string;
}> = [
  {
    // Increased model width (n_embd, d_model, hidden_size)
    test: (p) => hasIncreasedKey(p, ['n_embd', 'd_model', 'hidden_size', 'embed_dim', 'width']),
    actionClass: 'scale_up_width',
  },
  {
    // Increased model depth (n_layer, num_layers, depth)
    test: (p) => hasIncreasedKey(p, ['n_layer', 'num_layers', 'depth', 'n_blocks']),
    actionClass: 'scale_up_depth',
  },
  {
    // Decreased model size (any scaling down)
    test: (p) => hasDecreasedKey(p, ['n_embd', 'd_model', 'n_layer', 'num_layers', 'hidden_size']),
    actionClass: 'scale_down',
  },
  {
    // Increased learning rate
    test: (p) => hasIncreasedKey(p, ['lr', 'learning_rate', 'max_lr']),
    actionClass: 'lr_increase',
  },
  {
    // Decreased learning rate
    test: (p) => hasDecreasedKey(p, ['lr', 'learning_rate', 'max_lr']),
    actionClass: 'lr_decrease',
  },
  {
    // Increased batch size
    test: (p) => hasIncreasedKey(p, ['batch_size', 'total_batch_size', 'micro_batch']),
    actionClass: 'batch_size_increase',
  },
  {
    // Changed architecture type
    test: (p) => hasChangedKey(p, ['architecture', 'attention_type', 'activation', 'norm_type']),
    actionClass: 'architecture_swap',
  },
  {
    // Changed optimizer
    test: (p) => hasChangedKey(p, ['optimizer', 'optimizer_type']),
    actionClass: 'optimizer_change',
  },
];

// =============================================================================
// PARAMETER CHANGE DETECTION
// =============================================================================

/**
 * Track parameter changes between consecutive attempts.
 * The adapter needs to know which direction parameters moved.
 */
interface ParameterDelta {
  key: string;
  from: unknown;
  to: unknown;
  direction: 'increased' | 'decreased' | 'changed';
}

/** Check if any of the named keys increased (numeric comparison) */
function hasIncreasedKey(params: Record<string, unknown>, keys: string[]): boolean {
  for (const key of keys) {
    const val = params[key];
    const prev = params[`_prev_${key}`];
    if (typeof val === 'number' && typeof prev === 'number' && val > prev) {
      return true;
    }
    // Also check _delta markers
    const delta = params[`_delta`];
    if (delta && typeof delta === 'object' && (delta as Record<string, string>)[key] === 'increased') {
      return true;
    }
  }
  return false;
}

/** Check if any of the named keys decreased (numeric comparison) */
function hasDecreasedKey(params: Record<string, unknown>, keys: string[]): boolean {
  for (const key of keys) {
    const val = params[key];
    const prev = params[`_prev_${key}`];
    if (typeof val === 'number' && typeof prev === 'number' && val < prev) {
      return true;
    }
    const delta = params[`_delta`];
    if (delta && typeof delta === 'object' && (delta as Record<string, string>)[key] === 'decreased') {
      return true;
    }
  }
  return false;
}

/** Check if any of the named keys changed (any value) */
function hasChangedKey(params: Record<string, unknown>, keys: string[]): boolean {
  for (const key of keys) {
    const prev = params[`_prev_${key}`];
    if (prev !== undefined && prev !== params[key]) {
      return true;
    }
    const delta = params[`_delta`];
    if (delta && typeof delta === 'object' && (delta as Record<string, string>)[key] === 'changed') {
      return true;
    }
  }
  return false;
}

// =============================================================================
// ADAPTER IMPLEMENTATION
// =============================================================================

/**
 * Create an ML Training domain adapter.
 *
 * Usage:
 *   import { createMLTrainingAdapter } from '@sovereign-labs/narrowing/adapters/ml-training';
 *   const adapter = createMLTrainingAdapter();
 *   const loop = new NarrowingLoop({ adapter, direction: 'minimize' });
 */
export function createMLTrainingAdapter(): DomainAdapter {
  return {
    name: 'ml-training',

    extractSignature(error: string): string | undefined {
      for (const { pattern, signature } of ML_SIGNATURE_PATTERNS) {
        if (pattern.test(error)) return signature;
      }
      return undefined;
    },

    classifyBlame(error: string, context?: Record<string, unknown>): FailureKind {
      const e = error.toLowerCase();

      // GPU OOM — the agent chose dimensions too large for available VRAM.
      // This is an agent decision error, not infrastructure. OS-level OOM
      // (Killed/SIGKILL) is handled separately as harness_fault below.
      if (/cuda out of memory|torch\.cuda\.outofmemoryerror/i.test(error)) {
        return 'agent_failure';
      }

      // NCCL failures are always infrastructure (must be before timeout
      // because "NCCL timeout" contains "timeout")
      if (/nccl.*error|nccl.*timeout/i.test(error)) {
        return 'harness_fault';
      }

      // Timeout — the agent chose a model too large to train in time budget
      if (/timed?\s*out|timeout|exceeded.*time/i.test(error)) {
        return 'agent_failure';
      }

      // OS-level kills are infrastructure
      if (/killed|signal 9|sigkill|oom.kill/i.test(error)) {
        return 'harness_fault';
      }

      // cuBLAS errors are infrastructure
      if (/cublas_status|cublas.*error/i.test(error)) {
        return 'harness_fault';
      }

      // Syntax/import/runtime errors are the agent's fault
      if (/syntaxerror|indentationerror|taberror/i.test(error)) {
        return 'agent_failure';
      }
      if (/importerror|modulenotfounderror|no module named/i.test(error)) {
        return 'agent_failure';
      }
      if (/runtimeerror|valueerror|typeerror|attributeerror|keyerror|nameerror/i.test(error)) {
        // Runtime errors COULD be infrastructure (shape mismatch from hardware)
        // but are usually the agent's code
        return 'agent_failure';
      }

      // Training divergence is agent's fault (bad hyperparameters)
      if (/loss.*nan|loss.*inf|gradient.*nan/i.test(error)) {
        return 'agent_failure';
      }

      // Compile failures: could be either
      if (/torch\.compile|triton.*error/i.test(error)) {
        return 'unknown';
      }

      return 'unknown';
    },

    classifyAction(params: Record<string, unknown>, _targets: string[]): string | undefined {
      for (const { test, actionClass } of ACTION_CLASSIFIERS) {
        if (test(params)) return actionClass;
      }
      return undefined;
    },

    extractParameters(raw: Record<string, unknown>): Record<string, unknown> {
      // Extract known ML hyperparameter keys
      const KNOWN_KEYS = [
        'n_embd', 'd_model', 'hidden_size', 'embed_dim', 'width',
        'n_layer', 'num_layers', 'depth', 'n_blocks',
        'n_head', 'num_heads', 'head_dim',
        'lr', 'learning_rate', 'max_lr', 'min_lr',
        'batch_size', 'total_batch_size', 'micro_batch',
        'warmup_steps', 'warmdown_frac',
        'weight_decay', 'dropout',
        'optimizer', 'optimizer_type',
        'architecture', 'attention_type', 'activation', 'norm_type',
        'aspect_ratio', 'sequence_length', 'max_seq_len',
      ];

      const extracted: Record<string, unknown> = {};
      for (const key of KNOWN_KEYS) {
        if (raw[key] !== undefined) extracted[key] = raw[key];
      }

      // Preserve delta markers for action classification
      if (raw['_delta']) extracted['_delta'] = raw['_delta'];
      for (const key of Object.keys(raw)) {
        if (key.startsWith('_prev_')) extracted[key] = raw[key];
      }

      return extracted;
    },

    signaturePatterns: ML_SIGNATURE_PATTERNS,
  };
}

/**
 * Helper: compute parameter deltas between two parameter sets.
 * Returns delta markers that enable action class detection.
 */
export function computeParameterDeltas(
  prev: Record<string, unknown>,
  next: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = { ...next };
  const delta: Record<string, string> = {};

  for (const key of Object.keys(next)) {
    if (key.startsWith('_')) continue;
    const prevVal = prev[key];
    const nextVal = next[key];

    if (prevVal !== undefined && prevVal !== nextVal) {
      result[`_prev_${key}`] = prevVal;
      if (typeof prevVal === 'number' && typeof nextVal === 'number') {
        delta[key] = nextVal > prevVal ? 'increased' : 'decreased';
      } else {
        delta[key] = 'changed';
      }
    }
  }

  if (Object.keys(delta).length > 0) {
    result['_delta'] = delta;
  }

  return result;
}
