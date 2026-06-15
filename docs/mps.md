# Experimental Apple MPSGraph Backend

This branch adds an experimental Apple MPSGraph backend for macOS builds tagged with `mps`.

The current goal is correctness and integration with existing Gorgonia graph APIs. It is not yet a production-quality backend because tensors do not stay resident on GPU across op boundaries.

## Build Tags and Runtime

Use the backend with:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go test -tags mps .
```

The `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH` environment variable is required by the current `go4.org/unsafe/assume-no-moving-gc` dependency when testing with Go 1.26.

The backend is active only on builds with `-tags mps`. Non-MPS builds use the existing CPU/CUDA/OpenCL code paths and compile the MPS bridge as stubs.

## Architecture

The backend consists of three layers:

- `internal/mpsbridge`: cgo and Objective-C bridge to Metal/MPSGraph.
- MPS-aware op implementations: selected Gorgonia ops implement `MPSDoer`.
- VM integration: the tape VM detects `MPSDoer` ops and routes them through the MPS execution path.

Current execution is intentionally conservative:

- Inputs are ordinary CPU-accessible `tensor.Dense` values.
- The bridge copies input slices into `MTLBuffer` values.
- MPSGraph executes the operation.
- Outputs are read back into CPU-accessible `tensor.Dense` values.

This makes correctness testing straightforward, but it does not yet provide a GPU-resident tensor pipeline.

## Supported Operations

| Area | Supported | Current Scope |
| --- | --- | --- |
| Runtime | Metal device discovery | Default system Metal device |
| Buffers | Float32 buffer round-trip | Shared storage mode |
| Matrix multiply | `Mul` / `linAlgBinOp` | 2D, `float32`, non-transposed |
| Elementwise | `Add`, `Sub`, `HadamardProd` | Same-shaped `float32` tensors |
| Broadcast bias | `BroadcastAdd(matrix, bias, nil, []byte{0})` | Row-wise bias add |
| Activation | `Rectify` | `float32` tensors via MPSGraph ReLU |
| Softmax | `SoftMax`, `LogSoftMax` | 2D row-wise `float32` |
| Loss | `MPSCrossEntropy`, `MPSNLLLoss` | 2D `float32` logits/log-probs and `int32` labels |
| Autodiff | `Grad(loss, logits)` and two-layer MLP parameter gradients | MLP/classification path |

## Known Limitations

- No GPU-resident tensor representation yet.
- MPSGraph graphs are built per bridge call rather than cached or fused.
- Most op coverage is limited to `float32` tensors.
- Transposed matmul and batched matmul are not implemented.
- Shape-heavy ops such as `Reshape`, `Transpose`, `Slice`, and gather/embedding are not MPS-backed.
- Transformer building blocks such as attention masking, layer norm, and GELU are not implemented.
- The loss API is still experimental; `MPSCrossEntropy` should be reviewed before promoting it as public API.

## Verification

Run MPS tests:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go test -tags mps .
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go test -tags mps ./internal/mpsbridge
```

Run benchmark smoke tests:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go test -tags mps -run '^$' -bench 'BenchmarkMPS' -benchtime=1x .
```

## Benchmarks

The benchmark file `mps_benchmark_test.go` provides benchmark smoke coverage for selected supported ops:

- 2D matrix multiplication through the graph path.
- Row-wise softmax with a tensor CPU baseline and graph MPS path.
- Two-layer MLP forward pass through the graph path.

Because current MPS execution copies tensors to and from CPU on every op, benchmark results should be interpreted as end-to-end bridge overhead plus MPSGraph compute time. They are not representative of an optimized GPU-resident backend.

## Upstreaming Plan

This branch should not be upstreamed as one large PR. Suggested split:

1. Device, build tags, and VM extension points.
2. `internal/mpsbridge` runtime and buffer tests.
3. MatMul, elementwise, bias add, and Rectify integration.
4. SoftMax, LogSoftMax, loss, and autodiff integration.
5. Documentation, support matrix, benchmarks, and examples.

Before proposing a full backend PR, the highest-impact improvements are:

- Introduce a GPU-resident value/buffer representation.
- Cache or fuse MPSGraph execution where possible.
- Reduce MPS-specific public API surface.
- Add broader op coverage for common neural-network and transformer workloads.
