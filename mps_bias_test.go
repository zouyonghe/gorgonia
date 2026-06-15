//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSAddRowBiasFloat32ThroughTapeMachine(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	matrix := NewMatrix(g, tensor.Float32, WithShape(2, 3), WithName("matrix"))
	bias := NewVector(g, tensor.Float32, WithShape(3), WithName("bias"))
	z, err := BroadcastAdd(matrix, bias, nil, []byte{0})
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if _, ok := z.op.(mpsRowBiasAddOp); !ok {
		t.Fatalf("BroadcastAdd() op = %T, want mpsRowBiasAddOp", z.op)
	}

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected MPS row bias add to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, matrix, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6})))
	mustLet(t, bias, tensor.New(tensor.WithShape(3), tensor.WithBacking([]float32{10, 20, 30})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	assertNodeFloat32Value(t, z, []float32{11, 22, 33, 14, 25, 36})
}

func TestMPSAddRowBiasRegistersFloat32Values(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}
	if err := metadata.init(); err != nil {
		t.Fatalf("metadata.init() error = %v", err)
	}
	defer metadata.cleanup()

	matrix := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6}))
	bias := tensor.New(tensor.WithShape(3), tensor.WithBacking([]float32{10, 20, 30}))
	out, err := (mpsRowBiasAddOp{}).MPSDo(&metadata, AppleGPU(0), nil, matrix, bias)
	if err != nil {
		t.Fatalf("MPSDo() error = %v", err)
	}
	if out == nil {
		t.Fatalf("MPSDo() returned nil output")
	}
	if got := metadata.MPSValueCount(); got != 3 {
		t.Fatalf("MPSValueCount() = %d, want 3", got)
	}
}
