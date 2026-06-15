//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSReLUFloat32ThroughTapeMachine(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	x := NewMatrix(g, tensor.Float32, WithShape(2, 3), WithName("x"))
	y, err := MPSReLU(x)
	if err != nil {
		t.Fatalf("MPSReLU() error = %v", err)
	}

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected MPSReLU to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, x, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{-3, -0.5, 0, 2, 4, -9})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	assertNodeFloat32Value(t, y, []float32{0, 0, 0, 2, 4, 0})
}

func TestRectifyFloat32UsesMPSReLUThroughTapeMachine(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	x := NewMatrix(g, tensor.Float32, WithShape(2, 3), WithName("x"))
	y, err := Rectify(x)
	if err != nil {
		t.Fatalf("Rectify() error = %v", err)
	}
	if _, ok := y.op.(mpsReLUOp); !ok {
		t.Fatalf("Rectify() op = %T, want mpsReLUOp", y.op)
	}

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected Rectify to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, x, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{-3, -0.5, 0, 2, 4, -9})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	assertNodeFloat32Value(t, y, []float32{0, 0, 0, 2, 4, 0})
}

func TestMPSReLURegistersFloat32Values(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}
	if err := metadata.init(); err != nil {
		t.Fatalf("metadata.init() error = %v", err)
	}
	defer metadata.cleanup()

	x := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{-3, -0.5, 0, 2, 4, -9}))
	out, err := (mpsReLUOp{}).MPSDo(&metadata, AppleGPU(0), nil, x)
	if err != nil {
		t.Fatalf("MPSDo() error = %v", err)
	}
	if out == nil {
		t.Fatalf("MPSDo() returned nil output")
	}
	if got := metadata.MPSValueCount(); got != 2 {
		t.Fatalf("MPSValueCount() = %d, want 2", got)
	}
}
