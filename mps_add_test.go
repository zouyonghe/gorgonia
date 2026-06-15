//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSAddFloat32ThroughTapeMachine(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	a := NewMatrix(g, tensor.Float32, WithShape(2, 2), WithName("a"))
	b := NewMatrix(g, tensor.Float32, WithShape(2, 2), WithName("b"))
	z, err := Add(a, b)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected MPS add to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, a, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, -2, 3.5, 4})))
	mustLet(t, b, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{5, 6, -7.5, 8})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	gotT, ok := z.Value().(*tensor.Dense)
	if !ok {
		t.Fatalf("z.Value() = %T, want *tensor.Dense", z.Value())
	}
	got := gotT.Data().([]float32)
	want := []float32{6, 4, -4, 12}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v; got=%v", i, got[i], want[i], got)
		}
	}
}
