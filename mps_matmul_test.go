//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSMatMulFloat32ThroughTapeMachine(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	a := NewMatrix(g, tensor.Float32, WithShape(2, 2), WithName("a"))
	b := NewMatrix(g, tensor.Float32, WithShape(2, 2), WithName("b"))
	z, err := Mul(a, b)
	if err != nil {
		t.Fatalf("Mul() error = %v", err)
	}

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected MPS matmul to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, a, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4})))
	mustLet(t, b, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{5, 6, 7, 8})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	gotT, ok := z.Value().(*tensor.Dense)
	if !ok {
		t.Fatalf("z.Value() = %T, want *tensor.Dense", z.Value())
	}
	got := gotT.Data().([]float32)
	want := []float32{19, 22, 43, 50}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v; got=%v", i, got[i], want[i], got)
		}
	}
}

func mustLet(t *testing.T, n *Node, v interface{}) {
	t.Helper()
	if err := Let(n, v); err != nil {
		t.Fatalf("Let(%v) error = %v", n, err)
	}
}
