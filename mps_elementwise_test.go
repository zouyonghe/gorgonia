//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSElementwiseSubMulFloat32ThroughTapeMachine(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	a := NewMatrix(g, tensor.Float32, WithShape(2, 2), WithName("a"))
	b := NewMatrix(g, tensor.Float32, WithShape(2, 2), WithName("b"))
	sub, err := Sub(a, b)
	if err != nil {
		t.Fatalf("Sub() error = %v", err)
	}
	mul, err := HadamardProd(a, b)
	if err != nil {
		t.Fatalf("HadamardProd() error = %v", err)
	}

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected MPS elementwise ops to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, a, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, -2, 3.5, 4})))
	mustLet(t, b, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{5, 6, -7.5, 8})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	assertNodeFloat32Value(t, sub, []float32{-4, -8, 11, -4})
	assertNodeFloat32Value(t, mul, []float32{5, -12, -26.25, 32})
}

func assertNodeFloat32Value(t *testing.T, n *Node, want []float32) {
	t.Helper()
	gotT, ok := n.Value().(*tensor.Dense)
	if !ok {
		t.Fatalf("%v.Value() = %T, want *tensor.Dense", n, n.Value())
	}
	got := gotT.Data().([]float32)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v; got=%v", i, got[i], want[i], got)
		}
	}
}
