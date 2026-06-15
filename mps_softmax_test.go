//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"math"
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSSoftMaxFloat32ThroughTapeMachine(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	x := NewMatrix(g, tensor.Float32, WithShape(2, 3), WithName("x"))
	y, err := SoftMax(x)
	if err != nil {
		t.Fatalf("SoftMax() error = %v", err)
	}

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected SoftMax to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, x, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		1, 2, 3,
		1, 1, 1,
	})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	assertNodeFloat32ValueClose(t, y, []float32{
		0.09003057, 0.24472848, 0.66524094,
		0.33333334, 0.33333334, 0.33333334,
	}, 1e-5)
}

func TestMPSLogSoftMaxFloat32ThroughTapeMachine(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	x := NewMatrix(g, tensor.Float32, WithShape(2, 3), WithName("x"))
	y, err := LogSoftMax(x)
	if err != nil {
		t.Fatalf("LogSoftMax() error = %v", err)
	}

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected LogSoftMax to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, x, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		1, 2, 3,
		1, 1, 1,
	})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	assertNodeFloat32ValueClose(t, y, []float32{
		-2.407606, -1.407606, -0.407606,
		-1.0986123, -1.0986123, -1.0986123,
	}, 1e-5)
}

func assertNodeFloat32ValueClose(t *testing.T, node *Node, want []float32, tolerance float64) {
	t.Helper()
	value, ok := node.Value().(*tensor.Dense)
	if !ok {
		t.Fatalf("node value = %T, want *tensor.Dense", node.Value())
	}
	got, ok := value.Data().([]float32)
	if !ok {
		t.Fatalf("node backing = %T, want []float32", value.Data())
	}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("got[%d] = %v, want %v; got=%v", i, got[i], want[i], got)
		}
	}
}
