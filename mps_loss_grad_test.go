//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSCrossEntropyGradWRTLogits(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	logits := NewMatrix(g, tensor.Float32, WithShape(2, 3), WithName("logits"))
	labels := NewVector(g, tensor.Int32, WithShape(2), WithName("labels"))
	loss, err := MPSCrossEntropy(logits, labels)
	if err != nil {
		t.Fatalf("MPSCrossEntropy() error = %v", err)
	}
	if _, err := Grad(loss, logits); err != nil {
		t.Fatalf("Grad() error = %v", err)
	}

	m := NewTapeMachine(g, BindDualValues(logits))
	defer m.Close()
	mustLet(t, logits, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		1, 2, 3,
		1, 1, 1,
	})))
	mustLet(t, labels, tensor.New(tensor.WithShape(2), tensor.WithBacking([]int32{2, 0})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	gradValue, err := logits.Grad()
	if err != nil {
		t.Fatalf("logits.Grad() error = %v", err)
	}
	gradTensor, ok := gradValue.(*tensor.Dense)
	if !ok {
		t.Fatalf("grad = %T, want *tensor.Dense", gradValue)
	}
	got := gradTensor.Data().([]float32)
	want := []float32{
		0.045015285, 0.12236424, -0.16737953,
		-0.33333334, 0.16666667, 0.16666667,
	}
	assertFloat32Close(t, got, want, 1e-5)
}

func assertFloat32Close(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if float64(diff) > tolerance {
			t.Fatalf("got[%d] = %v, want %v; got=%v", i, got[i], want[i], got)
		}
	}
}
