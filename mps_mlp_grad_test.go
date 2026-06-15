//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSMLPGradWRTParameters(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	x := NewMatrix(g, tensor.Float32, WithShape(2, 3), WithName("x"))
	w1 := NewMatrix(g, tensor.Float32, WithShape(3, 4), WithName("w1"))
	b1 := NewVector(g, tensor.Float32, WithShape(4), WithName("b1"))
	w2 := NewMatrix(g, tensor.Float32, WithShape(4, 2), WithName("w2"))
	b2 := NewVector(g, tensor.Float32, WithShape(2), WithName("b2"))
	labels := NewVector(g, tensor.Int32, WithShape(2), WithName("labels"))

	l1, err := Mul(x, w1)
	if err != nil {
		t.Fatalf("Mul(x,w1) error = %v", err)
	}
	l1Bias, err := BroadcastAdd(l1, b1, nil, []byte{0})
	if err != nil {
		t.Fatalf("BroadcastAdd(l1,b1) error = %v", err)
	}
	hidden, err := Rectify(l1Bias)
	if err != nil {
		t.Fatalf("Rectify() error = %v", err)
	}
	logits, err := Mul(hidden, w2)
	if err != nil {
		t.Fatalf("Mul(hidden,w2) error = %v", err)
	}
	logitsBias, err := BroadcastAdd(logits, b2, nil, []byte{0})
	if err != nil {
		t.Fatalf("BroadcastAdd(logits,b2) error = %v", err)
	}
	loss, err := MPSCrossEntropy(logitsBias, labels)
	if err != nil {
		t.Fatalf("MPSCrossEntropy() error = %v", err)
	}
	if _, err := Grad(loss, w1, b1, w2, b2); err != nil {
		t.Fatalf("Grad() error = %v", err)
	}
}
