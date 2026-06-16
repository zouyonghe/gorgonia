//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"math"
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSNLLLossThroughTapeMachine(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	g := NewGraph()
	logProbs := NewMatrix(g, tensor.Float32, WithShape(2, 3), WithName("logProbs"))
	labels := NewVector(g, tensor.Int32, WithShape(2), WithName("labels"))
	loss, err := MPSNLLLoss(logProbs, labels)
	if err != nil {
		t.Fatalf("MPSNLLLoss() error = %v", err)
	}

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected MPSNLLLoss to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, logProbs, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		-2.407606, -1.407606, -0.407606,
		-1.098612, -1.098612, -1.098612,
	})))
	mustLet(t, labels, tensor.New(tensor.WithShape(2), tensor.WithBacking([]int32{2, 0})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	value, ok := loss.Value().(*F32)
	if !ok {
		t.Fatalf("loss value = %T, want *F32", loss.Value())
	}
	if math.Abs(float64(value.any()-0.753109)) > 1e-5 {
		t.Fatalf("loss = %v, want %v", value.any(), 0.753109)
	}
}

func TestMPSCrossEntropyThroughTapeMachine(t *testing.T) {
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

	m := NewTapeMachine(g)
	if m.Prog().gpulocs == 0 {
		t.Fatalf("expected MPSCrossEntropy to allocate GPU registers")
	}
	defer m.Close()

	mustLet(t, logits, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		1, 2, 3,
		1, 1, 1,
	})))
	mustLet(t, labels, tensor.New(tensor.WithShape(2), tensor.WithBacking([]int32{2, 0})))
	if err := m.RunAll(); err != nil {
		t.Fatalf("RunAll() error = %v", err)
	}

	value, ok := loss.Value().(*F32)
	if !ok {
		t.Fatalf("loss value = %T, want *F32", loss.Value())
	}
	if math.Abs(float64(value.any()-0.753109)) > 1e-5 {
		t.Fatalf("loss = %v, want %v", value.any(), 0.753109)
	}
}

func TestMPSNLLLossCachesLogProbBuffer(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}
	if err := metadata.init(); err != nil {
		t.Fatalf("metadata.init() error = %v", err)
	}
	defer metadata.cleanup()

	logProbs := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		-2.407606, -1.407606, -0.407606,
		-1.098612, -1.098612, -1.098612,
	}))
	labels := tensor.New(tensor.WithShape(2), tensor.WithBacking([]int32{2, 0}))
	loss, err := (mpsNLLLossOp{logProbShape: logProbs.Shape()}).MPSDo(&metadata, AppleGPU(0), nil, logProbs, labels)
	if err != nil {
		t.Fatalf("MPSDo() error = %v", err)
	}
	if loss == nil {
		t.Fatalf("MPSDo() returned nil loss")
	}
	if got := metadata.MPSValueCount(); got != 1 {
		t.Fatalf("MPSValueCount() = %d, want 1", got)
	}
}
