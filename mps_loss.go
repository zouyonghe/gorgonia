//go:build mps
// +build mps

package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/mpsbridge"
	"gorgonia.org/tensor"
)

// MPSNLLLossFloat32 computes mean negative log likelihood for row-wise log-probs.
func MPSNLLLossFloat32(logProbs []float32, labels []int32, rows, cols int) (float32, error) {
	return mpsbridge.NLLLossFloat32(logProbs, labels, rows, cols)
}

// MPSCrossEntropyLogitsGradFloat32 computes d(mean CE)/dlogits for row-wise logits and int32 labels.
func MPSCrossEntropyLogitsGradFloat32(logits []float32, labels []int32, rows, cols int) ([]float32, error) {
	return mpsbridge.CrossEntropyLogitsGradFloat32(logits, labels, rows, cols)
}

type mpsNLLLossOp struct {
	logProbShape tensor.Shape
}

// MPSNLLLoss creates a scalar float32 NLL loss node for row-wise log-probs and int32 labels.
func MPSNLLLoss(logProbs, labels *Node) (*Node, error) {
	op := mpsNLLLossOp{logProbShape: logProbs.Shape()}
	return ApplyOp(op, logProbs, labels)
}

// MPSCrossEntropy creates a scalar float32 cross-entropy loss node from logits and int32 labels.
func MPSCrossEntropy(logits, labels *Node) (*Node, error) {
	logProbs, err := LogSoftMax(logits)
	if err != nil {
		return nil, err
	}
	return MPSNLLLoss(logProbs, labels)
}

func (op mpsNLLLossOp) Arity() int { return 2 }

func (op mpsNLLLossOp) Type() hm.Type {
	return hm.NewFnType(matF32, vecI32(), tensor.Float32)
}

func vecI32() *TensorType { return newTensorType(1, tensor.Int32) }

func (op mpsNLLLossOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if len(inputs) != 2 {
		return nil, errors.Errorf("MPSNLLLoss expects two inputs")
	}
	logProbShape := inputs[0].(tensor.Shape)
	labelShape := inputs[1].(tensor.Shape)
	if logProbShape.Dims() != 2 {
		return nil, errors.Errorf("MPSNLLLoss expects 2D log-probs, got %v", logProbShape)
	}
	if labelShape.Dims() != 1 || labelShape[0] != logProbShape[0] {
		return nil, errors.Errorf("MPSNLLLoss expects labels shape (%d), got %v", logProbShape[0], labelShape)
	}
	return tensor.ScalarShape(), nil
}

func (op mpsNLLLossOp) ReturnsPtr() bool     { return false }
func (op mpsNLLLossOp) CallsExtern() bool    { return true }
func (op mpsNLLLossOp) OverwritesInput() int { return -1 }
func (op mpsNLLLossOp) DiffWRT(inputs int) []bool {
	return []bool{true, false}
}
func (op mpsNLLLossOp) SymDiff(inputs Nodes, output, gradNode *Node) (Nodes, error) {
	return nil, errors.Errorf("MPSNLLLoss symbolic differentiation is not implemented")
}
func (op mpsNLLLossOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	return errors.Errorf("MPSNLLLoss differentiation is not implemented")
}

func (op mpsNLLLossOp) Do(inputs ...Value) (Value, error) {
	return op.compute(inputs...)
}

func (op mpsNLLLossOp) MPSDo(extern External, dev Device, prealloc Value, inputs ...Value) (Value, error) {
	return op.compute(inputs...)
}

func (op mpsNLLLossOp) compute(inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	logProbs, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPSNLLLoss expected log-probs *tensor.Dense, got %T", inputs[0])
	}
	labels, ok := inputs[1].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPSNLLLoss expected labels *tensor.Dense, got %T", inputs[1])
	}
	if logProbs.Dtype() != tensor.Float32 {
		return nil, errors.Errorf("MPSNLLLoss supports float32 log-probs only; got %v", logProbs.Dtype())
	}
	if labels.Dtype() != tensor.Int32 {
		return nil, errors.Errorf("MPSNLLLoss supports int32 labels only; got %v", labels.Dtype())
	}
	if logProbs.Shape().Dims() != 2 {
		return nil, errors.Errorf("MPSNLLLoss expects 2D log-probs, got %v", logProbs.Shape())
	}
	logProbData, ok := logProbs.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPSNLLLoss expected []float32 backing, got %T", logProbs.Data())
	}
	labelData, ok := labels.Data().([]int32)
	if !ok {
		return nil, errors.Errorf("MPSNLLLoss expected []int32 backing, got %T", labels.Data())
	}
	loss, err := mpsbridge.NLLLossFloat32(logProbData, labelData, logProbs.Shape()[0], logProbs.Shape()[1])
	if err != nil {
		return nil, err
	}
	return NewF32(loss), nil
}

func (op mpsNLLLossOp) WriteHash(h hash.Hash) { fmt.Fprint(h, "MPSNLLLoss") }
func (op mpsNLLLossOp) Hashcode() uint32      { return simpleHash(op) }
func (op mpsNLLLossOp) String() string        { return "MPSNLLLoss" }
