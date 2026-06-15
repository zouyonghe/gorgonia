//go:build mps
// +build mps

package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/mpsbridge"
	"gorgonia.org/tensor"
)

func (op *softmaxOp) mpsSoftmaxRows() bool {
	return op.shape.Dims() == 2 && (op.axis == -1 || op.axis == 1)
}

func softmaxCallsExtern(op *softmaxOp) bool { return op.mpsSoftmaxRows() }

func (op *softmaxOp) MPSDo(extern External, dev Device, prealloc Value, inputs ...Value) (Value, error) {
	if !op.mpsSoftmaxRows() {
		return op.Do(inputs...)
	}
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, err
	}
	x, ok := inputTensor.(*tensor.Dense)
	if !ok {
		return op.Do(inputs...)
	}
	if x.Dtype() != tensor.Float32 {
		return op.Do(inputs...)
	}
	if x.Shape().Dims() != 2 {
		return op.Do(inputs...)
	}
	in, ok := x.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPS softmax expected []float32 backing, got %T", x.Data())
	}
	var out []float32
	if op.isLog {
		out, err = mpsbridge.LogSoftMaxRowsFloat32(in, x.Shape()[0], x.Shape()[1])
	} else {
		out, err = mpsbridge.SoftMaxRowsFloat32(in, x.Shape()[0], x.Shape()[1])
	}
	if err != nil {
		return nil, err
	}
	if reuse, ok := prealloc.(*tensor.Dense); ok && reuse.Dtype() == tensor.Float32 && reuse.Shape().Eq(x.Shape()) {
		copy(reuse.Data().([]float32), out)
		return reuse, nil
	}
	return tensor.New(tensor.WithShape(x.Shape()...), tensor.WithBacking(out)), nil
}
