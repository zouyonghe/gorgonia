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
	var outBuffer *mpsbridge.Float32Buffer
	metadata := mpsMetadataFromExternal(extern)
	rows, cols := x.Shape()[0], x.Shape()[1]
	if metadata != nil {
		inputValue, err := metadata.CacheFloat32Value(x)
		if err != nil {
			return nil, err
		}
		if op.isLog {
			outBuffer, err = mpsbridge.LogSoftMaxRowsFloat32Buffer(mpsFloat32ValueBuffer(inputValue), rows, cols)
		} else {
			outBuffer, err = mpsbridge.SoftMaxRowsFloat32Buffer(mpsFloat32ValueBuffer(inputValue), rows, cols)
		}
		if err != nil {
			return nil, err
		}
		out, err = outBuffer.Float32s()
	} else {
		if op.isLog {
			out, err = mpsbridge.LogSoftMaxRowsFloat32(in, rows, cols)
		} else {
			out, err = mpsbridge.SoftMaxRowsFloat32(in, rows, cols)
		}
	}
	if err != nil {
		return nil, err
	}
	if reuse, ok := prealloc.(*tensor.Dense); ok && reuse.Dtype() == tensor.Float32 && reuse.Shape().Eq(x.Shape()) {
		copy(reuse.Data().([]float32), out)
		if metadata != nil && outBuffer != nil {
			if _, err := metadata.TrackFloat32Value(reuse, outBuffer); err != nil {
				outBuffer.Close()
				return nil, err
			}
		} else {
			cacheMPSFloat32Value(extern, reuse)
		}
		return reuse, nil
	}
	retVal := tensor.New(tensor.WithShape(x.Shape()...), tensor.WithBacking(out))
	if metadata != nil && outBuffer != nil {
		if _, err := metadata.TrackFloat32Value(retVal, outBuffer); err != nil {
			outBuffer.Close()
			return nil, err
		}
	} else {
		cacheMPSFloat32Value(extern, retVal)
	}
	return retVal, nil
}
