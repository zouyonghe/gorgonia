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

type mpsRowBiasAddOp struct{}

func (op mpsRowBiasAddOp) Arity() int { return 2 }

func (op mpsRowBiasAddOp) Type() hm.Type { return hm.NewFnType(*matF32, *vecF32, *matF32) }

func (op mpsRowBiasAddOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return nil, errors.New("MPS row bias add expects matrix and bias inputs")
	}
	matrixShape, ok := inputs[0].(tensor.Shape)
	if !ok {
		return nil, nil
	}
	biasShape, ok := inputs[1].(tensor.Shape)
	if !ok {
		return nil, nil
	}
	if matrixShape.Dims() != 2 || biasShape.Dims() != 1 || matrixShape[1] != biasShape[0] {
		return nil, errors.Errorf("MPS row bias add expected matrix(rows, cols) and bias(cols), got %v and %v", matrixShape, biasShape)
	}
	return matrixShape.Clone(), nil
}

func (op mpsRowBiasAddOp) ReturnsPtr() bool     { return true }
func (op mpsRowBiasAddOp) CallsExtern() bool    { return true }
func (op mpsRowBiasAddOp) OverwritesInput() int { return -1 }
func (op mpsRowBiasAddOp) DiffWRT(i int) []bool { return []bool{true, true} }

func (op mpsRowBiasAddOp) SymDiff(inputs Nodes, output, gradNode *Node) (Nodes, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	biasGrad, err := Sum(gradNode, 0)
	if err != nil {
		return nil, err
	}
	return Nodes{gradNode, biasGrad}, nil
}

func (op mpsRowBiasAddOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	return errors.Errorf("MPS row bias add differentiation is not implemented")
}

func (op mpsRowBiasAddOp) Do(inputs ...Value) (Value, error) {
	return op.do(nil, CPU, nil, inputs...)
}

func (op mpsRowBiasAddOp) MPSDo(extern External, dev Device, prealloc Value, inputs ...Value) (Value, error) {
	return op.do(extern, dev, prealloc, inputs...)
}

func (op mpsRowBiasAddOp) do(extern External, dev Device, prealloc Value, inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	matrix, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPS row bias add expected matrix *tensor.Dense, got %T", inputs[0])
	}
	bias, ok := inputs[1].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPS row bias add expected bias *tensor.Dense, got %T", inputs[1])
	}
	if matrix.Dtype() != tensor.Float32 || bias.Dtype() != tensor.Float32 {
		return nil, errors.Errorf("MPS row bias add supports float32 only; got %v and %v", matrix.Dtype(), bias.Dtype())
	}
	if matrix.Shape().Dims() != 2 || bias.Shape().Dims() != 1 || matrix.Shape()[1] != bias.Shape()[0] {
		return nil, errors.Errorf("MPS row bias add expected matrix(rows, cols) and bias(cols), got %v and %v", matrix.Shape(), bias.Shape())
	}
	rows, cols := matrix.Shape()[0], matrix.Shape()[1]
	matrixData, ok := matrix.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPS row bias add expected matrix []float32 backing, got %T", matrix.Data())
	}
	biasData, ok := bias.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPS row bias add expected bias []float32 backing, got %T", bias.Data())
	}

	metadata := mpsMetadataFromExternal(extern)
	var out []float32
	var outBuffer *mpsbridge.Float32Buffer
	var err error
	if metadata != nil {
		matrixValue, err := metadata.CacheFloat32Value(matrix)
		if err != nil {
			return nil, err
		}
		biasValue, err := metadata.CacheFloat32Value(bias)
		if err != nil {
			return nil, err
		}
		outBuffer, err = mpsbridge.AddRowBiasFloat32Buffer(mpsFloat32ValueBuffer(matrixValue), mpsFloat32ValueBuffer(biasValue), rows, cols)
		if err != nil {
			return nil, err
		}
		out, err = outBuffer.Float32s()
	} else {
		out, err = mpsbridge.AddRowBiasFloat32(matrixData, biasData, rows, cols)
	}
	if err != nil {
		return nil, err
	}

	if reuse, ok := prealloc.(*tensor.Dense); ok && reuse.Dtype() == tensor.Float32 && reuse.Shape().Eq(matrix.Shape()) {
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
	retVal := tensor.New(tensor.WithShape(matrix.Shape()...), tensor.WithBacking(out))
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

func (op mpsRowBiasAddOp) WriteHash(h hash.Hash) { fmt.Fprint(h, "MPSRowBiasAdd") }
func (op mpsRowBiasAddOp) Hashcode() uint32      { return simpleHash(op) }
func (op mpsRowBiasAddOp) String() string        { return "MPSRowBiasAdd" }
