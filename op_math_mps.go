//go:build mps
// +build mps

package gorgonia

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/mpsbridge"
	"gorgonia.org/tensor"
)

func (op elemUnaryOp) CallsExtern() bool { return false }
func (op elemBinOp) CallsExtern() bool {
	switch op.binOpType() {
	case addOpType, subOpType, mulOpType:
		// supported below
	default:
		return false
	}
	a, ok := op.arg0.(TensorType)
	if !ok || a.Of != Float32 {
		return false
	}
	b, ok := op.arg1.(TensorType)
	return ok && b.Of == Float32
}

func (op linAlgBinOp) CallsExtern() bool {
	return op.āBinaryOperator == matMulOperator && !op.transA && !op.transB
}

// NewAddOp creates a new *ExternalOp that wraps an add op.
func NewAddOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	add := newElemBinOp(addOpType, a, b)
	op := NewExternalOp(add, ctx, nil)
	op.Device = CPU
	return op
}

// NewSubOp creates a new *ExternalOp that wraps a sub op.
func NewSubOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	sub := newEBOByType(subOpType, a.t, b.t)
	op := NewExternalOp(sub, ctx, nil)
	op.Device = CPU
	return op
}

// NewHadamardProdOp creates a new *ExternalOp that wraps a mul op.
func NewHadamardProdOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	mul := newEBOByType(mulOpType, a.t, b.t)
	op := NewExternalOp(mul, ctx, nil)
	op.Device = CPU
	return op
}

// MPSDo executes supported elementwise binary ops through Apple's MPSGraph backend.
// The supported slice is float32 tensor add/sub/mul with identical shapes.
func (op elemBinOp) MPSDo(extern External, dev Device, prealloc Value, inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	switch op.binOpType() {
	case addOpType, subOpType, mulOpType:
		// supported below
	default:
		return nil, errors.Errorf("MPS supports only float32 elementwise add/sub/mul for now; op=%v", op)
	}
	a, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPS add expected left *tensor.Dense, got %T", inputs[0])
	}
	b, ok := inputs[1].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPS add expected right *tensor.Dense, got %T", inputs[1])
	}
	if a.Dtype() != tensor.Float32 || b.Dtype() != tensor.Float32 {
		return nil, errors.Errorf("MPS add supports float32 only; got %v and %v", a.Dtype(), b.Dtype())
	}
	if !a.Shape().Eq(b.Shape()) {
		return nil, errors.Errorf("MPS add supports identical shapes only; got %v and %v", a.Shape(), b.Shape())
	}
	left, ok := a.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPS add expected left []float32 backing, got %T", a.Data())
	}
	right, ok := b.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPS add expected right []float32 backing, got %T", b.Data())
	}
	var out []float32
	var err error
	switch op.binOpType() {
	case addOpType:
		out, err = mpsbridge.AddFloat32(left, right)
	case subOpType:
		out, err = mpsbridge.SubFloat32(left, right)
	case mulOpType:
		out, err = mpsbridge.MulFloat32(left, right)
	}
	if err != nil {
		return nil, err
	}
	if reuse, ok := prealloc.(*tensor.Dense); ok && reuse.Dtype() == tensor.Float32 && reuse.Shape().Eq(a.Shape()) {
		copy(reuse.Data().([]float32), out)
		return reuse, nil
	}
	return tensor.New(tensor.WithShape(a.Shape()...), tensor.WithBacking(out)), nil
}

// MPSDo executes the supported linear algebra op through Apple's MPSGraph backend.
// The first supported slice is float32 matrix multiplication. Inputs and outputs
// are still surfaced as CPU-accessible tensors; the computation itself is done by
// MPSGraph through mpsbridge.
func (op linAlgBinOp) MPSDo(extern External, dev Device, prealloc Value, inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	if op.āBinaryOperator != matMulOperator || op.transA || op.transB {
		return nil, errors.Errorf("MPS supports only non-transposed float32 matrix multiplication for now; op=%v", op)
	}

	a, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPS matmul expected left *tensor.Dense, got %T", inputs[0])
	}
	b, ok := inputs[1].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPS matmul expected right *tensor.Dense, got %T", inputs[1])
	}
	if a.Dtype() != tensor.Float32 || b.Dtype() != tensor.Float32 {
		return nil, errors.Errorf("MPS matmul supports float32 only; got %v and %v", a.Dtype(), b.Dtype())
	}
	if a.Shape().Dims() != 2 || b.Shape().Dims() != 2 {
		return nil, errors.Errorf("MPS matmul supports 2D matrices only; got %v and %v", a.Shape(), b.Shape())
	}
	m, k := a.Shape()[0], a.Shape()[1]
	if b.Shape()[0] != k {
		return nil, fmt.Errorf("inner dimensions do not match: %v and %v", a.Shape(), b.Shape())
	}
	n := b.Shape()[1]

	left, ok := a.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPS matmul expected left []float32 backing, got %T", a.Data())
	}
	right, ok := b.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPS matmul expected right []float32 backing, got %T", b.Data())
	}
	cacheMPSFloat32Value(extern, a)
	cacheMPSFloat32Value(extern, b)
	out, err := mpsbridge.MatMulFloat32(left, right, m, k, n)
	if err != nil {
		return nil, err
	}

	if reuse, ok := prealloc.(*tensor.Dense); ok && reuse.Dtype() == tensor.Float32 && reuse.Shape().Eq(tensor.Shape{m, n}) {
		copy(reuse.Data().([]float32), out)
		cacheMPSFloat32Value(extern, reuse)
		return reuse, nil
	}
	retVal := tensor.New(tensor.WithShape(m, n), tensor.WithBacking(out))
	cacheMPSFloat32Value(extern, retVal)
	return retVal, nil
}
