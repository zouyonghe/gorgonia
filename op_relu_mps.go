//go:build mps
// +build mps

package gorgonia

import (
	"fmt"
	"hash"
	"math"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/mpsbridge"
	"gorgonia.org/tensor"
)

type mpsReLUOp struct{}

// MPSReLU creates a float32 ReLU node backed by MPSGraph when built with -tags mps.
func MPSReLU(x *Node) (*Node, error) { return ApplyOp(mpsReLUOp{}, x) }

func (op mpsReLUOp) Arity() int { return 1 }

func (op mpsReLUOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (op mpsReLUOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if len(inputs) != 1 || inputs[0] == nil {
		return nil, errors.New("MPSReLU expects one shaped input")
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op mpsReLUOp) ReturnsPtr() bool     { return true }
func (op mpsReLUOp) CallsExtern() bool    { return true }
func (op mpsReLUOp) OverwritesInput() int { return -1 }
func (op mpsReLUOp) DiffWRT(i int) []bool { return []bool{true} }
func (op mpsReLUOp) SymDiff(inputs Nodes, output, gradNode *Node) (Nodes, error) {
	return nil, errors.Errorf("MPSReLU symbolic differentiation is not implemented")
}
func (op mpsReLUOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	return errors.Errorf("MPSReLU differentiation is not implemented")
}

func (op mpsReLUOp) Do(inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	x, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPSReLU expected *tensor.Dense, got %T", inputs[0])
	}
	if x.Dtype() != tensor.Float32 {
		return nil, errors.Errorf("MPSReLU supports float32 only; got %v", x.Dtype())
	}
	in, ok := x.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPSReLU expected []float32 backing, got %T", x.Data())
	}
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(math.Max(0, float64(v)))
	}
	return tensor.New(tensor.WithShape(x.Shape()...), tensor.WithBacking(out)), nil
}

func (op mpsReLUOp) MPSDo(extern External, dev Device, prealloc Value, inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	x, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, errors.Errorf("MPSReLU expected *tensor.Dense, got %T", inputs[0])
	}
	if x.Dtype() != tensor.Float32 {
		return nil, errors.Errorf("MPSReLU supports float32 only; got %v", x.Dtype())
	}
	in, ok := x.Data().([]float32)
	if !ok {
		return nil, errors.Errorf("MPSReLU expected []float32 backing, got %T", x.Data())
	}
	out, err := mpsbridge.ReLUFloat32(in)
	if err != nil {
		return nil, err
	}
	if reuse, ok := prealloc.(*tensor.Dense); ok && reuse.Dtype() == tensor.Float32 && reuse.Shape().Eq(x.Shape()) {
		copy(reuse.Data().([]float32), out)
		return reuse, nil
	}
	return tensor.New(tensor.WithShape(x.Shape()...), tensor.WithBacking(out)), nil
}

func (op mpsReLUOp) WriteHash(h hash.Hash) { fmt.Fprint(h, "MPSReLU") }
func (op mpsReLUOp) Hashcode() uint32      { return simpleHash(op) }
func (op mpsReLUOp) String() string        { return "MPSReLU" }
