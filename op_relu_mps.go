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
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	var zero *Node
	dt, err := dtypeOf(inputs[0].t)
	if err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}
	switch dt {
	case Float32:
		zero = zerof32
	case Float64:
		zero = zerof64
	default:
		return nil, errors.Errorf(nyiFail, "MPSReLU SymDiff", dt)
	}
	cmp := newElemBinOp(gteOpType, inputs[0], zero)
	cmp.retSame = true
	mask, err := ApplyOp(cmp, inputs[0], zero)
	if err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}
	grad, err := HadamardProd(gradNode, mask)
	if err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}
	return Nodes{grad}, nil
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
	metadata := mpsMetadataFromExternal(extern)
	var out []float32
	var outBuffer *mpsbridge.Float32Buffer
	var err error
	if metadata != nil {
		inputValue, err := metadata.CacheFloat32Value(x)
		if err != nil {
			return nil, err
		}
		outBuffer, err = mpsbridge.ReLUFloat32Buffer(mpsFloat32ValueBuffer(inputValue))
		if err != nil {
			return nil, err
		}
		out, err = outBuffer.Float32s()
	} else {
		out, err = mpsbridge.ReLUFloat32(in)
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

func (op mpsReLUOp) WriteHash(h hash.Hash) { fmt.Fprint(h, "MPSReLU") }
func (op mpsReLUOp) Hashcode() uint32      { return simpleHash(op) }
func (op mpsReLUOp) String() string        { return "MPSReLU" }
