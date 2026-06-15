//go:build mps
// +build mps

package gorgonia

import "gorgonia.org/tensor"

func tryMPSBroadcastAdd(a, b *Node, leftPattern, rightPattern []byte) (*Node, bool, error) {
	if !isMPSRowBiasAdd(a, b, leftPattern, rightPattern) {
		return nil, false, nil
	}
	n, err := ApplyOp(mpsRowBiasAddOp{}, a, b)
	return n, true, err
}

func isMPSRowBiasAdd(a, b *Node, leftPattern, rightPattern []byte) bool {
	if a == nil || b == nil || len(leftPattern) != 0 || len(rightPattern) != 1 || rightPattern[0] != 0 {
		return false
	}
	at, ok := a.t.(TensorType)
	if !ok || at.Dims != 2 || at.Of != tensor.Float32 {
		return false
	}
	bt, ok := b.t.(TensorType)
	return ok && bt.Dims == 1 && bt.Of == tensor.Float32
}
