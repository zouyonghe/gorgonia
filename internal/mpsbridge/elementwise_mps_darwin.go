//go:build mps && darwin
// +build mps,darwin

package mpsbridge

/*
#include "bridge.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

const (
	mpsOpAdd = iota
	mpsOpSub
	mpsOpMul
)

// SubFloat32 computes elementwise A-B using MPSGraph.
func SubFloat32(a, b []float32) ([]float32, error) { return elementwiseFloat32(a, b, mpsOpSub, "sub") }

// MulFloat32 computes elementwise A*B using MPSGraph.
func MulFloat32(a, b []float32) ([]float32, error) { return elementwiseFloat32(a, b, mpsOpMul, "mul") }

func elementwiseFloat32(a, b []float32, op int, name string) ([]float32, error) {
	if len(a) == 0 {
		return nil, fmt.Errorf("cannot %s empty float32 slices", name)
	}
	if len(a) != len(b) {
		return nil, fmt.Errorf("mismatched %s lengths: %d and %d", name, len(a), len(b))
	}
	out := make([]float32, len(a))
	if ok := C.gorgonia_mps_elementwise_float32(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(len(a)),
		C.int(op),
	); ok != 1 {
		return nil, errors.New("MPSGraph float32 elementwise op failed")
	}
	return out, nil
}
