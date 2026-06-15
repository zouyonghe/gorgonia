//go:build mps && darwin
// +build mps,darwin

package mpsbridge

/*
#include "bridge.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

// ReLUFloat32 computes max(input, 0) using MPSGraph.
func ReLUFloat32(input []float32) ([]float32, error) {
	if len(input) == 0 {
		return nil, errors.New("cannot ReLU empty float32 slice")
	}
	out := make([]float32, len(input))
	if ok := C.gorgonia_mps_relu_float32(
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(len(input)),
	); ok != 1 {
		return nil, errors.New("MPSGraph float32 ReLU failed")
	}
	return out, nil
}
