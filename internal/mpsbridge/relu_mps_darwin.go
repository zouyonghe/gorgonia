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

// ReLUFloat32Buffer computes max(input, 0) from an existing MTLBuffer and returns an output buffer.
func ReLUFloat32Buffer(input *Float32Buffer) (*Float32Buffer, error) {
	if input == nil || input.ptr == nil {
		return nil, errors.New("MPS float32 buffer is closed")
	}
	if input.count == 0 {
		return nil, errors.New("cannot ReLU empty MPS float32 buffer")
	}
	ptr := C.gorgonia_mps_relu_float32_buffer(input.ptr, C.int(input.count))
	if ptr == nil {
		return nil, errors.New("MPSGraph float32 buffer ReLU failed")
	}
	return &Float32Buffer{ptr: ptr, count: input.count}, nil
}
