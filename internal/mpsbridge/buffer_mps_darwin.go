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

// Float32Buffer owns an MTLBuffer containing float32 values.
type Float32Buffer struct {
	ptr   unsafe.Pointer
	count int
}

// NewFloat32Buffer copies float32 data into an MTLBuffer using shared storage.
func NewFloat32Buffer(data []float32) (*Float32Buffer, error) {
	if len(data) == 0 {
		return nil, errors.New("cannot create MPS buffer from empty float32 slice")
	}
	ptr := C.gorgonia_mps_new_float32_buffer((*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
	if ptr == nil {
		return nil, errors.New("failed to create MPS float32 buffer")
	}
	return &Float32Buffer{ptr: ptr, count: len(data)}, nil
}

// Float32s copies the MTLBuffer contents back into Go memory.
func (b *Float32Buffer) Float32s() ([]float32, error) {
	if b == nil || b.ptr == nil {
		return nil, errors.New("MPS float32 buffer is closed")
	}
	out := make([]float32, b.count)
	if ok := C.gorgonia_mps_read_float32_buffer(b.ptr, (*C.float)(unsafe.Pointer(&out[0])), C.int(len(out))); ok != 1 {
		return nil, errors.New("failed to read MPS float32 buffer")
	}
	return out, nil
}

// Close releases the underlying MTLBuffer.
func (b *Float32Buffer) Close() {
	if b == nil || b.ptr == nil {
		return
	}
	C.gorgonia_mps_release_buffer(b.ptr)
	b.ptr = nil
}
