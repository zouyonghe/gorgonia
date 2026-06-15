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

// AddFloat32Buffers computes elementwise A+B from existing MTLBuffers.
func AddFloat32Buffers(a, b *Float32Buffer) (*Float32Buffer, error) {
	return elementwiseFloat32Buffers(a, b, mpsOpAdd, "add")
}

// SubFloat32Buffers computes elementwise A-B from existing MTLBuffers.
func SubFloat32Buffers(a, b *Float32Buffer) (*Float32Buffer, error) {
	return elementwiseFloat32Buffers(a, b, mpsOpSub, "sub")
}

// MulFloat32Buffers computes elementwise A*B from existing MTLBuffers.
func MulFloat32Buffers(a, b *Float32Buffer) (*Float32Buffer, error) {
	return elementwiseFloat32Buffers(a, b, mpsOpMul, "mul")
}

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

func elementwiseFloat32Buffers(a, b *Float32Buffer, op int, name string) (*Float32Buffer, error) {
	if a == nil || a.ptr == nil {
		return nil, fmt.Errorf("left MPS float32 buffer is closed for %s", name)
	}
	if b == nil || b.ptr == nil {
		return nil, fmt.Errorf("right MPS float32 buffer is closed for %s", name)
	}
	if a.count == 0 {
		return nil, fmt.Errorf("cannot %s empty MPS float32 buffers", name)
	}
	if a.count != b.count {
		return nil, fmt.Errorf("mismatched %s buffer lengths: %d and %d", name, a.count, b.count)
	}
	ptr := C.gorgonia_mps_elementwise_float32_buffers(a.ptr, b.ptr, C.int(a.count), C.int(op))
	if ptr == nil {
		return nil, errors.New("MPSGraph float32 buffer elementwise op failed")
	}
	return &Float32Buffer{ptr: ptr, count: a.count}, nil
}
