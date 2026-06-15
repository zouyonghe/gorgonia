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

// AddFloat32 computes elementwise A+B using MPSGraph.
func AddFloat32(a, b []float32) ([]float32, error) {
	if len(a) == 0 {
		return nil, errors.New("cannot add empty float32 slices")
	}
	if len(a) != len(b) {
		return nil, fmt.Errorf("mismatched add lengths: %d and %d", len(a), len(b))
	}
	out := make([]float32, len(a))
	if ok := C.gorgonia_mps_add_float32(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(len(a)),
	); ok != 1 {
		return nil, errors.New("MPSGraph float32 add failed")
	}
	return out, nil
}
