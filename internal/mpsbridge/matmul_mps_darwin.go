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

// MatMulFloat32 computes A(m x k) * B(k x n) using MPSGraph and returns row-major output.
func MatMulFloat32(a, b []float32, m, k, n int) ([]float32, error) {
	if m <= 0 || k <= 0 || n <= 0 {
		return nil, fmt.Errorf("invalid matmul dimensions m=%d k=%d n=%d", m, k, n)
	}
	if len(a) != m*k {
		return nil, fmt.Errorf("left matrix has %d elements, want %d", len(a), m*k)
	}
	if len(b) != k*n {
		return nil, fmt.Errorf("right matrix has %d elements, want %d", len(b), k*n)
	}
	out := make([]float32, m*n)
	if ok := C.gorgonia_mps_matmul_float32(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(m), C.int(k), C.int(n),
	); ok != 1 {
		return nil, errors.New("MPSGraph float32 matmul failed")
	}
	return out, nil
}
