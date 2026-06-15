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

// AddRowBiasFloat32 computes matrix(rows x cols) + bias(cols) using MPSGraph broadcasting.
func AddRowBiasFloat32(matrix, bias []float32, rows, cols int) ([]float32, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid row bias dimensions rows=%d cols=%d", rows, cols)
	}
	if len(matrix) != rows*cols {
		return nil, fmt.Errorf("matrix has %d elements, want %d", len(matrix), rows*cols)
	}
	if len(bias) != cols {
		return nil, fmt.Errorf("bias has %d elements, want %d", len(bias), cols)
	}
	out := make([]float32, len(matrix))
	if ok := C.gorgonia_mps_add_row_bias_float32(
		(*C.float)(unsafe.Pointer(&matrix[0])),
		(*C.float)(unsafe.Pointer(&bias[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(rows),
		C.int(cols),
	); ok != 1 {
		return nil, errors.New("MPSGraph float32 row bias add failed")
	}
	return out, nil
}
