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

// SoftMaxRowsFloat32 computes row-wise softmax using MPSGraph.
func SoftMaxRowsFloat32(input []float32, rows, cols int) ([]float32, error) {
	return softMaxRowsFloat32(input, rows, cols, false)
}

// LogSoftMaxRowsFloat32 computes row-wise log-softmax using MPSGraph.
func LogSoftMaxRowsFloat32(input []float32, rows, cols int) ([]float32, error) {
	return softMaxRowsFloat32(input, rows, cols, true)
}

func softMaxRowsFloat32(input []float32, rows, cols int, logOutput bool) ([]float32, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid softmax shape: rows=%d cols=%d", rows, cols)
	}
	if len(input) != rows*cols {
		return nil, fmt.Errorf("softmax input length %d does not match shape (%d, %d)", len(input), rows, cols)
	}
	out := make([]float32, len(input))
	if ok := C.gorgonia_mps_softmax_rows_float32(
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(rows),
		C.int(cols),
		C.int(boolToInt(logOutput)),
	); ok != 1 {
		if logOutput {
			return nil, errors.New("MPSGraph float32 row log-softmax failed")
		}
		return nil, errors.New("MPSGraph float32 row softmax failed")
	}
	return out, nil
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
