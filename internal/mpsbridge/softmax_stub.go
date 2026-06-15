//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

import "errors"

// SoftMaxRowsFloat32 is unavailable without MPS support.
func SoftMaxRowsFloat32(input []float32, rows, cols int) ([]float32, error) {
	return nil, errors.New("MPSGraph float32 row softmax is unavailable in this build")
}

// LogSoftMaxRowsFloat32 is unavailable without MPS support.
func LogSoftMaxRowsFloat32(input []float32, rows, cols int) ([]float32, error) {
	return nil, errors.New("MPSGraph float32 row log-softmax is unavailable in this build")
}
