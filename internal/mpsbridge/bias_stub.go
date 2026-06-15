//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

import "errors"

// AddRowBiasFloat32 is unavailable without MPS support.
func AddRowBiasFloat32(matrix, bias []float32, rows, cols int) ([]float32, error) {
	return nil, errors.New("MPSGraph float32 row bias add is unavailable in this build")
}
