//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

import "errors"

// MatMulFloat32 is unavailable without MPS support.
func MatMulFloat32(a, b []float32, m, k, n int) ([]float32, error) {
	return nil, errors.New("MPSGraph float32 matmul is unavailable in this build")
}
