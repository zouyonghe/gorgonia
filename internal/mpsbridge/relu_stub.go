//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

import "errors"

// ReLUFloat32 is unavailable without MPS support.
func ReLUFloat32(input []float32) ([]float32, error) {
	return nil, errors.New("MPSGraph float32 ReLU is unavailable in this build")
}
