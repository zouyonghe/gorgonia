//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

import "errors"

// SubFloat32 is unavailable without MPS support.
func SubFloat32(a, b []float32) ([]float32, error) {
	return nil, errors.New("MPSGraph float32 sub is unavailable in this build")
}

// MulFloat32 is unavailable without MPS support.
func MulFloat32(a, b []float32) ([]float32, error) {
	return nil, errors.New("MPSGraph float32 mul is unavailable in this build")
}
