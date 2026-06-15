//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

import "errors"

// AddFloat32 is unavailable without MPS support.
func AddFloat32(a, b []float32) ([]float32, error) {
	return nil, errors.New("MPSGraph float32 add is unavailable in this build")
}
