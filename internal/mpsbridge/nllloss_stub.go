//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

import "errors"

// NLLLossFloat32 is unavailable without MPS support.
func NLLLossFloat32(logProbs []float32, labels []int32, rows, cols int) (float32, error) {
	return 0, errors.New("MPSGraph float32 NLL loss is unavailable in this build")
}
