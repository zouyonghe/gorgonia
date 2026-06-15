//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

import "errors"

// CrossEntropyLogitsGradFloat32 is unavailable without MPS support.
func CrossEntropyLogitsGradFloat32(logits []float32, labels []int32, rows, cols int) ([]float32, error) {
	return nil, errors.New("MPSGraph float32 cross-entropy logits gradient is unavailable in this build")
}
