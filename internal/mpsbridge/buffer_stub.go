//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

import "errors"

// Float32Buffer owns an MPS buffer when built with MPS support.
type Float32Buffer struct{}

// NewFloat32Buffer is unavailable without MPS support.
func NewFloat32Buffer(data []float32) (*Float32Buffer, error) {
	return nil, errors.New("MPS float32 buffers are unavailable in this build")
}

// Float32s is unavailable without MPS support.
func (b *Float32Buffer) Float32s() ([]float32, error) {
	return nil, errors.New("MPS float32 buffers are unavailable in this build")
}

// Close releases the buffer.
func (b *Float32Buffer) Close() {}
