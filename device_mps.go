//go:build mps
// +build mps

package gorgonia

import "gorgonia.org/tensor"

// Device represents an execution device. MPS builds keep CPU as the zero value
// because registers throughout Gorgonia rely on the zero value being host
// execution. Apple GPU devices start at 1.
type Device int

const (
	// CPU is the default host execution device.
	CPU Device = 0
)

// AppleGPU returns the MPS device for the given index.
func AppleGPU(index int) Device { return Device(index + 1) }

// String implements fmt.Stringer and runtime.Stringer.
func (d Device) String() string {
	if d == CPU {
		return "CPU"
	}
	return "MPS"
}

// IsGPU reports whether the device is an Apple GPU device.
func (d Device) IsGPU() bool { return d != CPU }

// Alloc allocates memory on an MPS device.
func (d Device) Alloc(extern External, size int64) (tensor.Memory, error) {
	if d == CPU {
		return nil, nil
	}
	return extern.Get(d, size)
}

// Free releases memory associated with an MPS device.
func (d Device) Free(extern External, mem tensor.Memory, size int64) error {
	if d == CPU {
		return nil
	}
	extern.Put(d, mem, size)
	return nil
}
