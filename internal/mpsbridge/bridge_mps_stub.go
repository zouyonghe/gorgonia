//go:build !mps || !darwin
// +build !mps !darwin

package mpsbridge

// Available reports whether the MPSGraph runtime is available.
func Available() bool { return false }

// DeviceName returns the default Metal device name.
func DeviceName() string { return "" }
