//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "testing"

func TestAddFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	a := []float32{1, -2, 3.5, 4}
	b := []float32{5, 6, -7.5, 8}
	got, err := AddFloat32(a, b)
	if err != nil {
		t.Fatalf("AddFloat32() error = %v", err)
	}
	want := []float32{6, 4, -4, 12}
	assertFloat32sEqual(t, got, want)
}

func TestAddFloat32RejectsMismatchedLengths(t *testing.T) {
	if _, err := AddFloat32([]float32{1}, []float32{1, 2}); err == nil {
		t.Fatal("expected mismatched length error")
	}
}
