//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "testing"

func TestReLUFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	got, err := ReLUFloat32([]float32{-3, -0.5, 0, 2, 4})
	if err != nil {
		t.Fatalf("ReLUFloat32() error = %v", err)
	}
	want := []float32{0, 0, 0, 2, 4}
	assertFloat32sEqual(t, got, want)
}

func TestReLUFloat32RejectsEmptyInput(t *testing.T) {
	if _, err := ReLUFloat32(nil); err == nil {
		t.Fatal("expected empty input error")
	}
}
