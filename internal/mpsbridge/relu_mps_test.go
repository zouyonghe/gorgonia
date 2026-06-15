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

func TestReLUFloat32Buffer(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	input, err := NewFloat32Buffer([]float32{-3, -0.5, 0, 2, 4})
	if err != nil {
		t.Fatalf("NewFloat32Buffer() error = %v", err)
	}
	defer input.Close()
	out, err := ReLUFloat32Buffer(input)
	if err != nil {
		t.Fatalf("ReLUFloat32Buffer() error = %v", err)
	}
	defer out.Close()
	got, err := out.Float32s()
	if err != nil {
		t.Fatalf("Float32s() error = %v", err)
	}
	assertFloat32sEqual(t, got, []float32{0, 0, 0, 2, 4})
}
