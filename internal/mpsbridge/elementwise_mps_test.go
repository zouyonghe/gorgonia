//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "testing"

func TestSubFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}
	got, err := SubFloat32([]float32{5, 6, -7.5, 8}, []float32{1, -2, 3.5, 4})
	if err != nil {
		t.Fatalf("SubFloat32() error = %v", err)
	}
	assertFloat32sEqual(t, got, []float32{4, 8, -11, 4})
}

func TestMulFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}
	got, err := MulFloat32([]float32{1, -2, 3.5, 4}, []float32{5, 6, -7.5, 8})
	if err != nil {
		t.Fatalf("MulFloat32() error = %v", err)
	}
	assertFloat32sEqual(t, got, []float32{5, -12, -26.25, 32})
}

func TestElementwiseFloat32RejectsMismatchedLengths(t *testing.T) {
	if _, err := SubFloat32([]float32{1}, []float32{1, 2}); err == nil {
		t.Fatal("expected SubFloat32 mismatched length error")
	}
	if _, err := MulFloat32([]float32{1}, []float32{1, 2}); err == nil {
		t.Fatal("expected MulFloat32 mismatched length error")
	}
}
