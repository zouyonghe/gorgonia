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

func TestElementwiseFloat32Buffers(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	a, err := NewFloat32Buffer([]float32{1, -2, 3.5, 4})
	if err != nil {
		t.Fatalf("NewFloat32Buffer(a) error = %v", err)
	}
	defer a.Close()
	b, err := NewFloat32Buffer([]float32{5, 6, -7.5, 8})
	if err != nil {
		t.Fatalf("NewFloat32Buffer(b) error = %v", err)
	}
	defer b.Close()

	add, err := AddFloat32Buffers(a, b)
	if err != nil {
		t.Fatalf("AddFloat32Buffers() error = %v", err)
	}
	defer add.Close()
	addGot, err := add.Float32s()
	if err != nil {
		t.Fatalf("add.Float32s() error = %v", err)
	}
	assertFloat32sEqual(t, addGot, []float32{6, 4, -4, 12})

	sub, err := SubFloat32Buffers(a, b)
	if err != nil {
		t.Fatalf("SubFloat32Buffers() error = %v", err)
	}
	defer sub.Close()
	subGot, err := sub.Float32s()
	if err != nil {
		t.Fatalf("sub.Float32s() error = %v", err)
	}
	assertFloat32sEqual(t, subGot, []float32{-4, -8, 11, -4})

	mul, err := MulFloat32Buffers(a, b)
	if err != nil {
		t.Fatalf("MulFloat32Buffers() error = %v", err)
	}
	defer mul.Close()
	mulGot, err := mul.Float32s()
	if err != nil {
		t.Fatalf("mul.Float32s() error = %v", err)
	}
	assertFloat32sEqual(t, mulGot, []float32{5, -12, -26.25, 32})
}
