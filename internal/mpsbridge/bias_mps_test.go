//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "testing"

func TestAddRowBiasFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	matrix := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	bias := []float32{10, 20, 30}
	got, err := AddRowBiasFloat32(matrix, bias, 2, 3)
	if err != nil {
		t.Fatalf("AddRowBiasFloat32() error = %v", err)
	}
	want := []float32{
		11, 22, 33,
		14, 25, 36,
	}
	assertFloat32sEqual(t, got, want)
}

func TestAddRowBiasFloat32RejectsBadShapes(t *testing.T) {
	if _, err := AddRowBiasFloat32([]float32{1, 2}, []float32{1}, 2, 2); err == nil {
		t.Fatal("expected bad matrix length error")
	}
	if _, err := AddRowBiasFloat32([]float32{1, 2, 3, 4}, []float32{1}, 2, 2); err == nil {
		t.Fatal("expected bad bias length error")
	}
}

func TestAddRowBiasFloat32Buffer(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	matrix, err := NewFloat32Buffer([]float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("NewFloat32Buffer(matrix) error = %v", err)
	}
	defer matrix.Close()
	bias, err := NewFloat32Buffer([]float32{10, 20, 30})
	if err != nil {
		t.Fatalf("NewFloat32Buffer(bias) error = %v", err)
	}
	defer bias.Close()
	out, err := AddRowBiasFloat32Buffer(matrix, bias, 2, 3)
	if err != nil {
		t.Fatalf("AddRowBiasFloat32Buffer() error = %v", err)
	}
	defer out.Close()
	got, err := out.Float32s()
	if err != nil {
		t.Fatalf("Float32s() error = %v", err)
	}
	assertFloat32sEqual(t, got, []float32{11, 22, 33, 14, 25, 36})
}
