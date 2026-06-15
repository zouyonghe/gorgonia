//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "testing"

func TestMatMulFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	a := []float32{
		1, 2,
		3, 4,
	}
	b := []float32{
		5, 6,
		7, 8,
	}
	got, err := MatMulFloat32(a, b, 2, 2, 2)
	if err != nil {
		t.Fatalf("MatMulFloat32() error = %v", err)
	}
	want := []float32{
		19, 22,
		43, 50,
	}
	assertFloat32sEqual(t, got, want)
}

func TestMatMulFloat32RejectsBadShapes(t *testing.T) {
	if _, err := MatMulFloat32([]float32{1, 2}, []float32{3, 4}, 2, 2, 2); err == nil {
		t.Fatal("expected bad shape error")
	}
}

func TestMatMulFloat32Buffers(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	a, err := NewFloat32Buffer([]float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("NewFloat32Buffer(a) error = %v", err)
	}
	defer a.Close()
	b, err := NewFloat32Buffer([]float32{5, 6, 7, 8})
	if err != nil {
		t.Fatalf("NewFloat32Buffer(b) error = %v", err)
	}
	defer b.Close()

	out, err := MatMulFloat32Buffers(a, b, 2, 2, 2)
	if err != nil {
		t.Fatalf("MatMulFloat32Buffers() error = %v", err)
	}
	defer out.Close()
	got, err := out.Float32s()
	if err != nil {
		t.Fatalf("Float32s() error = %v", err)
	}
	assertFloat32sEqual(t, got, []float32{19, 22, 43, 50})
}

func assertFloat32sEqual(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v; got=%v", i, got[i], want[i], got)
		}
	}
}
