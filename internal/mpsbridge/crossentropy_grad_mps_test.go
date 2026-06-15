//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "testing"

func TestCrossEntropyLogitsGradFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	got, err := CrossEntropyLogitsGradFloat32([]float32{
		1, 2, 3,
		1, 1, 1,
	}, []int32{2, 0}, 2, 3)
	if err != nil {
		t.Fatalf("CrossEntropyLogitsGradFloat32() error = %v", err)
	}

	want := []float32{
		0.045015285, 0.12236424, -0.16737953,
		-0.33333334, 0.16666667, 0.16666667,
	}
	assertFloat32sClose(t, got, want, 1e-5)
}

func TestCrossEntropyLogitsGradFloat32RejectsBadShapes(t *testing.T) {
	if _, err := CrossEntropyLogitsGradFloat32([]float32{1, 2, 3}, []int32{0, 1}, 2, 2); err == nil {
		t.Fatal("expected bad logits shape error")
	}
	if _, err := CrossEntropyLogitsGradFloat32([]float32{1, 2, 3, 4}, []int32{0}, 2, 2); err == nil {
		t.Fatal("expected bad label length error")
	}
}
