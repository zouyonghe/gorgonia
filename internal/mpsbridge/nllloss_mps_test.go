//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "testing"

func TestNLLLossFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	got, err := NLLLossFloat32([]float32{
		-2.407606, -1.407606, -0.407606,
		-1.098612, -1.098612, -1.098612,
	}, []int32{2, 0}, 2, 3)
	if err != nil {
		t.Fatalf("NLLLossFloat32() error = %v", err)
	}

	assertFloat32sClose(t, []float32{got}, []float32{0.753109}, 1e-5)
}

func TestNLLLossFloat32RejectsBadShapes(t *testing.T) {
	if _, err := NLLLossFloat32([]float32{1, 2, 3}, []int32{0, 1}, 2, 2); err == nil {
		t.Fatal("expected bad log-prob shape error")
	}
	if _, err := NLLLossFloat32([]float32{1, 2, 3, 4}, []int32{0}, 2, 2); err == nil {
		t.Fatal("expected bad label length error")
	}
}
