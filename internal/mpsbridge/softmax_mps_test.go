//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import (
	"math"
	"testing"
)

func TestSoftMaxRowsFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	got, err := SoftMaxRowsFloat32([]float32{
		1, 2, 3,
		1, 1, 1,
	}, 2, 3)
	if err != nil {
		t.Fatalf("SoftMaxRowsFloat32() error = %v", err)
	}

	want := []float32{
		0.09003057, 0.24472848, 0.66524094,
		0.33333334, 0.33333334, 0.33333334,
	}
	assertFloat32sClose(t, got, want, 1e-5)
}

func TestLogSoftMaxRowsFloat32(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	got, err := LogSoftMaxRowsFloat32([]float32{
		1, 2, 3,
		1, 1, 1,
	}, 2, 3)
	if err != nil {
		t.Fatalf("LogSoftMaxRowsFloat32() error = %v", err)
	}

	want := []float32{
		-2.407606, -1.407606, -0.407606,
		-1.0986123, -1.0986123, -1.0986123,
	}
	assertFloat32sClose(t, got, want, 1e-5)
}

func TestSoftMaxRowsFloat32RejectsBadShapes(t *testing.T) {
	if _, err := SoftMaxRowsFloat32([]float32{1, 2, 3}, 2, 2); err == nil {
		t.Fatal("expected bad shape error")
	}
	if _, err := LogSoftMaxRowsFloat32([]float32{1, 2, 3}, 2, 2); err == nil {
		t.Fatal("expected bad shape error")
	}
}

func TestSoftMaxRowsFloat32Buffer(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	input, err := NewFloat32Buffer([]float32{1, 2, 3, 1, 1, 1})
	if err != nil {
		t.Fatalf("NewFloat32Buffer() error = %v", err)
	}
	defer input.Close()
	out, err := SoftMaxRowsFloat32Buffer(input, 2, 3)
	if err != nil {
		t.Fatalf("SoftMaxRowsFloat32Buffer() error = %v", err)
	}
	defer out.Close()
	got, err := out.Float32s()
	if err != nil {
		t.Fatalf("Float32s() error = %v", err)
	}
	assertFloat32sClose(t, got, []float32{0.09003057, 0.24472848, 0.66524094, 0.33333334, 0.33333334, 0.33333334}, 1e-5)
}

func TestLogSoftMaxRowsFloat32Buffer(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	input, err := NewFloat32Buffer([]float32{1, 2, 3, 1, 1, 1})
	if err != nil {
		t.Fatalf("NewFloat32Buffer() error = %v", err)
	}
	defer input.Close()
	out, err := LogSoftMaxRowsFloat32Buffer(input, 2, 3)
	if err != nil {
		t.Fatalf("LogSoftMaxRowsFloat32Buffer() error = %v", err)
	}
	defer out.Close()
	got, err := out.Float32s()
	if err != nil {
		t.Fatalf("Float32s() error = %v", err)
	}
	assertFloat32sClose(t, got, []float32{-2.407606, -1.407606, -0.407606, -1.0986123, -1.0986123, -1.0986123}, 1e-5)
}

func assertFloat32sClose(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("got[%d] = %v, want %v; got=%v", i, got[i], want[i], got)
		}
	}
}
