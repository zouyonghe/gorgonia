//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "testing"

func TestFloat32BufferRoundTrip(t *testing.T) {
	if !Available() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}

	in := []float32{1.25, -2.5, 3.75, 4}
	buf, err := NewFloat32Buffer(in)
	if err != nil {
		t.Fatalf("NewFloat32Buffer() error = %v", err)
	}
	defer buf.Close()

	out, err := buf.Float32s()
	if err != nil {
		t.Fatalf("Float32s() error = %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("len(out) = %d, want %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("out[%d] = %v, want %v", i, out[i], in[i])
		}
	}
}
