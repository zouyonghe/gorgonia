//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestMPSFloat32ValueRegistry(t *testing.T) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}
	if err := metadata.init(); err != nil {
		t.Fatalf("init() error = %v", err)
	}
	defer metadata.cleanup()

	value := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4}))
	mpsValue, err := metadata.CacheFloat32Value(value)
	if err != nil {
		t.Fatalf("CacheFloat32Value() error = %v", err)
	}
	if got := metadata.MPSValueCount(); got != 1 {
		t.Fatalf("MPSValueCount() = %d, want 1", got)
	}
	secondMPSValue, err := metadata.CacheFloat32Value(value)
	if err != nil {
		t.Fatalf("second CacheFloat32Value() error = %v", err)
	}
	if secondMPSValue != mpsValue {
		t.Fatalf("second CacheFloat32Value() returned different MPS value")
	}
	if got := metadata.MPSValueCount(); got != 1 {
		t.Fatalf("MPSValueCount() after repeated cache = %d, want 1", got)
	}

	readBack, err := mpsValue.Float32s()
	if err != nil {
		t.Fatalf("Float32s() error = %v", err)
	}
	assertNodeFloat32s(t, readBack, []float32{1, 2, 3, 4})

	metadata.ReleaseMPSValue(mpsValue)
	if got := metadata.MPSValueCount(); got != 0 {
		t.Fatalf("MPSValueCount() = %d, want 0", got)
	}
}

func assertNodeFloat32s(t *testing.T, got, want []float32) {
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
