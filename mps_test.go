//go:build mps
// +build mps

package gorgonia

import "testing"

func TestMPSRuntimeAvailabilitySmoke(t *testing.T) {
	var m ExternMetadata
	if err := m.init(); err != nil {
		t.Fatalf("init() error = %v", err)
	}
	if !m.MPSAvailable() {
		t.Skip("MPSGraph runtime is not available on this machine")
	}
	if name := m.MPSDeviceName(); name == "" {
		t.Fatalf("expected default Metal device name")
	} else {
		t.Logf("MPS device: %s", name)
	}
}
