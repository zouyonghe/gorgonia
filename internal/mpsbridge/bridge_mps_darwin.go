//go:build mps && darwin
// +build mps,darwin

package mpsbridge

/*
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
#include "bridge.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// Available reports whether Metal and MPSGraph can create their default runtime objects.
func Available() bool { return C.gorgonia_mps_available() == 1 }

// DeviceName returns the default Metal device name.
func DeviceName() string {
	name := C.gorgonia_mps_device_name()
	if name == nil {
		return ""
	}
	defer C.free(unsafe.Pointer(name))
	return C.GoString(name)
}
