//go:build mps
// +build mps

package gorgonia

import (
	"errors"
	"reflect"
	"sync"
	"unsafe"

	"gorgonia.org/gorgonia/internal/mpsbridge"
	"gorgonia.org/tensor"
)

// CUDA indicates whether this build is using CUDA. MPS builds are not CUDA
// builds, but keep the constant for existing callers that check CUDA support.
const CUDA = false

// MPS indicates whether this build includes the Apple Metal Performance Shaders
// external-device scaffolding.
const MPS = true

var _ External = &ExternMetadata{}

// MPSMachine is implemented by VMs that can host Apple MPS execution.
type MPSMachine interface {
	External
	MPSAvailable() bool
}

// ExternMetadata holds metadata for Apple MPS execution.
//
// This is currently a backend skeleton: it wires Gorgonia's device analysis and
// VM execution paths to a future MPSGraph/Metal implementation without changing
// CPU or CUDA behavior. Device memory and op execution return explicit errors
// until concrete MPS tensor storage and kernels are added.
type ExternMetadata struct {
	tensor.Engine
	sync.Mutex

	b             batchedBLAS
	workAvailable chan bool
	syncChan      chan struct{}
	initialized   bool
	available     bool
	deviceName    string
	memories      map[uintptr]mpsHostMemory
	values        map[*MPSFloat32Value]struct{}
}

type mpsHostMemory struct {
	data []byte
}

// MPSFloat32Value owns a Metal buffer for a float32 tensor value.
// It is the first step toward GPU-resident values; current ops still read back
// CPU tensors after execution, but this type gives the backend explicit buffer
// lifetime management independent of host tensor lifetimes.
type MPSFloat32Value struct {
	buffer *mpsbridge.Float32Buffer
	shape  tensor.Shape
	closed bool
}

// Shape returns the tensor shape represented by this MPS value.
func (v *MPSFloat32Value) Shape() tensor.Shape {
	if v == nil || v.shape == nil {
		return nil
	}
	return v.shape.Clone()
}

// Float32s copies this MPS value back to host memory.
func (v *MPSFloat32Value) Float32s() ([]float32, error) {
	if v == nil || v.closed || v.buffer == nil {
		return nil, errors.New("MPS float32 value is closed")
	}
	return v.buffer.Float32s()
}

// Close releases this MPS value's Metal buffer.
func (v *MPSFloat32Value) Close() {
	if v == nil || v.closed {
		return
	}
	v.closed = true
	if v.buffer != nil {
		v.buffer.Close()
		v.buffer = nil
	}
}

func (m mpsHostMemory) Uintptr() uintptr { return uintptr(unsafe.Pointer(&m.data[0])) }
func (m mpsHostMemory) MemSize() uintptr { return uintptr(len(m.data)) }

func bytesFromUintptr(ptr uintptr, size uintptr) []byte {
	var ret []byte
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&ret))
	hdr.Data = ptr
	hdr.Len = int(size)
	hdr.Cap = int(size)
	return ret
}

func (m *ExternMetadata) init() error {
	m.Lock()
	defer m.Unlock()
	if m.initialized {
		return nil
	}
	m.syncChan = make(chan struct{})
	if m.b != nil {
		m.workAvailable = make(chan bool)
		go m.collectBLASWork()
	}
	m.available = mpsbridge.Available()
	m.deviceName = mpsbridge.DeviceName()
	m.memories = make(map[uintptr]mpsHostMemory)
	m.values = make(map[*MPSFloat32Value]struct{})
	m.initialized = true
	return nil
}

func (m *ExternMetadata) initFail() { m.cleanup() }

func (m *ExternMetadata) cleanup() {
	m.Lock()
	defer m.Unlock()
	for value := range m.values {
		value.Close()
	}
	m.values = nil
	m.memories = nil
	m.initialized = false
}

// CacheFloat32Value copies a float32 dense tensor into a Metal buffer and tracks its lifetime.
func (m *ExternMetadata) CacheFloat32Value(value *tensor.Dense) (*MPSFloat32Value, error) {
	if value == nil {
		return nil, errors.New("cannot cache nil tensor as MPS value")
	}
	if value.Dtype() != tensor.Float32 {
		return nil, errors.New("MPS value cache supports float32 tensors only")
	}
	data, ok := value.Data().([]float32)
	if !ok {
		return nil, errors.New("MPS value cache expected []float32 backing")
	}
	buffer, err := mpsbridge.NewFloat32Buffer(data)
	if err != nil {
		return nil, err
	}
	mpsValue := &MPSFloat32Value{buffer: buffer, shape: value.Shape().Clone()}
	m.Lock()
	if m.values == nil {
		m.values = make(map[*MPSFloat32Value]struct{})
	}
	m.values[mpsValue] = struct{}{}
	m.Unlock()
	return mpsValue, nil
}

// ReleaseMPSValue releases a cached MPS value and removes it from the registry.
func (m *ExternMetadata) ReleaseMPSValue(value *MPSFloat32Value) {
	if value == nil {
		return
	}
	m.Lock()
	delete(m.values, value)
	m.Unlock()
	value.Close()
}

// MPSValueCount returns the number of tracked MPS values.
func (m *ExternMetadata) MPSValueCount() int {
	m.Lock()
	defer m.Unlock()
	return len(m.values)
}

func cacheMPSFloat32Value(extern External, value *tensor.Dense) {
	metadata := mpsMetadataFromExternal(extern)
	if metadata == nil || value == nil || value.Dtype() != tensor.Float32 {
		return
	}
	_, _ = metadata.CacheFloat32Value(value)
}

type mpsMetadataProvider interface {
	MPSMetadata() *ExternMetadata
}

func mpsMetadataFromExternal(extern External) *ExternMetadata {
	switch e := extern.(type) {
	case nil:
		return nil
	case *ExternMetadata:
		return e
	case mpsMetadataProvider:
		return e.MPSMetadata()
	default:
		return nil
	}
}

func (m *ExternMetadata) MPSMetadata() *ExternMetadata { return m }

// HasFunc reports whether an external function has been loaded.
func (m ExternMetadata) HasFunc(name string) bool { return false }

// MPSAvailable reports whether a concrete MPS runtime has been initialized.
func (m *ExternMetadata) MPSAvailable() bool { return mpsbridge.Available() }

// MPSDeviceName returns the default Metal device name, if available.
func (m *ExternMetadata) MPSDeviceName() string { return mpsbridge.DeviceName() }

// WorkAvailable returns a channel used to flush batched external work.
func (m *ExternMetadata) WorkAvailable() <-chan bool {
	if m.b != nil {
		return m.workAvailable
	}
	return nil
}

// Sync returns the synchronization channel for external work.
func (m *ExternMetadata) Sync() chan struct{} { return m.syncChan }

// DoWork flushes any queued external work.
func (m *ExternMetadata) DoWork() error {
	if m.b != nil {
		m.b.DoWork()
	}
	return nil
}

// Get allocates MPS device memory. Not implemented until MPS tensor storage lands.
func (m *ExternMetadata) Get(dev Device, size int64) (tensor.Memory, error) {
	if size <= 0 {
		return nil, noopError{}
	}
	mem := mpsHostMemory{data: make([]byte, size)}
	m.Lock()
	if m.memories == nil {
		m.memories = make(map[uintptr]mpsHostMemory)
	}
	m.memories[mem.Uintptr()] = mem
	m.Unlock()
	return mem, nil
}

// GetFromValue copies a host value to MPS device memory.
func (m *ExternMetadata) GetFromValue(dev Device, v Value) (tensor.Memory, error) {
	memsize := int64(v.MemSize())
	mem, err := m.Get(dev, memsize)
	if err != nil {
		return nil, err
	}
	copy(bytesFromUintptr(mem.Uintptr(), uintptr(memsize)), bytesFromUintptr(v.Uintptr(), uintptr(memsize)))
	return mem, nil
}

// Put releases MPS device memory.
func (m *ExternMetadata) Put(dev Device, mem tensor.Memory, size int64) {
	if mem == nil {
		return
	}
	m.Lock()
	delete(m.memories, mem.Uintptr())
	m.Unlock()
}

// PutValue releases memory associated with a value.
func (m *ExternMetadata) PutValue(dev Device, v Value) {}

// Transfer moves values between CPU and MPS devices.
func (m *ExternMetadata) Transfer(toDev, fromDev Device, v Value, synchronous bool) (Value, error) {
	if toDev == fromDev || toDev == CPU && fromDev == CPU {
		return v, nil
	}
	if toDev != CPU && fromDev == CPU {
		mem, err := m.GetFromValue(toDev, v)
		if err != nil {
			return nil, err
		}
		return makeValueFromMem(TypeOf(v), v.Shape(), mem)
	}
	if toDev == CPU && fromDev != CPU {
		retVal, err := makeValue(TypeOf(v), v.Shape())
		if err != nil {
			return nil, err
		}
		copy(bytesFromUintptr(retVal.Uintptr(), v.MemSize()), bytesFromUintptr(v.Uintptr(), v.MemSize()))
		return retVal, nil
	}
	return v, nil
}

// Reset clears external allocator state.
func (m *ExternMetadata) Reset() {
	m.Lock()
	defer m.Unlock()
	for value := range m.values {
		value.Close()
	}
	m.memories = make(map[uintptr]mpsHostMemory)
	m.values = make(map[*MPSFloat32Value]struct{})
}

// Cleanup cleans up ancillary allocations made during external execution.
func (m *ExternMetadata) Cleanup() { m.cleanup() }

// Signal flushes synchronous external work.
func (m *ExternMetadata) Signal() {
	m.signal()
	if m.workAvailable != nil {
		<-m.syncChan
	}
}

func (m *ExternMetadata) collectBLASWork() {
	if m.b != nil {
		for range m.b.WorkAvailable() {
			m.workAvailable <- false
		}
	}
}

func (m *ExternMetadata) signal() {
	if m.workAvailable != nil {
		m.workAvailable <- true
	}
}

func (m *ExternMetadata) setEngine(e tensor.Engine) { m.Engine = e }

// ValueOnDevice gets the node value on the requested device.
func (n *Node) ValueOnDevice(dev Device, extern External) (Value, bool, error) {
	if dev == CPU {
		return n.Value(), false, nil
	}
	mem, err := extern.GetFromValue(dev, n.Value())
	if err != nil {
		return nil, false, err
	}
	retVal, err := makeValueFromMem(TypeOf(n.Value()), n.Shape(), mem)
	return retVal, true, err
}

// GradOnDevice gets the node gradient on the requested device.
func (n *Node) GradOnDevice(dev Device, extern External) (retVal Value, allocOnExtern bool, err error) {
	if dev == CPU {
		retVal, err = n.Grad()
		return retVal, false, err
	}
	grad, err := n.Grad()
	if err != nil {
		return nil, false, err
	}
	mem, err := extern.GetFromValue(dev, grad)
	if err != nil {
		return nil, false, err
	}
	retVal, err = makeValueFromMem(TypeOf(grad), grad.Shape(), mem)
	return retVal, true, err
}
