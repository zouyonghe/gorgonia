//go:build mps && darwin
// +build mps,darwin

package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func BenchmarkMPSMatMulFloat32(b *testing.B) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		b.Skip("MPSGraph runtime is not available on this machine")
	}

	left := tensor.New(tensor.WithShape(64, 128), tensor.WithBacking(repeatFloat32(64*128, 0.01)))
	right := tensor.New(tensor.WithShape(128, 64), tensor.WithBacking(repeatFloat32(128*64, 0.02)))

	b.Run("Graph", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			g := NewGraph()
			x := NewMatrix(g, tensor.Float32, WithShape(64, 128), WithValue(left), WithName("x"))
			w := NewMatrix(g, tensor.Float32, WithShape(128, 64), WithValue(right), WithName("w"))
			if _, err := Mul(x, w); err != nil {
				b.Fatal(err)
			}
			m := NewTapeMachine(g)
			if err := m.RunAll(); err != nil {
				b.Fatal(err)
			}
			m.Close()
		}
	})
}

func BenchmarkMPSSoftMaxFloat32(b *testing.B) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		b.Skip("MPSGraph runtime is not available on this machine")
	}

	input := tensor.New(tensor.WithShape(128, 10), tensor.WithBacking(repeatFloat32(128*10, 0.1)))

	b.Run("TensorCPU", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			op := newSoftmaxOp(input.Shape())
			if _, err := op.Do(input); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("GraphMPS", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			g := NewGraph()
			x := NewMatrix(g, tensor.Float32, WithShape(128, 10), WithValue(input), WithName("x"))
			if _, err := SoftMax(x); err != nil {
				b.Fatal(err)
			}
			m := NewTapeMachine(g)
			if err := m.RunAll(); err != nil {
				b.Fatal(err)
			}
			m.Close()
		}
	})
}

func BenchmarkMPSMLPForwardFloat32(b *testing.B) {
	var metadata ExternMetadata
	if !metadata.MPSAvailable() {
		b.Skip("MPSGraph runtime is not available on this machine")
	}

	xData := tensor.New(tensor.WithShape(32, 784), tensor.WithBacking(repeatFloat32(32*784, 0.01)))
	w1Data := tensor.New(tensor.WithShape(784, 128), tensor.WithBacking(repeatFloat32(784*128, 0.001)))
	b1Data := tensor.New(tensor.WithShape(128), tensor.WithBacking(repeatFloat32(128, 0.01)))
	w2Data := tensor.New(tensor.WithShape(128, 10), tensor.WithBacking(repeatFloat32(128*10, 0.001)))
	b2Data := tensor.New(tensor.WithShape(10), tensor.WithBacking(repeatFloat32(10, 0.01)))

	b.Run("MPS", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			g := NewGraph()
			x := NewMatrix(g, tensor.Float32, WithShape(32, 784), WithValue(xData), WithName("x"))
			w1 := NewMatrix(g, tensor.Float32, WithShape(784, 128), WithValue(w1Data), WithName("w1"))
			b1 := NewVector(g, tensor.Float32, WithShape(128), WithValue(b1Data), WithName("b1"))
			w2 := NewMatrix(g, tensor.Float32, WithShape(128, 10), WithValue(w2Data), WithName("w2"))
			b2 := NewVector(g, tensor.Float32, WithShape(10), WithValue(b2Data), WithName("b2"))

			l1, err := Mul(x, w1)
			if err != nil {
				b.Fatal(err)
			}
			l1Bias, err := BroadcastAdd(l1, b1, nil, []byte{0})
			if err != nil {
				b.Fatal(err)
			}
			hidden, err := Rectify(l1Bias)
			if err != nil {
				b.Fatal(err)
			}
			logits, err := Mul(hidden, w2)
			if err != nil {
				b.Fatal(err)
			}
			if _, err := BroadcastAdd(logits, b2, nil, []byte{0}); err != nil {
				b.Fatal(err)
			}
			m := NewTapeMachine(g)
			if err := m.RunAll(); err != nil {
				b.Fatal(err)
			}
			m.Close()
		}
	})
}

func repeatFloat32(size int, value float32) []float32 {
	data := make([]float32, size)
	for i := range data {
		data[i] = value
	}
	return data
}
