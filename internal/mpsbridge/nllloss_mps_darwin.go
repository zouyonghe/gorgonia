//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "fmt"

// NLLLossFloat32 computes mean negative log likelihood for row-wise log-probs.
func NLLLossFloat32(logProbs []float32, labels []int32, rows, cols int) (float32, error) {
	if rows <= 0 || cols <= 0 {
		return 0, fmt.Errorf("invalid NLL loss shape: rows=%d cols=%d", rows, cols)
	}
	if len(logProbs) != rows*cols {
		return 0, fmt.Errorf("NLL loss log-prob length %d does not match shape (%d, %d)", len(logProbs), rows, cols)
	}
	if len(labels) != rows {
		return 0, fmt.Errorf("NLL loss label length %d does not match rows %d", len(labels), rows)
	}

	var sum float32
	for row, label := range labels {
		if label < 0 || int(label) >= cols {
			return 0, fmt.Errorf("NLL loss label %d at row %d is out of range [0, %d)", label, row, cols)
		}
		sum -= logProbs[row*cols+int(label)]
	}
	return sum / float32(rows), nil
}

// NLLLossFloat32Buffer computes mean negative log likelihood from a cached float32 MTLBuffer.
// Labels are still consumed from host memory; this is an incremental residency step.
func NLLLossFloat32Buffer(logProbs *Float32Buffer, labels []int32, rows, cols int) (float32, error) {
	if rows <= 0 || cols <= 0 {
		return 0, fmt.Errorf("invalid NLL loss shape: rows=%d cols=%d", rows, cols)
	}
	if logProbs == nil || logProbs.ptr == nil {
		return 0, fmt.Errorf("NLL loss log-prob MPS buffer is closed")
	}
	if logProbs.count != rows*cols {
		return 0, fmt.Errorf("NLL loss log-prob buffer length %d does not match shape (%d, %d)", logProbs.count, rows, cols)
	}
	data, err := logProbs.Float32s()
	if err != nil {
		return 0, err
	}
	return NLLLossFloat32(data, labels, rows, cols)
}
