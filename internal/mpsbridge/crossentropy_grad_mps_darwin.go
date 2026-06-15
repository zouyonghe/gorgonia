//go:build mps && darwin
// +build mps,darwin

package mpsbridge

import "fmt"

// CrossEntropyLogitsGradFloat32 computes d(mean CE)/dlogits for row-wise logits and int32 labels.
func CrossEntropyLogitsGradFloat32(logits []float32, labels []int32, rows, cols int) ([]float32, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid cross-entropy grad shape: rows=%d cols=%d", rows, cols)
	}
	if len(logits) != rows*cols {
		return nil, fmt.Errorf("cross-entropy grad logits length %d does not match shape (%d, %d)", len(logits), rows, cols)
	}
	if len(labels) != rows {
		return nil, fmt.Errorf("cross-entropy grad label length %d does not match rows %d", len(labels), rows)
	}
	grad, err := SoftMaxRowsFloat32(logits, rows, cols)
	if err != nil {
		return nil, err
	}
	scale := float32(1) / float32(rows)
	for row, label := range labels {
		if label < 0 || int(label) >= cols {
			return nil, fmt.Errorf("cross-entropy grad label %d at row %d is out of range [0, %d)", label, row, cols)
		}
		grad[row*cols+int(label)] -= 1
	}
	for i := range grad {
		grad[i] *= scale
	}
	return grad, nil
}
