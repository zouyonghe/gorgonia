//go:build !mps
// +build !mps

package gorgonia

func tryMPSBroadcastAdd(a, b *Node, leftPattern, rightPattern []byte) (*Node, bool, error) {
	return nil, false, nil
}
