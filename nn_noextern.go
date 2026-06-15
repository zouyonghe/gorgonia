//go:build !mps
// +build !mps

package gorgonia

func maybeMPSRectify(x *Node) (*Node, bool, error) { return nil, false, nil }
