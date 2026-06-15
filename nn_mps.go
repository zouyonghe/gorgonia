//go:build mps
// +build mps

package gorgonia

func maybeMPSRectify(x *Node) (*Node, bool, error) {
	dt, err := dtypeOf(x.t)
	if err != nil {
		return nil, false, err
	}
	if dt != Float32 || x.Shape().Dims() == 0 {
		return nil, false, nil
	}
	retVal, err := MPSReLU(x)
	return retVal, true, err
}
