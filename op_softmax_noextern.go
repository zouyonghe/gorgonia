//go:build !mps
// +build !mps

package gorgonia

func softmaxCallsExtern(op *softmaxOp) bool { return false }
