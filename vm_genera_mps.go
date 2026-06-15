//go:build mps
// +build mps

package gorgonia

func (m *lispMachine) init() error {
	if err := m.prepGraph(); err != nil {
		return err
	}
	return m.ExternMetadata.init()
}

func (m *lispMachine) execDevTrans(op devTrans, n *Node, children Nodes) error {
	return nil
}

func finalizeLispMachine(m *lispMachine) { m.cleanup() }

func (m *lispMachine) ForceCPU() {}
