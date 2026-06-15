//go:build mps
// +build mps

package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

func finalizeTapeMachine(m *tapeMachine) {
	m.cleanup()
	m.initFail()
}

// UseMPSFor is an option for *tapeMachine. It is currently a marker option;
// actual MPS placement is decided by ops implementing MPSDoer.
func UseMPSFor(ops ...string) VMOpt { return func(m VM) {} }

// UseCudaFor is a no-op in MPS builds.
func UseCudaFor(ops ...string) VMOpt { return func(m VM) {} }

func (m *tapeMachine) init() {
	var initMPS bool
	for _, instr := range m.p.instructions {
		if eo, ok := instr.(*execOp); ok {
			if _, ok := eo.op.(MPSDoer); ok {
				initMPS = true
				break
			}
		}
	}
	if !initMPS {
		return
	}
	if err := m.ExternMetadata.init(); err != nil {
		m.ExternMetadata.initFail()
		panic(err)
	}
}

func (m *tapeMachine) getEngine(dev Device) tensor.Engine { return m.Engine }

func (m *tapeMachine) MPSMetadata() *ExternMetadata { return &m.ExternMetadata }

func (instr *execOp) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v. Node is: %x", instr, instr.id)
	m.enterLogScope()
	defer m.leaveLogScope()

	m.watchedLogf("Inputs:")
	m.enterLogScope()
	var inputs []Value
	for _, reg := range instr.readFrom {
		v := m.getValue(reg)
		inputs = append(inputs, v)
		m.watchedLogf(m.valueFmt, v)
	}
	m.leaveLogScope()

	var v Value
	toDev := instr.writeTo.device
	if op, ok := instr.op.(MPSDoer); ok && toDev != CPU && instr.op.CallsExtern() {
		prealloc := m.getValue(instr.writeTo)
		if v, err = op.MPSDo(m, toDev, prealloc, inputs...); err != nil {
			return errors.Wrapf(err, "Happened while attempting to use MPS to execute %v. Node is %x. Register was %v", instr, instr.id, instr.writeTo.id)
		}
	} else {
		v, err = execCPUOp(instr, m, inputs)
		if err != nil {
			return err
		}
	}

	m.watchedLogf("Result:")
	m.enterLogScope()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLogScope()

	setEngine(v, m.getEngine(toDev))
	m.writeValue(instr.writeTo, v)
	node := m.p.g.Node(instr.id).(*Node)
	if m.trace() && (len(m.watchNodes) == 0 || m.watchNodes.Contains(node)) {
		if err = node.bindCopy(v); err != nil {
			return errors.Wrapf(err, "TraceExec failed to bind copy")
		}
		if node.op == (Iop{}) {
			v = node.Value()
			m.writeValue(instr.writeTo, v)
		}
	} else {
		node.bind(v)
	}

	if m.bindDV() && node.derivOf != nil {
		for _, src := range node.derivOf {
			if len(m.bindNodesDV) > 0 && !m.bindNodesDV.Contains(src) {
				continue
			}

			switch {
			case node.op == (Iop{}):
				closure := func() error {
					dv := dvUnit(src.boundTo)
					add := newEBOByType(addOpType, TypeOf(dv.d), TypeOf(v))
					if _, err := add.UnsafeDo(dv.d, v); err != nil {
						return err
					}
					return nil
				}
				m.closureQueue = append(m.closureQueue, closure)
			default:
				dv := dvUnit(src.boundTo)
				add := newEBOByType(addOpType, TypeOf(dv.d), TypeOf(v))
				if d, err := add.UnsafeDo(dv.d, v); err == nil {
					dv.SetDeriv(d)
					src.bind(dv)
				} else {
					return err
				}
			}
		}
	}

	m.watchedLogf("Written To: %v", instr.writeTo)
	m.enterLogScope()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLogScope()
	return nil
}

func execCPUOp(instr *execOp, m *tapeMachine, inputs []Value) (Value, error) {
	var v Value
	var err error
	var usePrealloc bool
	if m.cpumem[instr.writeTo.id] != nil {
		usePrealloc = true
	}
	switch {
	case instr.preAllocated:
		if pd, ok := instr.op.(UsePreallocDoer); ok {
			p := m.cpumem[instr.writeTo.id]
			if v, err = pd.UsePreallocDo(p, inputs...); err != nil {
				return nil, errors.Wrapf(err, "Happened while attempting to execute %v. Node is %x. Register was: %v ", instr, instr.id, instr.writeTo.id)
			}
		} else {
			if v, err = instr.op.Do(inputs...); err != nil {
				return nil, errors.Wrap(err, opDoFail)
			}
		}
	case usePrealloc:
		if pd, ok := instr.op.(UsePreallocDoer); ok {
			p := m.cpumem[instr.writeTo.id]
			if v, err = pd.UsePreallocDo(p, inputs...); err != nil {
				if v, err = instr.op.Do(inputs...); err != nil {
					return nil, errors.Wrap(err, opDoFail)
				}
			}
		} else {
			if v, err = instr.op.Do(inputs...); err != nil {
				return nil, errors.Wrap(err, opDoFail)
			}
		}
	case instr.useUnsafe:
		if ud, ok := instr.op.(UnsafeDoer); ok {
			if v, err = ud.UnsafeDo(inputs...); err != nil {
				return nil, errors.Wrap(err, "Failed to carry UnsafeDo()")
			}
		} else {
			if v, err = instr.op.Do(inputs...); err != nil {
				return nil, errors.Wrap(err, opDoFail)
			}
		}
	default:
		if v, err = instr.op.Do(inputs...); err != nil {
			return nil, errors.Wrap(err, opDoFail)
		}
	}
	return v, nil
}

func (instr deviceTransport) exec(m *tapeMachine) error {
	from := m.getValue(instr.from)
	if instr.from.device == instr.to.device {
		m.writeValue(instr.to, from)
		return nil
	}
	v, err := m.ExternMetadata.Transfer(instr.to.device, instr.from.device, from, true)
	if err != nil {
		return err
	}
	m.writeValue(instr.to, v)
	return nil
}
