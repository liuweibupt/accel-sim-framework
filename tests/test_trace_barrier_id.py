from pathlib import Path


def test_trace_barrier_uses_trace_immediate_not_pc_hash():
    source = Path('gpu-simulator/trace-driven/trace_driven.cc').read_text()
    op_bar_start = source.index('case OP_BAR:')
    op_bar_end = source.index('case OP_LDGDEPBAR:', op_bar_start)
    op_bar_block = source[op_bar_start:op_bar_end]

    assert 'trace.m_pc' not in op_bar_block
    assert 'bar_id = trace.imm' in op_bar_block
