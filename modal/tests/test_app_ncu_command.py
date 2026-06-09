import unittest

import app


class NcuCommandTest(unittest.TestCase):
    def test_cutlass_command_uses_existing_runner_and_metrics(self) -> None:
        cmd, binary = app._build_ncu_command('fp16', 512, 512, 512, 'cutlass')
        self.assertEqual(binary.name, 'cutlass_runner')
        self.assertIn('--csv', cmd)
        self.assertIn('dram__bytes_read.sum', ','.join(cmd))
        self.assertEqual(cmd[-9:], [str(binary), '--dtype', 'fp16', '--m', '512', '--n', '512', '--k', '512'])

    def test_cublaslt_command_uses_existing_runner(self) -> None:
        cmd, binary = app._build_ncu_command('bf16', 2048, 12288, 12288, 'cublaslt')
        self.assertEqual(binary.name, 'cublaslt_runner')
        self.assertEqual(cmd[-9:], [str(binary), '--dtype', 'bf16', '--m', '2048', '--n', '12288', '--k', '12288'])

    def test_rejects_runner_without_ncu_binary(self) -> None:
        with self.assertRaises(ValueError):
            app._build_ncu_command('fp16', 1, 1, 1, 'agent_kv')


if __name__ == '__main__':
    unittest.main()

class ModalResourceTest(unittest.TestCase):
    def test_a100_disk_request_matches_modal_lower_bound(self) -> None:
        self.assertGreaterEqual(app._A100_EPHEMERAL_DISK_MIB, 524288)

class NcuFailureReportingTest(unittest.TestCase):
    def test_ncu_failure_includes_stderr_tail(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            stderr_path = Path(tmp) / 'ncu.stderr'
            stderr_path.write_text('first line\nlast line\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'last line'):
                app._raise_for_ncu_failure(9, stderr_path)

    def test_ncu_success_does_not_raise(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            app._raise_for_ncu_failure(0, Path(tmp) / 'ncu.stderr')

class GpuDiagnosticsCommandTest(unittest.TestCase):
    def test_gpu_diagnostic_command_contains_profiler_checks(self) -> None:
        command = app._build_gpu_diagnostic_command()
        joined = ' '.join(command)
        self.assertIn('nvidia-smi', joined)
        self.assertIn('ncu --version', joined)
        self.assertIn('ldconfig -p', joined)
