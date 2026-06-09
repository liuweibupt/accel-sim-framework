from pathlib import Path
import unittest

import app


class AppCopyIgnoreTest(unittest.TestCase):
    def test_ignores_local_cmake_build_directories(self) -> None:
        self.assertTrue(app._ignore_modal_copy_path(Path('cutlass_runner/build/CMakeCache.txt')))
        self.assertTrue(app._ignore_modal_copy_path(Path('deepseek_v4_runner/build/kernel.o')))

    def test_keeps_runner_sources(self) -> None:
        self.assertFalse(app._ignore_modal_copy_path(Path('cutlass_runner/main.cu')))
        self.assertFalse(app._ignore_modal_copy_path(Path('cublaslt_runner/CMakeLists.txt')))


if __name__ == '__main__':
    unittest.main()
