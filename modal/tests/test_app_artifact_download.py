from pathlib import Path
import tempfile
import unittest

import app


class ArtifactDownloadDestinationTest(unittest.TestCase):
    def test_existing_artifact_directory_downloads_into_fresh_recreated_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts_root = Path(tmp)
            local_dest = artifacts_root / 'existing-job'
            stale_file = local_dest / 'stale.txt'
            local_dest.mkdir()
            stale_file.write_text('stale', encoding='utf-8')

            resolved_dest = app._prepare_artifact_download_destination(artifacts_root, 'existing-job')

            self.assertEqual(resolved_dest, local_dest)
            self.assertFalse(resolved_dest.exists())
            self.assertFalse(stale_file.exists())


if __name__ == '__main__':
    unittest.main()
