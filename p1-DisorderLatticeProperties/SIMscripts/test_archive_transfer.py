"""Offline full-archive transfer contracts; no SSH, PuTTY session or research data."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("B3_ABAQUS-archive.sh")
archive = types.ModuleType("archive_transfer")
code = SCRIPT.read_text(encoding="utf-8").split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
exec(compile(code, str(SCRIPT), "exec"), archive.__dict__)


class ArchiveTransfer(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="archive-transfer-test-")
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name) / "copy with spaces"
        self.entries = {"Ti": ["d", 0, "1.0"], "Ti/empty": ["d", 0, "1.0"],
                        "Ti/.hidden": ["f", 4, "1.0"],
                        "Ti/zero.txt": ["f", 0, "1.0"],
                        "Ti/sample [1].inp": ["f", 6, "1.0"],
                        "Al": ["d", 0, "1.0"], "Al/model.odb": ["f", 8, "1.0"]}
        self.arguments = ["--destination", str(self.destination)]
        self.batch = None
        self.clients = patch.object(archive, "putty_client", side_effect=lambda name: name)
        self.clients.start()
        self.addCleanup(self.clients.stop)

    def client(self, command, **kwargs):
        if "-b" in command:
            self.batch = Path(command[command.index("-b") + 1]).read_text(encoding="utf-8")
            for line in self.batch.splitlines():
                if line == "quit":
                    continue
                operation, remote, _, local, _ = line.split('"')
                path = Path(local)
                self.assertEqual(local, path.as_posix())
                name = path.relative_to(self.destination).as_posix()
                size = self.entries[name][1]
                if operation.strip() == "reget":
                    # Match the real client: resuming a nonexistent file is an error.
                    self.assertTrue(path.is_file(), "PSFTP reget needs an existing local file")
                    with path.open("ab") as target:
                        target.write(b"x" * (size - path.stat().st_size))
                else:
                    self.assertEqual(operation.strip(), "get")
                    self.assertFalse(path.exists() and path.stat().st_size,
                                     "get must not overwrite an existing partial prefix")
                    path.write_bytes(b"x" * size)
        return types.SimpleNamespace(poll=lambda: 0, wait=lambda: 0)

    def invoke(self, *extra, snapshot=None, client=None):
        with patch.object(archive, "inventory", return_value=snapshot or self.entries), \
                patch.object(archive.subprocess, "run", side_effect=client or self.client), \
                patch.object(archive.subprocess, "Popen", side_effect=client or self.client), \
                redirect_stdout(io.StringIO()):
            archive.main(self.arguments + list(extra))

    def test_dry_run_has_no_connections_or_writes(self):
        with patch.object(archive.subprocess, "run") as run, \
                patch.object(archive.subprocess, "check_output") as read, \
                redirect_stdout(io.StringIO()):
            archive.main(self.arguments + ["--dry-run"])
        run.assert_not_called()
        read.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_full_tree_hidden_empty_and_no_extra_p1_parent(self):
        self.invoke()
        self.assertTrue((self.destination / "Ti/empty").is_dir())
        self.assertEqual((self.destination / "Ti/.hidden").read_bytes(), b"xxxx")
        self.assertFalse((self.destination / "p1").exists())
        self.assertFalse(self.destination.with_name(self.destination.name + ".resume.json").exists())
        self.assertIn("/p1/Al/model.odb", self.batch)
        self.assertIn("sample \\[1\\].inp", self.batch)
        self.assertTrue(all(line.startswith("get ") or line == "quit" for line in self.batch.splitlines()))
        self.assertTrue((self.destination / "Ti/zero.txt").is_file())

    def interrupt(self, command, **kwargs):
        if "-b" in command:
            (self.destination / "Ti/.hidden").write_bytes(b"ab")
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    def test_failure_keeps_partial_and_resume_marker(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(client=self.interrupt)
        self.assertEqual((self.destination / "Ti/.hidden").stat().st_size, 2)
        self.assertTrue(self.destination.with_name(self.destination.name + ".resume.json").is_file())
        self.invoke()
        self.assertEqual((self.destination / "Ti/.hidden").read_bytes(), b"abxx")
        self.assertIn('reget "', self.batch)
        self.assertIn('get "', self.batch)

    def test_completed_file_is_not_downloaded_again(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(client=self.interrupt)
        (self.destination / "Al/model.odb").write_bytes(b"complete")
        self.invoke()
        self.assertNotIn("/p1/Al/model.odb", self.batch)
        self.assertEqual((self.destination / "Al/model.odb").read_bytes(), b"complete")

    def test_progress_measures_new_bytes_and_file_completion(self):
        path = Path(self.temporary.name) / "partial"
        path.write_bytes(b"ab")
        process = types.SimpleNamespace(poll=lambda: 0, wait=lambda: 0)
        output = io.StringIO()
        with patch.object(archive.subprocess, "Popen", return_value=process), redirect_stdout(output):
            archive.download(["psftp"], [(path, 4, 2)], 2)
        self.assertIn("0.00%", output.getvalue())
        self.assertIn("0/1 files complete", output.getvalue())
        path.write_bytes(b"abxx")
        output = io.StringIO()
        with patch.object(archive.subprocess, "Popen", return_value=process), redirect_stdout(output):
            archive.download(["psftp"], [(path, 4, 2)], 2)
        self.assertIn("100.00%", output.getvalue())
        self.assertIn("1/1 files complete", output.getvalue())

    def test_interrupt_terminates_only_transfer_client(self):
        process = unittest.mock.Mock()
        process.poll.return_value = None
        path = Path(self.temporary.name) / "partial"
        path.write_bytes(b"ab")
        with patch.object(archive.subprocess, "Popen", return_value=process), \
                patch.object(archive.time, "sleep", side_effect=KeyboardInterrupt), \
                redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            archive.download(["psftp"], [(path, 4, 2)], 2)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        self.assertEqual(path.read_bytes(), b"ab")

    def test_changed_source_refuses_old_partial_prefixes(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(client=self.interrupt)
        changed = copy.deepcopy(self.entries)
        changed["Ti/.hidden"][2] = "2.0"
        with self.assertRaisesRegex(RuntimeError, "Source inventory changed"):
            self.invoke(snapshot=changed)
        self.assertEqual((self.destination / "Ti/.hidden").stat().st_size, 2)

    def test_nonempty_foreign_destination_is_not_merged(self):
        self.destination.mkdir()
        (self.destination / "Al").mkdir()
        with self.assertRaisesRegex(RuntimeError, "Nonempty destination"):
            self.invoke()

    def test_check_only_does_not_download_or_recreate_marker(self):
        self.invoke()
        with patch.object(archive, "inventory", return_value=self.entries), \
                patch.object(archive.subprocess, "run") as run, redirect_stdout(io.StringIO()):
            archive.main(self.arguments + ["--check-only"])
        self.assertEqual(run.call_count, 1)  # sharing check only, never PSFTP
        (self.destination / "Ti/.hidden").write_bytes(b"x")
        with self.assertRaisesRegex(RuntimeError, "Incomplete copy"):
            self.invoke("--check-only")

    def test_unexpected_or_oversized_local_file_is_not_deleted(self):
        self.destination.mkdir()
        extra = self.destination / "unrelated.txt"
        extra.write_text("retain me")
        with self.assertRaisesRegex(RuntimeError, "Unexpected local entry"):
            self.invoke()
        self.assertEqual(extra.read_text(), "retain me")
        extra.unlink()
        (self.destination / "Al").mkdir()
        (self.destination / "Al/model.odb").write_bytes(b"x" * 9)
        with self.assertRaisesRegex(RuntimeError, "larger than its source"):
            self.invoke()

    def test_git_destination_is_rejected_even_in_dry_run(self):
        (Path(self.temporary.name) / ".git").mkdir()
        with self.assertRaisesRegex(RuntimeError, "outside every Git checkout"):
            self.invoke("--dry-run")

    def test_inventory_rejects_unsupported_or_colliding_names(self):
        for raw in (b"l\t1\t1\tlink\0", b"f\t1\t1\tCON.txt\0", b"f\t1\t1\t../escape\0",
                    b"f\t1\t1\tbad:name\0", b"f\t1\t1\tA\0f\t1\t1\ta\0"):
            with self.subTest(raw=raw), patch.object(archive.subprocess, "check_output", return_value=raw):
                with self.assertRaisesRegex(RuntimeError, "Cannot mirror"):
                    archive.inventory(["mock-plink"])

    def test_inventory_accepts_spaces_dotfiles_and_directories(self):
        raw = b"d\t0\t1\t\0d\t0\t1\tTi\0f\t4\t2\tTi/.hidden file\0"
        with patch.object(archive.subprocess, "check_output", return_value=raw):
            entries = archive.inventory(["mock-plink"])
        self.assertEqual(entries["Ti/.hidden file"], ["f", 4, "2"])

    def test_source_change_during_copy_keeps_resume_state_and_fails(self):
        changed = copy.deepcopy(self.entries)
        changed["Al/model.odb"][2] = "2.0"
        with patch.object(archive, "inventory", side_effect=[self.entries, changed]), \
                patch.object(archive.subprocess, "run", side_effect=self.client), \
                patch.object(archive.subprocess, "Popen", side_effect=self.client), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "remote archive changed during transfer"):
                archive.main(self.arguments)
        self.assertTrue(self.destination.with_name(self.destination.name + ".resume.json").exists())

    def test_insufficient_space_stops_before_creating_destination(self):
        with patch.object(archive.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient free space"):
                self.invoke()
        self.assertFalse(self.destination.exists())

    def test_missing_empty_directory_fails_final_verification(self):
        def remove_empty_directory(command, **kwargs):
            result = self.client(command, **kwargs)
            if "-b" in command:
                (self.destination / "Ti/empty").rmdir()
            return result

        with self.assertRaisesRegex(RuntimeError, "absent or incomplete"):
            self.invoke(client=remove_empty_directory)


if __name__ == "__main__":
    unittest.main()
