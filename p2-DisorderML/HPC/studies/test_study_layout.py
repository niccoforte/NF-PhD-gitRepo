"""Fast entry-point and real B1 staging checks; no research data or training."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


STUDIES = Path(__file__).resolve().parent
HPC = STUDIES.parent


class StudyLayoutTests(unittest.TestCase):
    def test_named_entry_points_preserve_common_runner(self):
        for name, initial, expected in (
            ("Independence", ["--independence-case", "1i"], ["--experiment", "baseline"]),
            ("Input", [], ["--experiment", "baseline", "--independence-case", "1i", "--followup-study", "inputs"]),
            ("Loss", [], ["--experiment", "baseline", "--independence-case", "1i", "--followup-study", "losses"]),
        ):
            module = runpy.run_path(str(STUDIES/name/f"A0-{name}.py"))
            with patch("runpy.run_path", return_value={"main": lambda args: args}) as routed:
                self.assertEqual(module["main"](initial), initial+expected)
                self.assertEqual(Path(routed.call_args.args[0]), HPC/"DualOutputs/A0-HPC-Dual-test.py")

    def test_independence_requires_explicit_case(self):
        with self.assertRaisesRegex(SystemExit, "explicit"):
            runpy.run_path(str(STUDIES/"Independence/A0-Independence.py"))["main"]([])

    def test_b1_stages_study_relative_layout(self):
        bash = shutil.which("bash")
        if sys.platform == "win32":
            candidate = Path("C:/Program Files/Git/bin/bash.exe")
            bash = str(candidate) if candidate.exists() else bash
        if not bash:
            self.skipTest("Bash unavailable")
        text = (HPC/"B1_ML-new.sh").read_text(encoding="utf-8")
        block = text[text.index("# Copy the framework"):text.index('cd "$SCRATCH_DIR"')]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root/"repo"
            scripts = repo/"p2-DisorderML/HPC"
            (repo/"resources").mkdir(parents=True)
            (repo/"resources/proof.txt").write_text("library", encoding="utf-8")
            (scripts/"DualOutputs").mkdir(parents=True)
            (scripts/"DualOutputs/A0-HPC-Dual-test.py").write_text(
                "import json\ndef main(args):\n    print(json.dumps(args))\n", encoding="utf-8")
            (scripts/"studies/Locality").mkdir(parents=True)
            (scripts/"studies/Locality/A0-Locality.py").write_text("# shared final generator\n")
            for name in ("Input", "Loss", "Independence", "FieldVariations", "SpatialBalance"):
                study = "Loss" if name == "SpatialBalance" else name
                directory = scripts/"studies"/study
                directory.mkdir(parents=True, exist_ok=True)
                source = directory/f"A0-{name}.py"
                shutil.copyfile(STUDIES/study/source.name, source)
                scratch = root/f"scratch-{name}"
                scratch.mkdir()
                variables = {
                    "REPO_ROOT": repo, "ML_CODE_DIR": scripts, "SCRIPT_SRC": source,
                    "SCRATCH_DIR": scratch, "SCRIPT_LOCAL": scratch/source.name,
                }
                setup = "\n".join(f"{key}={json.dumps(value.as_posix())}" for key, value in variables.items())
                # Mirror rsync's file/directory copy semantics without HPC or an rsync installation.
                mock = 'rsync() { shift; local src=$1 dst=$2; if [[ -d "$src" ]]; then mkdir -p "$dst"; cp -R "$src". "$dst"; else cp "$src" "$dst"; fi; }\n'
                subprocess.run([bash, "-c", "set -eu\n"+setup+"\n"+mock+block], check=True, capture_output=True, text=True)
                staged = scratch/"studies"/study/source.name
                self.assertTrue(staged.exists())
                self.assertTrue((scratch/"resources/proof.txt").exists())
                if name in ("FieldVariations", "SpatialBalance"):
                    self.assertTrue((scratch/"studies/Locality/A0-Locality.py").is_file())
                    continue
                args = ["--independence-case", "1i"] if name == "Independence" else []
                output = subprocess.run([sys.executable, str(staged), *args], check=True, capture_output=True, text=True)
                self.assertIn("--experiment", json.loads(output.stdout))

    def test_notebook_folder_has_no_study_python(self):
        # The maintained general figure-export regression is not a study CLI.
        self.assertEqual([p.name for p in (HPC.parent/"code").glob("*.py")
                          if p.name!="test_figure_exports.py"], [])

    def test_production_property_name(self):
        self.assertTrue((HPC/"FieldToProperty/A0-HPC_FieldToProperty.py").is_file())
        self.assertFalse((HPC/"FieldToProperty/A0-HPC_FieldToProperty-test.py").exists())


if __name__ == "__main__":
    unittest.main()
