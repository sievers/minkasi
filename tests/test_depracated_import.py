import importlib
import os
import sys
import unittest

MODULE = "minkasi.tods._depracated"


class TestDeprecatedImport(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.pop("MINKASI_USE_DEPRECATED", None)
        self.addCleanup(self._restore_env)
        sys.modules.pop(MODULE, None)

    def _restore_env(self):
        if self._saved_env is None:
            os.environ.pop("MINKASI_USE_DEPRECATED", None)
        else:
            os.environ["MINKASI_USE_DEPRECATED"] = self._saved_env
        sys.modules.pop(MODULE, None)

    def _fresh_import(self):
        # Drop any cached copy so the module body (and its imports) runs again.
        sys.modules.pop(MODULE, None)
        return importlib.import_module(MODULE)

    def test_deprecated_module_imports_without_distutils(self):
        # distutils was removed in Python 3.12; the module must import without it.
        mod = self._fresh_import()
        self.assertFalse(mod.use_dep)

    def test_env_var_true_enables_deprecated(self):
        os.environ["MINKASI_USE_DEPRECATED"] = "True"
        self.assertTrue(self._fresh_import().use_dep)

    def test_env_var_invalid_falls_back_to_false(self):
        os.environ["MINKASI_USE_DEPRECATED"] = "maybe"
        self.assertFalse(self._fresh_import().use_dep)

    def test_strtobool_accepted_values(self):
        mod = self._fresh_import()
        for val in ("y", "yes", "t", "true", "on", "1", "TRUE", "On"):
            self.assertTrue(mod._strtobool(val), val)
        for val in ("n", "no", "f", "false", "off", "0", "FALSE"):
            self.assertFalse(mod._strtobool(val), val)

    def test_strtobool_rejects_invalid(self):
        mod = self._fresh_import()
        with self.assertRaises(ValueError):
            mod._strtobool("maybe")


if __name__ == "__main__":
    unittest.main()
