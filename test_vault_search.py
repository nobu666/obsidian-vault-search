#!/usr/bin/env python3
"""Tests for vault-search. Stdlib only (unittest); no Ollama required — embed() is stubbed.

Run: python3 test_vault_search.py
Covers the parts selfcheck doesn't: incremental indexing (no dup chunks, mtime skip,
pruning deleted files), keyword matching, and hybrid search fusion.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path


def load_module(vault_dir, db_path):
    """Load the hyphenated, extension-less `vault-search` as a module with env wired up.

    VAULT and DB_PATH are read at import time, so env must be set before exec_module.
    """
    os.environ["VAULT_DIR"] = str(vault_dir)
    os.environ["VAULT_SEARCH_DB"] = str(db_path)
    # The script has no .py suffix, so name a SourceFileLoader explicitly.
    loader = importlib.machinery.SourceFileLoader(
        "vaultsearch", str(Path(__file__).resolve().parent / "vault-search"))
    spec = importlib.util.spec_from_loader("vaultsearch", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    # Deterministic 3-dim fake embeddings (no Ollama): flags for 'alpha' / 'beta'.
    m.embed = lambda texts: [
        [1.0 if "alpha" in t else 0.0, 1.0 if "beta" in t else 0.0, 0.1] for t in texts
    ]
    return m


def write(path, text, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


class VaultSearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name) / "vault"
        self.db = Path(self.tmp.name) / "index.db"
        self.vault.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_index(self, m, **kw):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            m.cmd_index(**kw)
        return out.getvalue()

    def count_chunks(self, m, path=None):
        con = m.db_connect()
        if path:
            return con.execute("SELECT COUNT(*) FROM chunks WHERE path=?", (str(path),)).fetchone()[0]
        return con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def test_index_incremental_and_prune(self):
        a = self.vault / "a.md"
        b = self.vault / "b.md"
        write(a, "# A\nalpha content\n", mtime=1000)
        write(b, "# B\nbeta content\n", mtime=1000)
        m = load_module(self.vault, self.db)

        self.assertIn("2 files updated", self.run_index(m))
        self.assertEqual(self.count_chunks(m), 2)

        # Re-index with no changes -> nothing updated, no duplicates.
        self.assertIn("0 files updated", self.run_index(m))
        self.assertEqual(self.count_chunks(m), 2)

        # Modify a.md (two headings now) and bump mtime -> only a re-embeds, chunks replaced not duplicated.
        write(a, "# A\nalpha one\n\n## A2\nalpha two\n", mtime=2000)
        self.assertIn("1 files updated", self.run_index(m))
        self.assertEqual(self.count_chunks(m, a), 2)   # replaced, not 1+2
        self.assertEqual(self.count_chunks(m), 3)

        # Delete b.md -> its chunks and file row are pruned.
        b.unlink()
        out = self.run_index(m)
        self.assertIn("1 removed", out)
        self.assertEqual(self.count_chunks(m, b), 0)

    def test_keyword_hits(self):
        write(self.vault / "k.md", "# Topic\nthe quick brown fox\n", mtime=1000)
        m = load_module(self.vault, self.db)
        self.run_index(m)
        hits = m.keyword_hits("quick fox")          # two distinct terms present
        self.assertEqual(len(hits), 1)
        self.assertEqual(list(hits.values())[0], 2)
        self.assertEqual(m.keyword_hits("absent zzzz"), {})

    def test_search_fusion_ranks_relevant_first(self):
        write(self.vault / "a.md", "# A\nalpha alpha alpha\n", mtime=1000)
        write(self.vault / "b.md", "# B\nbeta beta beta\n", mtime=1000)
        m = load_module(self.vault, self.db)
        self.run_index(m)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            m.cmd_search("alpha", k=2)
        lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
        self.assertTrue(lines, "search returned no output")
        self.assertTrue(lines[0].startswith("a.md"), f"expected a.md first, got: {lines[0]}")
        self.assertIn("kw", lines[0])               # 'alpha' is also a keyword hit -> emb+kw

    def test_dimension_mismatch_is_fatal(self):
        write(self.vault / "a.md", "# A\nalpha\n", mtime=1000)
        m = load_module(self.vault, self.db)
        self.run_index(m)
        # Query vector of a different dimension must not silently miscompute.
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                m.cosine_topn([1.0, 0.0], 5)        # index is 3-dim, query is 2-dim

    def test_snippet_strips_control_and_bidi(self):
        m = load_module(self.vault, self.db)
        s = m.make_snippet("alpha\x1b[31mRED\x1b[0m" + chr(0x202e) + "beta")
        self.assertNotIn("\x1b", s)            # ANSI escape removed
        self.assertNotIn(chr(0x202e), s)       # bidi override removed
        self.assertIn("alpha", s)              # visible text preserved


if __name__ == "__main__":
    unittest.main(verbosity=2)
