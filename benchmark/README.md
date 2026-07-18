# CleanBench

**CleanBench** is the repository's one canonical human-labeled data-quality
benchmark.

The local artifact is built at:

```text
benchmark/CleanBench/
```

Benchmark records, source tables, caches, and ZIP files are intentionally
ignored by Git. Only this documentation and `CleanBench/README.md` are tracked.

Build or refresh the benchmark from the current workbench labels with:

```bash
python -m scripts.export_cleanbench
```

See [`CleanBench/README.md`](CleanBench/README.md) for the file layout, schemas,
scoring contract, provenance, source pairing, and rebuild options.
