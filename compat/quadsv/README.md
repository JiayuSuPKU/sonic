# quadsv compatibility package

QuadSV has been renamed to **SONIC** (Spatial Organization through
Nonrandom-pattern Inference and Comparison).

Install the compatibility release candidate with:

```bash
pip install quadsv==1.0.0rc2
```

It installs `sonic-spatial` and preserves existing `quadsv` imports with a
deprecation warning. New code should install `sonic-spatial` and use
`import sonic`.
