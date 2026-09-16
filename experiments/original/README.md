# Original LiteDSPS-Mamba R5 experiment source

This directory preserves the exact original R5 direction-shared experiment source supplied for the LiteDSPS-Mamba paper.

The source is stored as seven split ZIP parts because binary payloads are transferred through the repository connector in bounded chunks:

```text
LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip.part01
...
LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip.part07
```

Reassemble and verify it with:

```bash
python experiments/original/reassemble.py
```

The reconstructed ZIP must have SHA-256:

```text
36df017861e61bbb3600591227ca6b6ac04dd72d90d426b42b05f15b05096dbb
```

It contains the exact uploaded source file:

```text
LiteDSPS_Mamba_R5_DirShare_Deterministic_Final(20260916-004726).py
```

The raw Python source has SHA-256:

```text
efc05bb4613bb27a2a5bf1c1f09707fd9bc8934bfdf02fefbd18c443a812ee48
```

The original evidence construction computes `abs(x[1:] - x[:-1])` as a length-`T-1` sequence and independently applies adaptive average/max pooling to the same effective evidence count `K_eff` used by the raw-feature statistics. The modular `litedsps/evidence.py` implementation is aligned to this behavior.
