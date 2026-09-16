# Original LiteDSPS-Mamba R5 experiment source

This directory preserves the exact original R5 direction-shared experiment source supplied for the LiteDSPS-Mamba paper.

- Archive: `LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip`
- Contained source: `LiteDSPS_Mamba_R5_DirShare_Deterministic_Final(20260916-004726).py`
- Raw source SHA-256: `efc05bb4613bb27a2a5bf1c1f09707fd9bc8934bfdf02fefbd18c443a812ee48`

The original evidence construction computes `abs(x[1:] - x[:-1])` as a length-`T-1` sequence and independently applies adaptive average/max pooling to the same effective evidence count `K_eff` used by the raw-feature statistics. The modular `litedsps/evidence.py` implementation is aligned to this behavior.
