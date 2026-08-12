# VBM artifact flow

The runtime streams every file in this table with NVFlare 2.8 `FileStreamer`;
only its opaque transfer ID, logical name, byte count, media type, stage,
direction, and SHA-256 digest are carried in a `Shareable`.

| Stage | Direction | Artifact | Historical format | Mathematical use |
| --- | --- | --- | --- | --- |
| Local mask | each site → central | `avg_nifti.nii` | NIfTI-1 | Central averages site images, thresholds the result, and establishes a common voxel space. |
| Global mask | central → all sites | `mask.nii`, `mni_downsampled.nii` | NIfTI-1 | Sites resample subject images into the shared mask and anatomical space. |
| Local model | each site → central | `mean_y_local.npy`, `count_local.npy`, `XTX.npy`, `XTy.npy` | Historical VBM text-header plus raw NumPy bytes | Central computes the weighted global mean and closed-form regularized coefficients. |
| Local model, optional | each site → central | local beta, p, t, and R² arrays | Historical VBM text-header plus raw NumPy bytes | Preserves `ShareLocalStats`; central relays these products in the final stage. |
| Global model | central → all sites | `avg_beta_vector.npy`, `mean_y_global.npy` | Historical VBM text-header plus raw NumPy bytes | Sites compute local SSE, SST, and design variance against global parameters. |
| Local metrics | each site → central | `SSE_local.npy`, `SST_local.npy`, `varX_matrix_local.npy` | Historical VBM text-header plus raw NumPy bytes | Central computes global R², MSE, t, and p arrays. |
| Global metrics | central → all sites | global R², MSE, t, and p arrays; optional per-site local-stat arrays | Historical VBM text-header plus raw NumPy bytes | Every site renders the historical NIfTI, PNG, and HTML outputs. |

Subject NIfTI files and `covariates.csv` remain at their originating site.
Every transferred product is derived, but averages, masks, sufficient
statistics, model parameters, and voxelwise statistics can still disclose
cohort or subject information. They must therefore be treated as sensitive.
`ShareLocalStats=true` expands the trust boundary: the named local maps from
each participant are sent to central and then to every participant.

For the repository's two-site fixture at `VoxelSize=4`, a source NIfTI is about
2.12 MB and therefore proves a multi-chunk transfer with the runtime's 1 MiB
default chunks. The shared mask contains 26,533 selected voxels. Representative
float64 artifacts are about 212 KB per voxel vector and 849 KB per four-column
voxel-by-covariate matrix; exact sizes depend on data, mask, and design width.
The configured defaults reject files above 512 MiB or more than 1 GiB of
artifacts for one workflow boundary.

The runtime, not computation code, owns source snapshots and destination roots.
It rejects traversal, symlinks, non-regular files, unexpected stages or peers,
duplicate manifest IDs, and size-limit violations. It writes mode-0600 files
under mode-0700 directories, verifies SHA-256 and byte count, atomically promotes
successful temporary files, retries twice after the initial attempt, times out
after 300 seconds per retrieval attempt, and removes staged files at run end.
Consortium-visible failures contain participant, task, stage, and return code;
artifact paths and subject details remain only in that participant's local log
and terminal-error marker.
