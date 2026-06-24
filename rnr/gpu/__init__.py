"""rnr.gpu — GPU port of the 3D vertex model + RNR (Okuda I<->H) for TissueForge.

Design study + staged plan: docs/2026-06-24_gpu-3d-vertex-model-exploration.md.

The novel target is a fully GPU-resident 3D vertex model whose *topology changes*
(reversible network reconnection) run on the GPU -- the gap left open by cellGPU (2D),
the Chaste/FLAME-GPU work (3D but topology-free), and VertAX (2D, not a dynamics engine).

Build order, each gated by a round-trip / equivalence test (mirrors the CPU-RNR
methodology in rnr/tests/test_roundtrip.py):

    Gate A  csr_mesh      index-based CSR/SoA mesh + round-trip vs TF  <-- Stage 0  [done]
    Gate B  device_mesh   bump allocator + padded mutable mesh                      [done B1]
            reconnect_csr  single count-changing I<->H round-trip (host reference)   [done B2]
            reconnect_warp single I<->H round-trip as a Warp kernel (fp64)           [done B3]
    Gate C  topology_csr   index-based [I]-config detector (no TF handles)          [done C0]
            schedule_csr   host-ref scheduler: veto + footprint + independent set    [done C1]
            schedule_warp  GPU atomic reservation (cellGPU protocol, 3D)             [done C2a]
            reconnect_warp i_to_h_batch_kernel: PARALLEL count-changing apply        [done C2b]
    Gate D  (next)         stream-compaction of dead slots
    Gate E  (next)         GPU force/integration kernels + sorting validation

Keep imports light: tissue_forge / warp are imported inside the functions that need them
so `import rnr.gpu` stays cheap.
"""
