"""Gate B substrate: a *padded, mutable* mesh representation + bump allocator + the local
topology-surgery primitives that the I<->H reconnection composes.

Why a second representation (vs the compact CSR of csr_mesh.py)?
  * The CSR (offsets + flat idx) is the compact *interchange* form -- great for Gate A and
    for force/geometry kernels that only READ connectivity, but awful to MUTATE (an
    insert/drop on one surface's ring would reflow every later offset).
  * Topology surgery needs O(1) local insert/drop on a vertex's ring and on a vertex's
    incident-surface list. So the working form is cellGPU's pattern, generalised to 3D:
    **fixed-width padded rows + a per-row length**, with spare capacity for births.

Allocation model (deliberately simple + GPU-safe): births **bump** a high-water counter
(atomic on the GPU -- proven in the feasibility smoke test); deaths just set `alive=0`.
There is NO concurrent free-list (a lock-free stack pop is the one genuinely hazardous GPU
primitive); dead slots are reclaimed later by Gate-D stream-compaction. This matches
cellGPU's grow-then-compact approach. Bodies are never created/destroyed by I<->H, so only
vertices and surfaces have capacity/among the allocator.

The host-side numpy methods here ARE the reference semantics; the Gate-B/C Warp kernels
mutate these same flat arrays on the device (to_warp/from_warp).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .csr_mesh import CSRMesh


# capacity headroom defaults (one I<->H: +1 vert, +1 surface, top/bottom rings grow by 1,
# new tri-verts pick up a few incidences). Generous for single ops; scale for batches.
_V_HEADROOM = 64
_S_HEADROOM = 64
_RING_PAD = 6     # extra columns beyond the largest face
_VS_PAD = 6       # extra columns beyond the largest vertex valence
_BS_PAD = 6       # extra columns beyond the largest cell face-count


@dataclass
class PaddedMesh:
    """Fixed-width padded, mutable mesh. Rows past `*_len` (or with alive==0) are dead.

    Arrays are host numpy; `to_warp()` uploads the identical layout to the device. -1 is
    the pad/empty sentinel throughout.
    """
    cap_v: int
    cap_s: int
    nb: int
    MAX_RING: int
    MAX_VS: int
    MAX_BS: int
    # high-water counters (next free slot = n_*_used; everything >= it is untouched space)
    n_v_used: int
    n_s_used: int
    # vertices
    vert_pos: np.ndarray    # (cap_v,3) f64
    vert_alive: np.ndarray  # (cap_v,) i32
    v2s: np.ndarray         # (cap_v,MAX_VS) i32   incident surfaces (unordered), -1 pad
    v2s_len: np.ndarray     # (cap_v,) i32
    # surfaces
    surf_alive: np.ndarray  # (cap_s,) i32
    s2v: np.ndarray         # (cap_s,MAX_RING) i32 ordered ring, -1 pad
    s2v_len: np.ndarray     # (cap_s,) i32
    s2b: np.ndarray         # (cap_s,2) i32  incident bodies, -1 pad
    # bodies (stable under I<->H)
    body_alive: np.ndarray  # (nb,) i32
    b2s: np.ndarray         # (nb,MAX_BS) i32 bounding surfaces, -1 pad
    b2s_len: np.ndarray     # (nb,) i32

    # ---------------------------------------------------------------- construction ----
    @staticmethod
    def from_csr(m: CSRMesh, v_headroom: int = _V_HEADROOM, s_headroom: int = _S_HEADROOM,
                 ring_pad: int = _RING_PAD, vs_pad: int = _VS_PAD,
                 bs_pad: int = _BS_PAD) -> "PaddedMesh":
        ring = np.diff(m.s2v_off)
        val = np.diff(m.v2s_off)
        bsz = np.diff(m.b2s_off)
        MAX_RING = int(ring.max()) + ring_pad if m.ns else ring_pad
        MAX_VS = int(val.max()) + vs_pad if m.nv else vs_pad
        MAX_BS = int(bsz.max()) + bs_pad if m.nb else bs_pad
        cap_v = m.nv + v_headroom
        cap_s = m.ns + s_headroom

        vert_pos = np.zeros((cap_v, 3), np.float64)
        vert_pos[:m.nv] = m.vert_pos
        vert_alive = np.zeros(cap_v, np.int32)
        vert_alive[:m.nv] = 1
        v2s = np.full((cap_v, MAX_VS), -1, np.int32)
        v2s_len = np.zeros(cap_v, np.int32)
        for v in range(m.nv):
            row = m.v2s_idx[m.v2s_off[v]:m.v2s_off[v + 1]]
            v2s[v, :len(row)] = row
            v2s_len[v] = len(row)

        surf_alive = np.zeros(cap_s, np.int32)
        surf_alive[:m.ns] = 1
        s2v = np.full((cap_s, MAX_RING), -1, np.int32)
        s2v_len = np.zeros(cap_s, np.int32)
        for s in range(m.ns):
            row = m.s2v_idx[m.s2v_off[s]:m.s2v_off[s + 1]]
            s2v[s, :len(row)] = row
            s2v_len[s] = len(row)
        s2b = np.full((cap_s, 2), -1, np.int32)
        s2b[:m.ns] = m.s2b

        b2s = np.full((m.nb, MAX_BS), -1, np.int32)
        b2s_len = np.zeros(m.nb, np.int32)
        for b in range(m.nb):
            row = m.b2s_idx[m.b2s_off[b]:m.b2s_off[b + 1]]
            b2s[b, :len(row)] = row
            b2s_len[b] = len(row)

        return PaddedMesh(
            cap_v=cap_v, cap_s=cap_s, nb=m.nb,
            MAX_RING=MAX_RING, MAX_VS=MAX_VS, MAX_BS=MAX_BS,
            n_v_used=m.nv, n_s_used=m.ns,
            vert_pos=vert_pos, vert_alive=vert_alive, v2s=v2s, v2s_len=v2s_len,
            surf_alive=surf_alive, s2v=s2v, s2v_len=s2v_len, s2b=s2b,
            body_alive=np.ones(m.nb, np.int32), b2s=b2s, b2s_len=b2s_len,
        )

    def to_csr(self) -> CSRMesh:
        """Compact the live elements back into a CSRMesh (dead slots dropped, indices
        renumbered to 0..n-1 in slot order). With no surgery this reproduces the source
        CSR exactly; after surgery it is the canonical post-state."""
        vlive = np.where(self.vert_alive[:self.n_v_used] == 1)[0]
        slive = np.where(self.surf_alive[:self.n_s_used] == 1)[0]
        vmap = {int(v): i for i, v in enumerate(vlive)}
        smap = {int(s): i for i, s in enumerate(slive)}
        nv, ns, nb = len(vlive), len(slive), self.nb

        vert_pos = self.vert_pos[vlive].copy()

        s2v_rows = [[vmap[int(x)] for x in self.s2v[s, :self.s2v_len[s]]] for s in slive]
        s2v_off, s2v_idx = _csr(s2v_rows, ns)
        s2b = np.full((ns, 2), -1, np.int32)
        for i, s in enumerate(slive):
            for k in range(2):
                b = int(self.s2b[s, k])
                s2b[i, k] = b if b >= 0 else -1

        v2s_rows = [[smap[int(x)] for x in self.v2s[v, :self.v2s_len[v]]] for v in vlive]
        v2s_off, v2s_idx = _csr(v2s_rows, nv)
        b2s_rows = [[smap[int(x)] for x in self.b2s[b, :self.b2s_len[b]] if int(x) in smap]
                    for b in range(nb)]
        b2s_off, b2s_idx = _csr(b2s_rows, nb)

        return CSRMesh(
            nv=nv, ns=ns, nb=nb, vert_pos=vert_pos,
            vert_id=np.array(vlive, np.int64),   # slot index doubles as a stable handle
            surf_id=np.array(slive, np.int64),
            body_id=np.arange(nb, dtype=np.int64),
            s2v_off=s2v_off, s2v_idx=s2v_idx, s2b=s2b,
            v2s_off=v2s_off, v2s_idx=v2s_idx, b2s_off=b2s_off, b2s_idx=b2s_idx,
            vert_alive=np.ones(nv, bool), surf_alive=np.ones(ns, bool),
            body_alive=np.ones(nb, bool),
        )

    @staticmethod
    def from_warp(g: dict) -> "PaddedMesh":
        """Read a device SoA (as produced by to_warp, after kernels mutated it) back into
        a host PaddedMesh. The inverse of to_warp; used by the Gate-B3 test to compare the
        Warp-kernel result against the host reference."""
        nu = g["n_used"].numpy()
        cap_v, cap_s, nb = g["cap_v"], g["cap_s"], g["nb"]
        return PaddedMesh(
            cap_v=cap_v, cap_s=cap_s, nb=nb,
            MAX_RING=g["MAX_RING"], MAX_VS=g["MAX_VS"], MAX_BS=g["MAX_BS"],
            n_v_used=int(nu[0]), n_s_used=int(nu[1]),
            vert_pos=g["vert_pos"].numpy().reshape(cap_v, 3).astype(np.float64),
            vert_alive=g["vert_alive"].numpy().reshape(cap_v).astype(np.int32),
            v2s=g["v2s"].numpy().reshape(cap_v, g["MAX_VS"]).astype(np.int32),
            v2s_len=g["v2s_len"].numpy().reshape(cap_v).astype(np.int32),
            surf_alive=g["surf_alive"].numpy().reshape(cap_s).astype(np.int32),
            s2v=g["s2v"].numpy().reshape(cap_s, g["MAX_RING"]).astype(np.int32),
            s2v_len=g["s2v_len"].numpy().reshape(cap_s).astype(np.int32),
            s2b=g["s2b"].numpy().reshape(cap_s, 2).astype(np.int32),
            body_alive=g["body_alive"].numpy().reshape(nb).astype(np.int32),
            b2s=g["b2s"].numpy().reshape(nb, g["MAX_BS"]).astype(np.int32),
            b2s_len=g["b2s_len"].numpy().reshape(nb).astype(np.int32),
        )

    def to_warp(self, device=None) -> dict:
        import warp as wp
        if device is None:
            cuda = [d for d in wp.get_devices() if d.is_cuda]
            device = cuda[0] if cuda else "cpu"
        f = lambda a, dt: wp.array(np.ascontiguousarray(a), dtype=dt, device=device)
        return dict(
            device=device, cap_v=self.cap_v, cap_s=self.cap_s, nb=self.nb,
            MAX_RING=self.MAX_RING, MAX_VS=self.MAX_VS, MAX_BS=self.MAX_BS,
            n_used=f(np.array([self.n_v_used, self.n_s_used], np.int32), wp.int32),
            vert_pos=f(self.vert_pos, wp.vec3d), vert_alive=f(self.vert_alive, wp.int32),
            v2s=f(self.v2s, wp.int32), v2s_len=f(self.v2s_len, wp.int32),
            surf_alive=f(self.surf_alive, wp.int32),
            s2v=f(self.s2v, wp.int32), s2v_len=f(self.s2v_len, wp.int32),
            s2b=f(self.s2b, wp.int32),
            body_alive=f(self.body_alive, wp.int32),
            b2s=f(self.b2s, wp.int32), b2s_len=f(self.b2s_len, wp.int32),
        )

    # ---------------------------------------------------------------- allocator -------
    def alloc_vertex(self, pos) -> int:
        """Bump-allocate a fresh vertex slot (host mirror of the device atomic_add)."""
        v = self.n_v_used
        if v >= self.cap_v:
            raise RuntimeError("vertex capacity exhausted (raise v_headroom / compact)")
        self.n_v_used += 1
        self.vert_pos[v] = pos
        self.vert_alive[v] = 1
        self.v2s_len[v] = 0
        self.v2s[v, :] = -1
        return v

    def alloc_surface(self) -> int:
        s = self.n_s_used
        if s >= self.cap_s:
            raise RuntimeError("surface capacity exhausted (raise s_headroom / compact)")
        self.n_s_used += 1
        self.surf_alive[s] = 1
        self.s2v_len[s] = 0
        self.s2v[s, :] = -1
        self.s2b[s, :] = -1
        return s

    def free_vertex(self, v: int) -> None:
        self.vert_alive[v] = 0
        self.v2s_len[v] = 0
    def free_surface(self, s: int) -> None:
        self.surf_alive[s] = 0
        self.s2v_len[s] = 0

    # ------------------------------------------------ low-level row edits (one side) --
    def _ring_pos(self, s: int, v: int) -> int:
        row = self.s2v[s, :self.s2v_len[s]]
        hits = np.where(row == v)[0]
        return int(hits[0]) if len(hits) else -1

    def _ring_replace(self, s: int, old_v: int, new_v: int) -> None:
        i = self._ring_pos(s, old_v)
        if i < 0:
            raise ValueError(f"vertex {old_v} not in surface {s} ring")
        self.s2v[s, i] = new_v

    def _ring_insert_after(self, s: int, new_v: int, after_v: int) -> None:
        L = int(self.s2v_len[s])
        i = self._ring_pos(s, after_v)
        if i < 0:
            raise ValueError(f"anchor {after_v} not in surface {s} ring")
        if L + 1 > self.MAX_RING:
            raise RuntimeError("ring overflow (raise ring_pad)")
        self.s2v[s, i + 2:L + 1] = self.s2v[s, i + 1:L]   # shift right after position i
        self.s2v[s, i + 1] = new_v
        self.s2v_len[s] = L + 1

    def _ring_drop(self, s: int, v: int) -> None:
        L = int(self.s2v_len[s])
        i = self._ring_pos(s, v)
        if i < 0:
            raise ValueError(f"vertex {v} not in surface {s} ring")
        self.s2v[s, i:L - 1] = self.s2v[s, i + 1:L]
        self.s2v[s, L - 1] = -1
        self.s2v_len[s] = L - 1

    def _v2s_add(self, v: int, s: int) -> None:
        L = int(self.v2s_len[v])
        if s in self.v2s[v, :L]:
            return
        if L + 1 > self.MAX_VS:
            raise RuntimeError("v2s overflow (raise vs_pad)")
        self.v2s[v, L] = s
        self.v2s_len[v] = L + 1

    def _v2s_remove(self, v: int, s: int) -> None:
        L = int(self.v2s_len[v])
        row = self.v2s[v, :L]
        hits = np.where(row == s)[0]
        if not len(hits):
            return
        i = int(hits[0])
        self.v2s[v, i:L - 1] = self.v2s[v, i + 1:L]
        self.v2s[v, L - 1] = -1
        self.v2s_len[v] = L - 1

    # ------------------------------------------------ composite primitives ------------
    # These mirror rnr/reconnect.py's _replace_v/_insert_between/_drop_v/_attach_body/
    # _detach_body, maintaining BOTH sides of each adjacency so the surgery is a direct
    # translation of the validated CPU op.
    def replace_v(self, s: int, old_v: int, new_v: int) -> None:
        self._ring_replace(s, old_v, new_v)
        self._v2s_add(new_v, s)
        self._v2s_remove(old_v, s)

    def insert_between(self, s: int, new_v: int, v1: int, v2: int) -> None:
        """Insert new_v between ring-adjacent v1,v2 (after whichever comes first)."""
        i1, i2 = self._ring_pos(s, v1), self._ring_pos(s, v2)
        L = int(self.s2v_len[s])
        after = v1 if ((i1 + 1) % L) == i2 else v2   # the one whose successor is the other
        self._ring_insert_after(s, new_v, after)
        self._v2s_add(new_v, s)

    def drop_v(self, s: int, v: int) -> None:
        self._ring_drop(s, v)
        self._v2s_remove(v, s)

    def attach_body(self, s: int, b: int) -> None:
        k = 0 if self.s2b[s, 0] < 0 else 1
        if self.s2b[s, 0] == b or self.s2b[s, 1] == b:
            return
        self.s2b[s, k] = b
        L = int(self.b2s_len[b])
        if L + 1 > self.MAX_BS:
            raise RuntimeError("b2s overflow (raise bs_pad)")
        self.b2s[b, L] = s
        self.b2s_len[b] = L + 1

    def detach_body(self, s: int, b: int) -> None:
        for k in range(2):
            if self.s2b[s, k] == b:
                self.s2b[s, k] = -1
        L = int(self.b2s_len[b])
        row = self.b2s[b, :L]
        hits = np.where(row == s)[0]
        if len(hits):
            i = int(hits[0])
            self.b2s[b, i:L - 1] = self.b2s[b, i + 1:L]
            self.b2s[b, L - 1] = -1
            self.b2s_len[b] = L - 1

    def set_ring(self, s: int, verts) -> None:
        """Populate a fresh surface's ordered vertex ring + both-sides v2s back-pointers.

        Mirrors TF's `SurfaceType(vertices=[...])` construction (used by reconnect.py to
        build the new triangular face): the surface starts empty (alloc_surface) and this
        installs its winding in one shot. Bodies are attached separately via attach_body.
        """
        L = len(verts)
        if L > self.MAX_RING:
            raise RuntimeError("ring overflow (raise ring_pad)")
        self.s2v[s, :] = -1
        for i, v in enumerate(verts):
            self.s2v[s, i] = int(v)
        self.s2v_len[s] = L
        for v in verts:
            self._v2s_add(int(v), s)

    def ring_neighbors(self, s: int, v: int):
        """The (prev, next) ring-neighbours of v in surface s (cyclic), as int indices.

        Host mirror of topology.ring_neighbors -- the surgery uses it to locate the two
        outer vertices flanking the short-edge vertex inside a top/bottom face.
        """
        L = int(self.s2v_len[s])
        i = self._ring_pos(s, v)
        if i < 0:
            raise ValueError(f"vertex {v} not in surface {s} ring")
        return int(self.s2v[s, (i - 1) % L]), int(self.s2v[s, (i + 1) % L])

    # ------------------------------------------------ consistency check ---------------
    def check_consistency(self) -> list:
        """Both-sides adjacency must agree; rings/lengths in range. Returns problems."""
        p = []
        for s in range(self.n_s_used):
            if not self.surf_alive[s]:
                continue
            L = int(self.s2v_len[s])
            if L < 0 or L > self.MAX_RING:
                p.append(f"surf {s} bad ring len {L}")
                continue
            for v in self.s2v[s, :L]:
                v = int(v)
                if v < 0 or not self.vert_alive[v]:
                    p.append(f"surf {s} ring references dead/invalid vert {v}")
                elif s not in self.v2s[v, :self.v2s_len[v]]:
                    p.append(f"surf {s} in ring of vert {v} but not in its v2s")
        for v in range(self.n_v_used):
            if not self.vert_alive[v]:
                continue
            for s in self.v2s[v, :self.v2s_len[v]]:
                s = int(s)
                if s < 0 or not self.surf_alive[s]:
                    p.append(f"vert {v} v2s references dead/invalid surf {s}")
                elif v not in self.s2v[s, :self.s2v_len[s]]:
                    p.append(f"vert {v} lists surf {s} but not in its ring")
        for s in range(self.n_s_used):
            if not self.surf_alive[s]:
                continue
            for k in range(2):
                b = int(self.s2b[s, k])
                if b < 0:
                    continue
                if s not in self.b2s[b, :self.b2s_len[b]]:
                    p.append(f"surf {s} names body {b} but not in its b2s")
        return p


def _csr(rows, n):
    off = np.zeros(n + 1, np.int32)
    for i, r in enumerate(rows):
        off[i + 1] = off[i] + len(r)
    idx = np.empty(int(off[-1]), np.int32)
    k = 0
    for r in rows:
        for x in r:
            idx[k] = x
            k += 1
    return off, idx
