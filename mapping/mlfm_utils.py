from __future__ import annotations
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, List, Tuple
import spacy
from spacy.matcher import Matcher
from dataclasses import dataclass
import cv2
import numpy as np
import torch
import re


def _global_remap(labels, top_k=None, seed=0):
    """
    Map original labels (any ints, -1=noise) -> [0..N], where 0= noise.
    If top_k is set, keep K largest clusters globally; others -> noise.
    Returns remapped_labels (same shape), ListedColormap.
    """
    lab = labels.ravel()
    mask = lab != -1
    vals, counts = np.unique(lab[mask], return_counts=True)

    if top_k is not None and top_k < len(vals):
        keep = set(vals[np.argsort(-counts)[:top_k]].tolist())
        lab2 = lab.copy()
        lab2[mask & ~np.isin(lab, list(keep))] = -1
        lab = lab2
        # recompute uniques on kept
        mask = lab != -1
        vals = np.unique(lab[mask])

    # build mapping: noise->0, clusters->1..N
    remap = {int(v): i+1 for i, v in enumerate(vals)}
    remapped = np.zeros_like(lab, dtype=np.int32)
    m = lab != -1
    # vectorized remap via searchsorted
    keys = np.array(list(remap.keys()), dtype=lab.dtype)
    order = np.argsort(keys)
    idx = np.searchsorted(keys[order], lab[m])
    remapped[m] = np.array(list(remap.values()))[order][idx]

    remapped = remapped.reshape(labels.shape)

    # deterministic random colors; index 0 reserved for noise (black)
    rng = np.random.default_rng(seed)
    n = int(remapped.max()) + 1
    colors = np.zeros((n, 4), dtype=float)
    colors[0] = (0, 0, 0, 1)  # noise
    if n > 1:
        colors[1:] = np.c_[rng.random((n-1, 3)), np.ones(n-1)]
    cmap = ListedColormap(colors)
    return remapped, cmap

def _boundaries2d(lab2d):
    # True where a pixel touches a different label (4-neighborhood)
    up    = lab2d != np.roll(lab2d,  1, axis=0)
    down  = lab2d != np.roll(lab2d, -1, axis=0)
    left  = lab2d != np.roll(lab2d,  1, axis=1)
    right = lab2d != np.roll(lab2d, -1, axis=1)
    b = up | down | left | right
    b[0,:] = b[-1,:] = b[:,0] = b[:,-1] = True
    return b

# ---------- main entry points ----------

def save_clusters_per_slice(
    labels, out_dir,
    top_k=None, seed=0, with_outlines=True,
    dpi=300, prefix="clusters"
):
    """
    Save one PNG per z-slice.
    labels: (H, W, L) int array; -1 = noise
    """
    H, W, L = labels.shape
    os.makedirs(out_dir, exist_ok=True)

    # global color mapping so the same cluster id has the same color across slices
    mapped, cmap = _global_remap(labels, top_k=top_k, seed=seed)

    for z in range(L):
        lab2d = mapped[:, :, z]
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        ax = plt.gca()
        im = ax.imshow(lab2d, cmap=cmap, vmin=0, vmax=mapped.max(),
                       interpolation='nearest')
        ax.set_axis_off()

        if with_outlines:
            bd = _boundaries2d(labels[:, :, z])  # outlines from original ids
            ax.contour(bd.astype(float), levels=[0.5], linewidths=0.5, colors='white')

        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(out_dir, f"{prefix}_z{z}.png"),
                    dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

def hsv_lut(n=101, s=0.70, v=0.95):
    # Evenly spaced hues → bright distinct colors
    h = np.linspace(0, 1, n, endpoint=False)
    c = np.stack([h, np.full(n, s), np.full(n, v)], axis=1)  # HSV
    # HSV → RGB
    import colorsys
    rgb = np.array([colorsys.hsv_to_rgb(*t) for t in c], dtype=float)  # in [0,1]
    return rgb  # (n,3) floats

def save_clusters_grid_figure(
    labels, out_path, cols=None,
    top_k=None, seed=0, with_outlines=True, dpi=200
):
    """
    Make a single tiled figure with all z-slices.
    """
    H, W, L = labels.shape
    cols = cols or min(L, 3)
    rows = int(np.ceil(L / cols))
    cmap = ListedColormap(hsv_lut(101))

    fig_w = 4 * cols
    fig_h = 4 * rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    for z in range(L):
        ax = plt.subplot(rows, cols, z+1)
        ax.imshow(labels[:,:,z], cmap=cmap, interpolation='nearest')
        # img = cv2.applyColorMap((labels[:,:,z] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        # ax.imshow(img)
        # ax.imshow(mapped[:, :, z], cmap=cmap, vmin=0, vmax=mapped.max(),
        #           interpolation='nearest')
        # if with_outlines:
        #     bd = _boundaries2d(labels[:, :, z])
        #     ax.contour(bd.astype(float), levels=[0.5], linewidths=0.5, colors='white')
        ax.set_title(f"z={z}")
        ax.set_axis_off()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_ab_dominance_map(
    sim2_hw: torch.tensor,
    out_path: str = "ab_map.png",
    top_percent: float = 1.0,
    per_channel_minmax: bool = True,
) -> None:
    """
    Create a colored image from a (2, H, W, l) similarity map.
    - Channel 0 (A) -> red
    - Channel 1 (B) -> blue
    - Intensity proportional to normalized scores in the winning channel
    - Optionally highlight only the top X% pixels per object via `top_percent`.

    Args:
        sim2_hw: np.ndarray of shape (2, H, W, l), cosine similarities for A and B.
        out_path: where to save the image (PNG recommended).
        top_percent: if set (e.g., 0.05), only top X% highest-score pixels per object are colored.
        per_channel_minmax: normalize each channel by its own min/max (True), or use global min/max (False).
    """
    assert sim2_hw.ndim == 4 and sim2_hw.shape[0] == 2, "Expected shape (2, H, W, l)."
    A = torch.max(sim2_hw[0], axis=-1)[0].cpu().numpy().astype(np.float32)
    B = torch.max(sim2_hw[1], axis=-1)[0].cpu().numpy().astype(np.float32)

    # --- Normalize to [0, 1] (robust to constant maps) ---
    def norm01(x, lo=None, hi=None):
        if lo is None or hi is None:
            lo, hi = (float(x.min()), float(x.max()))
        if hi <= lo + 1e-12:
            return np.zeros_like(x, dtype=np.float32)  # all equal
        return (x - lo) / (hi - lo)

    if per_channel_minmax:
        An = norm01(A)
        Bn = norm01(B)
    else:
        gmin = float(min(A.min(), B.min()))
        gmax = float(max(A.max(), B.max()))
        An = norm01(A, gmin, gmax)
        Bn = norm01(B, gmin, gmax)

    H, W = An.shape

    # --- Build masks where each object "wins" the pixel ---
    maskA = An >= Bn
    maskB = ~maskA  # strictly B > A

    # --- Optional: restrict to top X% pixels per object (by that object's score) ---
    if top_percent is not None:
        top_percent = float(top_percent)
        top_percent = max(0.0, min(1.0, top_percent))
        if top_percent > 0:
            # Thresholds at top X% within each channel
            kA = max(1, int(np.ceil(top_percent * H * W)))
            kB = max(1, int(np.ceil(top_percent * H * W)))
            thrA = np.partition(An.ravel(), -kA)[-kA]
            thrB = np.partition(Bn.ravel(), -kB)[-kB]
            maskA &= (An >= thrA)
            maskB &= (Bn >= thrB)
        else:
            # top_percent == 0 -> nothing highlighted
            maskA[:] = False
            maskB[:] = False

    # --- Compose RGB image (float in [0,1]) ---
    # Red channel = An where A wins; Blue channel = Bn where B wins
    R = np.where(maskA, An, 0.0)
    G = np.zeros((H, W), dtype=np.float32)
    Bc = np.where(maskB, Bn, 0.0)

    rgb = np.stack([R, G, Bc], axis=-1)  # (H, W, 3) in RGB

    # Convert to BGR uint8 for OpenCV and save
    bgr = (rgb[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, bgr)

# ---------------- canon ----------------
def relation_label(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw.lower().strip())
    canonical = {
        "next to": "next to",
        "in front of": "in front of",
        "on top of": "on",
        "to the left of": "left of",
        "to the right of": "right of",
        # NEW ego-centric forms
        "to your left": "left of",
        "to your right": "right of",
        "on your left": "left of",
        "on your right": "right of",
    }
    return canonical.get(raw, raw)

# ---------------- matcher ----------------
def build_relation_matcher(nlp) -> Matcher:
    m = Matcher(nlp.vocab)
    add = m.add
    def p(*ws): return [{"LOWER": w} for w in ws]

    add("LEFT_OF",  [p("to","the","left","of"),  p("left","of")])
    add("RIGHT_OF", [p("to","the","right","of"), p("right","of")])
    add("IN_FRONT_OF", [p("in","front","of")])
    add("ON_TOP_OF", [p("on","top","of"), p("on")])
    add("UNDER",   [p("under"), p("below"), p("beneath")])
    add("ABOVE",   [p("above"), p("over")])
    add("NEAR",    [p("next","to"), p("beside"), p("near")])
    add("IN",      [p("inside"), p("in")])
    # NEW: ego-centric left/right
    add("EGO_RIGHT", [p("to","your","right"), p("on","your","right")])
    add("EGO_LEFT",  [p("to","your","left"),  p("on","your","left")])
    return m

# ---------------- helpers ----------------
def normalize_obj(text: str) -> str:
    toks = [t for t in text.strip().lower().split()]
    if toks and toks[0] in {"the","a","an","this","that","these","those"}:
        toks = toks[1:]
    return " ".join(toks) if toks else text.strip().lower()

def nearest_np_left_of(doc, idx_start) -> str | None:
    # closest noun chunk fully ending at/before idx_start
    cand = [nc for nc in doc.noun_chunks if nc.end <= idx_start]
    return normalize_obj(cand[-1].text) if cand else None

def first_np_right_of(doc, idx_end) -> List[str]:
    # first noun chunk starting at/after idx_end
    cand = [nc for nc in doc.noun_chunks if nc.start >= idx_end]
    return [normalize_obj(cand[0].text)] if cand else []

def find_facing_np(doc) -> str | None:
    """
    Detect phrases like 'Facing the table lamp', 'Face the TV', 'Turn toward the sofa'
    and return that NP (normalized).
    """
    for tok in doc:
        low = tok.lower_
        lem = tok.lemma_.lower()
        if lem in {"face","turn"} or low in {"facing"}:
            # prefer a noun chunk immediately after the token
            right_ncs = [nc for nc in doc.noun_chunks if nc.start >= tok.i+1]
            if right_ncs:
                return normalize_obj(right_ncs[0].text)
        if low in {"toward","towards"} and tok.dep_ == "prep" and tok.head.lemma_.lower() in {"turn","face"}:
            # 'turn toward the sofa'
            ncs = [nc for nc in doc.noun_chunks if nc.start >= tok.i+1]
            if ncs:
                return normalize_obj(ncs[0].text)
    return None

@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str

# ---------------- main ----------------
def extract_graph_from_text(text: str, nlp=None) -> Dict:
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    matcher = build_relation_matcher(nlp)
    doc = nlp(text)

    facing_np = find_facing_np(doc)  # may be None
    edges_set = set()
    nodes_set = set()

    matches = matcher(doc)
    spans = spacy.util.filter_spans([doc[s:e] for _, s, e in matches])

    # block tokens inside matched spans
    rel_token_idxs = {i for sp in spans for i in range(sp.start, sp.end)}

    for sp in spans:
        rel = relation_label(sp.text)

        # Ego-centric handling: "… NP … to your right/left"
        if rel in {"right of", "left of"} and sp[0].lower_ in {"to","on"} and sp[-1].lower_ in {"right","left"}:
            L = nearest_np_left_of(doc, sp.start)   # target object immediately before the phrase
            R = facing_np or "you"                  # anchor: faced object if present, else 'you'
            if L and R and L != R:
                edges_set.add((L, rel, R))
                nodes_set.update([L, R])
            continue

        # Default (explicit relational phrase): left NP <--rel--> first NP to the right
        L = nearest_np_left_of(doc, sp.start)
        Rs = first_np_right_of(doc, sp.end)
        if L and Rs:
            for R in Rs:
                if R and R != L:
                    edges_set.add((L, rel, R))
                    nodes_set.update([L, R])

    # Fallback: dependency prepositions not already consumed
    for tok in doc:
        if tok.dep_ != "prep":
            continue
        if tok.i in rel_token_idxs or tok.head.i in rel_token_idxs:
            continue
        pobj = [c for c in tok.children if c.dep_ in {"pobj","pcomp"}]
        if not pobj:
            continue
        L = None
        for nc in doc.noun_chunks:
            if nc.root == tok.head:
                L = normalize_obj(nc.text); break
        Rlist = []
        for c in pobj:
            expanded = next((nc.text for nc in doc.noun_chunks if nc.start <= c.i < nc.end), c.text)
            Rlist.append(normalize_obj(expanded))
        if L and Rlist:
            rel = relation_label(tok.text)
            for R in Rlist:
                if R and R != L:
                    edges_set.add((L, rel, R))
                    nodes_set.update([L, R])

    graph = {
        "nodes": sorted(nodes_set),
        "edges": [{"from": s, "to": t, "relation": r} for (s, r, t) in sorted(edges_set)],
    }
    return graph


if __name__ == "__main__":
    # ---------------------------
    # Example usage
    examples = [
        "Facing the table lamp, go to the potted_plant to your right.",
        "the potted succulent plant next to the tv",
        "the mug is on the wooden table",
        "a chair to the left of the desk and near the window",
        "the lamp is in front of the sofa",
        "the book is on top of the shelf",
        "a vase between the lamp and the tv",
    ]
    for s in examples:
        print(s, "→")
        print(extract_graph_from_text(s))
        print()

    # H, W = 1000, 1000
    # # Example dummy data: two Gaussian bumps
    # yy, xx = np.mgrid[0:H, 0:W]
    # Amap = np.exp(-((yy-300)**2 + (xx-400)**2) / (2*80**2))
    # Bmap = np.exp(-((yy-700)**2 + (xx-600)**2) / (2*100**2))
    # sim = np.stack([Amap, Bmap], axis=0)

    # # Full dominance map (no thresholding)
    # save_ab_dominance_map(sim, out_path="ab_dominance.png", top_percent=None)

    # # Only top 5% highest-score pixels per object
    # save_ab_dominance_map(sim, out_path="ab_top5.png", top_percent=0.05)

    # map_extent = [
    #     (torch.tensor(415), torch.tensor(572), torch.tensor(2)),
    #     (torch.tensor(414), torch.tensor(571), torch.tensor(3)),
    #     (torch.tensor(415), torch.tensor(572), torch.tensor(3)),
    #     (torch.tensor(415), torch.tensor(572), torch.tensor(3)),
    #     (torch.tensor(414), torch.tensor(571), torch.tensor(3)),
    #     (torch.tensor(415), torch.tensor(572), torch.tensor(3)),
    #     (torch.tensor(414), torch.tensor(571), torch.tensor(3)),
    #     (torch.tensor(415), torch.tensor(572), torch.tensor(3)),
    #     (torch.tensor(414), torch.tensor(571), torch.tensor(3)),
    #     (torch.tensor(415), torch.tensor(572), torch.tensor(3)),
    #     (torch.tensor(414), torch.tensor(571), torch.tensor(3)),
    #     (torch.tensor(412), torch.tensor(567), torch.tensor(3)),
    #     (torch.tensor(417), torch.tensor(577), torch.tensor(3)),
    # ]
    # map_extent = np.array([[t[0].item(),t[1].item(),t[2].item()] for t in map_extent])
    # A_min, A_max = map_extent.min(axis=0), map_extent.max(axis=0)


