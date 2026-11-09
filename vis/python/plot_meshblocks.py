#! /usr/bin/env python
"""
Visualize MeshBlock structure from a 'mesh_structure.dat' file.

Generate the file by running AthenaK with the '-m' flag (writes the file and exits
without evolving):
    ./athena -i athinput.xxx -m

This extends plot_mesh.py by also parsing the logical level and MPI rank of every
MeshBlock (from the '#MeshBlock ...' comment lines), so blocks can be colored by
refinement level or rank, and 2D slices through the 3D mesh can be drawn.

Usage (command line):
    python plot_meshblocks.py -i mesh_structure.dat                 # 3D + slices
    python plot_meshblocks.py -i mesh_structure.dat --color rank -o mesh.png
    python plot_meshblocks.py --slice z=0 --xlim -10 10 --ylim -10 10

Usage (notebook):
    import sys; sys.path.append('/path/to/athenak_andrew/vis/python')
    import plot_meshblocks as pm
    blocks = pm.read_mesh_structure('mesh_structure.dat')
    pm.summary(blocks)
    pm.plot_slice(blocks, axis='z', value=0.0)
    pm.plot_3d(blocks)
"""

import argparse
import re

import numpy as np


def read_mesh_structure(filename='mesh_structure.dat'):
    """Parse mesh_structure.dat into a list of dicts.

    Each dict has: id, rank, cost, level, lloc (lx1,lx2,lx3), ndim,
    xmin (array of length 3), xmax (array of length 3).
    For 2D meshes, x3 bounds are set to (0, 0).
    """
    re_head = re.compile(r'#MeshBlock\s+(\d+)\s+on rank=(\d+)\s+with cost=(\S+)')
    re_lev = re.compile(r'#\s*Logical level\s+(-?\d+),\s*location\s*=\s*'
                        r'\((-?\d+)\s+(-?\d+)\s+(-?\d+)\)')
    blocks = []
    cur = None
    pts = []

    def finish():
        if cur is not None and pts:
            p = np.array(pts)
            if p.shape[1] == 2:
                p = np.column_stack([p, np.zeros(len(p))])
            cur['ndim'] = len(pts[0])
            cur['xmin'] = p.min(axis=0)
            cur['xmax'] = p.max(axis=0)
            blocks.append(cur)

    with open(filename) as f:
        for line in f:
            m = re_head.match(line)
            if m:
                finish()
                cur = dict(id=int(m.group(1)), rank=int(m.group(2)),
                           cost=float(m.group(3)), level=0, lloc=(0, 0, 0))
                pts = []
                continue
            m = re_lev.match(line)
            if m and cur is not None:
                cur['level'] = int(m.group(1))
                cur['lloc'] = tuple(int(m.group(k)) for k in (2, 3, 4))
                continue
            s = line.split()
            if s and not line.startswith('#'):
                pts.append([float(v) for v in s])
    finish()
    return blocks


def read_meshblock_nx(athinput):
    """Return (nx1, nx2, nx3) cells per MeshBlock from the <meshblock> block of an
    AthenaK input file. Missing entries (e.g. nx3 in 2D) default to 1."""
    nx = {'nx1': 1, 'nx2': 1, 'nx3': 1}
    in_block = False
    with open(athinput) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line.startswith('<'):
                in_block = (line == '<meshblock>')
                continue
            if in_block and '=' in line:
                key, val = [t.strip() for t in line.split('=', 1)]
                if key in nx:
                    nx[key] = int(val)
    return nx['nx1'], nx['nx2'], nx['nx3']


def summary(blocks):
    """Print number of MeshBlocks per level and per rank."""
    levels = np.array([b['level'] for b in blocks])
    ranks = np.array([b['rank'] for b in blocks])
    print(f"Total MeshBlocks: {len(blocks)}  (ndim = {blocks[0]['ndim']})")
    lmin = levels.min()
    for lev in np.unique(levels):
        sel = [b for b in blocks if b['level'] == lev]
        dx = sel[0]['xmax'] - sel[0]['xmin']
        print(f"  level {lev} (physical level {lev - lmin}): {len(sel):6d} blocks, "
              f"block size = {dx[:sel[0]['ndim']]}")
    print(f"  ranks used: {len(np.unique(ranks))}, "
          f"blocks/rank min/max = {np.bincount(ranks).min()}/{np.bincount(ranks).max()}")


def _colors(blocks, color_by, cmap_name):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    vals = np.array([b[color_by] for b in blocks])
    uniq = np.unique(vals)
    cmap = plt.get_cmap(cmap_name)
    bounds = np.append(uniq - 0.5, uniq[-1] + 0.5)
    norm = BoundaryNorm(bounds, cmap.N)
    return vals, uniq, cmap, norm


def plot_slice(blocks, axis='z', value=0.0, color_by='level', ax=None,
               cmap='viridis', fill=True, xlim=None, ylim=None, lw=0.6,
               nx=None, cell_lw=0.15, cell_color='0.3'):
    """Plot outlines of all MeshBlocks intersecting the plane <axis> = value.

    For 2D meshes, use axis='z' (any value) to plot the whole mesh.

    nx: cells per MeshBlock, either an int (same in every direction) or a tuple
        (nx1, nx2, nx3) -- the <meshblock> nx1/nx2/nx3 values from the input file
        (see read_meshblock_nx). If given, every grid cell in the plane is outlined
        with thin lines of width cell_lw and color cell_color.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    iax = {'x': 0, 'y': 1, 'z': 2}[axis]
    ia, ib = [i for i in range(3) if i != iax]
    labels = ['x', 'y', 'z']

    if blocks[0]['ndim'] == 2:
        sel = blocks
    else:
        # include a block if value lies in [xmin, xmax); on the upper mesh edge use <=
        sel = [b for b in blocks if b['xmin'][iax] <= value < b['xmax'][iax]]
        if not sel:
            sel = [b for b in blocks if b['xmin'][iax] <= value <= b['xmax'][iax]]
    if not sel:
        raise ValueError(f"No MeshBlocks intersect {axis} = {value}")

    vals, uniq, cmap_, norm = _colors(sel, color_by, cmap)
    patches = [Rectangle((b['xmin'][ia], b['xmin'][ib]),
                         b['xmax'][ia] - b['xmin'][ia],
                         b['xmax'][ib] - b['xmin'][ib]) for b in sel]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    pc = PatchCollection(patches, cmap=cmap_, norm=norm,
                         edgecolor='k', linewidth=lw,
                         alpha=0.6 if fill else 1.0)
    if fill:
        pc.set_array(vals)
    else:
        pc.set_facecolor('none')
    ax.add_collection(pc)

    if nx is not None:
        from matplotlib.collections import LineCollection
        if np.isscalar(nx):
            nx = (nx, nx, nx)
        segs = []
        for b in sel:
            a0, a1 = b['xmin'][ia], b['xmax'][ia]
            b0, b1 = b['xmin'][ib], b['xmax'][ib]
            # interior cell faces only; block edges are already drawn by the patches
            for xa in np.linspace(a0, a1, nx[ia] + 1)[1:-1]:
                segs.append([(xa, b0), (xa, b1)])
            for xb in np.linspace(b0, b1, nx[ib] + 1)[1:-1]:
                segs.append([(a0, xb), (a1, xb)])
        ax.add_collection(LineCollection(segs, colors=cell_color, linewidths=cell_lw))

    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_xlabel(labels[ia])
    ax.set_ylabel(labels[ib])
    title = f"{len(sel)} MeshBlocks"
    if blocks[0]['ndim'] == 3:
        title += f" intersecting {axis} = {value:g}"
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if fill:
        cb = plt.colorbar(pc, ax=ax, ticks=uniq if len(uniq) <= 20 else None)
        cb.set_label(color_by)
    return ax


def plot_3d(blocks, color_by='level', ax=None, cmap='viridis', levels=None, lw=0.5):
    """3D wireframe of all MeshBlocks, colored by level or rank.

    levels: optional list of levels to draw (e.g. only the finest ones).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if levels is not None:
        blocks = [b for b in blocks if b['level'] in levels]
    vals, uniq, cmap_, norm = _colors(blocks, color_by, cmap)

    # 12 edges of a box, as pairs of corner indices
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    segs, cvals = [], []
    for b, v in zip(blocks, vals):
        lo, hi = b['xmin'], b['xmax']
        corners = np.array([[x, y, z] for z in (lo[2], hi[2])
                            for y in (lo[1], hi[1]) for x in (lo[0], hi[0])])
        for e in edges:
            segs.append(corners[list(e)])
            cvals.append(v)

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection='3d')
    lc = Line3DCollection(segs, cmap=cmap_, norm=norm, linewidths=lw)
    lc.set_array(np.array(cvals))
    ax.add_collection3d(lc)
    allmin = np.min([b['xmin'] for b in blocks], axis=0)
    allmax = np.max([b['xmax'] for b in blocks], axis=0)
    ax.set_xlim(allmin[0], allmax[0])
    ax.set_ylim(allmin[1], allmax[1])
    ax.set_zlim(allmin[2], allmax[2])
    try:
        ax.set_box_aspect(allmax - allmin + 1e-30)
    except Exception:
        pass
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"{len(blocks)} MeshBlocks")
    cb = plt.colorbar(lc, ax=ax, shrink=0.7, ticks=uniq if len(uniq) <= 20 else None)
    cb.set_label(color_by)
    return ax


def main(**kw):
    if kw['output'] != 'show':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt

    blocks = read_mesh_structure(kw['input'])
    summary(blocks)
    nx = kw['nx']
    if nx is None and kw['athinput'] is not None:
        nx = read_meshblock_nx(kw['athinput'])

    if blocks[0]['ndim'] == 2:
        plot_slice(blocks, color_by=kw['color'], xlim=kw['xlim'], ylim=kw['ylim'],
                   nx=nx)
    elif kw['slice'] is not None:
        axis, val = kw['slice'].split('=')
        plot_slice(blocks, axis=axis.strip(), value=float(val), color_by=kw['color'],
                   xlim=kw['xlim'], ylim=kw['ylim'], nx=nx)
    else:
        fig = plt.figure(figsize=(18, 5.5))
        plot_3d(blocks, color_by=kw['color'], ax=fig.add_subplot(1, 3, 1, projection='3d'))
        plot_slice(blocks, 'z', 0.0, color_by=kw['color'], ax=fig.add_subplot(1, 3, 2),
                   xlim=kw['xlim'], ylim=kw['ylim'], nx=nx)
        plot_slice(blocks, 'y', 0.0, color_by=kw['color'], ax=fig.add_subplot(1, 3, 3),
                   xlim=kw['xlim'], ylim=kw['ylim'], nx=nx)
        fig.tight_layout()

    if kw['output'] == 'show':
        plt.show()
    else:
        plt.savefig(kw['output'], bbox_inches='tight', dpi=200)
        print(f"Saved {kw['output']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-i', '--input', default='mesh_structure.dat')
    p.add_argument('-o', '--output', default='show',
                   help='image filename; omit to display to screen')
    p.add_argument('--color', default='level', choices=['level', 'rank', 'cost'],
                   help='quantity used to color blocks')
    p.add_argument('--slice', default=None,
                   help="single 2D slice for 3D meshes, e.g. 'z=0' or 'x=1.5'")
    p.add_argument('--xlim', type=float, nargs=2, default=None)
    p.add_argument('--ylim', type=float, nargs=2, default=None)
    p.add_argument('--nx', type=int, nargs=3, default=None,
                   help='cells per MeshBlock (nx1 nx2 nx3); outlines every grid cell')
    p.add_argument('--athinput', default=None,
                   help='input file to read <meshblock> nx1/nx2/nx3 from (instead of --nx)')
    main(**vars(p.parse_args()))
