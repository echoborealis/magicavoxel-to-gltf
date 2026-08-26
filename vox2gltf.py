#!/usr/bin/env python3
"""
vox2gltf.py - Convert MagicaVoxel .vox files to glTF 2.0 (.gltf with embedded buffer).

Features:
  * Full scene-graph support (nTRN / nGRP / nSHP) for multi-model files (frame 0)
  * Greedy meshing (merged quads, only visible faces)
  * Material preservation from MATL chunks:
      _diffuse -> baseColorFactor (+ _rough -> roughnessFactor)
      _metal   -> metallicFactor (_weight), roughnessFactor (_rough)
      _glass   -> KHR_materials_transmission (_weight), KHR_materials_ior (_ior), _rough
      _emit    -> emissiveFactor + KHR_materials_emissive_strength (_emit * 2^_flux)
  * MagicaVoxel Z-up coordinates converted to glTF Y-up

Usage:
  python vox2gltf.py input.vox [output.gltf]

No third-party dependencies (Python 3.8+ stdlib only).
"""

import base64
import json
import struct
import sys

# ---------------------------------------------------------------- default palette
# Used only if the RGBA chunk is absent (0xAABBGGRR per spec).
DEFAULT_PALETTE = [
    0x00000000, 0xffffffff, 0xffccffff, 0xff99ffff, 0xff66ffff, 0xff33ffff, 0xff00ffff, 0xffffccff,
    0xffccccff, 0xff99ccff, 0xff66ccff, 0xff33ccff, 0xff00ccff, 0xffff99ff, 0xffcc99ff, 0xff9999ff,
    0xff6699ff, 0xff3399ff, 0xff0099ff, 0xffff66ff, 0xffcc66ff, 0xff9966ff, 0xff6666ff, 0xff3366ff,
    0xff0066ff, 0xffff33ff, 0xffcc33ff, 0xff9933ff, 0xff6633ff, 0xff3333ff, 0xff0033ff, 0xffff00ff,
    0xffcc00ff, 0xff9900ff, 0xff6600ff, 0xff3300ff, 0xff0000ff, 0xffffffcc, 0xffccffcc, 0xff99ffcc,
    0xff66ffcc, 0xff33ffcc, 0xff00ffcc, 0xffffcccc, 0xffcccccc, 0xff99cccc, 0xff66cccc, 0xff33cccc,
    0xff00cccc, 0xffff99cc, 0xffcc99cc, 0xff9999cc, 0xff6699cc, 0xff3399cc, 0xff0099cc, 0xffff66cc,
    0xffcc66cc, 0xff9966cc, 0xff6666cc, 0xff3366cc, 0xff0066cc, 0xffff33cc, 0xffcc33cc, 0xff9933cc,
    0xff6633cc, 0xff3333cc, 0xff0033cc, 0xffff00cc, 0xffcc00cc, 0xff9900cc, 0xff6600cc, 0xff3300cc,
    0xff0000cc, 0xffffff99, 0xffccff99, 0xff99ff99, 0xff66ff99, 0xff33ff99, 0xff00ff99, 0xffffcc99,
    0xffcccc99, 0xff99cc99, 0xff66cc99, 0xff33cc99, 0xff00cc99, 0xffff9999, 0xffcc9999, 0xff999999,
    0xff669999, 0xff339999, 0xff009999, 0xffff6699, 0xffcc6699, 0xff996699, 0xff666699, 0xff336699,
    0xff006699, 0xffff3399, 0xffcc3399, 0xff993399, 0xff663399, 0xff333399, 0xff003399, 0xffff0099,
    0xffcc0099, 0xff990099, 0xff660099, 0xff330099, 0xff000099, 0xffffff66, 0xffccff66, 0xff99ff66,
    0xff66ff66, 0xff33ff66, 0xff00ff66, 0xffffcc66, 0xffcccc66, 0xff99cc66, 0xff66cc66, 0xff33cc66,
    0xff00cc66, 0xffff9966, 0xffcc9966, 0xff999966, 0xff669966, 0xff339966, 0xff009966, 0xffff6666,
    0xffcc6666, 0xff996666, 0xff666666, 0xff336666, 0xff006666, 0xffff3366, 0xffcc3366, 0xff993366,
    0xff663366, 0xff333366, 0xff003366, 0xffff0066, 0xffcc0066, 0xff990066, 0xff660066, 0xff330066,
    0xff000066, 0xffffff33, 0xffccff33, 0xff99ff33, 0xff66ff33, 0xff33ff33, 0xff00ff33, 0xffffcc33,
    0xffcccc33, 0xff99cc33, 0xff66cc33, 0xff33cc33, 0xff00cc33, 0xffff9933, 0xffcc9933, 0xff999933,
    0xff669933, 0xff339933, 0xff009933, 0xffff6633, 0xffcc6633, 0xff996633, 0xff666633, 0xff336633,
    0xff006633, 0xffff3333, 0xffcc3333, 0xff993333, 0xff663333, 0xff333333, 0xff003333, 0xffff0033,
    0xffcc0033, 0xff990033, 0xff660033, 0xff330033, 0xff000033, 0xffffff00, 0xffccff00, 0xff99ff00,
    0xff66ff00, 0xff33ff00, 0xff00ff00, 0xffffcc00, 0xffcccc00, 0xff99cc00, 0xff66cc00, 0xff33cc00,
    0xff00cc00, 0xffff9900, 0xffcc9900, 0xff999900, 0xff669900, 0xff339900, 0xff009900, 0xffff6600,
    0xffcc6600, 0xff996600, 0xff666600, 0xff336600, 0xff006600, 0xffff3300, 0xffcc3300, 0xff993300,
    0xff663300, 0xff333300, 0xff003300, 0xffff0000, 0xffcc0000, 0xff990000, 0xff660000, 0xff330000,
    0xff0000ee, 0xff0000dd, 0xff0000bb, 0xff0000aa, 0xff000088, 0xff000077, 0xff000055, 0xff000044,
    0xff000022, 0xff000011, 0xff00ee00, 0xff00dd00, 0xff00bb00, 0xff00aa00, 0xff008800, 0xff007700,
    0xff005500, 0xff004400, 0xff002200, 0xff001100, 0xffee0000, 0xffdd0000, 0xffbb0000, 0xffaa0000,
    0xff880000, 0xff770000, 0xff550000, 0xff440000, 0xff220000, 0xff110000, 0xffeeeeee, 0xffdddddd,
    0xffbbbbbb, 0xffaaaaaa, 0xff888888, 0xff777777, 0xff555555, 0xff444444, 0xff222222, 0xff111111,
]

# Coordinate change: MagicaVoxel (x right, y depth, z up) -> glTF (x right, y up, z back)
# v_gltf = C @ v_mv
C_MAT = [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]]
C_MAT_T = [[1, 0, 0],
           [0, 0, -1],
           [0, 1, 0]]

IDENT3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

def read_string(buf, off):
    (n,) = struct.unpack_from('<i', buf, off)
    off += 4
    s = buf[off:off + n].decode('utf-8', 'replace')
    return s, off + n


def read_dict(buf, off):
    (n,) = struct.unpack_from('<i', buf, off)
    off += 4
    d = {}
    for _ in range(n):
        k, off = read_string(buf, off)
        v, off = read_string(buf, off)
        d[k] = v
    return d, off


def parse_chunks(buf, pos, end):
    chunks = []
    while pos + 12 <= end:
        cid = buf[pos:pos + 4].decode('ascii', 'replace')
        n, m = struct.unpack_from('<II', buf, pos + 4)
        content = buf[pos + 12: pos + 12 + n]
        cstart = pos + 12 + n
        cend = cstart + m
        children = parse_chunks(buf, cstart, cend) if m else []
        chunks.append((cid, content, children))
        pos = cend
    return chunks

def mat3_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat3_vec(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def mat4_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def t4(v):
    return [[1, 0, 0, v[0]], [0, 1, 0, v[1]], [0, 0, 1, v[2]], [0, 0, 0, 1]]


def decode_rotation(r):
    """Decode ROTATION byte into a row-major 3x3 matrix."""
    i0 = r & 3
    i1 = (r >> 2) & 3
    if i0 == i1:
        return [row[:] for row in IDENT3]
    i2 = 3 - i0 - i1
    s0 = -1 if (r >> 4) & 1 else 1
    s1 = -1 if (r >> 5) & 1 else 1
    s2 = -1 if (r >> 6) & 1 else 1
    m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    m[0][i0] = s0
    m[1][i1] = s1
    m[2][i2] = s2
    return m


def clamp01(x):
    return max(0.0, min(1.0, x))


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def load_vox(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'VOX ':
        raise ValueError('not a .vox file: ' + path)
    if data[8:12] != b'MAIN':
        raise ValueError('missing MAIN chunk: ' + path)
    _, m = struct.unpack_from('<II', data, 12)
    children = parse_chunks(data, 20, 20 + m)

    models = []          # [(size_xyz, {(x,y,z): color_index})]
    palette = None       # [256 x (r,g,b,a)]
    materials = {}       # material id (= palette index) -> {prop: value}
    nodes = {}           # node id -> tuple
    layers = {}          # layer id -> attrs

    cur_size = None
    for cid, content, _ in children:
        if cid == 'SIZE':
            cur_size = struct.unpack('<iii', content)
        elif cid == 'XYZI':
            (nvox,) = struct.unpack_from('<i', content, 0)
            vox = {}
            off = 4
            for _ in range(nvox):
                x, y, z, c = content[off:off + 4]
                off += 4
                if c:
                    vox[(x, y, z)] = c
            models.append((cur_size, vox))
        elif cid == 'RGBA':
            palette = [tuple(content[i * 4:i * 4 + 4]) for i in range(256)]
        elif cid == 'MATL':
            (mid,) = struct.unpack_from('<i', content, 0)
            props, _ = read_dict(content, 4)
            materials[mid] = props
        elif cid == 'nTRN':
            (nid,) = struct.unpack_from('<i', content, 0)
            attrs, off = read_dict(content, 4)
            child, _reserved, layer, nframes = struct.unpack_from('<iiii', content, off)
            off += 16
            frames = []
            for _ in range(nframes):
                fr, off = read_dict(content, off)
                frames.append(fr)
            nodes[nid] = ('T', attrs, child, layer, frames)
        elif cid == 'nGRP':
            (nid,) = struct.unpack_from('<i', content, 0)
            attrs, off = read_dict(content, 4)
            (nch,) = struct.unpack_from('<i', content, off)
            off += 4
            ids = list(struct.unpack_from('<%di' % nch, content, off))
            nodes[nid] = ('G', attrs, ids)
        elif cid == 'nSHP':
            (nid,) = struct.unpack_from('<i', content, 0)
            attrs, off = read_dict(content, 4)
            (nmodels,) = struct.unpack_from('<i', content, off)
            off += 4
            mids = []
            for _ in range(nmodels):
                (mid,) = struct.unpack_from('<i', content, off)
                off += 4
                _mattrs, off = read_dict(content, off)
                mids.append(mid)
            nodes[nid] = ('S', attrs, mids)
        elif cid == 'LAYR':
            (lid,) = struct.unpack_from('<i', content, 0)
            attrs, _ = read_dict(content, 4)
            layers[lid] = attrs

    if palette is None:
        palette = []
        for v in DEFAULT_PALETTE:
            palette.append((v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF))

    return models, palette, materials, nodes, layers


def collect_instances(nodes, layers, num_models):
    """Flatten the scene graph into a list of (model_id, R, t, name, hidden)."""
    child_ids = set()
    for node in nodes.values():
        if node[0] == 'T':
            child_ids.add(node[2])
        elif node[0] == 'G':
            child_ids.update(node[2])
    roots = [nid for nid in nodes if nid not in child_ids]

    instances = []

    def visit(nid, rp, tp, hidden):
        node = nodes.get(nid)
        if node is None:
            return
        kind, attrs = node[0], node[1]
        if attrs.get('_hidden') == '1':
            hidden = True
        if kind == 'T':
            _, _, child, layer, frames = node
            if layers.get(layer, {}).get('_hidden') == '1':
                hidden = True
            fr = frames[0] if frames else {}
            rl = decode_rotation(int(fr['_r'])) if '_r' in fr else IDENT3
            tl = [int(v) for v in fr.get('_t', '0 0 0').split()]
            r = mat3_mul(rp, rl)
            t = [tp[i] + v for i, v in enumerate(mat3_vec(rp, tl))]
            visit(child, r, t, hidden)
        elif kind == 'G':
            for c in node[2]:
                visit(c, rp, tp, hidden)
        elif kind == 'S':
            if hidden:
                return
            for mid in node[2]:
                if 0 <= mid < num_models:
                    instances.append((mid, rp, tp, attrs.get('_name', '')))

    for r in roots:
        visit(r, IDENT3, (0, 0, 0), False)
    return instances


def greedy_mesh(size, voxels):
    """Return {color_index: [positions, normals, indices]} with merged quads."""
    faces = {}
    dims = size
    for d in range(3):
        u, v = (d + 1) % 3, (d + 2) % 3
        w_dim, h_dim = dims[u], dims[v]
        for s in range(-1, dims[d]):
            mask = [None] * (w_dim * h_dim)
            for iv in range(h_dim):
                for iu in range(w_dim):
                    pa = [0, 0, 0]
                    pa[d], pa[u], pa[v] = s, iu, iv
                    pb = [0, 0, 0]
                    pb[d], pb[u], pb[v] = s + 1, iu, iv
                    a = voxels.get((pa[0], pa[1], pa[2]), 0)
                    b = voxels.get((pb[0], pb[1], pb[2]), 0)
                    if a and b:
                        continue
                    if a:
                        mask[iu + iv * w_dim] = (a, 1)
                    elif b:
                        mask[iu + iv * w_dim] = (b, -1)
            iv = 0
            while iv < h_dim:
                iu = 0
                while iu < w_dim:
                    mkey = mask[iu + iv * w_dim]
                    if mkey is None:
                        iu += 1
                        continue
                    w = 1
                    while iu + w < w_dim and mask[iu + w + iv * w_dim] == mkey:
                        w += 1
                    h = 1
                    grow = True
                    while iv + h < h_dim and grow:
                        for k in range(w):
                            if mask[iu + k + (iv + h) * w_dim] != mkey:
                                grow = False
                                break
                        if grow:
                            h += 1
                    for j in range(h):
                        for k in range(w):
                            mask[iu + k + (iv + j) * w_dim] = None
                    color, sign = mkey
                    faces.setdefault(color, []).append((d, sign, s + 1, iu, iv, w, h))
                    iu += w
                iv += 1

    out = {}
    for color, quads in faces.items():
        pos, nrm, idx = [], [], []
        for (d, sign, plane, iu, iv, w, h) in quads:
            u, v = (d + 1) % 3, (d + 2) % 3
            base = [0, 0, 0]
            base[d], base[u], base[v] = plane, iu, iv
            du = [0, 0, 0]
            du[u] = w
            dv = [0, 0, 0]
            dv[v] = h
            p0 = base
            p1 = [base[i] + du[i] for i in range(3)]
            p2 = [base[i] + du[i] + dv[i] for i in range(3)]
            p3 = [base[i] + dv[i] for i in range(3)]
            corners = [p0, p1, p2, p3] if sign > 0 else [p0, p3, p2, p1]
            n = [0, 0, 0]
            n[d] = sign
            i0 = len(pos)
            for p in corners:
                # bake coordinate conversion into vertices: (x, y, z)_mv -> (x, z, -y)_gltf
                pos.append((float(p[0]), float(p[2]), float(-p[1])))
                nrm.append((float(n[0]), float(n[2]), float(-n[1])))
            idx += [i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3]
        out[color] = (pos, nrm, idx)
    return out


def build_material(color_index, palette, props):
    r, g, b, a = palette[color_index - 1]
    if a == 0:
        a = 255
    mtype = props.get('_type', '_diffuse')

    def fnum(key, default):
        try:
            return float(props[key])
        except (KeyError, ValueError):
            return default

    weight = fnum('_weight', 1.0)
    base = [srgb_to_linear(r / 255.0), srgb_to_linear(g / 255.0),
            srgb_to_linear(b / 255.0), a / 255.0]
    mat = {
        'name': 'mat_%d' % color_index,
        'pbrMetallicRoughness': {
            'baseColorFactor': base,
            'metallicFactor': 0.0,
            'roughnessFactor': 1.0,
        },
    }
    exts = {}
    if mtype == '_metal':
        mat['pbrMetallicRoughness']['metallicFactor'] = clamp01(weight)
        mat['pbrMetallicRoughness']['roughnessFactor'] = clamp01(fnum('_rough', 0.1))
    elif mtype == '_glass':
        mat['pbrMetallicRoughness']['roughnessFactor'] = clamp01(fnum('_rough', 0.05))
        base[3] = 1.0  # transparency is expressed via transmission
        exts['KHR_materials_transmission'] = {'transmissionFactor': clamp01(weight)}
        exts['KHR_materials_ior'] = {'ior': max(1.0, fnum('_ior', 1.5))}
    elif mtype == '_emit':
        mat['pbrMetallicRoughness']['roughnessFactor'] = clamp01(fnum('_rough', 1.0))
        emit_amt = fnum('_emit', weight)
        strength = min(emit_amt * (2.0 ** fnum('_flux', 0.0)), 100.0)
        mat['emissiveFactor'] = base[:3]
        if abs(strength - 1.0) > 1e-6:
            exts['KHR_materials_emissive_strength'] = {'emissiveStrength': strength}
    else:  # _diffuse
        if '_rough' in props:
            mat['pbrMetallicRoughness']['roughnessFactor'] = clamp01(fnum('_rough', 1.0))
    if a < 255 and mtype != '_glass':
        mat['alphaMode'] = 'BLEND'
    if exts:
        mat['extensions'] = exts
    return mat, list(exts.keys())


def convert(in_path, out_path):
    models, palette, matl, nodes, layers = load_vox(in_path)
    if not models:
        raise ValueError('no models found in ' + in_path)

    if nodes:
        instances = collect_instances(nodes, layers, len(models))
    else:
        instances = []
    if not instances:
        # legacy fallback: stagger models along +x, sitting on the ground
        # (t is the model center position, so offset by half size)
        x_off = 0
        for mid, (size, _vox) in enumerate(models):
            t = (x_off + size[0] // 2, size[1] // 2, size[2] // 2)
            instances.append((mid, IDENT3, t, ''))
            x_off += size[0] + 2

    # mesh every referenced model once
    mesh_per_model = {}
    for mid, _r, _t, _n in instances:
        if mid not in mesh_per_model:
            mesh_per_model[mid] = greedy_mesh(*models[mid])

    used_colors = sorted({c for mesh in mesh_per_model.values() for c in mesh})

    materials = []
    mat_index = {}
    extensions_used = []
    for c in used_colors:
        mat, exts = build_material(c, palette, matl.get(c, {}))
        mat_index[c] = len(materials)
        materials.append(mat)
        for e in exts:
            if e not in extensions_used:
                extensions_used.append(e)

    bin_data = bytearray()
    buffer_views = []
    accessors = []

    def add_view(payload, target):
        off = len(bin_data)
        bin_data.extend(payload)
        while len(bin_data) % 4:
            bin_data.append(0)
        buffer_views.append({'buffer': 0, 'byteOffset': off,
                             'byteLength': len(payload), 'target': target})
        return len(buffer_views) - 1

    def add_accessor(view, comp_type, count, acc_type, mn=None, mx=None):
        acc = {'bufferView': view, 'componentType': comp_type,
               'count': count, 'type': acc_type}
        if mn is not None:
            acc['min'] = mn
            acc['max'] = mx
        accessors.append(acc)
        return len(accessors) - 1

    meshes = []
    mesh_index_per_model = {}
    total_tris = 0
    for mid in sorted(mesh_per_model):
        primitives = []
        for color in sorted(mesh_per_model[mid]):
            pos, nrm, idx = mesh_per_model[mid][color]
            if not idx:
                continue
            total_tris += len(idx) // 3
            idx_bv = add_view(struct.pack('<%dI' % len(idx), *idx), 34963)
            pos_bv = add_view(struct.pack('<%df' % (len(pos) * 3),
                                          *[c for p in pos for c in p]), 34962)
            nrm_bv = add_view(struct.pack('<%df' % (len(nrm) * 3),
                                          *[c for n in nrm for c in n]), 34962)
            pos_acc = add_accessor(
                pos_bv, 5126, len(pos), 'VEC3',
                [min(p[i] for p in pos) for i in range(3)],
                [max(p[i] for p in pos) for i in range(3)])
            nrm_acc = add_accessor(nrm_bv, 5126, len(nrm), 'VEC3')
            idx_acc = add_accessor(idx_bv, 5125, len(idx), 'SCALAR')
            primitives.append({
                'attributes': {'POSITION': pos_acc, 'NORMAL': nrm_acc},
                'indices': idx_acc,
                'material': mat_index[color],
            })
        if primitives:
            mesh_index_per_model[mid] = len(meshes)
            meshes.append({'name': 'model_%d' % mid, 'primitives': primitives})

    gltf_nodes = []
    for mid, r, t, name in instances:
        if mid not in mesh_index_per_model:
            continue
        size = models[mid][0]
        # MV transform: world = R @ (p - c) + t, where c = model center and
        # t is the world position of the model's center (verified against
        # sample files: t.z == size_z/2 for models standing on the ground).
        c = [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0]
        rp = mat3_mul(C_MAT, mat3_mul(r, C_MAT_T))          # R'  = C R C^T
        cp = mat3_vec(C_MAT, c)                             # c'  = C c
        tp = mat3_vec(C_MAT, t)                             # t'  = C t
        r4 = [row[:] + [0.0] for row in rp] + [[0.0, 0.0, 0.0, 1.0]]
        r4 = [[float(v) for v in row] for row in r4]
        m = mat4_mul(t4(tp), mat4_mul(r4, t4([-v for v in cp])))
        node = {
            'name': name or ('model_%d' % mid),
            'matrix': [float(m[row][col]) for col in range(4) for row in range(4)],
            'mesh': mesh_index_per_model[mid],
        }
        gltf_nodes.append(node)

    uri = 'data:application/octet-stream;base64,' + base64.b64encode(bytes(bin_data)).decode('ascii')
    gltf = {
        'asset': {'version': '2.0', 'generator': 'vox2gltf.py'},
        'scene': 0,
        'scenes': [{'name': 'Scene', 'nodes': list(range(len(gltf_nodes)))}],
        'nodes': gltf_nodes,
        'meshes': meshes,
        'materials': materials,
        'accessors': accessors,
        'bufferViews': buffer_views,
        'buffers': [{'uri': uri, 'byteLength': len(bin_data)}],
    }
    if extensions_used:
        gltf['extensionsUsed'] = extensions_used

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(gltf, f, separators=(',', ':'))

    return {
        'models': len(models),
        'instances': len(gltf_nodes),
        'meshes': len(meshes),
        'primitives': sum(len(m['primitives']) for m in meshes),
        'triangles': total_tris,
        'materials': len(materials),
        'extensions': extensions_used,
    }


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    in_path = argv[1]
    out_path = argv[2] if len(argv) > 2 else in_path.rsplit('.', 1)[0] + '.gltf'
    stats = convert(in_path, out_path)
    print('%s -> %s' % (in_path, out_path))
    for k, v in stats.items():
        print('  %-10s %s' % (k + ':', v))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
