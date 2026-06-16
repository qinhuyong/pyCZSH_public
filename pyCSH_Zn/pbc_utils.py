from __future__ import print_function

import math


def parse_box_line(box, parts):
    if len(parts) >= 4 and parts[2:4] == ["xlo", "xhi"]:
        box["xlo_bound"], box["xhi_bound"] = float(parts[0]), float(parts[1])
        return True
    if len(parts) >= 4 and parts[2:4] == ["ylo", "yhi"]:
        box["ylo_bound"], box["yhi_bound"] = float(parts[0]), float(parts[1])
        return True
    if len(parts) >= 4 and parts[2:4] == ["zlo", "zhi"]:
        box["zlo_bound"], box["zhi_bound"] = float(parts[0]), float(parts[1])
        return True
    if len(parts) >= 6 and parts[3:6] == ["xy", "xz", "yz"]:
        box["xy"], box["xz"], box["yz"] = float(parts[0]), float(parts[1]), float(parts[2])
        return True
    return False


def finalize_box(box):
    xy = float(box.get("xy", 0.0))
    xz = float(box.get("xz", 0.0))
    yz = float(box.get("yz", 0.0))
    xlo_bound = float(box.get("xlo_bound", box.get("xlo", 0.0)))
    xhi_bound = float(box.get("xhi_bound", box.get("xhi", 0.0)))
    ylo_bound = float(box.get("ylo_bound", box.get("ylo", 0.0)))
    yhi_bound = float(box.get("yhi_bound", box.get("yhi", 0.0)))
    zlo = float(box.get("zlo_bound", box.get("zlo", 0.0)))
    zhi = float(box.get("zhi_bound", box.get("zhi", 0.0)))

    xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
    xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
    ylo = ylo_bound - min(0.0, yz)
    yhi = yhi_bound - max(0.0, yz)

    box.update({
        "xlo": xlo,
        "xhi": xhi,
        "ylo": ylo,
        "yhi": yhi,
        "zlo": zlo,
        "zhi": zhi,
        "xy": xy,
        "xz": xz,
        "yz": yz,
        "lx": xhi - xlo,
        "ly": yhi - ylo,
        "lz": zhi - zlo,
        "triclinic": bool(abs(xy) > 0.0 or abs(xz) > 0.0 or abs(yz) > 0.0),
    })
    return box


def box_volume(box):
    b = finalize_box(dict(box))
    return abs(b["lx"] * b["ly"] * b["lz"])


def minimum_image_vector(coord_i, coord_j, box):
    b = finalize_box(dict(box))
    dx = float(coord_j[0]) - float(coord_i[0])
    dy = float(coord_j[1]) - float(coord_i[1])
    dz = float(coord_j[2]) - float(coord_i[2])

    lx = b["lx"]
    ly = b["ly"]
    lz = b["lz"]
    xy = b["xy"]
    xz = b["xz"]
    yz = b["yz"]

    if lx == 0.0 or ly == 0.0 or lz == 0.0:
        return dx, dy, dz

    # Restricted triclinic h matrix columns are:
    # a=(lx,0,0), b=(xy,ly,0), c=(xz,yz,lz).
    fz = dz / lz
    fy = (dy - yz * fz) / ly
    fx = (dx - xy * fy - xz * fz) / lx
    fx -= round(fx)
    fy -= round(fy)
    fz -= round(fz)
    return (
        lx * fx + xy * fy + xz * fz,
        ly * fy + yz * fz,
        lz * fz,
    )


def minimum_image_distance(atom_i, atom_j, box):
    vec = minimum_image_vector(
        (atom_i["x"], atom_i["y"], atom_i["z"]),
        (atom_j["x"], atom_j["y"], atom_j["z"]),
        box,
    )
    return math.sqrt(sum(x * x for x in vec))
