"""
test_assembly.py
Unit tests for src/assembly.py

Synthetic pieces are built directly (bypassing the earlier pipeline
stages) with hand-crafted 'sides' lists so the correct grid solution is
known in advance. All synthetic pieces share one fixed 16x16 mask
(centroid (7.5, 7.5)), and sides are built from FLAT/TAB/BLANK point
templates per compass direction so that tab bulges point away from the
shared centroid and blank dips point toward it -- i.e. real, geometrically
consistent tab/blank pairs, not arbitrary numbers.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import assembly as asm


# ---------------------------------------------------------------------------
# synthetic piece builder
# ---------------------------------------------------------------------------

_MASK = np.zeros((16, 16), dtype=np.uint8)
_MASK[2:14, 2:14] = 255

_FLAT = {
    "N": [(2, 2), (2, 7), (2, 13)],
    "E": [(2, 13), (7, 13), (13, 13)],
    "S": [(13, 13), (13, 7), (13, 2)],
    "W": [(13, 2), (7, 2), (2, 2)],
}
_TAB = {
    "N": [(2, 2), (-1, 7), (2, 13)],
    "E": [(2, 13), (7, 16), (13, 13)],
    "S": [(13, 13), (16, 7), (13, 2)],
    "W": [(13, 2), (7, -1), (2, 2)],
}
_BLANK = {
    "N": [(2, 2), (5, 7), (2, 13)],
    "E": [(2, 13), (7, 10), (13, 13)],
    "S": [(13, 13), (10, 7), (13, 2)],
    "W": [(13, 2), (7, 5), (2, 2)],
}
_COLOR = np.full((32, 3), 100, dtype=np.uint8)


def _make_side(kind, direction):
    pts = {"flat": _FLAT, "tab": _TAB, "blank": _BLANK}[kind][direction]
    return {"type": kind, "points": pts, "color_strip": _COLOR.copy()}


def _make_piece(n_kind, e_kind, s_kind, w_kind, shift=0):
    natural = [_make_side(n_kind, "N"), _make_side(e_kind, "E"),
               _make_side(s_kind, "S"), _make_side(w_kind, "W")]
    if shift == 0:
        stored = natural
    else:
        stored = natural[-shift:] + natural[:-shift]  # right-rotate by `shift`
    return {"mask": _MASK, "sides": stored}


def _make_2x2_puzzle(b_shift=0):
    """
    Known-solution 2x2 layout:
        A(0,0) B(0,1)
        C(1,0) D(1,1)
    """
    A = _make_piece("flat", "tab", "tab", "flat")     # top-left corner
    B = _make_piece("flat", "flat", "tab", "blank", shift=b_shift)  # top-right corner
    C = _make_piece("blank", "tab", "flat", "flat")   # bottom-left corner
    D = _make_piece("blank", "flat", "flat", "blank")  # bottom-right corner
    return [A, B, C, D]


# ---------------------------------------------------------------------------
# side / rotation helpers
# ---------------------------------------------------------------------------

def test_get_side_rotation_zero_is_identity():
    piece = _make_piece("flat", "tab", "tab", "flat")
    assert asm.get_side(piece, 0, "N")["type"] == "flat"
    assert asm.get_side(piece, 0, "E")["type"] == "tab"
    assert asm.get_side(piece, 0, "S")["type"] == "tab"
    assert asm.get_side(piece, 0, "W")["type"] == "flat"


def test_count_flat_sides():
    corner = _make_piece("flat", "tab", "tab", "flat")
    border = _make_piece("flat", "tab", "tab", "blank")
    interior = _make_piece("tab", "tab", "blank", "blank")
    assert asm.count_flat_sides(corner) == 2
    assert asm.count_flat_sides(border) == 1
    assert asm.count_flat_sides(interior) == 0


def test_choose_start_piece_prefers_corner():
    corner = _make_piece("flat", "tab", "tab", "flat")
    border = _make_piece("flat", "tab", "tab", "blank")
    idx, rot = asm.choose_start_piece([border, corner])
    assert idx == 1  # the corner piece, even though listed second
    assert asm.get_side(corner, rot, "N")["type"] == "flat"
    assert asm.get_side(corner, rot, "W")["type"] == "flat"


# ---------------------------------------------------------------------------
# full assembly: known 2x2 solution
# ---------------------------------------------------------------------------

def test_assemble_solves_known_2x2_puzzle():
    pieces = _make_2x2_puzzle()
    result = asm.assemble(pieces, alpha=0.6, beta=0.4, num_samples=16)

    assert result["complete"] is True
    assert result["unplaced_pieces"] == []
    assert result["shape"] == (2, 2)

    grid = result["grid"]
    assert grid[(0, 0)]["piece"] == 0  # A
    assert grid[(0, 1)]["piece"] == 1  # B
    assert grid[(1, 0)]["piece"] == 2  # C
    assert grid[(1, 1)]["piece"] == 3  # D
    # all scores should be (near) zero -- perfect synthetic fit
    for info in grid.values():
        assert info["score"] < 1e-6


def test_assemble_resolves_rotation_when_piece_stored_shifted():
    # piece B's sides are stored right-rotated by 1 -- assembly must
    # discover rotation=1 to recover the correct orientation.
    pieces = _make_2x2_puzzle(b_shift=1)
    result = asm.assemble(pieces, alpha=0.6, beta=0.4, num_samples=16)

    assert result["complete"] is True
    grid = result["grid"]
    assert grid[(0, 1)]["piece"] == 1
    assert grid[(0, 1)]["rotation"] == 1


def test_assemble_all_four_rotations_resolved_correctly():
    for shift in (0, 1, 2, 3):
        pieces = _make_2x2_puzzle(b_shift=shift)
        result = asm.assemble(pieces, alpha=0.6, beta=0.4, num_samples=16)
        assert result["complete"] is True
        assert result["grid"][(0, 1)]["rotation"] == shift % 4


# ---------------------------------------------------------------------------
# dead end / unplaceable piece handling
# ---------------------------------------------------------------------------

def test_assemble_leaves_unplaceable_piece_unplaced_but_returns_rest():
    pieces = _make_2x2_puzzle()
    # a 5th piece whose 4 sides are all "tab" can never mate with anything
    # once the 2x2 solution consumes all 4 matching blanks amongst A-D.
    extra = _make_piece("tab", "tab", "tab", "tab")
    pieces_with_extra = pieces + [extra]

    result = asm.assemble(pieces_with_extra, alpha=0.6, beta=0.4, num_samples=16)

    assert result["complete"] is False
    assert result["unplaced_pieces"] == [4]
    # the 4 solvable pieces should still have been placed correctly
    assert result["shape"] == (2, 2)
    assert len(result["grid"]) == 4


def test_assemble_empty_piece_list():
    result = asm.assemble([])
    assert result["grid"] == {}
    assert result["shape"] == (0, 0)
    assert result["complete"] is True


def test_assemble_single_piece():
    piece = _make_piece("flat", "flat", "flat", "flat")
    result = asm.assemble([piece])
    assert result["complete"] is True
    assert result["shape"] == (1, 1)
    assert result["grid"][(0, 0)]["piece"] == 0


def test_assemble_reports_start_cell():
    pieces = _make_2x2_puzzle()
    result = asm.assemble(pieces, num_samples=16)
    assert result["start_cell"] in result["grid"]
    assert result["grid"][result["start_cell"]]["piece"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
