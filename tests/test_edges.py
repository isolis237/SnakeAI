# tests/test_edges.py
import pygame as pg

def _rgb(c):
    cc = pg.Color(c)
    return (cc.r, cc.g, cc.b)

def test_edge_endpoints_clip_true(node_factory, edge_factory):
    a = node_factory(pos=(100, 120), r=10, color=(0, 255, 0))
    b = node_factory(pos=(200, 120), r=20, color=(0, 255, 0))
    e = edge_factory(a, b, clip_to_circle=True, aa=False, width=1)
    p1, p2 = e._endpoints()
    # With horizontal alignment, clipping should move endpoints by radii
    assert p1 == (100 + a.r, 120)   # 110, 120
    assert p2 == (200 - b.r, 120)   # 180, 120

def test_edge_endpoints_clip_false(node_factory, edge_factory):
    a = node_factory(pos=(100, 120), r=10)
    b = node_factory(pos=(200, 120), r=20)
    e = edge_factory(a, b, clip_to_circle=False, aa=False, width=1)
    p1, p2 = e._endpoints()
    assert p1 == a.rect.center
    assert p2 == b.rect.center

def test_edge_draw_colors_midpoint(node_factory, edge_factory, screen):
    bg = (0, 0, 0, 0)
    screen.fill(bg)
    a = node_factory(pos=(100, 120), r=10)
    b = node_factory(pos=(200, 120), r=10)
    color = (255, 0, 0)  # bright red
    e = edge_factory(a, b, color=color, width=3, aa=False, clip_to_circle=True)
    e.draw(screen)
    # sample a point halfway between the clipped endpoints (should be on the line)
    midx = (100 + a.r + 200 - b.r) // 2
    midy = 120
    pix = screen.get_at((midx, midy))
    assert _rgb(pix) == _rgb(color)

def test_edge_moves_with_nodes(node_factory, edge_factory):
    a = node_factory(pos=(100, 120), r=10)
    b = node_factory(pos=(200, 120), r=10)
    e = edge_factory(a, b, clip_to_circle=True, aa=False, width=1)
    p1_before, p2_before = e._endpoints()
    a.rect.x += 20  # move left endpoint right
    p1_after, p2_after = e._endpoints()
    assert p1_after != p1_before
    assert p2_after == p2_before  # b didn't move
