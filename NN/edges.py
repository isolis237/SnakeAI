import pygame as pg
def weight_to_color(weight):
    """
    Map weight in [0, 1] to RGB.
    0 = red (240, 40, 40)
    1 = green (40, 240, 140)
    Clamp anything outside [0, 1].
    """
    w = max(0.0, min(1.0, weight))  # clamp to [0, 1]

    r_start, g_start, b_start = 240, 40, 40
    r_end, g_end, b_end = 40, 240, 140

    r = int(r_start + (r_end - r_start) * w)
    g = int(g_start + (g_end - g_start) * w)
    b = int(b_start + (b_end - b_start) * w)

    return (r, g, b)


class Edge:
    def __init__(self, nodeA, nodeB, weight=0, color=(40,240,140), width=3, aa=True, clip_to_circle=True):
        if not (hasattr(nodeA, "rect") and hasattr(nodeB, "rect")):
            raise TypeError("Edge endpoints must have a .rect")
        self.weight = weight
        self.a, self.b = nodeA, nodeB
        self.p1c, self.p2c = self.a.rect.center, self.b.rect.center
        self.color = weight_to_color(weight)
        self.width = width
        self.aa = aa
        self.clip = clip_to_circle  # True = clip to node radii if available

    def _update_color(self):
        self.color = weight_to_color(self.weight)

    def _endpoints(self):
        p1 = pg.Vector2(self.a.rect.center)
        p2 = pg.Vector2(self.b.rect.center)
        if not self.clip:
            return (round(p1.x), round(p1.y)), (round(p2.x), round(p2.y))

        # Use node.r if present, else approximate from rect size
        r1 = getattr(self.a, "r", min(self.a.rect.w, self.a.rect.h) // 2)
        r2 = getattr(self.b, "r", min(self.b.rect.w, self.b.rect.h) // 2)

        v = p2 - p1
        if v.length_squared() == 0:
            return (round(p1.x), round(p1.y)), (round(p2.x), round(p2.y))
        u = v.normalize()

        p1c = p1 + u * r1
        p2c = p2 - u * r2

        self.p1c, self.p2c = p1c, p2c
        return (round(p1c.x), round(p1c.y)), (round(p2c.x), round(p2c.y))

    def draw(self, surface):
        p1, p2 = self._endpoints()

        # Draw the edge
        if self.width > 1:
            pg.draw.line(surface, self.color, p1, p2, self.width)
            if self.aa:
                pg.draw.aaline(surface, self.color, p1, p2)
        else:
            if self.aa:
                pg.draw.aaline(surface, self.color, p1, p2)
            else:
                pg.draw.line(surface, self.color, p1, p2, 1)

        # === Weight label ===
        label_str = f"{self.weight:.2f}"
        font = pg.font.Font(None, max(14, int(12 + self.width)))
        text_surf = font.render(label_str, True, (100, 100, 20))

        pad = 4
        bg = pg.Surface((text_surf.get_width() + pad*2, text_surf.get_height() + pad*2), pg.SRCALPHA)
        pg.draw.rect(bg, (255, 255, 255, 170), bg.get_rect(), border_radius=6)
        bg.blit(text_surf, (pad, pad))

        # Use the clipped endpoints (on the circle borders) to get the edge midpoint
        # self._endpoints() already set self.p1c / self.p2c
        edge_mid = (self.p1c + self.p2c) * 0.5

        # Find which node is on the left by x, then use THAT NODE'S CENTER
        a_center = pg.Vector2(self.a.rect.center)
        b_center = pg.Vector2(self.b.rect.center)
        left_center = a_center if a_center.x <= b_center.x else b_center

        # Target is halfway between left node center and edge midpoint
        target = (left_center + edge_mid) * 0.525

        # Small perpendicular nudge off the line to improve readability (optional)
        v = self.p2c - self.p1c
        if v.length_squared() != 0:
            n = pg.Vector2(-v.y, v.x).normalize()
            target += n * (4 + 0.5 * self.width)

        bg_rect = bg.get_rect(center=(int(target.x), int(target.y)))
        surface.blit(bg, bg_rect)


