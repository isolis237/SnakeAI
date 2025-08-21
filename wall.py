# wall.py
import pygame as pg
import random
from snake import SnakeGame

class SnakeWall:
    def __init__(self, rows=10, cols=10, grid_w=None, grid_h=None, cell=None,
                win_w=800, win_h=800, fps=12, seed=None, restart_finished=True):

        self.rows, self.cols = rows, cols
        self.fps = fps
        self.restart_finished = restart_finished
        self.rng = random.Random(seed)
        self.show_score = True if self.rows * self.cols < 50 else False

        # decide tile size
        self.tile_w = win_w // cols
        self.tile_h = win_h // rows

        # case 1: cell given → compute grid size
        if cell is not None:
            self.cell = cell
            self.grid_w = self.tile_w // cell if grid_w is None else grid_w
            self.grid_h = self.tile_h // cell if grid_h is None else grid_h
        # case 2: grid size given → compute cell size
        elif grid_w and grid_h:
            self.grid_w = grid_w
            self.grid_h = grid_h
            self.cell = min(self.tile_w // grid_w, self.tile_h // grid_h)
        else:
            raise ValueError("Must provide either cell size or grid_w+grid_h")

        # recompute tile size so everything fits neatly
        self.tile_w = self.grid_w * self.cell
        self.tile_h = self.grid_h * self.cell
        self.win_w = self.tile_w * cols
        self.win_h = self.tile_h * rows

        pg.init()
        self.screen = pg.display.set_mode((self.win_w, self.win_h))
        self.screen = pg.display.set_mode((self.win_w, self.win_h))
        pg.display.set_caption("Snake - Multi")
        self.clock = pg.time.Clock()
        self.font_small = pg.font.Font(None, 18)
        self.font_big = pg.font.Font(None, 28)

        # build subsurfaces
        self.tiles = [
            [self.screen.subsurface(pg.Rect(c * self.tile_w, r * self.tile_h, self.tile_w, self.tile_h))
             for c in range(self.cols)]
            for r in range(self.rows)
        ]

        # create games
        self.games = []
        for r in range(self.rows):
            for c in range(self.cols):
                g = SnakeGame(grid_w=self.grid_w, grid_h=self.grid_h, cell=self.cell, fps=self.fps,
                              seed=self.rng.randint(0, 1_000_000))
                g.init(surface=self.tiles[r][c])
                g.reset()
                self.games.append(g)

        self.running = True
        self.best_ever = 0  # track overall high score for the wall

    def _overall_best_current(self):
        """Return (best_score, indices_with_that_score)."""
        scores = [g.score for g in self.games]
        if not scores:
            return 0, []
        best = max(scores)
        leaders = [i for i, s in enumerate(scores) if s == best]
        return best, leaders

    def run(self):
        while self.running:
            # events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False

            # step all games
            alive = 0
            for g in self.games:
                if not g.done and g.running:
                    action = self.rng.choice([0, 1, 2])
                    _, _, done, _ = g.step(action)
                    if not done:
                        alive += 1

            # compute leaders
            best_now, leaders = self._overall_best_current()
            if best_now > self.best_ever:
                self.best_ever = best_now

            # draw overlays (per tile)
            for i, g in enumerate(self.games):
                g.draw_overlay(self.font_small, is_leader=(i in leaders), show_score=self.show_score)

            # draw overall banner on main screen
            banner = self.font_big.render(f"Current Best: {best_now}   Overall Best: {self.best_ever}", True, (245, 245, 245))
            self.screen.blit(banner, (8, 8))

            pg.display.flip()
            self.clock.tick(self.fps)

            # keep wall alive by restarting finished games (optional)
            if self.restart_finished and alive < len(self.games):
                for g in self.games:
                    if g.done and g.running:
                        g.reset()

        pg.quit()
