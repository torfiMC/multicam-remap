import ctypes
from OpenGL import GL
from src.geometry import make_grid_lines


class Grid:
    def __init__(self, extent: float = 25.0, spacing: float = 1.0):
        verts = make_grid_lines(extent=extent, spacing=spacing)
        self.vert_count = len(verts)
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)

    def draw(self, loc_pos: int):
        if self.vert_count == 0:
            return
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableVertexAttribArray(loc_pos)
        GL.glVertexAttribPointer(loc_pos, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glLineWidth(1.0)
        GL.glDrawArrays(GL.GL_LINES, 0, self.vert_count)
