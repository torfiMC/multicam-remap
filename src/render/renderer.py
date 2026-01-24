import ctypes
import math
import numpy as np
from OpenGL import GL
from src.shaders import (
	compile_shader,
	link_program,
	VERT_SRC,
	FRAG_SRC_FLOAT,
	FRAG_SRC_FLOAT_SOFTBORDER,
	FRAG_SRC_RAW,
	FRAG_SRC_PANO,
	GRID_VERT_SRC,
	GRID_FRAG_SRC,
)
from src.math_utils import (
	mat4_perspective,
	mat4_from_yaw_pitch_roll,
	mat4_from_yaw_pitch_roll_turret,
	mat3_from_yaw_pitch_roll_turret,
)


class Renderer:
	def __init__(self, softborder: bool):
		self.softborder = softborder

		vs = compile_shader(VERT_SRC, GL.GL_VERTEX_SHADER)
		fs_src = FRAG_SRC_FLOAT_SOFTBORDER if softborder else FRAG_SRC_FLOAT
		fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
		self.prog = link_program(vs, fs)

		raw_fs = compile_shader(FRAG_SRC_RAW, GL.GL_FRAGMENT_SHADER)
		self.raw_prog = link_program(vs, raw_fs)

		pano_fs = compile_shader(FRAG_SRC_PANO, GL.GL_FRAGMENT_SHADER)
		self.pano_prog = link_program(vs, pano_fs)

		grid_vs = compile_shader(GRID_VERT_SRC, GL.GL_VERTEX_SHADER)
		grid_fs = compile_shader(GRID_FRAG_SRC, GL.GL_FRAGMENT_SHADER)
		self.grid_prog = link_program(grid_vs, grid_fs)

		self.lens_locs = {
			'a_pos': GL.glGetAttribLocation(self.prog, "a_pos"),
			'a_uv': GL.glGetAttribLocation(self.prog, "a_uv"),
			'u_mvp': GL.glGetUniformLocation(self.prog, "u_mvp"),
			'u_src': GL.glGetUniformLocation(self.prog, "u_src"),
			'u_lookup': GL.glGetUniformLocation(self.prog, "u_lookup"),
			'u_uv_offset_x': GL.glGetUniformLocation(self.prog, "u_uv_offset_x"),
			'u_uv_scale_x': GL.glGetUniformLocation(self.prog, "u_uv_scale_x"),
			'u_view_scale': GL.glGetUniformLocation(self.prog, "u_view_scale"),
			'u_view_offset': GL.glGetUniformLocation(self.prog, "u_view_offset"),
		}
		if softborder:
			self.lens_locs['u_mask'] = GL.glGetUniformLocation(self.prog, "u_mask")

		self.raw_locs = {
			'a_pos': GL.glGetAttribLocation(self.raw_prog, "a_pos"),
			'a_uv': GL.glGetAttribLocation(self.raw_prog, "a_uv"),
			'u_mvp': GL.glGetUniformLocation(self.raw_prog, "u_mvp"),
			'u_src': GL.glGetUniformLocation(self.raw_prog, "u_src"),
			'u_orientation_rad': GL.glGetUniformLocation(self.raw_prog, "u_orientation_rad"),
			'u_uv_offset_x': GL.glGetUniformLocation(self.raw_prog, "u_uv_offset_x"),
			'u_uv_scale_x': GL.glGetUniformLocation(self.raw_prog, "u_uv_scale_x"),
		}

		self.grid_locs = {
			'a_pos': GL.glGetAttribLocation(self.grid_prog, "a_pos"),
			'u_mvp': GL.glGetUniformLocation(self.grid_prog, "u_mvp"),
			'u_color': GL.glGetUniformLocation(self.grid_prog, "u_color"),
		}

		self.pano_locs = {
			'a_pos': GL.glGetAttribLocation(self.pano_prog, "a_pos"),
			'a_uv': GL.glGetAttribLocation(self.pano_prog, "a_uv"),
			'u_src': GL.glGetUniformLocation(self.pano_prog, "u_src"),
			'u_mvp': GL.glGetUniformLocation(self.pano_prog, "u_mvp"),
			'u_cam_rot': GL.glGetUniformLocation(self.pano_prog, "u_cam_rot"),
			'u_focal_x': GL.glGetUniformLocation(self.pano_prog, "u_focal_x"),
			'u_focal_y': GL.glGetUniformLocation(self.pano_prog, "u_focal_y"),
			'u_uv_offset_x': GL.glGetUniformLocation(self.pano_prog, "u_uv_offset_x"),
			'u_uv_scale_x': GL.glGetUniformLocation(self.pano_prog, "u_uv_scale_x"),
			'u_view_scale': GL.glGetUniformLocation(self.pano_prog, "u_view_scale"),
			'u_view_offset': GL.glGetUniformLocation(self.pano_prog, "u_view_offset"),
			'u_model': GL.glGetUniformLocation(self.pano_prog, "u_model"),
		}

	def draw_frame(self, fb_size, scene, sphere_mesh, quad_mesh, grid, lenses):
		fb_w, fb_h = fb_size
		GL.glViewport(0, 0, fb_w, fb_h)
		GL.glClearColor(0.0, 0.0, 0.0, 1.0)
		GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

		mode = scene.view_mode if scene.view_mode in ("inside", "orbit", "equirect", "all", "pano") else "inside"

		if mode == "pano":
			self._draw_pano(fb_w, fb_h, quad_mesh, lenses, scene)
			self._reset_state()
			return

		if mode == "equirect":
			self._draw_equirect(quad_mesh, lenses, view_scale=1.0, view_offset=(0.0, 0.0))
			self._reset_state()
			return
		if mode == "all":
			self._draw_all(fb_w, fb_h, quad_mesh, lenses)
			self._reset_state()
			return

		aspect = fb_w / float(fb_h if fb_h else 1)
		proj = mat4_perspective(scene.fov, aspect, 0.1, 100.0)
		view = scene.view_matrix()

		if mode == "inside":
			self._draw_inside(proj, view, sphere_mesh, lenses)
		else:
			self._draw_grid(proj, view, grid)
			self._draw_orbit(proj, view, sphere_mesh, lenses)

		self._reset_state()

	def _bind_lens_uniforms(self, mvp, lens):
		GL.glUniformMatrix4fv(self.lens_locs['u_mvp'], 1, GL.GL_TRUE, mvp)
		GL.glUniform1f(self.lens_locs['u_uv_offset_x'], lens.uv_offset_x)
		GL.glUniform1f(self.lens_locs['u_uv_scale_x'], lens.uv_scale_x)

		GL.glActiveTexture(GL.GL_TEXTURE0)
		GL.glBindTexture(GL.GL_TEXTURE_2D, lens.camera.tex_id)

		GL.glActiveTexture(GL.GL_TEXTURE1)
		GL.glBindTexture(GL.GL_TEXTURE_2D, lens.tex_lookup)

		if self.softborder:
			GL.glActiveTexture(GL.GL_TEXTURE2)
			GL.glBindTexture(GL.GL_TEXTURE_2D, getattr(lens, 'tex_mask', 0))

	def _draw_inside(self, proj, view, sphere_mesh, lenses):
		GL.glDisable(GL.GL_DEPTH_TEST)
		GL.glEnable(GL.GL_CULL_FACE)
		GL.glFrontFace(GL.GL_CCW)
		GL.glCullFace(GL.GL_BACK)
		if self.softborder:
			GL.glEnable(GL.GL_BLEND)
			GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

		GL.glUseProgram(self.prog)
		GL.glUniform1i(self.lens_locs['u_src'], 0)
		GL.glUniform1i(self.lens_locs['u_lookup'], 1)
		if self.softborder:
			GL.glUniform1i(self.lens_locs['u_mask'], 2)
		self._set_view_uniforms(1.0, 0.0, 0.0)

		sphere_mesh.bind(self.lens_locs['a_pos'], self.lens_locs['a_uv'])

		for lens in reversed(lenses):
			m_world = mat4_from_yaw_pitch_roll_turret(lens.world_yaw, lens.world_pitch, lens.world_roll)
			m_orient = mat4_from_yaw_pitch_roll_turret(0.0, 0.0, lens.orientation)
			mvp = proj @ view @ (m_world @ m_orient)
			self._bind_lens_uniforms(mvp, lens)
			GL.glDrawElements(GL.GL_TRIANGLES, sphere_mesh.index_count, GL.GL_UNSIGNED_INT, None)

	def _draw_orbit(self, proj, view, sphere_mesh, lenses):
		GL.glDisable(GL.GL_DEPTH_TEST)
		GL.glDepthMask(GL.GL_TRUE)
		GL.glEnable(GL.GL_CULL_FACE)
		GL.glFrontFace(GL.GL_CW)
		if self.softborder:
			GL.glEnable(GL.GL_BLEND)
			GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

		GL.glUseProgram(self.prog)
		GL.glUniform1i(self.lens_locs['u_src'], 0)
		GL.glUniform1i(self.lens_locs['u_lookup'], 1)
		if self.softborder:
			GL.glUniform1i(self.lens_locs['u_mask'], 2)
		self._set_view_uniforms(1.0, 0.0, 0.0)

		sphere_mesh.bind(self.lens_locs['a_pos'], self.lens_locs['a_uv'])

		layer_bias = -0.001
		layer_step = -1.0
		GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
		for cull_face in (GL.GL_FRONT, GL.GL_BACK):
			GL.glCullFace(cull_face)
			for idx, lens in enumerate(reversed(lenses)):
				m_world = mat4_from_yaw_pitch_roll_turret(lens.world_yaw, lens.world_pitch, lens.world_roll)
				m_orient = mat4_from_yaw_pitch_roll_turret(0.0, 0.0, lens.orientation)
				mvp = proj @ view @ (m_world @ m_orient)

				GL.glPolygonOffset(0.0, layer_bias + layer_step * idx)
				self._bind_lens_uniforms(mvp, lens)
				GL.glDrawElements(GL.GL_TRIANGLES, sphere_mesh.index_count, GL.GL_UNSIGNED_INT, None)
		GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

	def _draw_grid(self, proj, view, grid):
		GL.glEnable(GL.GL_BLEND)
		GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
		GL.glUseProgram(self.grid_prog)
		GL.glUniformMatrix4fv(self.grid_locs['u_mvp'], 1, GL.GL_TRUE, proj @ view)
		GL.glUniform4f(self.grid_locs['u_color'], 0.22, 0.22, 0.22, 0.6)
		grid.draw(self.grid_locs['a_pos'])
		GL.glDisable(GL.GL_BLEND)

	def _draw_equirect(self, quad_mesh, lenses, view_scale: float, view_offset: tuple[float, float]):
		# Render full-frame equirect using the pano shader so lens orientation is honored.
		self._draw_pano_common(
			fb_w=None,
			fb_h=None,
			quad_mesh=quad_mesh,
			lenses=lenses,
			view_scale=view_scale,
			view_offset=view_offset,
			square=False,
		)

	def _draw_pano(self, fb_w, fb_h, quad_mesh, lenses, scene):
		if not lenses:
			return

		self._draw_pano_common(
			fb_w=fb_w,
			fb_h=fb_h,
			quad_mesh=quad_mesh,
			lenses=lenses,
			view_scale=scene.pano_view()[0],
			view_offset=(scene.pano_view()[1], scene.pano_view()[2]),
			square=True,
		)

	def _draw_pano_common(self, fb_w, fb_h, quad_mesh, lenses, view_scale: float, view_offset: tuple[float, float], square: bool):
		GL.glDisable(GL.GL_DEPTH_TEST)
		GL.glDisable(GL.GL_CULL_FACE)
		if self.softborder:
			GL.glEnable(GL.GL_BLEND)
			GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

		GL.glUseProgram(self.pano_prog)
		GL.glUniform1i(self.pano_locs['u_src'], 0)
		GL.glUniform1f(self.pano_locs['u_view_scale'], float(view_scale))
		GL.glUniform2f(self.pano_locs['u_view_offset'], float(view_offset[0]), float(view_offset[1]))

		if square and fb_w is not None and fb_h is not None:
			side = max(1, min(fb_w, fb_h))
			offset_x = (fb_w - side) // 2
			offset_y = (fb_h - side) // 2
			GL.glViewport(int(offset_x), int(offset_y), int(side), int(side))
		elif fb_w is not None and fb_h is not None:
			GL.glViewport(0, 0, fb_w, fb_h)

		mvp = np.eye(4, dtype=np.float32)
		GL.glUniformMatrix4fv(self.pano_locs['u_mvp'], 1, GL.GL_TRUE, mvp)

		quad_mesh.bind(self.pano_locs['a_pos'], self.pano_locs['a_uv'])

		for lens in reversed(lenses):
			cam_rot = _cam_rotation_world_to_camera(lens)
			GL.glUniformMatrix3fv(self.pano_locs['u_cam_rot'], 1, GL.GL_TRUE, cam_rot)
			GL.glUniform1f(self.pano_locs['u_focal_x'], getattr(lens, 'focal_scale_x', 1.0))
			GL.glUniform1f(self.pano_locs['u_focal_y'], getattr(lens, 'focal_scale_y', 1.0))
			GL.glUniform1f(self.pano_locs['u_uv_offset_x'], lens.uv_offset_x)
			GL.glUniform1f(self.pano_locs['u_uv_scale_x'], lens.uv_scale_x)
			GL.glUniform1i(self.pano_locs['u_model'], int(getattr(lens, 'model_code', 0)))

			GL.glActiveTexture(GL.GL_TEXTURE0)
			GL.glBindTexture(GL.GL_TEXTURE_2D, lens.camera.tex_id)

			GL.glDrawElements(GL.GL_TRIANGLES, quad_mesh.index_count, GL.GL_UNSIGNED_INT, None)

	def _draw_all(self, fb_w, fb_h, quad_mesh, lenses):
		if not lenses:
			return

		GL.glDisable(GL.GL_DEPTH_TEST)
		GL.glDisable(GL.GL_CULL_FACE)
		GL.glUseProgram(self.raw_prog)
		GL.glUniform1i(self.raw_locs['u_src'], 0)
		GL.glUniform1f(self.raw_locs['u_uv_offset_x'], 0.0)
		GL.glUniform1f(self.raw_locs['u_uv_scale_x'], 1.0)
		GL.glUniform1f(self.raw_locs['u_orientation_rad'], 0.0)

		mvp = np.eye(4, dtype=np.float32)
		GL.glUniformMatrix4fv(self.raw_locs['u_mvp'], 1, GL.GL_TRUE, mvp)

		quad_mesh.bind(self.raw_locs['a_pos'], self.raw_locs['a_uv'])

		count = len(lenses)
		cols = max(1, count)
		rows = 1

		cell_w = max(1, fb_w // cols)
		cell_h = max(1, fb_h)

		for idx, lens in enumerate(lenses):
			cam = lens.camera
			cam_w = getattr(cam, 'actual_w', None) or getattr(cam, 'width', None) or 1
			cam_h = getattr(cam, 'actual_h', None) or getattr(cam, 'height', None) or 1
			slice_scale = getattr(lens, 'uv_scale_x', 1.0) or 1.0
			cam_w = max(1, int(round(float(cam_w) * slice_scale)))
			cam_h = max(1, int(round(float(cam_h))))
			cam_aspect = cam_h / float(cam_w)

			GL.glUniform1f(self.raw_locs['u_uv_offset_x'], getattr(lens, 'uv_offset_x', 0.0))
			GL.glUniform1f(self.raw_locs['u_uv_scale_x'], getattr(lens, 'uv_scale_x', 1.0))
			GL.glUniform1f(
				self.raw_locs['u_orientation_rad'],
				math.radians(-1.0 * getattr(lens, 'orientation', 0.0)),
			)

			row = idx // cols
			col = idx % cols

			base_x = col * cell_w
			base_y = (rows - 1 - row) * cell_h

			draw_w = cell_w
			draw_h = cell_h
			cell_aspect = cell_w / float(cell_h if cell_h else 1.0)

			if cam_aspect > cell_aspect:
				draw_w = cell_w
				draw_h = max(1, int(round(cell_w / cam_aspect)))
				base_y += (cell_h - draw_h) // 2
			else:
				draw_h = cell_h
				draw_w = max(1, int(round(cell_h * cam_aspect)))
				base_x += (cell_w - draw_w) // 2

			GL.glViewport(int(base_x), int(base_y), int(draw_w), int(draw_h))
			GL.glActiveTexture(GL.GL_TEXTURE0)
			GL.glBindTexture(GL.GL_TEXTURE_2D, cam.tex_id)
			GL.glDrawElements(GL.GL_TRIANGLES, quad_mesh.index_count, GL.GL_UNSIGNED_INT, None)

	def _reset_state(self):
		GL.glFrontFace(GL.GL_CCW)
		GL.glDisable(GL.GL_CULL_FACE)
		GL.glDisable(GL.GL_DEPTH_TEST)
		GL.glDepthMask(GL.GL_TRUE)
		GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
		GL.glDisable(GL.GL_BLEND)

	def _set_view_uniforms(self, scale: float, offset_u: float, offset_v: float):
		loc_scale = self.lens_locs.get('u_view_scale', -1)
		loc_offset = self.lens_locs.get('u_view_offset', -1)
		if loc_scale is not None and loc_scale >= 0:
			GL.glUniform1f(loc_scale, float(scale))
		if loc_offset is not None and loc_offset >= 0:
			GL.glUniform2f(loc_offset, float(offset_u), float(offset_v))


def _cam_rotation_world_to_camera(lens) -> np.ndarray:
	"""Return a world->camera 3x3 rotation for the pano shader."""
	R_world = mat3_from_yaw_pitch_roll_turret(lens.world_yaw, lens.world_pitch, lens.world_roll)
	R_orient = mat3_from_yaw_pitch_roll_turret(0.0, 0.0, lens.orientation)
	cam_to_world = R_world @ R_orient
	world_to_cam = cam_to_world.T
	return world_to_cam.astype(np.float32)
