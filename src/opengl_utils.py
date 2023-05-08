import OpenGL.GL as gl
import numpy as np
import numpy.typing as npt




def create_framebuffer(shape: tuple[int, int]):
    """creates a framebuffer with rgba uint8 renderbuffer color
    attachment and a depth attachment"""

    height, width = shape
    framebuffer = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)

    renderbuffer = gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, renderbuffer)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA, width, height)
    gl.glFramebufferRenderbuffer(
        gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, renderbuffer
    )

    depthbuffer = gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depthbuffer)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, width, height)
    gl.glFramebufferRenderbuffer(
        gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depthbuffer
    )

    status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("failed to create framebuffer")

    return framebuffer, renderbuffer, depthbuffer


def create_shader(vertex_source: str, fragment_source: str):
    """loads and compiles the shader from the given fragment
    and vertex shader source files"""

    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, vertex_source)
    gl.glCompileShader(vertex_shader)

    # checking for compiler errors omitted

    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, fragment_source)
    gl.glCompileShader(fragment_shader)

    # checking for compiler errors omitted

    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)

    # checking for linker errors omitted

    return program


def allocate_buffers(vertices: np.ndarray):
    """allocates the vbo and vao, copies the vertices
    into it and set the vertex attributes"""

    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(
        gl.GL_ARRAY_BUFFER,
        vertices.nbytes,
        vertices,
        gl.GL_STATIC_DRAW,
    )

    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    gl.glVertexAttribPointer(
        0, 2, gl.GL_FLOAT, gl.GL_FALSE, 2 * vertices.itemsize, None
    )
    gl.glEnableVertexAttribArray(0)

    return vbo, vao


def render(fbo: gl.GLuint, shader: gl.GLuint, vao: gl.GLuint, n_vertices: int):
    """renders the vao on the fbo with shader"""

    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # type: ignore
    gl.glUseProgram(shader)
    gl.glBindVertexArray(vao)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, n_vertices)
    gl.glBindVertexArray(0)


def blit_framebuffer(fbo: gl.GLuint, shape: tuple[int, int]):
    """blits the fbo into the default framebuffer"""

    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
    gl.glBlitFramebuffer(
        0,
        0,
        shape[1],
        shape[0],
        0,
        0,
        shape[1],
        shape[0],
        gl.GL_COLOR_BUFFER_BIT,
        gl.GL_NEAREST,
    )


def read_pixels(fbo: gl.GLuint, shape: tuple[int, int]) -> npt.NDArray:
    """reads pixels from a framebuffer into a numpy array"""

    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    pixels = gl.glReadPixels(0, 0, shape[1], shape[0], gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    return np.frombuffer(pixels, dtype=np.uint8).reshape(shape[0], shape[1], 4)  # type: ignore
