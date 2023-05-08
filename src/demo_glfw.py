import glfw
import numpy as np
from opengl_utils import *
from cuda_utils import *
import OpenGL.GL as gl
from ctypes import c_int


VERTEX_SHADER_SOURCE = """
#version 460

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 460

layout(location = 0) out vec4 color;

void main() {
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

VERTICES = np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]], dtype=np.float32)


def create_gl_context(shape: tuple[int, int]):
    """creates a glfw window and sets up the opengl context"""
    # glfw boilerplate to create a window
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    # creates opengl context
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    window = glfw.create_window(shape[1], shape[0], "cuda interop demo", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    version_major = gl.glGetIntegerv(gl.GL_MAJOR_VERSION)
    version_minor = gl.glGetIntegerv(gl.GL_MINOR_VERSION)

    print(f"OpenGL version: {version_major}.{version_minor}")

    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glDepthFunc(gl.GL_LESS)
    gl.glViewport(0, 0, shape[1], shape[0])

    return window


def cleanup_cuda(
    array: cudart.cudaArray_t,
    surface: cudart.cudaSurfaceObject_t,
    resource: cudart.cudaGraphicsResource_t,
    module: cuda.CUmodule,
    context: cuda.CUcontext
) -> None:
    """cleanup all the cuda resources"""

    cudart.cudaGraphicsUnregisterResource(resource)
    cudart.cudaDestroySurfaceObject(surface)
    cudart.cudaFreeArray(array)
    cuda.cuModuleUnload(module)
    cuda.cuCtxDestroy(context)


def cleanup_gl(
    fbo: gl.GLuint,
    rbo: gl.GLuint,
    dbo: gl.GLuint,
    vbo: gl.GLuint,
    vao: gl.GLuint,
    shader: gl.GLuint,
) -> None:
    """cleanup all the opengl and egl resources"""

    gl.glDeleteBuffers(1, gl.GLuint(vbo))
    gl.glDeleteVertexArrays(1, gl.GLuint(vao))
    gl.glDeleteFramebuffers(1, gl.GLuint(fbo))
    gl.glDeleteRenderbuffers(1, gl.GLuint(rbo))
    gl.glDeleteRenderbuffers(1, gl.GLuint(dbo))
    gl.glDeleteProgram(gl.GLuint(shader))


def main():
    shape = height, width = 600, 800

    # create a window with glfw and set up the opengl context
    window = create_gl_context(shape)

    # allocate a framebufer with colorbuffer (renderbuffer) and depthbuffer (renderbuffer)
    # we can not use the default framebuffer created with the window, because it doesn't
    # expose the internal renderbuffers/textures which we need to register with cuda
    fbo, rbo, dbo = create_framebuffer(shape)
    # allocate the vbo and vao and copy the vertices into the vbo
    vbo, vao = allocate_buffers(VERTICES)
    # load and compile the shader
    shader = create_shader(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)

    cuda_device, cuda_context = create_cuda_context()
    # load and compile the edge detection kernel
    kernel, module = create_kernel("./src/edge.cu", "edge", cuda_device)

    # allocate an array for the output of the kernel
    output_array = create_empty_array(shape, np.uint32)
    # and bind a surface to it
    output_surface = create_surface(output_array)

    # register the renderbuffer as image in cuda
    image = cuda_check_errors(
        cudart.cudaGraphicsGLRegisterImage(
            rbo,
            gl.GL_RENDERBUFFER,
            cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
    )

    while not glfw.window_should_close(window):
        # bind the fbo, vao and shader and make a draw call
        render(fbo, shader, vao, VERTICES.shape[0])

        # map the resource, could be seen as cuda taking ownership of it
        # opengl operations on it after this call are undefined
        # this needs to happen each iteration because the underlying data
        # is not guaranteed to be stationary
        cuda_check_errors(cudart.cudaGraphicsMapResources(1, image, None))
        # get the underlying array
        input_array = cuda_check_errors(
            cudart.cudaGraphicsSubResourceGetMappedArray(image, 0, 0)
        )
        # and bind a texture to the mapped array
        input_texture = create_texture(input_array, np.uint8)

        # launch the edge detection kernel
        cuda_check_errors(
            cuda.cuLaunchKernel(
                kernel,
                int(np.ceil(width / 32)),
                int(np.ceil(height / 32)),
                1,
                32,
                32,
                1,
                0,
                0,
                (
                    (input_texture, output_surface, width, height),
                    (None, None, c_int, c_int),
                ),
                0,
            )
        )

        # copy the result from the output array back to the input array
        # so that the changes are visible in the opengl renderbuffer
        cuda_check_errors(
            cudart.cudaMemcpy2DArrayToArray(
                input_array,
                0,
                0,
                output_array,
                0,
                0,
                width * np.dtype(np.uint8).itemsize * 4,
                height,
                cudart.cudaMemcpyKind.cudaMemcpyDefault
            )
        )

        # unmap the resource, operations on it in cuda after this call are undefined
        # but opengl can use it again safely
        cuda_check_errors(cudart.cudaGraphicsUnmapResources(1, image, None))

        # destroy the texture because it is not valid anymore
        cudart.cudaDestroyTextureObject(input_texture)

        # blit the manually created framebuffer into the default
        # framebuffer tied to the window
        blit_framebuffer(fbo, shape)

        glfw.poll_events()
        glfw.swap_buffers(window)

    # cleanup all the resources
    cleanup_cuda(
        output_array,
        output_surface,
        image,
        module,
        cuda_context
    )

    cleanup_gl(
        fbo,
        rbo,
        dbo,
        vbo,
        vao,
        shader,
    )

    glfw.terminate()


if __name__ == "__main__":
    main()
