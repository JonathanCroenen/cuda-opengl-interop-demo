from cuda import cuda, cudart  # type: ignore
import OpenGL.EGL as egl
import OpenGL.GL as gl
from ctypes import c_int, pointer

from opengl_utils import *
from cuda_utils import *
import numpy as np
import matplotlib.pyplot as plt

import os
# so that PyOpenGL prioritizes egl over the native api
os.environ["PYOPENGL_PLATFORM"] = "egl"


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

# this triangle is upsidedown, but when reading the buffer to the host, it is flipped back
# to the correct orientation. I left it like this so that I don't have the flip it after reading
VERTICES = np.array([[-0.5, 0.5], [0.5, 0.5], [0.0, -0.5]], dtype=np.float32)


def create_gl_context(shape: tuple[int, int]):
    """creates and egl context which also sets up the opengl context
    for offscreen rendering"""

    height, width = shape
    egl_display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)

    major, minor = egl.EGLint(), egl.EGLint()
    egl.eglInitialize(egl_display, pointer(major), pointer(minor))

    config_attribs = [
        egl.EGL_SURFACE_TYPE,
        egl.EGL_PBUFFER_BIT,
        egl.EGL_BLUE_SIZE,
        8,
        egl.EGL_GREEN_SIZE,
        8,
        egl.EGL_RED_SIZE,
        8,
        egl.EGL_DEPTH_SIZE,
        24,
        egl.EGL_RENDERABLE_TYPE,
        egl.EGL_OPENGL_BIT,
        egl.EGL_NONE,
    ]

    # multiplying a ctype by an int makes a new array type of that ctype
    # for example: (c_int * 3) results in the type c_int_Array_3 and (c_int * 3)(1, 2, 3)
    # then instantiates an object of that type
    config_attribs = (egl.EGLint * len(config_attribs))(*config_attribs)

    egl_config = egl.EGLConfig()
    num_configs = egl.EGLint()
    egl.eglChooseConfig(
        egl_display, config_attribs, pointer(egl_config), 1, pointer(num_configs)
    )

    pbuffer_attribs = [
        egl.EGL_WIDTH,
        width,
        egl.EGL_HEIGHT,
        height,
        egl.EGL_NONE,
    ]

    pbuffer_attribs = (egl.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
    # creates an offscreen pixel buffer for the default framebuffer if using opengl
    surface = egl.eglCreatePbufferSurface(egl_display, egl_config, pbuffer_attribs)

    # tell it to also create an opengl context
    egl.eglBindAPI(egl.EGL_OPENGL_API)

    egl_context = egl.eglCreateContext(
        egl_display, egl_config, egl.EGL_NO_CONTEXT, None
    )
    # binds the context to the current thread
    egl.eglMakeCurrent(egl_display, surface, surface, egl_context)

    version_major = gl.glGetIntegerv(gl.GL_MAJOR_VERSION)
    version_minor = gl.glGetIntegerv(gl.GL_MINOR_VERSION)

    print(f"OpenGL version: {version_major}.{version_minor}")

    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glDepthFunc(gl.GL_LESS)
    gl.glViewport(0, 0, width, height)

    return egl_display, egl_context


def cleanup_cuda(
    array: cudart.cudaArray_t,
    texture: cudart.cudaTextureObject_t,
    surface: cudart.cudaSurfaceObject_t,
    resource: cudart.cudaGraphicsResource_t,
    module: cuda.CUmodule,
    context: cuda.CUcontext
) -> None:
    """cleanup all the cuda resources"""

    cudart.cudaGraphicsUnregisterResource(resource)
    cudart.cudaDestroySurfaceObject(surface)
    cudart.cudaDestroyTextureObject(texture)
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
    context: egl.EGLContext,
    display: egl.EGLDisplay
) -> None:
    """cleanup all the opengl and egl resources"""

    gl.glDeleteBuffers(1, gl.GLuint(vbo))
    gl.glDeleteVertexArrays(1, gl.GLuint(vao))
    gl.glDeleteFramebuffers(1, gl.GLuint(fbo))
    gl.glDeleteRenderbuffers(1, gl.GLuint(rbo))
    gl.glDeleteRenderbuffers(1, gl.GLuint(dbo))
    gl.glDeleteProgram(gl.GLuint(shader))
    egl.eglDestroyContext(display, context)
    egl.eglTerminate(display)


def main():
    shape = height, width = 600, 800
    # create the opengl context with egl
    egl_display, egl_context = create_gl_context(shape)
    # create the cuda context
    cuda_device, cuda_context = create_cuda_context()

    # allocate a framebufer with colorbuffer (renderbuffer) and depthbuffer (renderbuffer)
    fbo, rbo, dbo = create_framebuffer(shape)
    # allocate the vbo and vao and copy the vertices into the vbo
    vbo, vao = allocate_buffers(VERTICES)
    # load and compile the shader
    shader = create_shader(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)

    # bind the fbo, vao and shader and make a draw call
    render(fbo, shader, vao, VERTICES.shape[0])
    # read the resulting image back to the host
    img1 = read_pixels(fbo, shape)

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
    # map the resource, could be seen as cuda taking ownership of it
    # opengl operations on it after this call are undefined
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

    # prep a host array to copy the result into
    values = np.empty((*shape, 4), dtype=np.uint8)

    # copy the result from the ouput array to the host so we can look at it
    cuda_check_errors(
        cudart.cudaMemcpy2DFromArray(
            values.ctypes.data,
            values.shape[1] * values.itemsize * 4,
            output_array,
            0,
            0,
            values.shape[1] * values.itemsize * 4,
            values.shape[0],
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
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
    # read the pixels from the renderbuffer again, which should now contain
    # the result of the edge detection kernel
    img2 = read_pixels(fbo, shape)

    plt.figure()
    ax1 = plt.subplot(311)
    ax1.title.set_text("after opengl draw")
    plt.imshow(img1)
    ax2 = plt.subplot(312)
    ax2.title.set_text("after cuda kernel (from cuda array)")
    plt.imshow(values)
    ax3 = plt.subplot(313)
    ax3.title.set_text("after cuda kernel (from opengl renderbuffer)")
    plt.imshow(img2)
    plt.tight_layout(pad=0.5)
    plt.show()

    # cleanup all the resources
    cleanup_cuda(
        output_array,
        input_texture,
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
        egl_context,
        egl_display
    )




if __name__ == "__main__":
    main()
