from cuda import cuda, nvrtc, cudart  # type: ignore
import OpenGL.EGL as egl
import OpenGL.GL as gl
from ctypes import pointer, c_int
from cuda_utils import cuda_check_errors
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path
import os




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


def read_pixels(fbo: gl.GLuint, shape: tuple[int, int]) -> npt.NDArray:
    """reads pixels from a framebuffer into a numpy array"""

    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    pixels = gl.glReadPixels(0, 0, shape[1], shape[0], gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    return np.frombuffer(pixels, dtype=np.uint8).reshape(shape[0], shape[1], 4)  # type: ignore


def create_cuda_context() -> tuple[cuda.CUdevice, cuda.CUcontext]:
    """creates a cuda context on the first available device"""

    cuda_check_errors(cuda.cuInit(0))
    device = cuda_check_errors(cuda.cuDeviceGet(0))
    context = cuda_check_errors(cuda.cuCtxCreate(0, device))

    return device, context


def create_kernel(
    source_file: Path, name: str, device: cuda.CUdevice
) -> tuple[cuda.CUfunction, cuda.CUmodule]:
    """loades and compiles a cuda kernel from a source file and then retrieves
    the function by name"""

    with open(source_file, "r") as file:
        source = file.read()

    program = cuda_check_errors(
        nvrtc.nvrtcCreateProgram(source.encode(), b"temp.cu", 0, [], [])
    )

    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home is None:
        raise RuntimeError("CUDA_HOME not set")

    include_dir = os.path.join(cuda_home, "include")
    include_arg = f"--include-path={include_dir}".encode()

    major = cuda_check_errors(
        cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device
        )
    )
    minor = cuda_check_errors(
        cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device
        )
    )

    arch_arg = f"--gpu-architecture=compute_{major}{minor}".encode()
    opts = [b"--fmad=true", arch_arg, include_arg, b"--std=c++17", b"-default-device"]

    try:
        cuda_check_errors(nvrtc.nvrtcCompileProgram(program, len(opts), opts))
    except RuntimeError:
        log_size = cuda_check_errors(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * log_size  # type: ignore

        cuda_check_errors(nvrtc.nvrtcGetProgramLog(program, log))
        raise RuntimeError(log.decode())

    data_size = cuda_check_errors(nvrtc.nvrtcGetPTXSize(program))
    data = b" " * data_size  # type: ignore

    cuda_check_errors(nvrtc.nvrtcGetPTX(program, data))
    module = cuda_check_errors(cuda.cuModuleLoadData(np.char.array(data)))
    cuda_check_errors(nvrtc.nvrtcDestroyProgram(program))

    return cuda_check_errors(cuda.cuModuleGetFunction(module, name.encode())), module


def __dtype_to_channel_format(dtype: npt.DTypeLike) -> cudart.cudaChannelFormatKind:
    """helper to convert numpy dtype to cuda channel format kind"""

    if np.issubdtype(dtype, np.floating):
        return cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat
    if np.issubdtype(dtype, np.signedinteger):
        return cudart.cudaChannelFormatKind.cudaChannelFormatKindSigned

    return cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned


def create_empty_array(
    shape: tuple[int, int], dtype: npt.DTypeLike
) -> cudart.cudaArray_t:
    """creates and empty 2d array on the device, dtype is used to determine
    memory size of each element in the array"""

    format_kind = __dtype_to_channel_format(dtype)

    rows, cols = shape
    channel_desc = cuda_check_errors(
        cudart.cudaCreateChannelDesc(8 * np.dtype(dtype).itemsize, 0, 0, 0, format_kind)
    )

    return cuda_check_errors(
        cudart.cudaMallocArray(channel_desc, cols, rows, cudart.cudaArrayDefault)
    )


def create_texture(
    array: cudart.cudaArray_t, dtype: npt.DTypeLike
) -> cudart.cudaTextureObject_t:
    """creates a texture object from the given array through which we
    can read the array with some caching benefits"""

    resource_desc = cudart.cudaResourceDesc()
    resource_desc.resType = cudart.cudaResourceType.cudaResourceTypeArray
    resource_desc.res.array.array = array

    read_normalized = dtype == np.uint8 or dtype == np.uint16
    texture_desc = cudart.cudaTextureDesc()
    texture_desc.addressMode[0] = cudart.cudaTextureAddressMode.cudaAddressModeBorder
    texture_desc.addressMode[1] = cudart.cudaTextureAddressMode.cudaAddressModeBorder
    texture_desc.borderColor[0] = 0
    texture_desc.borderColor[1] = 0
    texture_desc.borderColor[2] = 0
    texture_desc.borderColor[3] = 0
    texture_desc.filterMode = cudart.cudaTextureFilterMode.cudaFilterModePoint
    texture_desc.readMode = (
        cudart.cudaTextureReadMode.cudaReadModeNormalizedFloat
        if read_normalized
        else cudart.cudaTextureReadMode.cudaReadModeElementType
    )
    texture_desc.normalizedCoords = 0

    return cuda_check_errors(
        cudart.cudaCreateTextureObject(resource_desc, texture_desc, None)
    )


def create_surface(array: cudart.cudaArray_t) -> cudart.cudaSurfaceObject_t:
    """creates a surface object from the given array through which we can edit the array"""

    resource_desc = cudart.cudaResourceDesc()
    resource_desc.resType = cudart.cudaResourceType.cudaResourceTypeArray
    resource_desc.res.array.array = array

    return cuda_check_errors(cudart.cudaCreateSurfaceObject(resource_desc))


def register_image(renderbuffer: gl.GLuint) -> cudart.cudaGraphicsResource_t:
    """registers the renderbuffer as a cudaGraphicsResource_t in cuda"""

    return cuda_check_errors(
        cudart.cudaGraphicsGLRegisterImage(
            renderbuffer,
            gl.GL_RENDERBUFFER,
            cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
    )


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
    egl_display, egl_context = create_gl_context(shape)
    cuda_device, cuda_context = create_cuda_context()

    # allocate a framebufer with colorbuffer (renderbuffer) and depthbuffer (renderbuffer)
    fbo, rbo, dbo = create_framebuffer(shape)
    # allocate the vbo and vao and copy the vertices into the vbo
    vbo, vao = allocate_buffers(VERTICES)
    # load and compile the shader
    shader = create_shader(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)

    # bind the fbo, vao and shader and draw call
    render(fbo, shader, vao, VERTICES.shape[0])
    # read the resulting image back to the host
    img1 = read_pixels(fbo, shape)

    # load and compile the edge detection kernel
    kernel, module = create_kernel(Path("./src/edge.cu"), "edge", cuda_device)

    # allocate an array for the output of the kernel
    output_array = create_empty_array(shape, np.uint32)
    # and bind a surface to it
    output_surface = create_surface(output_array)

    # register the renderbuffer as image in cuda
    image = register_image(rbo)
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
