from cuda import cuda, nvrtc, cudart  # type: ignore
import os
import numpy as np
import numpy.typing as npt


def cuda_get_error_enum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return (
            name.decode()
            if err == cuda.CUresult.CUDA_SUCCESS
            else "<unknown driver error>"
        )
    elif isinstance(error, cudart.cudaError_t):
        err, name = cudart.cudaGetErrorName(error)
        return (
            name.decode()
            if err == cudart.cudaError_t.cudaSuccess
            else "<unknown runtime error>"
        )
    elif isinstance(error, nvrtc.nvrtcResult):
        err, name = nvrtc.nvrtcGetErrorString(error)
        return (
            name.decode()
            if err == nvrtc.nvrtcResult.NVRTC_SUCCESS
            else "<unknown nvrtc error>"
        )
    else:
        return "<unknown error>"


def cuda_check_errors(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA Error code: {} ({})".format(
                result[0].value, cuda_get_error_enum(result[0])
            )
        )

    if len(result) == 2:
        return result[1]
    elif len(result) > 2:
        return result[1:]


def create_cuda_context() -> tuple[cuda.CUdevice, cuda.CUcontext]:
    """creates a cuda context on the first available device"""

    cuda_check_errors(cuda.cuInit(0))
    device = cuda_check_errors(cuda.cuDeviceGet(0))
    context = cuda_check_errors(cuda.cuCtxCreate(0, device))

    return device, context


def create_kernel(
    source_file: str, name: str, device: cuda.CUdevice
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
    can read the array with some caching benefits, dtype is used to get
    the memory size of a single component"""

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
