from cuda import cuda, nvrtc, cudart # type: ignore



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
