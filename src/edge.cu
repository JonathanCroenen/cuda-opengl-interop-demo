


extern "C" __global__
void edge(
    cudaTextureObject_t render,
    cudaSurfaceObject_t output,
    int width,
    int height
) {
    unsigned int sx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int stridex = blockDim.x * gridDim.x;
    unsigned int stridey = blockDim.y * gridDim.y;

    for (int x = sx; x < width; x += stridex) {
        for (int y = sy; y < height; y += stridey) {
            float4 top_left = tex2D<float4>(render, x - 1, y - 1);
            float4 top_center = tex2D<float4>(render, x, y - 1);
            float4 top_right = tex2D<float4>(render, x + 1, y - 1);

            float4 center_left = tex2D<float4>(render, x - 1, y);
            // float4 center_center = tex2D<float4>(render, x, y);
            float4 center_right = tex2D<float4>(render, x + 1, y);

            float4 bottom_left = tex2D<float4>(render, x - 1, y + 1);
            float4 bottom_center = tex2D<float4>(render, x, y + 1);
            float4 bottom_right = tex2D<float4>(render, x + 1, y + 1);

            // [-1  0  1]  [-1 -1 -1]
            // [-1  0  1]  [ 0  0  0]
            // [-1  0  1]  [ 1  1  1]
            float mag_vx = - top_left.x - center_left.x - bottom_left.x
                + top_right.x + center_right.x + bottom_right.x;

            float mag_vy = - top_left.y - center_left.y - bottom_left.y
                + top_right.y + center_right.y + bottom_right.y;

            float mag_vz = - top_left.z - center_left.z - bottom_left.z
                + top_right.z + center_right.z + bottom_right.z;

            float mag_hx = - top_left.x - top_center.x - top_right.x
                + bottom_left.x + bottom_center.x + bottom_right.x;

            float mag_hy = - top_left.y - top_center.y - top_right.y
                + bottom_left.y + bottom_center.y + bottom_right.y;

            float mag_hz = - top_left.z - top_center.z - top_right.z
                + bottom_left.z + bottom_center.z + bottom_right.z;

            float mag = sqrt(mag_hx * mag_hx + mag_vx * mag_vx
                             + mag_hy * mag_hy + mag_vy * mag_vy
                             + mag_hz * mag_hz + mag_vz * mag_vz) / 3;

            // convert the float value back to rgba uint8 (grayscale)
            unsigned char mag_char = (unsigned char) (mag * 255);
            uchar4 mag_char4 = make_uchar4(mag_char, mag_char, mag_char, 255);
            surf2Dwrite(mag_char4, output, 4 * x, y, cudaBoundaryModeZero);
        }
    }
}
