// D5a 首步噪声协议 cuda_philox4090_v1，fresh generator offset=0。
// Philox4x32-10 计数器算法；normal4 使用 cuRAND 的单精度变换顺序。
// 参考：PyTorch v2.8.0 DistributionTemplates.h；NVIDIA cuRAND device API。
// 虚拟布局固定为4090参考布局，实际CUDA线程数量不参与随机数定位。
struct UInt4 { unsigned int x, y, z, w; };
struct Float2 { float x, y; };
struct Float4 { float x, y, z, w; };

__device__ __forceinline__ UInt4 philox_round(UInt4 ctr, unsigned int kx, unsigned int ky) {
    unsigned int hi0 = __umulhi(0xD2511F53u, ctr.x);
    unsigned int hi1 = __umulhi(0xCD9E8D57u, ctr.z);
    unsigned int lo0 = 0xD2511F53u * ctr.x;
    unsigned int lo1 = 0xCD9E8D57u * ctr.z;
    UInt4 result = {hi1 ^ ctr.y ^ kx, lo1, hi0 ^ ctr.w ^ ky, lo0};
    return result;
}

__device__ __forceinline__ UInt4 philox10(unsigned long long seed, unsigned long long subsequence,
                                        unsigned long long round_index) {
    UInt4 counter = {(unsigned int)round_index, (unsigned int)(round_index >> 32),
                     (unsigned int)subsequence, (unsigned int)(subsequence >> 32)};
    unsigned int kx = (unsigned int)seed;
    unsigned int ky = (unsigned int)(seed >> 32);
    #pragma unroll
    for (int r = 0; r < 9; ++r) {
        counter = philox_round(counter, kx, ky);
        kx += 0x9E3779B9u;
        ky += 0xBB67AE85u;
    }
    return philox_round(counter, kx, ky);
}

__device__ __forceinline__ Float2 normal_pair(unsigned int x, unsigned int y) {
    const float inv32 = 2.3283064e-10f;
    const float inv32_2pi = 2.3283064e-10f * 6.2831855f;
    float u = x * inv32 + (inv32 / 2);
    float v = y * inv32_2pi + (inv32_2pi / 2);
    float s = sqrtf(-2.0f * logf(u));
    Float2 result;
    __sincosf(v, &result.x, &result.y);
    result.x *= s;
    result.y *= s;
    return result;
}

__device__ __forceinline__ Float4 normal4(UInt4 x) {
    Float2 first = normal_pair(x.x, x.y);
    Float2 second = normal_pair(x.z, x.w);
    Float4 result = {first.x, first.y, second.x, second.y};
    return result;
}

extern "C" __global__ void restoration_normal(float* out, long long n, unsigned long long seed,
                                               long long reference_stride) {
    long long actual_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long actual_stride = (long long)blockDim.x * gridDim.x;
    for (long long j = actual_index; j < reference_stride && j < n; j += actual_stride) {
        for (unsigned long long r = 0; j + (long long)(4 * r) * reference_stride < n; ++r) {
            Float4 z = normal4(philox10(seed, (unsigned long long)j, r));
            long long i = j + (long long)(4 * r) * reference_stride;
            if (i < n) out[i] = z.x;
            if (i + reference_stride < n) out[i + reference_stride] = z.y;
            if (i + 2 * reference_stride < n) out[i + 2 * reference_stride] = z.z;
            if (i + 3 * reference_stride < n) out[i + 3 * reference_stride] = z.w;
        }
    }
}
