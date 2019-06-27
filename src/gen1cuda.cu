
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif /* USE_CUDA */

__global__
void fitness_score_add(
   float* out, const uint8_t* buffer_tgt,
   const uint8_t* buffer_test, int byte_width
) {
   int i = threadIdx.x;
   int stride = blockDim.x;

   for( ; byte_width > i ; i += stride ) {
      *out += abs( (buffer_tgt[i] - buffer_test[i]) );
   }
}

extern "C" float fitness_score(
   int width, const uint8_t* buffer_tgt, const uint8_t* buffer_test
) {

   uint8_t* d_line_master = NULL;
   uint8_t* d_candidate = NULL;
   float* d_score_total = NULL;
   float out = 0;

   cudaMalloc(
      (void**)&d_score_total, sizeof( float ) );
   cudaMalloc(
      (void**)&d_candidate, sizeof( uint8_t ) * width );
   cudaMalloc(
      (void**)&d_score_total, sizeof( uint8_t ) * width );

   cudaMemcpy( d_candidate, buffer_test, width, cudaMemcpyHostToDevice );
   cudaMemcpy( d_line_master, buffer_tgt, width, cudaMemcpyHostToDevice );

   fitness_score_add<<<0, 256>>>(
      d_score_total, d_line_master, d_candidate, width );

   cudaMemcpy( &out, d_score_total, sizeof( float ), cudaMemcpyDeviceToHost );

   return out;
}

