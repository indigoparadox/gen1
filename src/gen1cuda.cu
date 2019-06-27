
#include "gen1.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>

__global__
void setup_curand( int* init, curandState *state ) {
   int idx = threadIdx.x;

   curand_init( *init, idx, 0, &state[idx] );
}

__global__
void generate_line_c( curandState* state, uint8_t* buffer, int width ) {
   int idx = threadIdx.x;
   int i = 0;
   float f_rand;

   for( i = 0 ; width > i ; i++ ) {
      f_rand = curand_uniform( state + idx );
      f_rand *= (UINT8_MAX + 0.999999);

      buffer[i] = (uint8_t)truncf( f_rand );
   }
}

extern "C" void generate_line( uint8_t* buffer, int width ) {
   curandState *d_state = NULL;
   int init_time = 0;

   cudaMalloc( &d_state, sizeof( curandState ) * 256 );

   init_time = time( NULL );

   setup_curand<<<1, 1>>>( &init_time, d_state );

   generate_line_c<<<0, 256>>>( d_state, buffer, width );

   cudaFree( d_state );
}

__global__
void fitness_score_add(
   float* out, const float* buffer_tgt,
   const float* buffer_test, int float_width
) {
   int i = threadIdx.x;
   int stride = blockDim.x;

   for( ; float_width > i ; i += stride ) {
      out[i] = 5; //abs( (buffer_tgt[i] - buffer_test[i]) );
   }
}

#include <stdio.h>
extern "C" float fitness_score(
   int width, const uint8_t* buffer_tgt, const uint8_t* buffer_test
) {
   float* d_line_master = NULL;
   float* d_candidate = NULL;
   float* d_out_diffs = NULL;
   float* out_diffs = NULL;
   float out = 0;
   int i = 0;
   int float_width = 0;

   assert( sizeof( float ) == 4 * sizeof( uint8_t ) );
   assert( 0 == width % 4 );
   assert( sizeof( uint8_t ) == 1 );

   float_width = width / 4;

   out_diffs = (float*)calloc( 1, width );

   cudaMalloc( (void**)&d_out_diffs, width );
   cudaMalloc( (void**)&d_candidate, width );
   cudaMalloc( (void**)&d_line_master, width );

   cudaMemcpy( d_candidate, buffer_test, width, cudaMemcpyHostToDevice );
   cudaMemcpy( d_line_master, buffer_tgt, width, cudaMemcpyHostToDevice );

   fitness_score_add<<<0, 256>>>(
      d_out_diffs, d_line_master, d_candidate, float_width );

   cudaMemcpy( &out_diffs, d_out_diffs, width, cudaMemcpyDeviceToHost );

   for( i = 0 ; width > i ; i++ ) {
      printf( "%f\n", out_diffs[i] );
   }

   cudaFree( d_line_master );
   cudaFree( d_candidate );
   cudaFree( d_out_diffs );
   free( out_diffs );

   return out;
}

__global__
void combine_lines_c( uint8_t* line_dest, const uint8_t* line_src, int width ) {
   int i = threadIdx.x;
   int stride = blockDim.x;

   for( ; width > i ; i += stride ) {
      if( 0 == i % 2 ) {
         line_dest[i] = line_src[i];
      }
   }
}

extern "C"
void combine_lines( uint8_t* line_dest, const uint8_t* line_src, int width ) {
   uint8_t* d_source = NULL;
   uint8_t* d_dest = NULL;

   cudaMalloc(
      (void**)&d_source, sizeof( uint8_t ) * width );
   cudaMalloc(
      (void**)&d_dest, sizeof( uint8_t ) * width );

   cudaMemcpy( d_source, line_src, width, cudaMemcpyHostToDevice );
   cudaMemcpy( d_dest, line_dest, width, cudaMemcpyHostToDevice );

   combine_lines_c<<<0, 256>>>( d_dest, d_source, width );

   cudaMemcpy(
      &line_dest, d_dest, sizeof( uint8_t ) * width, cudaMemcpyDeviceToHost );

   cudaFree( d_source );
   cudaFree( d_dest );
}

