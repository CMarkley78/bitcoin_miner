#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <library.h>

#define max_uint32_t (((uint32_t)0)-1)
#define gridSize 1000000
#define threadSize 16
#define test_cts ((int)ceil(max_uint32_t/(gridSize*threadSize)))

void str__char_arr(const char* str, unsigned char* out) { //Remember to take this out once the library is finished.
    int len = strlen(str) / 2;
    for (int out_idx = 0, str_idx = 0; out_idx < len; out_idx++, str_idx += 2) {
        char hexpair[3] = {str[str_idx], str[str_idx + 1], '\0'};
        out[out_idx] = (unsigned char)strtol(hexpair, NULL, 16);
    }
}

//Function for rotating a 32 bit datatype x n bits to the right. Used for SHA256 calculations
__device__ uint32_t right_rotate(uint32_t x, int n) {
  return (x>>n)|(x<<(32-n));
}

//Function to reverse the endianness of a 32 bit number. Used to fix endian fuckery.
__device__ uint32_t reverse32(uint32_t x) {
  return ((x>>24)&0x000000FF)|((x>>8)&0x0000FF00)|((x<<8)&0x00FF0000)|((x<<24)&0xFF000000);
}
uint32_t cpu_reverse32(uint32_t x) {
  return ((x>>24)&0x000000FF)|((x>>8)&0x0000FF00)|((x<<8)&0x00FF0000)|((x<<24)&0xFF000000);
}

//Returns 1 if test1 < test2, and 0 otherwise. Used to compare the generated hash to the target.
__device__ unsigned char compareHashes(const uint32_t* test1, const uint32_t* test2) {
  for (int i=0; i<8; i++) {
      if (reverse32(test1[7-i]) < test2[i]) {
          return 1;
      } else if (reverse32(test1[7-i]) > test2[i]) {
          return 0;
      }
  }

  return 0;
}

__constant__ uint32_t k_vals[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

//The hashing of the 80 byte input header, stores the output in hash
__device__ void d_first_hash(unsigned char* header, uint32_t* hash) {

  //Initialization of hash parts
  hash[0] = 0x6a09e667;
  hash[1] = 0xbb67ae85;
  hash[2] = 0x3c6ef372;
  hash[3] = 0xa54ff53a;
  hash[4] = 0x510e527f;
  hash[5] = 0x9b05688c;
  hash[6] = 0x1f83d9ab;
  hash[7] = 0x5be0cd19;

  //Construction of the 2 data blocks
  unsigned char block_1[64], block_2[64];
  memcpy(block_1,header,64);
  memcpy(block_2,&header[64],16);
  block_2[16] = 0x80;
  for (int i=17;i<64;i++) {
    block_2[i] = 0x00;
  }
  block_2[62] = 0x02;
  block_2[63] = 0x80;

  //Initialization of all of the calculation variables I need
  uint32_t message_schedule[64], s0, s1, temp1, temp2, a, b, c, d, e, f, g, h, ch, maj;

  //BEGIN BLOCK 1 PROCESSING
  //Message schedule generation
  memcpy(message_schedule,block_1,64);
  //The next three lines were almost the end of my sanity in v1. Endianness is stupid. It should all be big endian.
  for (int i=0;i<16;i++) {
    message_schedule[i] = reverse32(message_schedule[i]);
  }
  //Actual schedule gen
  for (int i=16;i<64;i++) {
    s0 = right_rotate(message_schedule[i-15],7)^right_rotate(message_schedule[i-15],18)^(message_schedule[i-15]>>3);
    s1 = right_rotate(message_schedule[i-2],17)^right_rotate(message_schedule[i-2],19)^(message_schedule[i-2]>>10);
    message_schedule[i] = message_schedule[i-16]+s0+message_schedule[i-7]+s1;
  }

  //Next up is the main compression function
  a = hash[0];
  b = hash[1];
  c = hash[2];
  d = hash[3];
  e = hash[4];
  f = hash[5];
  g = hash[6];
  h = hash[7];
  for (int i=0;i<64;i++) {
    s1 = right_rotate(e,6)^right_rotate(e,11)^right_rotate(e,25);
    ch = (e&f)^((~e)&g);
    temp1 = h + s1 + ch + k_vals[i] + message_schedule[i];
    s0 = right_rotate(a,2)^right_rotate(a,13)^right_rotate(a,22);
    maj = (a&b)^(a&c)^(b&c);
    temp2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1+temp2;
  }
  hash[0] = hash[0] + a;
  hash[1] = hash[1] + b;
  hash[2] = hash[2] + c;
  hash[3] = hash[3] + d;
  hash[4] = hash[4] + e;
  hash[5] = hash[5] + f;
  hash[6] = hash[6] + g;
  hash[7] = hash[7] + h;

  //BEGIN BLOCK 2
  //Message schedule gen... Already been through this.
  memcpy(&message_schedule[0],&block_2[0],64);
  for (int i=0;i<16;i++) {
    message_schedule[i] = reverse32(message_schedule[i]);
  }
  for (int i=16;i<64;i++) {
    s0 = right_rotate(message_schedule[i-15],7)^right_rotate(message_schedule[i-15],18)^(message_schedule[i-15]>>3);
    s1 = right_rotate(message_schedule[i-2],17)^right_rotate(message_schedule[i-2],19)^(message_schedule[i-2]>>10);
    message_schedule[i] = message_schedule[i-16]+s0+message_schedule[i-7]+s1;
  }
  //Main compression function
  a = hash[0];
  b = hash[1];
  c = hash[2];
  d = hash[3];
  e = hash[4];
  f = hash[5];
  g = hash[6];
  h = hash[7];
  for (int i=0;i<64;i++) {
    s1 = right_rotate(e,6)^right_rotate(e,11)^right_rotate(e,25);
    ch = (e&f)^((~e)&g);
    temp1 = h + s1 + ch + k_vals[i] + message_schedule[i];
    s0 = right_rotate(a,2)^right_rotate(a,13)^right_rotate(a,22);
    maj = (a&b)^(a&c)^(b&c);
    temp2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1+temp2;
  }
  hash[0] = hash[0] + a;
  hash[1] = hash[1] + b;
  hash[2] = hash[2] + c;
  hash[3] = hash[3] + d;
  hash[4] = hash[4] + e;
  hash[5] = hash[5] + f;
  hash[6] = hash[6] + g;
  hash[7] = hash[7] + h;


}

__device__ void d_second_hash(uint32_t* hash) {
  //Initialization of working variables
  uint32_t message_schedule[64], s0, s1, temp1, temp2, a, b, c, d, e, f, g, h, ch, maj;

  //Because all of the hashed data now fits in one block, it's actually easier to just build up the message schedule rather than to build a block then copy it over.
  for (int i=8;i<16;i++) {
    message_schedule[i] = 0;
  }
  message_schedule[8] = 0x80000000;
  message_schedule[15] = 0x00000100;
  memcpy(message_schedule,hash,32);

  //Initialization of hash values
  hash[0] = 0x6a09e667;
  hash[1] = 0xbb67ae85;
  hash[2] = 0x3c6ef372;
  hash[3] = 0xa54ff53a;
  hash[4] = 0x510e527f;
  hash[5] = 0x9b05688c;
  hash[6] = 0x1f83d9ab;
  hash[7] = 0x5be0cd19;

  for (int i=16;i<64;i++) {
    s0 = right_rotate(message_schedule[i-15],7)^right_rotate(message_schedule[i-15],18)^(message_schedule[i-15]>>3);
    s1 = right_rotate(message_schedule[i-2],17)^right_rotate(message_schedule[i-2],19)^(message_schedule[i-2]>>10);
    message_schedule[i] = message_schedule[i-16]+s0+message_schedule[i-7]+s1;
  }

  //And then, like normal, we do our main compression loop.
  a = hash[0];
  b = hash[1];
  c = hash[2];
  d = hash[3];
  e = hash[4];
  f = hash[5];
  g = hash[6];
  h = hash[7];
  for (int i=0;i<64;i++) {
    s1 = right_rotate(e,6)^right_rotate(e,11)^right_rotate(e,25);
    ch = (e&f)^((~e)&g);
    temp1 = h + s1 + ch + k_vals[i] + message_schedule[i];
    s0 = right_rotate(a,2)^right_rotate(a,13)^right_rotate(a,22);
    maj = (a&b)^(a&c)^(b&c);
    temp2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1+temp2;
  }
  hash[0] += a;
  hash[1] += b;
  hash[2] += c;
  hash[3] += d;
  hash[4] += e;
  hash[5] += f;
  hash[6] += g;
  hash[7] += h;
}

__constant__ unsigned char d_header_template[76];

__constant__ uint32_t d_target[8];

__global__ void d_test_span(int * flag, int n) {
  //First thing that needs to be done is to get the number that the thread is testing
  uint32_t nonce = ((threadSize*gridSize*n)+(threadSize*blockIdx.x)+(threadIdx.x));
  //nonce = 2083236893;

  //Create our shared memory space for the header and the hash.
  unsigned char header[80];
  uint32_t hash[8];

  //Populate the header
  for (int i=0;i<76;i++) {
    header[i] = d_header_template[i];
  }
  memcpy(&header[76],&nonce,4);

  //Perform the first hash
  d_first_hash(header,hash);

  //Perform the second hash
  d_second_hash(hash);

  //Increment the hit value for our chunk by the value of compareHashes (1 if found, 0 otherwise, array starts at 0s)
  atomicAdd(&flag[n],(int)compareHashes(hash,d_target));
}

__global__ void d_search_specific(unsigned char* flag, int n) {
  //First thing that needs to be done is to get the number that the thread is testing
  uint32_t nonce = ((threadSize*gridSize*n)+(threadSize*blockIdx.x)+(threadIdx.x));
  //nonce = 2083236893;

  //Create our shared memory space for the header and the hash.
  unsigned char header[80];
  uint32_t hash[8];

  //Populate the header
  for (int i=0;i<76;i++) {
    header[i] = d_header_template[i];
  }
  memcpy(&header[76],&nonce,4);

  //Perform the first hash
  d_first_hash(header,hash);

  //Perform the second hash
  d_second_hash(hash);

  //Set our corresponding search section memory to our state
  flag[(threadSize*blockIdx.x)+(threadIdx.x)] = compareHashes(hash,d_target);
}

void search (unsigned char * header_info, unsigned char* output) {
  //Copy header into constant memory
  cudaMemcpyToSymbol(d_header_template, header_info, sizeof(unsigned char)*76);

  //Target generation and placement
  uint32_t nbits;
  memcpy(&nbits, &header_info[72], 4);

  uint32_t target[8] = {0};
  //nbits = cpu_reverse32(nbits);
  uint32_t bits_shift = 256 - (((nbits >> 24)) * 8);
  uint32_t word_skip = bits_shift / 32;
  uint32_t bit_skip = bits_shift % 32;
  uint64_t significand = (uint64_t)(nbits & 0xFFFFFF) << (40 - bit_skip);
  significand = (significand >> 32) | (significand << 32);
  memcpy(&target[word_skip], &significand, sizeof(uint64_t));

  cudaMemcpyToSymbol(d_target,target,sizeof(uint32_t)*8);

  //Allocate some memory for the flag variables
  int* chunk_hit_cts = (int*)malloc(sizeof(int)*test_cts);
  for (int i=0;i<test_cts;i++) {
    chunk_hit_cts[i] = 0;
  }
  int* d_flag;
  cudaMalloc((void**)&d_flag,sizeof(int)*test_cts);
  cudaMemcpy(d_flag,chunk_hit_cts,test_cts*sizeof(int),cudaMemcpyHostToDevice);

  //Actually perform the search
  for (int n=0;n<test_cts;n++) {
    d_test_span<<<gridSize,threadSize>>>(d_flag,n);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(chunk_hit_cts,d_flag,test_cts*sizeof(int),cudaMemcpyDeviceToHost);

  bool solved = false;
  uint32_t solve_nonce;
  for (int test_n=0;test_n<test_cts && !solved;test_n++) {
    if (chunk_hit_cts[test_n]!=0) {
      solved = true;
      unsigned char* search_flag = (unsigned char*)malloc(gridSize*threadSize*sizeof(unsigned char));
      unsigned char* d_search_flag;
      cudaMalloc((void**)&d_search_flag,gridSize*threadSize*sizeof(unsigned char));
      d_search_specific<<<gridSize,threadSize>>>(d_search_flag,test_n);
      cudaDeviceSynchronize();
      cudaMemcpy(search_flag,d_search_flag,gridSize*threadSize*sizeof(unsigned char),cudaMemcpyDeviceToHost);
      cudaFree(d_search_flag);
      for (int offset=0;offset<threadSize*gridSize;offset++) {
        if (search_flag[offset]==1) {
          solve_nonce = (test_n*threadSize*gridSize)+offset;
        }
      }
      free(search_flag);
    }
  }
  memcpy(output,&solved,sizeof(bool));
  memcpy(output+1,&solve_nonce,sizeof(uint32_t));
  free(chunk_hit_cts);
  cudaFree(d_flag);
}
