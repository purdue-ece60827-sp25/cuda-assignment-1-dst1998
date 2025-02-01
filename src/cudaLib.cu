
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size) return;
    y[i] = scale * x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	// std::cout << "Lazy, you are!\n";
	// std::cout << "Write code, you must\n";

	size_t size = vectorSize * sizeof(float);
    float *device_x, *device_y, scale = 2.0f;

    // Initialize host
    std::vector<float> host_x(vectorSize);
    std::vector<float> host_y(vectorSize);
    std::vector<float> host_result(vectorSize);

    // Initialize vector values (vector.data() returns pointer of the first data in the vector)
    vectorInit(host_x.data(), vectorSize);
    vectorInit(host_y.data(), vectorSize);

    // Copy host_y data to host_result vector
    std::memcpy(host_result.data(), host_y.data(), size);

    // Malloc space for x and y in GPU, and use device_x, device_y pointers to point at them
    gpuAssert(cudaMalloc((void **)&device_x, size), __FILE__, __LINE__);
    gpuAssert(cudaMalloc((void **)&device_y, size), __FILE__, __LINE__);

    // Copy the data from host to GPU
    gpuAssert(cudaMemcpy( device_x, host_x.data(), size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    gpuAssert(cudaMemcpy( device_y, host_result.data(), size, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // Configure CUDA kernel launch parameters - to have the device to perform actual vector multi-add
    int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
	// Launch SAXPY kernel
    saxpy_gpu<<<blocksPerGrid, threadsPerBlock>>>(device_x, device_y, scale, vectorSize);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // Computation finished, copy the result from GPU to host, verifyVector, free device vectors
    gpuAssert(cudaMemcpy(host_result.data(), device_y, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    int errorCount = verifyVector(host_x.data(), host_y.data(), host_result.data(), scale, vectorSize);
    std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";
    gpuAssert(cudaFree(device_x), __FILE__, __LINE__);
    gpuAssert(cudaFree(device_y), __FILE__, __LINE__);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pSumSize) return;
    uint64_t hit_count = 0;
    curandState_t rng;
    curand_init(clock64(), idx, 0, &rng);
    // int randInt = curand_uniform(&rng);
    for(int i = 0; i < sampleSize; i++){
        float x = curand_uniform(&rng);
        float y = curand_uniform(&rng);
        if((x*x) + (y*y) <= 1.0f){
            ++ hit_count;
        }
    }
    pSums[idx] = hit_count;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t reduceThreadCount = pSumSize / reduceSize;
    if(idx >= reduceThreadCount) return;
    uint64_t sum = 0;
    for (uint64_t i = 0; i < reduceSize; i++) {
        sum += pSums[idx * reduceSize + i];
    }
    totals[idx] = sum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	// std::cout << "Sneaky, you are ...\n";
	// std::cout << "Compute pi, you must!\n";
	size_t generate_size = generateThreadCount * sizeof(uint64_t);
    size_t reduce_size = reduceThreadCount * sizeof(uint64_t);
    std::vector<uint64_t> host_total(reduceThreadCount);
    uint64_t *device_pSums = nullptr;
    uint64_t *device_totals = nullptr;
    gpuAssert(cudaMalloc((void **)&device_pSums, generate_size), __FILE__, __LINE__);
    gpuAssert(cudaMalloc((void **)&device_totals, reduce_size), __FILE__, __LINE__);

    // Kernel setup for generatePoints (generate random points in GPU)
    int generateThreadsPerBlock = 256;
    int gernerateBlocksPerGrid = (generateThreadCount + generateThreadsPerBlock - 1) / generateThreadsPerBlock;
    generatePoints<<<gernerateBlocksPerGrid, generateThreadsPerBlock>>>(device_pSums, generateThreadCount, sampleSize);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // Kernel setup for reduceCounts
    int reduceThreadsPerBlock = 256;
    int reduceBlocksPerGrid = (reduceThreadCount + reduceThreadsPerBlock - 1) / reduceThreadsPerBlock;
    reduceCounts<<<reduceBlocksPerGrid, reduceThreadsPerBlock>>> (device_pSums, device_totals, generateThreadCount, reduceSize);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // Computation finished, copy the result from GPU to host
    gpuAssert(cudaMemcpy(host_total.data(), device_totals, reduce_size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    
	uint64_t totalHitCount = 0;
    for (uint64_t i = 0; i < reduceThreadCount; i++) {
        totalHitCount += host_total[i];
    }
    approxPi = (((double)totalHitCount / (double)sampleSize) / (double)generateThreadCount) * 4.0f;

	//Free device vectors
    gpuAssert(cudaFree(device_pSums), __FILE__, __LINE__);
    gpuAssert(cudaFree(device_totals), __FILE__, __LINE__);
	return approxPi;
}
