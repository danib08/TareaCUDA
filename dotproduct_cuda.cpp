#include <iostream>
#include <cstdlib>

// kernel que llevara a cabo la operación del producto punto
__global__ void dotProductKernel(float* A, float* B, float* C, int N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int j = id; j < N; j += blockDim.x * gridDim.x) {
        // se utiliza la funcion atomicAdd para que no se de el problema
        // de la condicion de carrera. antes de utilizarla,
        // el resultado del producto punto daba 40, lo cual es incorrecto
        atomicAdd(&C[0], A[j] * B[j]);
    }
}

float dotProductCUDA(float* h_A, float* h_B, int N) {
    // tamaño del array 
    size_t size = N * sizeof(float);

    // se asigna la memoria necesaria para los arrays en el device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, size);
    cudaMalloc((void**) &d_B, size);
    cudaMalloc((void**) &d_C, sizeof(float));

    // se copian los arrays como tal al device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // se define el tamaño de los bloques y de los grids
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // el resultado de blocksPerGrid sera 5, por lo que en total
    // threads = 5 * 256 = 1280, el cual es un numero mayor que N = 1040
    // lo que implica que alcanzan de sobra los threads para realizar el producto punto

    // se invoca al kernel
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    float result;
    // se copia el resultado del producto punto del device al host
    cudaMemcpy(&result, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    // se libera la memoria del device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}

int main() {
    // Se define el tamaño N de los arrays multiplo de 4 y mayor que 500
    const int N = 1040;

    // Se definen los dos arrays para el producto punto
    float array1[N];
    float array2[N];

    // Se llenan los arrays con valores definidos ya que se necesita 
    // que el resultado del producto punto sea igual que en el quiz SIMD.
    for (int i = 0; i < N; ++i) {
        array1[i] = 5.0f;
        array2[i] = 8.0f;
    }

    // Se realiza la operación del producto punto utilizando CUDA
    float result = dotProductCUDA(array1, array2, N);

    // Se imprime el resultado del producto punto y el tiempo de ejecución
    std::cout << "Producto punto: " << result << std::endl;

    return 0;
}
