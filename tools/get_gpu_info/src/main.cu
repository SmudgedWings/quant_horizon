#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>


//显示CUDA设备信息
void show_GPU_info(void)
{
    int deviceCount;
    //获取CUDA设备总数
    cudaGetDeviceCount(&deviceCount);
    //分别获取每个CUDA设备的信息
    for(int i=0;i<deviceCount;i++)
    {
        //定义存储信息的结构体
        cudaDeviceProp devProp;
        //将第i个CUDA设备的信息写入结构体中
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
        std::cout << "设备全局内存总量：" << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "每个SM里面最大的block数：" << devProp.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "设备上一个线程块（Block）中可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
        std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "设备上多处理器的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "======================================================" << std::endl;     
        
    }
}
//用于查看显卡信息
int main(int argc, char ** argv)
{
    show_GPU_info();
    return 0;
}
