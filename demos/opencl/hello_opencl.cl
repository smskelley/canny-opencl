//__constant unsigned long MAX = 3000;

__kernel void hello_opencl(__global unsigned char *data)
{
    size_t tid = get_global_id(0);
    data[tid] /= 2;
}