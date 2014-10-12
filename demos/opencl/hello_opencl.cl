//__constant unsigned long MAX = 3000;

__kernel void hello_opencl(__global unsigned char *data,
                           size_t row_step,
                           size_t col_step)
{
    size_t row = get_global_id(0);
    size_t col = get_global_id(1);
    size_t pos = row * row_step + col * col_step;

    data[pos] /= 2;
}