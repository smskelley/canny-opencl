__constant float gaus[3][3] = { {0.0625, 0.125, 0.0625},
                                {0.1250, 0.125, 0.1250},
                                {0.0625, 0.125, 0.0625} };

// Gaussian Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__kernel void gaussian_kernel(__global uchar *data,
                              __global uchar *out,
                                       size_t rows,
                                       size_t cols)
{
    int sum = 0;
    size_t row = get_global_id(0);
    size_t col = get_global_id(1);
    size_t pos = row * cols + col;
    
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            sum += gaus[i][j] *
                   data[ (i+row+-1)*cols + (j+col+-1) ];

    out[pos] = min(255,max(0,sum));
}
