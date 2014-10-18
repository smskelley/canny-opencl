// Some of the available convolution kernels
__constant float gaus[3][3] = { {0.0625, 0.125, 0.0625},
                                {0.1250, 0.125, 0.1250},
                                {0.0625, 0.125, 0.0625} };
__constant int sobx[3][3] = { {-1, 0, 1},
                               {-2, 0, 2},
                               {-1, 0, 1} };
__constant int soby[3][3] = { {-1, -2, -1},
                               { 0,  0,  0},
                               {1,   2,  1} };
__constant int edge[3][3] = { {-1, -1, -1},
                               {-1,  8, -1},
                               {-1, -1, -1} };
__constant int edg2[3][3] = { { 1, 0, -1},
                                { 0, 0,  0},
                                {-1, 0,  1} };


// Convolution Kernel Example
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__kernel void convolution_kernel(__global uchar *data,
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
            sum += edge[i][j] *
                   data[ ((i+row+rows-1)%rows)*cols + (j+col+cols-1)%cols ];

    //out[pos] = sum;
    out[pos] = min(255,max(0,sum));
}
