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
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = get_local_id(0) + 1;
    size_t l_col = get_local_id(1) + 1;
    
    size_t pos = g_row * cols + g_col;
    
    __local int a[18][18];

    // copy to local
    a[l_row][l_col] = data[pos];

    // top most row
    if (l_row == 1)
    {
        a[0][l_col] = data[pos-cols];
        // top left
        if (l_col == 1)
            a[0][0] = data[pos-cols-1];

        // top right
        else if (l_col == 16)
            a[0][17] = data[pos-cols+1];
    }
    // bottom most row
    else if (l_row == 16)
    {
        a[17][l_col] = data[pos+cols];
        // bottom left
        if (l_col == 1)
            a[17][0] = data[pos+cols-1];

        // bottom right
        else if (l_col == 16)
            a[17][17] = data[pos+cols+1];
    }

    if (l_col == 1)
        a[l_row][0] = data[pos-1];
    else if (l_col == 16)
        a[l_row][17] = data[pos+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            sum += gaus[i][j] * a[i+l_row-1][j+l_col-1];
                   //a[ (i+l_row-1)*18 + (j+l_col-1) ];

    out[pos] = min(255,max(0,sum));

    return;
}
