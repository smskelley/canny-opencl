// Some of the available convolution kernels
__constant int sobx[3][3] = { {-1, 0, 1},
                              {-2, 0, 2},
                              {-1, 0, 1} };

__constant int soby[3][3] = { {-1,-2,-1},
                              { 0, 0, 0},
                              { 1, 2, 1} };

// Sobel kernel. Apply sobx and soby separately, then find the sqrt of their
//               squares.
// data:  image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out:   image output data (8B1C)
// theta: angle output data
__kernel void sobel_kernel(__global uchar *data,
                           __global uchar *out,
                           __global uchar *theta,
                                    size_t rows,
                                    size_t cols)
{
    // collect sums separately. we're storing them into floats because that
    // is what hypot and atan2 will expect.
    const float PI = 3.14159265;
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = get_local_id(0) + 1;
    size_t l_col = get_local_id(1) + 1;
    
    size_t pos = g_row * cols + g_col;
    
    __local int l_data[18][18];

    // copy to local
    l_data[l_row][l_col] = data[pos];

    // top most row
    if (l_row == 1)
    {
        l_data[0][l_col] = data[pos-cols];
        // top left
        if (l_col == 1)
            l_data[0][0] = data[pos-cols-1];

        // top right
        else if (l_col == 16)
            l_data[0][17] = data[pos-cols+1];
    }
    // bottom most row
    else if (l_row == 16)
    {
        l_data[17][l_col] = data[pos+cols];
        // bottom left
        if (l_col == 1)
            l_data[17][0] = data[pos+cols-1];

        // bottom right
        else if (l_col == 16)
            l_data[17][17] = data[pos+cols+1];
    }

    // left
    if (l_col == 1)
        l_data[l_row][0] = data[pos-1];
    // right
    else if (l_col == 16)
        l_data[l_row][17] = data[pos+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sumx = 0, sumy = 0, angle = 0;
    // find x and y derivatives
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            sumx += sobx[i][j] * l_data[i+l_row-1][j+l_col-1];
            sumy += soby[i][j] * l_data[i+l_row-1][j+l_col-1];
        }
    }

    // The output is now the square root of their squares, but they are
    // constrained to 0 <= value <= 255. Note that hypot is a built in function
    // defined as: hypot(x,y) = sqrt(x*x, y*y).
    out[pos] = min(255,max(0, (int)hypot(sumx,sumy) ));

    // Compute the direction angle theta in radians
    // atan2 has a range of (-PI, PI) degrees
    angle = atan2(sumy,sumx);

    // If the angle is negative, 
    // shift the range to (0, 2PI) by adding 2PI to the angle, 
    // then perform modulo operation of 2PI
    if (angle < 0)
    {
        angle = fmod((angle + 2*PI),(2*PI));
    }

    // Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
    // then store it in the theta buffer at the proper position
    theta[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
}
