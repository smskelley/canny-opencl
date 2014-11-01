// Some of the available convolution kernels
__constant int sobx[3][3] = { {-1, 0, 1},
                              {-2, 0, 2},
                              {-1, 0, 1} };

__constant int soby[3][3] = { {-1,-2,-1},
                              { 0, 0, 0},
                              { 1, 2, 1} };

// Sobel kernel. Apply sobx and soby separately, then find the sqrt of their
//               squares.
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__kernel void sobel_kernel(__global uchar *data,
                           __global uchar *out,
                           __global uchar *theta,
                                    size_t rows,
                                    size_t cols)
{
    // collect sums separately. we're storing them into floats because that
    // is what hypot and atan2 will expect.
    const float PI = 3.14159265;
    float sumx = 0, sumy = 0, angle = 0;
    size_t row = get_global_id(0);
    size_t col = get_global_id(1);
    size_t pos = row * cols + col;
    
    // find x and y derivatives
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            sumx += sobx[i][j] *
                    data[ ((i+row+rows-1)%rows)*cols + (j+col+cols-1)%cols ];
            sumy += soby[i][j] *
                    data[ ((i+row+rows-1)%rows)*cols + (j+col+cols-1)%cols ];
        }
    }

    // The output is now the square root of their squares, but they are
    // constrained to 0 <= value <= 255. Note that hypot is a built in function
    // defined as: hypot(x,y) = sqrt(x*x, y*y).
    out[pos] = min(255,max(0, (int)hypot(sumx,sumy) ));

    // Compute the direction angle theta in radians
    // atan2(y,x) = arc tan(y/x)
    // arc tan has a range of (-90, 90) degrees
    angle = atan2(sumy, sumx);

    // Shift the range to (0, 180) degrees by adding 90 degrees
    angle = angle + PI/2;
    
    // Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
    // then store it in the theta buffer at the proper position
    if (angle <= PI/8) 
    {
        theta[pos] = 0;
    }
    else if (angle <= 3*PI/8)
    {
        theta[pos] = 45;
    }
    else if (angle <= 5*PI/8)
    {
        theta[pos] = 90;
    }
    else if (angle <= 7*PI/8)
    {
        theta[pos] = 135;
    }
    else // (angle <= PI)
    {
        theta[pos] = 0;
    }
}
