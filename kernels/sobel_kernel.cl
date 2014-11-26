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
                    data[ (i+row+-1)*cols + (j+col+-1) ];
            sumy += soby[i][j] *
                    data[ (i+row+-1)*cols + (j+col+-1) ];
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
    else if (angle <= 9*PI/8)
    {
        theta[pos] = 0;
    }
    else if (angle <= 11*PI/8)
    {
        theta[pos] = 45;
    }
    else if (angle <= 13*PI/8)
    {
        theta[pos] = 90;
    }
    else if (angle <= 15*PI/8)
    {
        theta[pos] = 135;
    }
    else // (angle <= 16*PI/8)
    {
        theta[pos] = 0;
    }
}
