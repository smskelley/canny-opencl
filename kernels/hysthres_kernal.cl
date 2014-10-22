// Hysteresis Thresholding kernel.
// data: 
// out: 
__kernel void hysthresh_kernel(__global uchar *data,
                           __global uchar *out,
                                    size_t rows,
                                    size_t cols)
{
    // collect sums separately. we're storing them into floats because that
    // is what hypot will expect.
    float sumx = 0, sumy = 0;
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
}
