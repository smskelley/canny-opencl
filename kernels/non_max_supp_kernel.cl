// Non-maximum Supression Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
// theta: angle input data
__kernel void non_max_supp_kernel(__global uchar *data,
                                  __global uchar *out,
                                  __global uchar *theta,
                                           size_t rows,
                                           size_t cols)
{
    // These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
    size_t row = get_global_id(0) + 1;
    size_t col = get_global_id(1) + 1;

    // The following variables are used to address the matrices more easily
    const size_t POS = row * cols + col;
    const size_t N = (row - 1) * cols + col;
    const size_t NE = (row - 1) * cols + (col + 1);
    const size_t E = row * cols + (col + 1);
    const size_t SE = (row + 1) * cols + (col + 1);
    const size_t S = (row + 1) * cols + col;
    const size_t SW = (row + 1) * cols + (col - 1);
    const size_t W = row * cols + (col - 1);
    const size_t NW = (row - 1) * cols + (col - 1);

    switch (theta[POS])
    {
        // A gradient angle of 0 degrees = an edge that is North/South
        // Check neighbors to the East and West
        case 0:
            // supress me if my neighbor has larger magnitude
            if (data[POS] <= data[E])
            {
                out[POS] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[POS] = data[POS];
            }
                    
            // supress me if my neighbor has larger magnitude
            if (data[POS] <= data[W])
            {
                out[POS] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[POS] = data[POS];
            }
            break;
                
        // A gradient angle of 45 degrees = an edge that is NW/SE
        // Check neighbors to the NE and SW
        case 45:
            // supress me if my neighbor has larger magnitude
            if (data[POS] <= data[NE])
            {
                out[POS] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[POS] = data[POS];
            }
                    
            // supress me if my neighbor has larger magnitude
            if (data[POS] <= data[SW])
            {
                out[POS] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[POS] = data[POS];
            }
            break;
                    
        // A gradient angle of 90 degrees = an edge that is E/W
        // Check neighbors to the North and South
        case 90: ;
            // supress me if my neighbor has larger magnitude
            if (data[POS] <= data[N])
            {
                out[POS] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[POS] = data[POS];
            }
                    
            // supress me if my neighbor has larger magnitude
            if (data[POS] <= data[S])
            {
                out[POS] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                 out[POS] = data[POS];
            }
            break;
                    
        // A gradient angle of 135 degrees = an edge that is NE/SW
        // Check neighbors to the NW and SE
        case 135:
            // supress me if my neighbor has larger magnitude
            if (data[POS] <= data[NW])
            {
                out[POS] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[POS] = data[POS];
            }
                    
            // supress me if my neighbor has larger magnitude
            if (data[POS] <= data[SE])
            {
                out[POS] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[POS] = data[POS];
            }
            break;
                    
        defaut:
            out[POS] = data[POS];
            break;
    } 
}
