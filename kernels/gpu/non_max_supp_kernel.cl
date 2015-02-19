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
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = get_local_id(0) + 1;
    size_t l_col = get_local_id(1) + 1;
    
    size_t pos = g_row * cols + g_col;
    
    __local int l_data[18][18];

    // copy to l_data
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

    if (l_col == 1)
        l_data[l_row][0] = data[pos-1];
    else if (l_col == 16)
        l_data[l_row][17] = data[pos+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    uchar my_magnitude = l_data[l_row][l_col];

    // The following variables are used to address the matrices more easily
    switch (theta[pos])
    {
        // A gradient angle of 0 degrees = an edge that is North/South
        // Check neighbors to the East and West
        case 0:
            // supress me if my neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row][l_col+1] || // east
                my_magnitude <= l_data[l_row][l_col-1])   // west
            {
                out[pos] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[pos] = my_magnitude;
            }
            break;
                
        // A gradient angle of 45 degrees = an edge that is NW/SE
        // Check neighbors to the NE and SW
        case 45:
            // supress me if my neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row-1][l_col+1] || // north east
                my_magnitude <= l_data[l_row+1][l_col-1])   // south west
            {
                out[pos] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[pos] = my_magnitude;
            }
            break;
                    
        // A gradient angle of 90 degrees = an edge that is E/W
        // Check neighbors to the North and South
        case 90:
            // supress me if my neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row-1][l_col] || // north
                my_magnitude <= l_data[l_row+1][l_col])   // south
            {
                out[pos] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[pos] = my_magnitude;
            }
            break;
                    
        // A gradient angle of 135 degrees = an edge that is NE/SW
        // Check neighbors to the NW and SE
        case 135:
            // supress me if my neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row-1][l_col-1] || // north west
                my_magnitude <= l_data[l_row+1][l_col+1])   // south east
            {
                out[pos] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[pos] = my_magnitude;
            }
            break;
                    
        default:
            out[pos] = my_magnitude;
            break;
    } 
}
