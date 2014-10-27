// This is a practice kernel for OpenCL
__kernel void add_1(__global float* input, __global float* output,                                             
                    const unsigned int count)                                           
{                                                                      
    int i = get_global_id(0);   // find out which process I am                                           
    if(i < count)               // as long as I'm less than the count
    {                           // perform my operation                            
       output[i] = input[i] + 1;                                
    }
}                                                                     
