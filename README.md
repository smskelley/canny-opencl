#canny-opencl

##Summary
This is an implementation of the Canny Edge Detection algorithm using OpenCL in
C++. It uses OpenCV for some utility features, such as capturing an image from
a webcam, opening/writing an image file, and converting from BGR to Grayscale.
Note that it was considered outside of the scope of this project to maintain
cross platform compatability, so it has only been tested in OSX 10.10.
Additionally, because OpenCL applications can be heavily optimized depending on
the target hardware, the range of target hardware has been significantly
limited. While this should run on most modern macs, it has only really been
optimized and tested on my early 2011 Macbook Pro 15".

##Executables
It contains two executables: live-capture and benchmark-suite. Live capture
allows the user to view the results of canny edge detection in near real time
using the webcam as input. The benchmark suite uses several implementations
of the canny edge detection algorithm on multiple images. Each implementation
is run multiple times on each image. Next, the stages of the algorithm are
timed in isolation to better understand which steps are taking the longest.

##Code Layout
The code is separated into three logical groups: image processors, live capture,
and benchmarking.

###Image Processors
All Image Processors are contained within the ImageProcessors namespace and
inherit from the ImageProcessor abstract base class. Each image processor
implements the canny edge detection algorithm, but each one does it in a
different way. The focus of this project was canny edge detection in opencl,
so the vast majority of effort was put into the OpenclImageProcessor class.
Other image processors include SerialImageProcessor and CvImageProcessor.

####OpenclImageProcessor
Based upon a boolean accepted by the constructor, this class can either
target the CPU or GPU. When using the GPU, it will use 16x16 sized workgroups
so that it may copy chunks of data to faster memory on the GPU and make fewer
calls to global memory. While this is advantageous on the GPU, CPUs generally
don't support two demensional work groups, so the same kernels cannot be used.
As such, the **kernels** directory contains two sets. These kernels are only
used for the OpenclImageProcessor.

####SerialImageProcessor
The serial image processor was constructed by serializing the
OpenclImageProcessor. It is used so that one may determine the speedup we get
from using the OpenclImageProcessor.

####CvImageProcessor
The Cv Image Processor simply uses OpenCV's canny edge detection algorithm.
Unlike the other classes, it does not support executing the different stages
of the algorithm in isolation.

###Live Capture
    usage: ./live-capture [cpu|serial]
    If no argument is given, live-capture will use the GPU.

An image processor is created based upon arguments if any exist. When operating
in CPU and GPU mode, live-capture uses the OpenclImageProcessor. Live capture
is performed using OpenCV to capture frames from the webcam. The frames are
then fed into the image processor and its result is displayed on screen. The
result is a near realtime video of canny edge detection algorithm running on
the webcam's data.

###Benchmarking
    usage: ./benchmark-suite
    Note: This currently requires images to be fetched and on the local path
          ./images/. If you haven't already, run tools/fetch-images.sh to obtain
          these images.

Because the focus of this project was to produce a reasonably performant
implementation of the canny edge detection algorithm, a fair amount of effort
went into creating a flexible benchmarking system. It is desirable to know
how this implementation compares to others, how it performs at different
resolutions, and how much time each stage of the algorithm consumes. As such,
the benchmarking-suite runs several iterations of each pair of image and
implementation to find the mean and standard deviation of time taken. After
benchmarking the full algorithm, each stage is run in isolation to capture the
amount of time each stage consumes.
