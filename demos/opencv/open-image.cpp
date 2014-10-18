#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<"Please specify the image you want to open: ./open-image image-name.jpg" << endl;
     return -1;
    }

    cout << "Press any key in the image window to close the image." << endl;

    Mat image;	// Create a matrix to hold the image
    
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   	// Read the image into the matrix

    if(! image.data )                              	// Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE );	// Create a window for display
    imshow( "Display window", image );                 	// Show the image inside it

    waitKey(0);                                       	// Wait for a keystroke in the window
    return 0;
	}
