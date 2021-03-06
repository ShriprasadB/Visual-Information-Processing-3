#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace std;
using namespace cv;


int xGradient(Mat image, int x, int y)
{
    return image.at<Vec3b>(y-1, x-1)[2] + 2*image.at<Vec3b>(y, x-1)[2] +image.at<Vec3b>(y+1, x-1)[2] -image.at<Vec3b>(y-1, x+1)[2] -2*image.at<Vec3b>(y, x+1)[2] - image.at<Vec3b>(y+1, x+1)[2];
}
 
// y component of the gradient vector
 
int yGradient(Mat image, int x, int y)
{
    return image.at<Vec3b>(y-1, x-1)[2] + 2*image.at<Vec3b>(y-1, x)[2] + image.at<Vec3b>(y-1, x+1)[2] - image.at<Vec3b>(y+1, x-1)[2] - 2*image.at<Vec3b>(y+1, x)[2] - image.at<Vec3b>(y+1, x+1)[2];
}

Mat toHSI(Mat src_image)
{
	Mat hsi_image(src_image.rows, src_image.cols, src_image.type());

	float red, green, blue, hue, sat, in;

	for (int i = 0; i < src_image.rows; i++)
	{
		for (int j = 0; j < src_image.cols; j++)
		{
			blue = src_image.at<Vec3b>(i, j)[0];			
			green = src_image.at<Vec3b>(i, j)[1];			//green channel
			red = src_image.at<Vec3b>(i, j)[2];			//red channel

			in = (blue + green + red) / 3;			//intensity value

			int min_value = 0;
			min_value = std::min(red, std::min(blue, green));
			sat = 1 - 3 * (min_value / (blue + green + red));		//saturation value
			if (sat < 0.00001)
			{
				sat = 0;
			}
			else if (sat > 0.99999)
			{
				sat = 1;
			}

			if (sat != 0)
			{
				// hue value
				hue = 0.5 * ((red - green) + (red - blue)) / sqrt(((red - green)*(red - green)) + ((red - blue)*(green - blue)));
				hue = acos(hue);

				if (blue <= green)
				{
					hue = hue;
				}
				else
				{
					hue = ((360 * 3.14159265) / 180.0) - hue;
				}
			}

			//hsi_image.at<Vec3b>(i, j)[0] = (hue * 180) / 3.14159265;
			//hsi_image.at<Vec3b>(i, j)[1] = sat * 100;
			hsi_image.at<Vec3b>(i, j)[2] = in;					//intensity image
		}

	}

	/*namedWindow("RGB image", CV_WINDOW_AUTOSIZE);
	imshow("RGB image", src_image);

	namedWindow("HSI image", CV_WINDOW_AUTOSIZE);
	imshow("HSI image", hsi_image);

	waitKey(0);
	
	cvDestroyAllWindows();*/
	return hsi_image;

}


int main(int argc, char **argv) {
  


  
  	string filename;
	cout << "Enter image name with full path " << endl;
	cin >> filename;
	Mat f1 = imread(filename, 1);	//Load the image


  //Mat f1= imread(argv[1], CV_LOAD_IMAGE_COLOR);
  
  
  Mat *image = &f1;

  unsigned char *input = (unsigned char*)(f1.data);
 
  
  if(!f1.data) {
    cout << "ERROR: Could not load image data." << endl;
    return -1;
  }


  Mat hsi = toHSI(f1);

  


  // Sobel operator 
  

  int gx, gy, sum;
 Mat f_hsi = hsi.clone();
  for(int y = 0; y < hsi.rows; y++){
    for(int x = 0; x < hsi.cols; x++){

      f_hsi.at<Vec3b>(y, x)[0]= 0.0;
      f_hsi.at<Vec3b>(y, x)[1]= 0.0;
      f_hsi.at<Vec3b>(y, x)[2]= 0.0;
    }
  }

	
 
  for(int y = 1; y < hsi.rows - 1; y++){
      for(int x = 1; x < hsi.cols - 1; x++){
        gx = xGradient(hsi, x, y);

        gy = yGradient(hsi, x, y);
        
        
        sum = abs(gx) + abs(gy);
        if(sum>255)
          sum=255;
        else if(sum<0)
          sum=0;
        f_hsi.at<Vec3b>(y, x)[2]= sum;
      }
  }

  
  
 
  namedWindow("Roi");
 
  bool loop = true;
  while(loop) {
    imshow("Roi", *image);
    
    switch(cvWaitKey(15)) {
      case 27:  //Exit 
        loop = false;
        break;
      case 32:  //Swap image pointer if spacebar is pressed
        if(image == &f1) image = &f_hsi;
        else image = &f1;
break;
      default:
        break;
    }
  }
}


