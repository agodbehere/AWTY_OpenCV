/*
 * FGBGTest.cpp
 *
 *  Created on: May 7, 2012
 *      Author: Andrew B. Godbehere
 */

#include <opencv2/opencv.hpp>
#include "BackgroundSubtractorGBH.hpp"
#include <iostream>
#include <sstream>

// TODO: Rename this to BackgroundSubtractorGMG (godbehere matsukawa goldberg)

using namespace cv;

void help()
{
	std::cout <<
	"\nA program demonstrating the use and capabilities of a particular BackgroundSubtraction\n"
	"algorithm described in Godbehere, Matsukawa, Goldberg (2012), used in an interactive\n"
	"installation at the Contemporary Jewish Museum in San Francisco, CA from March 31 through\n"
	"July 31, 2011.\n"
	"Call:\n"
	"./BackgroundSubtractorGMG_sample\n"
	"Using OpenCV version %s\n" << CV_VERSION << "\n"<<std::endl;
}

int main(char *argc, char** argv)
{
	setUseOptimized(true);
	setNumThreads(8);

	Ptr<BackgroundSubtractorGMG> fgbg = Algorithm::create<BackgroundSubtractorGMG>("BackgroundSubtractor.SparseBayes");
	if (fgbg == NULL)
	{
		CV_Error(CV_StsError,"Failed to create Algorithm\n");
	}

	fgbg->set("smoothingRadius",7);
	fgbg->set("decisionThreshold",0.7);

	VideoCapture cap;
	cap.open("ThreeVisitors_Raw.avi");
	if (!cap.isOpened())
	{
		CV_Error(CV_StsError, "Fatal error, cannot read video\n");
		return -1;
	}

	Mat img, downimg, fgmask, upfgmask, posterior, upposterior;

	bool first = true;
	namedWindow("posterior");
	namedWindow("fgmask");
	namedWindow("annotated image");
	int i = 0;
	while (true)
	{
		std::stringstream txt;
		txt << "frame: ";
		txt << i++;

		cap >> img;
		putText(img,txt.str(),Point(20,40),FONT_HERSHEY_SIMPLEX,0.8,Scalar(1.0,0.0,0.0));

		resize(img,downimg,Size(160,120),0,0,INTER_NEAREST);   // Size(cols, rows) or Size(width,height)
		if (first)
		{
			fgbg->initializeType(downimg,0,255);
			first = false;
		}
		if (img.empty())
		{
			return 0;
		}
		(*fgbg)(downimg,fgmask);
		fgbg->updateBackgroundModel(Mat::zeros(120,160,CV_8U));
		fgbg->getPosteriorImage(posterior);
		resize(fgmask,upfgmask,Size(640,480),0,0,INTER_NEAREST);
		Mat coloredFG = Mat::zeros(480,640,CV_8UC3);
		coloredFG.setTo(Scalar(100,100,0),upfgmask);

		resize(posterior,upposterior,Size(640,480),0,0,INTER_NEAREST);
		imshow("posterior",upposterior);
		imshow("fgmask",upfgmask);
		imshow("annotated image",img+coloredFG);
		if (waitKey(2) == 'q')
			break;
	}

}

