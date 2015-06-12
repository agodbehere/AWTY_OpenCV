/*
 * BackgroundSubtractorGBH_test.cpp
 *
 *  Created on: Jun 14, 2012
 *      Author: andrewgodbehere
 */

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ts/ts.hpp"
#include "BackgroundSubtractorGBH.hpp"

using namespace cv;

class CV_BackgroundSubtractorTest : public cvtest::BaseTest
{
public:
	CV_BackgroundSubtractorTest();
protected:
	void run(int);
};

CV_BackgroundSubtractorTest::CV_BackgroundSubtractorTest()
{
}

void CV_BackgroundSubtractorTest::run(int)
{
	int code = cvtest::TS::OK;
	RNG& rng = ts->get_rng();
	int type = ((unsigned int)rng)%7;  // pick a random type, 0 - 6, defined in types_c.h
	int channels = 1 + ((unsigned int)rng)%4;  // random number of channels from 1 to 4.
	int channelsAndType = CV_MAKETYPE(type,channels);
	int width = 2 + ((unsigned int)rng)%98; // Mat will be 2 to 100 in width and height
	int height = 2 + ((unsigned int)rng)%98;

	Ptr<BackgroundSubtractorGBH> fgbg = Algorithm::create<BackgroundSubtractorGBH>("BackgroundSubtractor.SparseBayes");
	Mat fgmask;

	if (fgbg == NULL)
	{
		CV_Error(CV_StsError,"Failed to create Algorithm\n");
	}
	fgbg->set("smoothingRadius",7);
	fgbg->set("decisionThreshold",0.7);
	fgbg->set("initializationFrames",120);

	/*
	 * Generate bounds for the values in the matrix
	 */
	flexitype max,min;
	if (type == CV_8U)
	{
		unsigned char half = UCHAR_MAX/2;
		max.uc = (unsigned char)rng.uniform(half+32,UCHAR_MAX);
		min.uc = (unsigned char)rng.uniform(0,half-32);
	}
	else if (type == CV_8S)
	{
		char half = CHAR_MAX/2 + CHAR_MIN/2;
		max.c = (char)rng.uniform(half+32,CHAR_MAX);
		min.c = (char)rng.uniform(CHAR_MIN,half-32);
	}
	else if (type == CV_16U)
	{
		uint half = UINT_MAX/2;
		max.ui = (unsigned int)rng.uniform((int)half+32,UINT_MAX);
		min.ui = (unsigned int)rng.uniform(0,(int)half-32);
	}
	else if (type == CV_16S)
	{
		int half = INT_MAX/2 + INT_MIN/2;
		max.i = rng.uniform(half+32,INT_MAX);
		min.i = rng.uniform(INT_MIN,half-32);
	}
	else if (type == CV_32S)
	{
		long int half = LONG_MAX/2 + LONG_MIN/2;
		max.li = rng.uniform((int)half+32,(int)LONG_MAX);
		min.li = rng.uniform((int)LONG_MIN,(int)half-32);
	}
	else if (type == CV_32F)
	{
		float half = FLT_MAX/2.0 + FLT_MIN/2.0;
		max.f = rng.uniform(half+(float)32.0*FLT_EPSILON,FLT_MAX);
		min.f = rng.uniform(FLT_MIN,half-(float)32.0*FLT_EPSILON);
	}
	else if (type == CV_64F)
	{
		double half = DBL_MAX/2.0 + DBL_MIN/2.0;
		max.d = rng.uniform(half+(double)32.0*DBL_EPSILON,DBL_MAX);
		min.d = rng.uniform(DBL_MIN,half-(double)32.0*DBL_EPSILON);
	}

	Mat simImage = Mat::zeros(height,width,channelsAndType);
	const uint numLearningFrames = 120;
	for (uint i = 0; i < numLearningFrames; ++i)
	{
		/*
		 * Genrate simulated "image"
		 */
		if (type == CV_8U)
			rng.fill(simImage,RNG::UNIFORM,(unsigned char)(min.uc/2+max.uc/2),max.uc);
		else if (type == CV_8S)
			rng.fill(simImage,RNG::UNIFORM,(char)(min.c/2+max.c/2),max.c);
		else if (type == CV_16U)
			rng.fill(simImage,RNG::UNIFORM,(unsigned int)(min.ui/2+max.ui/2),max.ui);
		else if (type == CV_16S)
			rng.fill(simImage,RNG::UNIFORM,(int)(min.i/2+max.i/2),max.i);
		else if (type == CV_32F)
			rng.fill(simImage,RNG::UNIFORM,(float)(min.f/2.0+max.f/2.0),max.f);
		else if (type == CV_32S)
			rng.fill(simImage,RNG::UNIFORM,(long int)(min.li/2+max.li/2),max.li);
		else if (type == CV_64F)
			rng.fill(simImage,RNG::UNIFORM,(double)(min.d/2.0+max.d/2.0),max.d);

		/*
		 * Feed images into background subtractor
		 */
		if (i == 0)
		{
			fgbg->initializeType(simImage,min,max);
		}
		(*fgbg)(simImage,fgmask);
		Mat fullbg = Mat::zeros(Size(simImage.cols,simImage.rows),CV_8U);
		fgbg->updateBackgroundModel(fullbg);

		// fgmask should be entirely background during training
		code = cvtest::cmpEps2( ts, fgmask, fullbg, 0, false, "The training foreground mask" );
		if (code < 0)
			ts->set_failed_test_info( code );
	}
	// last one!
	if (type == CV_8U)
		rng.fill(simImage,RNG::UNIFORM,min.uc,min.uc);
	else if (type == CV_8S)
		rng.fill(simImage,RNG::UNIFORM,min.c,min.c);
	else if (type == CV_16U)
		rng.fill(simImage,RNG::UNIFORM,min.ui,min.ui);
	else if (type == CV_16S)
		rng.fill(simImage,RNG::UNIFORM,min.i,min.i);
	else if (type == CV_32F)
		rng.fill(simImage,RNG::UNIFORM,min.f,min.f);
	else if (type == CV_32S)
		rng.fill(simImage,RNG::UNIFORM,min.li,min.li);
	else if (type == CV_64F)
		rng.fill(simImage,RNG::UNIFORM,min.d,min.d);

	(*fgbg)(simImage,fgmask);
	// now fgmask should be entirely foreground
	Mat fullfg = 255*Mat::ones(Size(simImage.cols,simImage.rows),CV_8U);
	code = cvtest::cmpEps2( ts, fgmask, fullfg, 0, false, "The final foreground mask" );
	if (code < 0)
	{
		ts->set_failed_test_info( code );
	}

}


TEST(VIDEO_BGSUBGMG, accuracy) { CV_BackgroundSubtractorTest test; test.safe_run(); }
