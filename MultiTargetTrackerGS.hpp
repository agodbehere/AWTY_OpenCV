/*
 * MultiTargetTrackerGS.hpp
 *
 *  Created on: May 21, 2012
 *      Author: andrewgodbehere
 */

#ifndef MULTITARGETTRACKERGS_HPP_
#define MULTITARGETTRACKERGS_HPP_

#include "opencv2/opencv.hpp"

namespace cv
{

/*
 * Image plane object tracking
 */
class CV_EXPORTS_W MultiTargetTracker: public Algorithm
{
public:
	virtual ~MultiTargetTracker(){};

	CV_WRAP_AS(apply) virtual void operator()(InputArray MaskImage, OutputArray objects, OutputArray covariances, OutputArray labels);
};

class CV_EXPORTS_W MultiTargetTrackerGS: public MultiTargetTracker
{
public:
	CV_WRAP MultiTargetTrackerGS();
	virtual ~MultiTargetTrackerGS();
	virtual AlgorithmInfo* info() const;
	virtual void 	operator()(InputArray MaskImage, OutputArray objects, OutputArray covariances,
								OutputArray labels);
	// input mask image, vector of tracks to delete. Output vector of rectangles (object locations) and vector of numerical object labels
	virtual void	getMaskImage(OutputArray maskimg);  // TODO: Don't return Mat, make it an OutputArray in argument.
	virtual void	deleteTracks(vector<int> objectIndices); // TODO: delete by index, not by object

protected:
	int		maxObjects;
	Mat		A,C,Q,R;  // Kalman filter parameters. Same across the board
	Mat		initialCovariance;
	vector<KalmanFilter> tracks;
	//vector<Mat> states;  // estimated states
	vector<Mat> covariances;
	vector<int> labels;

private:
	Mat		CTC_Inverse;  // calculated at initialization for efficiency
	bool	precomputed;  // whether or not certain matrices have been precomputed for efficiency
	bool	useRotatedRectangles; // if rank of specified C is ==5, extra observation.
	int		dimObservations;
	int		dimState;
	vector<int> toDelete;  // store track labels to delete
	Size	imgSize;
	bool 	validate();  // run if not precomputed to generate initial values, and validate parameters

	struct MarriedPair
	{
		// i,j indices in correspondence matrix
		int i;
		int j;
		double weight; // preference relation weight
		MarriedPair(){i = -1; j = -1; weight=0.0;}
	};
};



// TODO: Separate module for filtering out tracks, decide which should be deleted.

} /* namespace cv */
#endif /* MULTITARGETTRACKERGS_HPP_ */
