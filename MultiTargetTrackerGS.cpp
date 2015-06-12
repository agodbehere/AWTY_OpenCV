/*
 * MultiTargetTrackerGS.cpp
 *
 *  Created on: May 21, 2012
 *      Author: andrewgodbehere
 */

// TODO: Change name to MultiBlobTrackerGS

#include "MultiTargetTrackerGS.hpp"
#include <queue>

namespace cv
{

void MultiTargetTracker::operator ()(InputArray maskImage,OutputArray objects,OutputArray covariances,OutputArray labels)
{

}


static Algorithm* createMultiTargetTrackerGS()
{
    return new MultiTargetTrackerGS;
}

static AlgorithmInfo MultiTargetTrackerGS_info("MultiTargetTracker.GaleShapley", createMultiTargetTrackerGS);

bool initModule_MultiTargetTrackerGS(void)
{
    Ptr<Algorithm> sb = createMultiTargetTrackerGS();
    return sb->info() != 0;
}

AlgorithmInfo* MultiTargetTrackerGS::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        MultiTargetTrackerGS obj;
        MultiTargetTrackerGS_info.addParam(obj, "A", obj.A,false,0,0,"State dynamics matrix. x(k+1) = Ax(k) + noise");
        MultiTargetTrackerGS_info.addParam(obj, "C", obj.C,false,0,0,"Observation matrix. y(k) = Cx(k) + noise");
        MultiTargetTrackerGS_info.addParam(obj, "Q", obj.Q,false,0,0,"Process noise covariance. x(k+1) = A(k) + process noise");
        MultiTargetTrackerGS_info.addParam(obj, "R", obj.R,false,0,0,"Observation noise covariance. y(k) = Cx(k) + observation noise");
        MultiTargetTrackerGS_info.addParam(obj, "initialCovariance",obj.initialCovariance,false,0,0,"Initial error covariance for the Kalman Filters.");
        MultiTargetTrackerGS_info.addParam(obj, "maxObjects",obj.maxObjects,false,0,0,"Maximum number of tracks to maintain.");
        initialized = true;
    }
    return &MultiTargetTrackerGS_info;
}

MultiTargetTrackerGS::MultiTargetTrackerGS()
{
	/*
	 * Default Parameters
	 */
	maxObjects = 200;
    A = (Mat_<float>(6,6) << 1,1,0,0,0,0,
                             0,1,0,0,0,0,
                             0,0,1,1,0,0,
                             0,0,0,1,0,0,
                             0,0,0,0,1,0,
                             0,0,0,0,0,1);

    /*
     * TODO: Specification of C should correlate directly to rectangle. First row: x. Second row: y. Third row: width. Fourth row: height. If using angle, fifth row, angle.
     */
    C = (Mat_<float>(4,6) << 1,0,0,0,0,0,
                             0,0,1,0,0,0,
                             0,0,0,0,1,0,
                             0,0,0,0,0,1);


    Q = (Mat_<float>(6,6) << 1/3.0,1/2.0,0,0,0,0,
                             1/2.0,1,0,0,0,0,
                             0,0,1/3.0,1/2.0,0,0,
                             0,0,1/2.0,1,0,0,
                             0,0,0,0,35.0,0,
                             0,0,0,0,0,35.0);

    R = (Mat_<float>(4,4) << 15.0,0,0,0,
                             0,15.0,0,0,
                             0,0,15.0,0,
                             0,0,0,15.0);


	/*
	 * System state variables
	 */
	precomputed = false;
}

bool MultiTargetTrackerGS::validate()
{
	/*
	 * If we've already validated and precomputed, we're done.
	 */
	if (precomputed)
		return true;

	dimState = A.cols;
	dimObservations = C.rows;

	/*
	 * Validate parameters
	 */
	if (A.cols != C.cols)
	{
		CV_Error(CV_StsBadSize,"A and C must have same number of columns.");
		return false;
	}
	if (C.rows != R.rows)
	{
		CV_Error(CV_StsBadSize,"C and R must have the same number of rows.");
		return false;
	}

	if (R.rows != R.cols)
	{
		CV_Error(CV_StsBadSize,"R must be square.");
		return false;
	}

	if (Q.rows != Q.cols)
	{
		CV_Error(CV_StsBadSize,"Q must be square.");
		return false;
	}
	if (A.cols != A.rows)
	{
		CV_Error(CV_StsBadSize,"A must be square.");
		return false;
	}
	if (A.cols != Q.cols)
	{
		CV_Error(CV_StsBadSize,"A and Q must have the same dimensions.");
		return false;
	}
	if (C.rows == 4)
	{
		useRotatedRectangles = false;
	}
	else if (C.rows == 5)
	{
		useRotatedRectangles = true;  // fifth dimension for angle
	}
	else
	{
		CV_Error(CV_StsBadSize,"C must have 4 or 5 rows to represent observations with rectangles.");
		return false;
	}

	/*
	 * TODO: Check that Q and R are positive definite
	 */

	/*
	 * Pre-compute initial covariance matrix
	 */
	Mat CTC = C.t()*C;
	CTC_Inverse = CTC.inv(DECOMP_SVD);
	//initialCovariance = C.t()*R*C+Q;
	initialCovariance = CTC_Inverse*C.t()*R*C*CTC_Inverse.t() + Q;
	precomputed = true;
	return true;
}

// TODO: Filter tracks, delete old tracks based on error matrix determinant being too large, etc.
// Have a function for dynamic filtering
void MultiTargetTrackerGS::operator ()(InputArray _MaskImage, OutputArray Objects, OutputArray covariances, OutputArray labels)
{
	/*
	 * Test if values have been pre-computed
	 */
	if (!validate())
		return;



	/*
	 * STEP 1: OBSERVATIONS =======================================================================
	 * Accept MaskImage and run connected components to return the set of observations
	 */
	// Observation vector [center(x),center(y),width,height,<angle>], CV_32F. 4 or 5 dimensions
	vector<Mat> observations;
	Mat MaskImage = _MaskImage.getMat();
	imgSize = MaskImage.size();
	vector<vector<Point> > contours;
	findContours(MaskImage,contours,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	vector<vector<Point> >::iterator contour;
	vector<vector<Point> >::iterator contour_end = contours.end();
	for (contour=contours.begin(); contour!=contour_end; ++contour)
	{
		if (useRotatedRectangles)
		{
			RotatedRect obsRect = minAreaRect(*contour);
			Mat obsMat = Mat(5,1,CV_32F);
			Point2f center = obsRect.center;
			Size2f size = obsRect.size;
			obsMat.at<float>(0,0) = center.x;
			obsMat.at<float>(1,0) = center.y;
			obsMat.at<float>(2,0) = size.width;
			obsMat.at<float>(3,0) = size.height;
			obsMat.at<float>(4,0) = obsRect.angle;
			observations.push_back(obsMat);
		}
		else
		{
			/*
			 * Find bounding box (not rotated).
			 * Transform data structure into more general rotated rectangle.
			 * Add rectangle to objects data structure.
			 */
			Rect obsRect = boundingRect(*contour);
			Mat obsMat(4,1,CV_32F);
			obsMat.at<float>(0,0) = obsRect.x + obsRect.width/2.0;
			obsMat.at<float>(1,0) = obsRect.y + obsRect.height/2.0;

			Size2f size;
			obsMat.at<float>(2,0) = obsRect.width;
			obsMat.at<float>(3,0) = obsRect.height;

			observations.push_back(obsMat);
		}
	}

	/*
	 * STEP 2: PREDICTIONS ========================================================================
	 */
	vector<Mat> predictedObservations;
	//vector<Mat> predictedObservationCovariances;
	vector<KalmanFilter>::iterator track;
	vector<KalmanFilter>::iterator lastTrack = tracks.end();
	for (track = tracks.begin(); track != lastTrack; ++track)
	{
		const Mat predictedState = track->predict();
		Mat predictedObs = C*predictedState;
		predictedObservations.push_back(predictedObs);
		//Mat predictedObservationCovariance = C*track->errorCovPre*C.t()+R;
		//predictedObservationCovariances.push_back(predictedObservationCovariance);
	}

	/*
	 * STEP 3: TRACK TO OBSERVATION MATCHING (GALE-SHAPLEY) =======================================
	 */

	// (a) If there were no observations, all existing tracks are updated with no observation.
	// See Sinopoli, Schenato,Franceschetti, Poola, Jordan, and Sastry (2004).
	if (observations.empty())
	{
		if (tracks.empty())  // nothing to do
			return;
		vector<KalmanFilter>::iterator track;
		vector<KalmanFilter>::iterator lastTrack = tracks.end();
		for (track = tracks.begin(); track != lastTrack; ++track)
		{
			track->errorCovPre.copyTo(track->errorCovPost);
			track->statePre.copyTo(track->statePost);
		}
		return;
	}

	// (b) If there are no existing tracks, then we simply create new tracks.
	if (tracks.empty())
	{
		// Iterate through observations
		vector<Mat>::iterator observation;
		vector<Mat>::iterator lastObservation = observations.end();
		for (observation = observations.begin(); observation!= lastObservation; ++observation)
		{
			KalmanFilter newTrack(dimState,dimObservations);
			// set linear system parameters
			A.copyTo(newTrack.transitionMatrix);
			C.copyTo(newTrack.measurementMatrix);
			Q.copyTo(newTrack.processNoiseCov);
			R.copyTo(newTrack.measurementNoiseCov);
			// set initial state and covariance
			Mat initialState = CTC_Inverse*(*observation);
			initialState.copyTo(newTrack.statePost);
			initialCovariance.copyTo(newTrack.errorCovPost);
			tracks.push_back(newTrack);
		}
		return;
	}

	/* (c) General Gale-Shapley Matching. */
	/* See Gale, Shapley (1962) and Godbehere, Matsukawa, Goldberg (2012) */

	// (i) Calculate corresponence matrix
	Mat correspondence = Mat::zeros(predictedObservations.size(),observations.size(),CV_64F);
	vector<Mat>::iterator predictedObs;
	vector<Mat>::iterator lastPredictedObs = predictedObservations.end();
	vector<Mat>::iterator observation;
	vector<Mat>::iterator lastObservation = observations.end();
	unsigned int i;
	int j = 0;
	for (i = 0, predictedObs = predictedObservations.begin(); predictedObs!= lastPredictedObs; ++predictedObs, ++i)
	{
		// predicted observation covariances...
		Mat cov = C*track->errorCovPre*C.t()+R; // cov is PSD
		// TODO: Test PSD with determinant > 0
		Mat covinv = cov.inv(DECOMP_CHOLESKY);  // TODO: IF THIS DOESN'T WORK, USE DECOMP_SVD (more accurate but slower)
		for (j = 0, observation = observations.begin(); observation != lastObservation; ++observation, ++j)
		{
			correspondence.at<double>(i,j) = (double)(exp(-Mahalanobis((*predictedObs),(*observation),covinv)));
		}
	}
	threshold(correspondence,correspondence,0.00001,0.0,THRESH_TOZERO);  // TODO: Make correspondence threshold a parameter

	Mat pairMatrix = Mat::zeros(predictedObservations.size(),observations.size(),CV_64F);

	// TODO: Check this stable marriage implementation.
	// I'm concerned that I don't make the marriedpair vector large enough at the beginning
	vector<MarriedPair> pairs;  // TODO: get rid of MarriedPair vector in favor of matrix, easier to lookup in matrix
	std::queue<int> unpairedTracks;  // store indices of tracks for pairing
	//int j;
	// create the necessary pairs (reserve space)
	/*for (j = 0; j < observations.size() + predictedObservations.size(); ++j)
	{
		//MarriedPair pair;
		//pairs.push_back(pair);
		unpairedTracks.push(j);  // remember tracks (and null-tracks) that need pairing.
	}*/
	for (i=0; i<predictedObservations.size(); ++i)
	{
		unpairedTracks.push(i);
	}
	while (!unpairedTracks.empty())  // continue until all tracks have been paired
	{
		int i = unpairedTracks.front();
		Mat preferences = correspondence.row(i);
		int maxIndex;
		double maxVal;
		while (true)
		{
			minMaxIdx(preferences,NULL,&maxVal,NULL,&maxIndex); // find max val and index in row, corresponds to preference and observation
			// we have our preference, now let's see if that pair is already taken
			double pairval;
			int pairindex;
			Mat pairvals = pairMatrix.col(maxIndex);
			minMaxIdx(pairvals,NULL,&pairval,NULL,&pairindex);

			if (maxVal == 0.0)
			{
				// then we best keep this track unpaired. Too far from anything else.
				unpairedTracks.pop(); // remove track from unpaired tracks list. we are pairing it with null observation.
				break;
			}

			if (pairval < maxVal)
			{
				// ours will replace it!
				unpairedTracks.pop();
				pairMatrix.at<double>(pairindex,maxIndex) = 0.0;
				pairMatrix.at<double>(i,maxIndex) = maxVal;
				unpairedTracks.push(pairindex);  // pairindex track got dumped, put back in unpaired queue
				// since pairindex got dumped, must set appropriate element in correspondence matrix to 0, won't pair with that one again, broken heart
				correspondence.at<double>(pairindex,maxIndex) = 0.0;
				break; // made the pair for current track, go to the next
			}
			// if observation is taken and preference is stronger, look for next stronger
			else
			{
				correspondence.at<double>(i,maxIndex) = 0.0;  // this attempted pairing failed
			}
		}
	}
	// All pairings complete by now.
	/*
     * STEP 4: TRACK UPDATES ======================================================================
	 */
	//vector<KalmanFilter>::iterator track;
	//vector<KalmanFilter>::iterator lastTrack = tracks.end();
	lastTrack = tracks.end();
	vector<int> observationIndices(observations.size(),1);
	for (i=0,track = tracks.begin(); track != lastTrack; ++track,++i)
	{
		// (i) Find the pair
		Mat pairRow = pairMatrix.row(i);
		int pairIndex;
		double pairWeight;
		minMaxIdx(pairRow,NULL,&pairWeight,NULL,&pairIndex);
		if (pairWeight == 0.0)
		{
			// unpaired track. Update with no observation.
			track->errorCovPre.copyTo(track->errorCovPost);
			track->statePre.copyTo(track->statePost);
			continue; // next iteration
		}
		else
		{
			observationIndices[pairIndex] = 0; // observation is accounted for
			// normal update
			track->correct(observations[pairIndex]);
		}
	}

	vector<int>::iterator observationIndex;
	vector<int>::iterator lastObservationIndex = observationIndices.end();
	for (i=0,observationIndex = observationIndices.begin(); observationIndex != lastObservationIndex; ++observationIndex,++i)
	{
		if (*observationIndex == 1)
		{
			// then observaiton is unpaired. Create a new track.
			KalmanFilter newTrack(dimState,dimObservations);
			// set linear system parameters
			A.copyTo(newTrack.transitionMatrix);
			C.copyTo(newTrack.measurementMatrix);
			Q.copyTo(newTrack.processNoiseCov);
			R.copyTo(newTrack.measurementNoiseCov);
			// set initial state and covariance
			Mat initialState = CTC_Inverse*(observations[i]);
			initialState.copyTo(newTrack.statePost);
			initialCovariance.copyTo(newTrack.errorCovPost);
		}
	}
}
// TODO: Delete tracks based on confidence model.
void MultiTargetTrackerGS::deleteTracks(vector<int> trackIndices)
{
	// specify the tracks to delete. Let's convert that to tracks to keep.
	vector<bool> tracksToKeep (tracks.size(),true);
	vector<KalmanFilter> keptTracks;

	vector<int>::iterator trackIndex;
	vector<int>::iterator lastTrackIndex = trackIndices.end();
	for (trackIndex = trackIndices.begin(); trackIndex != lastTrackIndex; ++trackIndex)
	{
		tracksToKeep[*trackIndex]=false;
	}
	unsigned int i;
	for (i = 0; i < tracks.size(); ++i)
	{
		if (tracksToKeep[i])
		{
			keptTracks.push_back(tracks[i]);
		}
	}
	tracks.clear();
	tracks = keptTracks;
}

void MultiTargetTrackerGS::getMaskImage(OutputArray _maskimg)
{
	_maskimg.create(imgSize,CV_8U);
	Mat maskimg = _maskimg.getMat();
	vector<KalmanFilter>::iterator track;
	vector<KalmanFilter>::iterator lastTrack = tracks.end();
	for (track = tracks.begin(); track != lastTrack; ++track)
	{
		Point2f vertices[4];
		Mat obsMat = track->measurementMatrix*track->statePost;
		RotatedRect obsRect;

		obsRect.center.x = obsMat.at<float>(0,0);
		obsRect.center.y = obsMat.at<float>(1,0);
		obsRect.size.width = obsMat.at<float>(2,0);
		obsRect.size.height = obsMat.at<float>(3,0);
		if (useRotatedRectangles)
			obsRect.angle = obsMat.at<float>(4,0);
		else
			obsRect.angle = 0.0;

		obsRect.points(vertices);
		Point2i vertexInts[4];

		for (uint i = 0; i < 4; ++i)
		{
			vertexInts[i].x = int(vertices[i].x);
			vertexInts[i].y = int(vertices[i].y);
		}

		fillConvexPoly(maskimg,vertexInts,4,Scalar(255)); // fill region of image with 255 (foreground value)
	}

	maskimg.copyTo(_maskimg);
}

MultiTargetTrackerGS::~MultiTargetTrackerGS()
{

}

} /* namespace cv */
