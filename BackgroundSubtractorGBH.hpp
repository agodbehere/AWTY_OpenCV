/*
Copyright (c) 2012, Andrew B. Godbehere
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution. Neither the name of UC Berkeley nor the names of its
contributors may be used to endorse or promote products derived from this software without specific
prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef BACKGROUNDSUBTRACTORGMG_HPP_
#define BACKGROUNDSUBTRACTORGMG_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include <list>
#include <stdint.h>

namespace cv {




/**
 * Background Subtractor module. Takes a series of images and returns a sequence of mask (8UC1)
 * images of the same size, where 255 indicates Foreground and 0 represents Background.
 * This class implements an algorithm described in "Visual Tracking of Human Visitors under
 * Variable-Lighting Conditions for a Responsive Audio Art Installation," A. Godbehere,
 * A. Matsukawa, K. Goldberg, American Control Conference, Montreal, June 2012.
 */
class CV_EXPORTS BackgroundSubtractorGMG: public cv::BackgroundSubtractor
{
private:
	/**
	 * 	A general flexible datatype.
	 *
	 * 	Used internally to enable background subtraction algorithm to be robust to any input Mat type.
	 * 	Datatype can be char, unsigned char, int, unsigned int, long int, float, or double.
	 */
	union flexitype{
		char c;
		uchar uc;
		int i;
		unsigned int ui;
		long int li;
		float f;
		double d;

		flexitype(){d = 0.0;}  //!< Default constructor, set all bits of the union to 0.
		flexitype(char cval){c = cval;} //!< Char type constructor

		bool operator ==(flexitype& rhs)
		{
			return d == rhs.d;
		}

		//! Char type assignment operator
		flexitype& operator =(char cval){
			if (this->c == cval){return *this;}
			c = cval; return *this;
		}
		flexitype(unsigned char ucval){uc = ucval;} //!< unsigned char type constructor

		//! unsigned char type assignment operator
		flexitype& operator =(unsigned char ucval){
			if (this->uc == ucval){return *this;}
			uc = ucval; return *this;
		}
		flexitype(int ival){i = ival;} //!< int type constructor
		//! int type assignment operator
		flexitype& operator =(int ival){
			if (this->i == ival){return *this;}
			i = ival; return *this;
		}
		flexitype(unsigned int uival){ui = uival;} //!< unsigned int type constructor

		//! unsigned int type assignment operator
		flexitype& operator =(unsigned int uival){
			if (this->ui == uival){return *this;}
			ui = uival; return *this;
		}
		flexitype(float fval){f = fval;} //!< float type constructor
		//! float type assignment operator
		flexitype& operator =(float fval){
			if (this->f == fval){return *this;}
			f = fval; return *this;
		}
		flexitype(long int lival){li = lival;} //!< long int type constructor
		//! long int type assignment operator
		flexitype& operator =(long int lival){
			if (this->li == lival){return *this;}
			li = lival; return *this;
		}

		flexitype(double dval){d=dval;} //!< double type constructor
		//! double type assignment operator
		flexitype& operator =(double dval){
			if (this->d == dval){return *this;}
		d = dval; return *this;
		}
	};
	/**
	 *	Used internally to represent a single feature in a histogram.
	 *	Feature is a color and an associated likelihood (weight in the histogram).
	 */
	struct HistogramFeatureGMG
	{
		/**
		 * Default constructor.
		 * Initializes likelihood of feature to 0, color remains uninitialized.
		 */
		HistogramFeatureGMG(){likelihood = 0.0;}

		/**
		 * Copy constructor.
		 * Required to use HistogramFeatureGMG in a std::vector
		 * @see operator =()
		 */
		HistogramFeatureGMG(const HistogramFeatureGMG& orig){
			color = orig.color; likelihood = orig.likelihood;
		}

		/**
		 * Assignment operator.
		 * Required to use HistogramFeatureGMG in a std::vector
		 */
		HistogramFeatureGMG& operator =(const HistogramFeatureGMG& orig){
			color = orig.color; likelihood = orig.likelihood; return *this;
		}

		/**
		 * Tests equality of histogram features.
		 * Equality is tested only by matching the color (feature), not the likelihood.
		 * This operator is used to look up an observed feature in a histogram.
		 */
		bool operator ==(HistogramFeatureGMG &rhs);

		//! Regardless of the image datatype, it is quantized and mapped to an integer and represented as a vector.
		vector<size_t>			color;

		//! Represents the weight of feature in the histogram.
		float 					likelihood;
		friend class PixelModelGMG;
	};

	/**
	 * 	Representation of the statistical model of a single pixel for use in the background subtraction
	 * 	algorithm.
	 */
	class PixelModelGMG
	{
	public:
		PixelModelGMG();
		virtual ~PixelModelGMG();

		/**
		 * 	Incorporate the last observed feature into the statistical model.
		 *
		 * 	@param learningRate	The adaptation parameter for the histogram. -1.0 to use default. Value
		 * 						should be between 0.0 and 1.0, the higher the value, the faster the
		 * 						adaptation. 1.0 is limiting case where fast adaptation means no memory.
		 */
		void 	insertFeature(double learningRate = -1.0);

		/**
		 * 	Set the feature last observed, to save before incorporating it into the statistical
		 * 	model with insertFeature().
		 *
		 * 	@param feature		The feature (color) just observed.
		 */
		void	setLastObservedFeature(BackgroundSubtractorGMG::HistogramFeatureGMG feature);
		/**
		 *	Set the upper limit for the number of features to store in the histogram. Use to adjust
		 *	memory requirements.
		 *
		 *	@param max			size_t representing the max number of features.
		 */
		void	setMaxFeatures(size_t max) {
			maxFeatures = max; histogram.resize(max); histogram.clear();
		}
		/**
		 * 	Normalize the histogram, so sum of weights of all features = 1.0
		 */
		void	normalizeHistogram();
		/**
		 * 	Return the weight of a feature in the histogram. If the feature is not represented in the
		 * 	histogram, the weight returned is 0.0.
		 */
		double 	getLikelihood(HistogramFeatureGMG f);
		PixelModelGMG& operator *=(const float &rhs);
		//friend class BackgroundSubtractorGMG;
		friend class HistogramFeatureGMG;
	protected:
		size_t numFeatures;  //!< number of features in histogram
		size_t maxFeatures; //!< max allowable features in histogram
		std::list<HistogramFeatureGMG> histogram; //!< represents the histogram as a list of features
		HistogramFeatureGMG			   lastObservedFeature;
		//!< store last observed feature in case we need to add it to histogram
	};

public:
	BackgroundSubtractorGMG();
	virtual ~BackgroundSubtractorGMG();
	virtual AlgorithmInfo* info() const;

	/**
	 * Performs single-frame background subtraction and builds up a statistical background image
	 * model.
	 * @param image Input image
	 * @param fgmask Output mask image representing foreground and background pixels
	 */
	virtual void operator()(InputArray image, OutputArray fgmask, double learningRate=-1.0);

	/**
	 * Validate parameters and set up data structures for appropriate image type. Must call before
	 * running on data.
	 * @param image One sample image from dataset
	 * @param min	minimum value taken on by pixels in image sequence. Usually 0
	 * @param max	maximum value taken on by pixels in image sequence. e.g. 1.0 or 255
	 */
	void	initializeType(InputArray image, flexitype min, flexitype max);
	/**
	 * Selectively update the background model. Only update background model for pixels identified
	 * as background.
	 * @param mask 	Mask image same size as images in sequence. Must be 8UC1 matrix, 255 for foreground
	 * and 0 for background.
	 */
	void	updateBackgroundModel(InputArray mask);
	/**
	 * Retrieve the greyscale image representing the probability that each pixel is foreground given
	 * the current estimated background model. Values are 0.0 (black) to 1.0 (white).
	 * @param img The 32FC1 image representing per-pixel probabilities that the pixel is foreground.
	 */
	void	getPosteriorImage(OutputArray img);

protected:
	//! Total number of distinct colors to maintain in histogram.
	int		maxFeatures;
	//! Set between 0.0 and 1.0, determines how quickly features are "forgotten" from histograms.
	double	learningRate;
	//! Number of frames of video to use to initialize histograms.
	int		numInitializationFrames;
	//! Number of discrete levels in each channel to be used in histograms.
	int		quantizationLevels;
	//! Prior probability that any given pixel is a background pixel. A sensitivity parameter.
	double	backgroundPrior;

	double	decisionThreshold; //!< value above which pixel is determined to be FG.
	int		smoothingRadius;  //!< smoothing radius, in pixels, for cleaning up FG image.

	flexitype maxVal, minVal;

	/*
	 * General Parameters
	 */
	size_t		imWidth;		//!< width of image.
	size_t		imHeight;		//!< height of image.
	size_t		numPixels;

	int					imageDepth;  //!< Depth of image, e.g. CV_8U
	unsigned int		numChannels; //!< Number of channels in image.

	bool	isDataInitialized;
	//!< After general parameters are set, data structures must be initialized.

	size_t	elemSize;  //!< store image mat element sizes
	size_t	elemSize1;

	/*
	 * Data Structures
	 */
	vector<PixelModelGMG>		pixels; //!< Probabilistic background models for each pixel in image.
	int							frameNum; //!< Frame number counter, used to count frames in training mode.
	Mat							posteriorImage;  //!< Posterior probability image.
	Mat							fgMaskImage;   //!< Foreground mask image.

};

}

#endif /* BACKGROUNDSUBTRACTORGMG_HPP_ */
