/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */


#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

int block_estimation = 0;
int min_estimation = 0;
int max_estimation = 0;
int num_estimation_real = 0;
int num_estimation = 0;
int p1_estimation = 0;
int p2_estimation = 0;
Mat disp, disp8;

static void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|var] [--blocksize=<block_size>]\n"
			"[--max-disparity=<max_disparity>] --min-disparity=<min_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
			"[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}

static void showXYZ(const Mat& mat, const Mat& rgb, const string& outfile="")
{
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	cloud.reserve(mat.cols*mat.rows);

	const double max_z = 2e3; // Disregard points farther away than 2 m
	//const double max_z = 1.24e3; // Disregard points farther away than 2 m
	//FILE* fp = fopen(filename, "wt");
	for(int y = 0; y < mat.rows; y++)
	{
		for(int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);

			// This omits zero points
			if(point[0] == 0 && point[1] == 0 && point[2] == 0)
				continue;

			// This omits points equal to or larger than max_z
			if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
				continue;
			//fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);

			// Point to write to
			pcl::PointXYZRGB p;

			// Scale position from mm to m
			p.x = 0.001*point[0];
			p.y = 0.001*point[1];
			p.z = 0.001*point[2];

			// OpenCV reads in images in BGR order, so we must switch to BGR for PCL
			Vec3b pbgr = rgb.at<Vec3b>(y,x);
			p.b = pbgr[0];
			p.g = pbgr[1];
			p.r = pbgr[2];

			cloud.push_back(p);
		}
	}

	cout << "Showing " << cloud.size() << " points" << endl;
	// Show using PCL
	pcl::visualization::PCLVisualizer visu;
	visu.addPointCloud(cloud.makeShared());
	visu.addCoordinateSystem(0.25);
	visu.setBackgroundColor(0.5,0.5,0.5);
	visu.spin();

	if(!outfile.empty())
		pcl::io::savePCDFile(outfile, cloud);
}

int main(int argc, char** argv) {

	const char* algorithm_opt = "--algorithm=";
	const char* maxdisp_opt = "--max-disparity=";
	const char* mindisp_opt = "--min-disparity=";
	const char* blocksize_opt = "--blocksize=";
	const char* nodisplay_opt = "--no-display=";
	const char* scale_opt = "--scale=";

	if(argc < 3) {
		print_help();
		return 0;
	}
	const char* img1_filename = 0; //left image
	const char* img2_filename = 0; //right image
	const char* intrinsic_filename = 0; //intrinsic parameters
	const char* extrinsic_filename = 0; //extrinsic parameters
	const char* disparity_filename = 0;
	const char* point_cloud_filename = 0;

	int iterator=0;

	 // Create a window
	namedWindow("Disparity", WINDOW_NORMAL);
	resizeWindow("Disparity", 900,900);

	 //Create trackbar to change block size
	int block_slider = 3;
	createTrackbar("Block size", "Disparity", &block_slider, 50);

	 //Create trackbar to change minimum disparity
	int min_slider = 0;
	createTrackbar("Min disparity", "Disparity", &min_slider, 1000);

	 //Create trackbar to change maximum disparity
	int max_slider = 199;
	createTrackbar("Max disparity", "Disparity", &max_slider, 1000);

	 //Create trackbar to change the P1 parameter
	int p1_slider = 120;
	createTrackbar("P1", "Disparity", &p1_slider, 1000);

	 //Create trackbar to change the P1 parameter
	int p2_slider = 500;
	createTrackbar("P2", "Disparity", &p2_slider, 1000);

	enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
	int alg = STEREO_SGBM;
	int SADWindowSize = 0, numberOfDisparities = 0, min_Disparities = 0;
	bool no_display = false;
	float scale = 1.f;

	StereoSGBM sgbm;

	//read commandline options
	for( int i = 1; i < argc; i++ )
	{
		if( argv[i][0] != '-' )
		{
			if( !img1_filename )
				img1_filename = argv[i];
			else
				img2_filename = argv[i];
		}
		else if( strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0 )
		{
			if( sscanf( argv[i] + strlen(maxdisp_opt), "%d", &numberOfDisparities ) != 1 ||
					numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
			{
				printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
				print_help();
				return -1;
			}
		}
		else if( strncmp(argv[i], mindisp_opt, strlen(mindisp_opt)) == 0 )
		{
			if( sscanf( argv[i] + strlen(mindisp_opt), "%d", &min_Disparities ) != 1 )
			{
				printf("Command-line parameter error: The min disparity\n");
				print_help();
				return -1;
			}
		}
		else if( strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0 )
		{
			if( sscanf( argv[i] + strlen(blocksize_opt), "%d", &SADWindowSize ) != 1 ||
					SADWindowSize < 1 || SADWindowSize % 2 != 1 )
			{
				printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
				return -1;
			}
		}
		else if( strncmp(argv[i], scale_opt, strlen(scale_opt)) == 0 )
		{
			if( sscanf( argv[i] + strlen(scale_opt), "%f", &scale ) != 1 || scale < 0 )
			{
				printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
				return -1;
			}
		}
		else if( strcmp(argv[i], nodisplay_opt) == 0 )
			no_display = true;
		else if( strcmp(argv[i], "-i" ) == 0 )
			intrinsic_filename = argv[++i];
		else if( strcmp(argv[i], "-e" ) == 0 )
			extrinsic_filename = argv[++i];
		else if( strcmp(argv[i], "-o" ) == 0 )
			disparity_filename = argv[++i];
		else if( strcmp(argv[i], "-p" ) == 0 )
			point_cloud_filename = argv[++i];
		else
		{
			printf("Command-line parameter error: unknown option %s\n", argv[i]);
			return -1;
		}
	}

	if( !img1_filename || !img2_filename )
	{
		printf("Command-line parameter error: both left and right images must be specified\n");
		return -1;
	}

	if( (intrinsic_filename != 0) ^ (extrinsic_filename != 0) )
	{
		printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
		return -1;
	}

	if( extrinsic_filename == 0 && point_cloud_filename )
	{
		printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
		return -1;
	}

	int color_mode = alg == STEREO_BM ? 0 : -1;
	Mat img1 = imread(img1_filename, color_mode);
	Mat img2 = imread(img2_filename, color_mode);

	Mat img_colored = imread(img1_filename, -1);

	if( scale != 1.f ) {
		Mat temp1, temp2;
		int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
		resize(img1, temp1, Size(), scale, scale, method);
		img1 = temp1;
		resize(img2, temp2, Size(), scale, scale, method);
		img2 = temp2;
	}

	Size img_size = img1.size();

	Rect roi1, roi2;
	Mat Q;

	Mat R, T, R1, P1, R2, P2;

	if( intrinsic_filename )
	{
		// reading intrinsic parameters
		FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
		if(!fs.isOpened())
		{
			printf("Failed to open file %s\n", intrinsic_filename);
			return -1;
		}

		Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		M1 *= scale;
		M2 *= scale;

		fs.open(extrinsic_filename, CV_STORAGE_READ);
		if(!fs.isOpened())
		{
			printf("Failed to open file %s\n", extrinsic_filename);
			return -1;
		}

		//Mat R, T, R1, P1, R2, P2;
		fs["R"] >> R;
		fs["T"] >> T;

		//stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );
		stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, 0, 0, img_size, &roi1, &roi2 );

		if(iterator<1)
		{
			std::cout<<"R1 matrix:\n "<<R1<<endl<<endl;
			std::cout<<"R2 matrix:\n "<<R2<<endl<<endl;
			std::cout<<"P1 matrix:\n "<<P1<<endl<<endl;
			std::cout<<"P2 matrix:\n "<<P2<<endl<<endl;
			std::cout<<"P1 (0 3): "<<P1.at<double>(0,2)<<endl;
		}


		Mat map11, map12, map21, map22;
		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

		Mat img1r, img2r, img_cr;

		remap(img1, img1r, map11, map12, INTER_CUBIC);
		remap(img2, img2r, map21, map22, INTER_CUBIC);

		remap(img_colored, img_cr, map11, map12, INTER_CUBIC);

		img1 = img1r;
		img2 = img2r;
		img_colored = img_cr;

		if(iterator<1)
		{
			imwrite("left_rect.png",img1);
			imwrite("right_rect.png",img2);
		}
	}

	// Start of trackbar functionality
	while(true)
	{
		numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

		if(iterator<1)
		{
			sgbm.preFilterCap = 63;
			sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
			int cn = img1.channels();
			sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
			sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
			sgbm.minDisparity = min_Disparities;
			sgbm.numberOfDisparities = numberOfDisparities;
			sgbm.uniquenessRatio = 0;
			sgbm.speckleWindowSize = 0;
			sgbm.speckleRange = 0;
			sgbm.disp12MaxDiff = -1;
			sgbm.fullDP = alg == STEREO_HH;
		}
		else
		{
			block_estimation=getTrackbarPos("Block size", "Disparity");
			if(block_estimation<1)
				block_estimation=1;
			min_estimation = getTrackbarPos("Min disparity", "Disparity");
			max_estimation = getTrackbarPos("Max disparity", "Disparity");
			num_estimation_real = max_estimation-min_estimation;
			num_estimation = num_estimation_real-(num_estimation_real%16);
			p1_estimation = getTrackbarPos("P1", "Disparity");
			p2_estimation = getTrackbarPos("P2", "Disparity");
			if(max_estimation<16)
				max_estimation=16;
			sgbm.preFilterCap = 63;
			sgbm.SADWindowSize = block_estimation;
			int cn = img1.channels();
			sgbm.P1 = p1_estimation;
			sgbm.P2 = p2_estimation;
			sgbm.minDisparity = min_estimation;
			sgbm.numberOfDisparities = num_estimation;
			sgbm.uniquenessRatio = 10;  //5-15
			sgbm.speckleWindowSize = 100; //50-200
			sgbm.speckleRange = 1; //1-2
			sgbm.disp12MaxDiff = -1;
			sgbm.fullDP = alg == STEREO_HH;
		}

		int64 t = getTickCount();

		sgbm(img1, img2, disp);
		t = getTickCount() - t;

		disp.convertTo(disp8, CV_8U);
		//Show result
		imshow("Disparity", disp);

		iterator+=1;

		// Wait until user press some key for 50ms
		int iKey = waitKey(50);
		// If user press 'ESC' key
		if (iKey == 1048603) // 27 = ENG keyboard;  1048603 = DANISH keyboard
			{
				cout<<"Program finished!"<<endl;
				printf("Time elapsed for stereo matching: %fms\n", t*1000/getTickFrequency());
				cout<<"The best found parameters are:"<<endl;
				cout<<"-> Block size = "<<block_estimation<<endl;
				cout<<"-> Minimum disparity = "<<min_estimation<<endl;
				cout<<"-> Number of disparities = "<<num_estimation<<endl;
				cout<<"-> P1 = "<<p1_estimation<<endl;
				cout<<"-> P2 = "<<p2_estimation<<endl;
				break;
			}

	}

	if(disparity_filename)
		imwrite(disparity_filename, disp8);

	if(point_cloud_filename) {
		printf("storing the point cloud...");
		fflush(stdout);

		if(iterator<1)
		{
			std::cout<<"Q reproject matrix:\n "<<Q<<endl;
			std::cout<<"Q (0 3): "<<Q.at<double>(0,3)<<endl;
		}

		// Get parameters for reconstruction
		float f = P1.at<double>(0,0); // Focal length
		float B = P2.at<double>(0,3)/f; // Baseline in the x direction
		float cx = P1.at<double>(0,2); // Center x coordinate
		float cy = P1.at<double>(1,2); // Center y coordinate

		float cx2 = P2.at<double>(0,2); // Center x coordinate of right image
		float dcx = cx-cx2; // Difference in center x coordinates
		int temp = disp.at<int16_t>(0,0);
		int maxdisp = 0;
		for(int y = 0; y < disp.rows; ++y) {
			for(int x = 0; x<disp.cols; ++x) {
				if(temp > disp.at<int16_t>(y,x))
					temp = disp.at<int16_t>(y,x);
				if(maxdisp < disp.at<int16_t>(y,x))
					maxdisp = disp.at<int16_t>(y,x);
			}
		}

		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr out (new pcl::PointCloud<pcl::PointXYZRGBA>());
		out->height = disp.cols;
		out->width = disp.rows;
		out->points.resize(out->height * out->width);


		for (int i = 0; i < out->size(); i++){
			(*out)[i].x = std::numeric_limits<float>::quiet_NaN();
			(*out)[i].y = std::numeric_limits<float>::quiet_NaN();
			(*out)[i].z = std::numeric_limits<float>::quiet_NaN();
		}

		Mat_<Vec3f> xyz(disp.rows, disp.cols, Vec3f(0,0,0)); // Resulting point cloud, initialized to zero
		for(int y = 0; y < disp.rows; ++y) {
			for(int x = 0; x < disp.cols; ++x) {
				pcl::PointXYZRGBA point;

				// Avoid invalid disparities
				if(disp.at<int16_t>(y,x) == temp) continue;
				if(disp.at<int16_t>(y,x) == 0) continue;

				float d = float(disp.at<int16_t>(y,x)) / 16.0f; // Disparity
				float W = B/(-d+dcx); // Weighting

				point.x = (float(x)-cx) * W;
				point.y = (float(y)-cy) * W;
				point.z = f * W;
				//skip 0 points
				if (point.x== 0 && point.y == 0 && point.z == 0) continue;
				// disregard points farther then 2m
				const double max_z = 2e3;
				if (fabs(point.y - max_z) < FLT_EPSILON || fabs(point.y) > max_z) continue;
				//scale position from mm to m
				point.x = 0.001*point.x;
				point.y = 0.001*point.y;
				point.z = 0.001*point.z;
				//add color
				Vec3b bgr = img_colored.at<Vec3b>(y,x);
				point.b = bgr[0];
				point.g = bgr[1];
				point.r = bgr[2];

				out->at(y, x) = point;
			}
		}
		pcl::io::savePCDFile("stereo_out.pcd", *out);

		printf("\n");
	}

	return 0;
}
