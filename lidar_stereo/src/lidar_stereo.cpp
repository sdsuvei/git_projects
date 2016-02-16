#include "lidar_stereo.h"

int main (int argc, char** argv)
{

	//const char* algorithm_opt = "--algorithm=";
	//const char* maxdisp_opt = "--max-disparity=";
	//const char* mindisp_opt = "--min-disparity=";
	//const char* blocksize_opt = "--blocksize=";
	//const char* nodisplay_opt = "--no-display=";
	//const char* scale_opt = "--scale=";

	if(argc < 3) {
		print_help();
		return 0;
	}

	const char* img1_filename = 0; //left image
	const char* img2_filename = 0; //right image
	const char* intrinsic_filename = 0; //intrinsic parameters
	const char* extrinsic_filename = 0; //extrinsic parameters
	const char* disparity_filename1 = 0;
	const char* disparity_filename2 = 0;
	const char* experiment_filename_1 = 0;
	const char* experiment_filename_2 = 0;
	const char* point_cloud_filename = 0;

	// Elementes for the filling option -- fill the empty disparity pixels with LIDAR information
	const char* improvement = "--improvement=";
	enum {NO_FILL = 0, FILL = 1};
	int imp = 0;

	// Elementes for the threading option
	const char* thread = "--thread=";
	enum {SINGLE = 0, MULTI = 1};
	int thr = 1;

	for( int i = 1; i < argc; i++ )
	{
		if( argv[i][0] != '-' )
		{
			if( !img1_filename )
				img1_filename = argv[i];
			else
				img2_filename = argv[i];
		}
		else if( strncmp(argv[i], improvement, strlen(improvement)) == 0 )
		{
			char* _imp = argv[i] + strlen(improvement);
			 imp = strcmp(_imp, "no_fill") == 0 ? NO_FILL :
					strcmp(_imp, "fill") == 0 ? FILL : -1;
			if( imp < 0 )
			{
				printf("Command-line parameter error: Unknown improvement option\n\n");
				print_help();
				return -1;
			}
		}
		else if(strncmp(argv[i], thread, strlen(thread)) == 0 )
		{
			char* _thr = argv[i] + strlen(thread);
			 thr = strcmp(_thr, "single") == 0 ? SINGLE :
					strcmp(_thr, "multi") == 0 ? MULTI : -1;
			if( thr < 0 )
			{
				printf("Command-line parameter error: Unknown thread option\n\n");
				print_help();
				return -1;
			}
		}
	}

	int option = 0;

	cout<<"Choose test platform: "<<endl;
	cout<<"1. SAFE"<<endl;
	cout<<"2. Care-O-Bot"<<endl;
	cout<<"3. ACAT "<<endl;
	cout<<"Option: ";
	cin>>option;

	switch(option){
	case 1:
		intrinsic_filename = "stereo_parameters/stereo_parameters_SAFE/int.yml";
		extrinsic_filename = "stereo_parameters/stereo_parameters_SAFE/ent.yml";
		disparity_filename1 = "results/SAFE/DISP1.png";
		disparity_filename2 = "results/SAFE/DISP2.png";
		point_cloud_filename = "pointclouds/pointclouds_SAFE/stereo_out.pcd";
		break;
	case 2:
		intrinsic_filename = "stereo_parameters/stereo_parameters_COB/int.yml";
		extrinsic_filename = "stereo_parameters/stereo_parameters_COB/ent.yml";
		disparity_filename1 = "results/COB/DISP1.png";
		disparity_filename2 = "results/COB/DISP2.png";
		point_cloud_filename = "pointclouds/pointclouds_COB/stereo_out.pcd";
		break;
	case 3:
		intrinsic_filename = "stereo_parameters/stereo_parameters_ACAT/int.yml";
		extrinsic_filename = "stereo_parameters/stereo_parameters_ACAT/ent.yml";
		disparity_filename1 = "results/ACAT/DISP1.png";
		disparity_filename2 = "results/ACAT/DISP2.png";
		point_cloud_filename = "pointclouds/pointclouds_ACAT/stereo_out.pcd";
		break;
	default:
		cout<<"Incorect option!"<<endl;
		exit(0);
	}

	if( !img1_filename || !img2_filename )
	{
		printf("Command-line parameter error: both left and right images must be specified\n");
		return -1;
	}

	// Select threading option -- single or multi
	if(thr==SINGLE)
	{
		cv::setNumThreads(0);
		cout<<"Multithreading OFF"<<endl;
	}
	else
	{
		cv::setNumThreads(1);
		cout<<"Multithreading ON"<<endl;
	}

	int iterator=0;

	enum { STEREO_BM=0 };
	int alg = STEREO_BM;
	int SADWindowSize = 0, numberOfDisparities = 0, min_Disparities = 0;
	bool no_display = false;
	float scale = 1.f; // Original is 1.f;

	cv::Mat img_colored = cv::imread(img1_filename, -1);

	int color_mode = alg == STEREO_BM ? 0 : -1;
	cv::Mat img1 = cv::imread(img1_filename, color_mode);
	cv::Mat img2 = cv::imread(img2_filename, color_mode);

	//img1 = img1( cv::Rect(0,215,1917,830) );
	//img2 = img2( cv::Rect(0,215,1917,830) );

	//imshow("ROI",rol_1);cv::waitKey();

	if( scale != 1.f )
	{
		cv::Mat temp1, temp2;
		int method = scale < 1 ? cv::INTER_AREA : cv::INTER_CUBIC;
		cv::resize(img1, temp1, cv::Size(), scale, scale, method);
		img1 = temp1;
		cv::imwrite("image_color.png",img1);
		cv::resize(img2, temp2, cv::Size(), scale, scale, method);
		img2 = temp2;
	}

	//cv::Mat img_colored = cv::imread("image_color.png", -1);

	cv::Size img_size = img1.size();

	cv::Rect roi1, roi2;
	cv::Mat Q;

	cv::Mat R, T, R1, P1, R2, P2;

	if( intrinsic_filename )
	{
		// reading intrinsic parameters
		cv::FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
		if(!fs.isOpened())
			{
				printf("Failed to open file %s\n", intrinsic_filename);
				return -1;
			}

		cv::Mat M1, D1, M2, D2;
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

		fs["R"] >> R;
		fs["T"] >> T;

		stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, 0, 0, img_size, &roi1, &roi2 );

		cv::Mat map11, map12, map21, map22;
		cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

		cv::Mat img1r, img2r, img_cr;

		cv::remap(img1, img1r, map11, map12, cv::INTER_CUBIC);
		cv::remap(img2, img2r, map21, map22, cv::INTER_CUBIC);

		cv::remap(img_colored, img_cr, map11, map12, cv::INTER_CUBIC);

		img1 = img1r;
		img2 = img2r;
		img_colored = img_cr;

		switch(option){
		case 1:
			cv::imwrite("results/SAFE/left_rect.png",img1);
			cv::imwrite("results/SAFE/right_rect.png",img2);
			break;
		case 2:
			cv::imwrite("results/COB/left_rect.png",img1);
			cv::imwrite("results/COB/right_rect.png",img2);
			break;
		case 3:
			cv::imwrite("results/ACAT/left_rect.png",img1);
			cv::imwrite("results/ACAT/right_rect.png",img2);
			break;
		default:
			cout<<"Wrong option!"<<endl;
			break;
		}
		//cout<<"R1:"<<endl<<R1<<endl;
	}

	LEFT = img1;
	RIGHT = img2;

	 //// 1) READ THE LIDAR POINT CLOUD AND EXTRACT THE POINTS
	pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	//// Loading the point clouds

	switch(option){
	case 1:
		if (pcl::io::loadPCDFile<pcl::PointXYZ> ("pointclouds/pointclouds_SAFE/lidar.pcd", *lidar_cloud) == -1) //* load the file
			{
				PCL_ERROR ("Couldn't read the LIDAR point cloud! \n");
				return (-1);
			}
		break;
	case 2:
		if (pcl::io::loadPCDFile<pcl::PointXYZ> ("pointclouds/pointclouds_COB/carmine_cob.pcd", *lidar_cloud) == -1) //* load the file
			{
				PCL_ERROR ("Couldn't read the Carmine point cloud. \n");
				return (-1);
			}
		break;
	case 3:
		if (pcl::io::loadPCDFile<pcl::PointXYZ> ("pointclouds/pointclouds_ACAT/carmine_acat.pcd", *lidar_cloud) == -1) //* load the file
			{
				PCL_ERROR ("Couldn't read the Carmine point cloud. \n");
				return (-1);
			}
		break;
	default:
		cout<<"Wrong option!"<<endl;
		break;
	}

	Eigen::Matrix4f depthTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f depthTs_R = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_lTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_lTs_R = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_cobTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_cobTs_R = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_cTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_cTs_R = Eigen::Matrix4f::Identity();

	switch(option){
	case 1:
		//// The Transform Matrix which moves the lidar points in the stereo camera frame
		//// This matrix comes from the ICP algorithm appliead before-hand
		transform_lTs << -0.0298508, 0.999563, -0.00252324, 0.752321,
				0.242289, 0.00478399, -0.970206, 0.128226,
				-0.969755, -0.0295702, -0.242321, 2.05342,
				0, 0, 0, 1;

		//// This is the R matrix from the calibration process - it will overlap the LIDAR onto the STEREO view
		transform_lTs_R << 0.9999898012149877, -0.004498446772515268, -0.0004017992584439221, 0,
				0.004497161244118079, 0.9999849421569889, -0.003144996029905281, 0,
				0.0004159408054540794, 0.00314315699871394, 0.9999949737660324, 0,
				0, 0, 0, 1;

		depthTs = transform_lTs;
		depthTs_R = transform_lTs_R;
		break;
	case 2:
		//// COB carmine data

		transform_cobTs <<  1.00001, -0.00124211, 0.000743923, 0.0322781,
				0.00124286, 0.999932, -0.0126477, -0.0188326,
				-0.00073585, 0.0126586, 0.999934, -0.0153805,
				 0, 0, 0, 1;

		transform_cobTs_R << 0.9999067228415842, 0.003405266333134126, 0.01322685818339805, 0,
				-0.003407517169981614, 0.9999941834978157, 0.0001476389963970085, 0,
				-0.01322627849924465, -0.0001926959714158296, 0.999912510385445, 0,
				0, 0, 0, 1;

		depthTs = transform_cobTs;
		depthTs_R  = transform_cobTs_R;
		break;
	case 3:
		//// ACAT carmine data
	/*	transform_cTs <<  0.998995, -0.0210776, -0.0400272, 0.0354567,
				 0.0206785, 0.999749, -0.0103799, 0.0811172,
				 0.0402364, 0.00953686, 0.999161, 0.0228925,
				0, 0, 0, 1; */

		transform_cTs << 0.999408, -0.00767449, -0.0345354, 0.0454886,
		 0.00751068, 0.999998, -0.00488497, 0.0786888,
		  0.0345857, 0.00461929, 0.999417, 0.0503483,
		          0, 0, 0, 1;


		transform_cTs_R << 0.999813317139531, -0.0167718601851281, 0.00959351741432884, 0,
				0.0165824843680798, 0.9996725239970849, 0.0194901507813638, 0,
				-0.00991726185148418, -0.019327427951707, 0.999764021380116, 0,
				0, 0, 0, 1;

		depthTs = transform_cTs;
		depthTs_R = transform_cTs_R;
		break;
	default:
		cout<<"Wrong option!"<<endl;
		break;
	}

	//// Executing the transformation
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud2 (new pcl::PointCloud<pcl::PointXYZ> ());
	//// You can either apply transform_1 or transform_2; they are the same
	pcl::transformPointCloud (*lidar_cloud, *transformed_cloud, depthTs );
	pcl::transformPointCloud (*transformed_cloud, *transformed_cloud2, depthTs_R);

	pcl::PointCloud<pcl::PointXYZ> Lcloud=*transformed_cloud2;
	pcl::PointCloud<pcl::PointXYZ> saved_cloud;
	saved_cloud.height=Lcloud.height;
	saved_cloud.width=Lcloud.width;

	for(int s=0;s<saved_cloud.points.size(); ++s) {
		saved_cloud.points[s].x=0;
		saved_cloud.points[s].y=0;
		saved_cloud.points[s].z=0;
	}

	//// Create empty disparity map for Lidar data
	cv::Mat lidar_l=cv::Mat::zeros(img1.rows, img1.cols, CV_16S);

	float f = P1.at<double>(0,0); // Focal length
	float B = P2.at<double>(0,3)/f; // Baseline in the x direction
	float cx = P1.at<double>(0,2); // Center x coordinate
	float cy = P1.at<double>(1,2); // Center y coordinate

	float cx2 = P2.at<double>(0,2); // Center x coordinate of right image
	float cy2 = P2.at<double>(1,2); // Center y coordinate of right image
	float dcx = cx-cx2;

	for (int i = 0; i < Lcloud.points.size (); ++i)  //try with cloud height and width
	{
		//// Cross reference the 2D points
		double x_=806.6065420854103*Lcloud.points[i].x+982.1147003173828*Lcloud.points[i].z;
		double y_=806.6065420854103*Lcloud.points[i].y+515.74658203125*Lcloud.points[i].z;
		double z_=Lcloud.points[i].z;

		//// If x_2d and y_2d match x and y (from below), than the tranformation was correctly performed
		int x_2d=x_/z_;
		int y_2d=y_/z_;

		//// Scale position from m to mm;
		Lcloud.points[i].x = Lcloud.points[i].x/0.001;
		Lcloud.points[i].y = Lcloud.points[i].y/0.001;
		Lcloud.points[i].z = Lcloud.points[i].z/0.001;

		float d = (Lcloud.points[i].z*dcx-f*B)/Lcloud.points[i].z; // disparity
		float W = B/(-d+dcx); // Weighting
		int x = (Lcloud.points[i].x+cx*W)/W; // x value
		int y = (Lcloud.points[i].y+cy*W)/W; // y value

		//// Filter out all LIDAR points which are outside the camera view
		if(y>=0 && y<lidar_l.rows && x>=0 && x<lidar_l.cols)
			{
				lidar_l.at<int16_t>(y,x)=d;
				//std::cout<<"x: "<<x<<" y: "<<y<<" d: "<<d<<endl;
				saved_cloud.points.push_back(Lcloud.points[i]);
			}
	}

	saved_cloud.width = 1;
	saved_cloud.height = saved_cloud.points.size();

	switch(option){
	case 1:
		pcl::io::savePCDFileASCII ("pointclouds/pointclouds_SAFE/saved_lidar.pcd", saved_cloud);
		cv::imwrite("results/SAFE/lidar_disp.png",lidar_l);
		break;
	case 2:
		pcl::io::savePCDFileASCII ("pointclouds/pointclouds_COB/saved_carmine.pcd", saved_cloud);
		cv::imwrite("results/COB/carmine_disp.png",lidar_l);
		break;
	case 3:
		pcl::io::savePCDFileASCII ("pointclouds/pointclouds_ACAT/saved_carmine.pcd", saved_cloud);
		cv::imwrite("results/ACAT/carmine_disp.png",lidar_l);
		break;
	default:
		cout<<"Wrong option!"<<endl;
		break;
	}

	//// 2) USE THE LIDAR POINTS TO INFLUENCE THE STEREO_BM METHOD
	//// ACCESS THE DISPARITY POINTS
	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

	// Initialize global LIDAR matrix
	cv::Mat lidar_DISP=cv::Mat::ones(img1.rows, img1.cols, CV_16S);

	cv::Mat src = lidar_l; // cv::imread("lidar_disp.png");
	cv::Mat dst = cv::Mat::zeros(img1.rows, img1.cols, CV_16S);

	cv::Mat kernel = cv::Mat::ones(3,3,CV_16S);

	switch(option){
	case 1:
		cv::dilate(src,dst, kernel, cv::Point(-1, -1), 11, 1, 1); // 11 for SAFE!!
		// Apply the specified morphology operation
		cv::imwrite("results/SAFE/dilated_lidar.png",dst);
		DISP = dst;
		break;
	case 2:
		cv::dilate(src,dst, kernel, cv::Point(-1, -1), 2, 1, 1); // 2 for COB!!
		// Apply the specified morphology operation
		//morphologyEx( src, dst, MORPH_TOPHAT, element ); // here iteration=1
		cv::imwrite("results/COB/dilated_carmine.png",dst);
		DISP = dst;
		break;
	case 3:
		cv::dilate(src,dst, kernel, cv::Point(-1, -1), 1, 1, 1); // 1 for ACAT!!
		// Apply the specified morphology operation
		//morphologyEx( src, dst, MORPH_TOPHAT, element ); // here iteration=1
		cv::imwrite("results/ACAT/dilated_carmine.png",dst);
		DISP = dst;
		break;
	default:
		cout<<"Wrong option!"<<endl;
		break;
	}

	/*
	//// Normalize the disparity map (for viewing)
	cv::Mat temp;
	cv::normalize(DISP,temp,0,255,cv::NORM_MINMAX,CV_8U);
	imshow("disp",temp);cv::waitKey(); */

	cv::StereoBM bm;

	bm.state->roi1 = roi1;
	bm.state->roi2 = roi2;
	bm.state->preFilterCap = 63;
	bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
	bm.state->minDisparity = min_Disparities;
	bm.state->numberOfDisparities = numberOfDisparities;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;  //5-15
	bm.state->speckleWindowSize = 100; //50-200
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 1; // positive
	int cn = img1.channels();
	INTERVAL = 60; // changes the search range inside the BM method

	struct timeval start, end;
	long mtime, seconds, useconds;
	gettimeofday(&start, NULL);

	bm(img1, img2, disp); // <-----------------------------------

	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	printf("Elapsed time for BM: %f seconds\n", (float)mtime/1000);

	disp = disp/16; //because it is multiplied by 16 inside the BM method!!

	if(disparity_filename1)
		imwrite(disparity_filename1, disp);


	//// Infill the disparity
	////  Fill in the -1 pixels with Lidar points
	if(imp==FILL){
		cout<<"Filling disparity with LIDAR information..."<<endl;
		for(int w = 0; w < disp.rows; ++w) {
					for(int v = 0; v < disp.cols; ++v) {
						if(disp.at<int16_t>(w,v)==-1 && DISP.at<int16_t>(w,v)>1)
							{
								disp.at<int16_t>(w,v) = DISP.at<int16_t>(w,v);
							}
						}
					}

				if(disparity_filename2)
					imwrite(disparity_filename2, disp);
	}

	if(point_cloud_filename) {
		printf("Storing the point cloud...");
		cout<<endl;
		fflush(stdout);
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
		out->height = disp.rows;
		out->width = disp.cols;
		out->points.resize(out->height * out->width);


		for (int i = 0; i < out->size(); i++){
			(*out)[i].x = std::numeric_limits<float>::quiet_NaN();
			(*out)[i].y = std::numeric_limits<float>::quiet_NaN();
			(*out)[i].z = std::numeric_limits<float>::quiet_NaN();
		}

		cv::Mat_<cv::Vec3f> xyz(disp.rows, disp.cols, cv::Vec3f(0,0,0)); // Resulting point cloud, initialized to zero
		for(int y = 0; y < disp.rows; ++y) {
			for(int x = 0; x < disp.cols; ++x) {
				pcl::PointXYZRGBA point;

				// Avoid invalid disparities
				if(disp.at<int16_t>(y,x) == temp) continue;
				if(disp.at<int16_t>(y,x) == 0) continue;

				float d = float(disp.at<int16_t>(y,x)); // Disparity
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
				cv::Vec3b bgr = img_colored.at<cv::Vec3b>(y,x);
				point.b = bgr[0];
				point.g = bgr[1];
				point.r = bgr[2];

				out->at(x,y) = point;
			}
		}
		pcl::io::savePCDFile(point_cloud_filename, *out);

		//saveXYZ(point_cloud_filename, xyz);
		//showXYZ(xyz, img_colored, point_cloud_filename);
		printf("\n");
	}

	printf("PROGRAM DONE!! \n");
	return 0;
}
