#include "safe_node.h"

using namespace cv;
using namespace std;

cv::Mat disp, DISP;
int INTERVAL = 50;
std::vector<double> gauss;
double gaussarr[3000];
double std_d = 300;

int main(int argc, char** argv) {

	cv::setNumThreads(0);

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
	const char* img1_filename = 0;
	const char* img2_filename = 0;
	const char* intrinsic_filename = 0;
	const char* extrinsic_filename = 0;
	const char* disparity_filename1 = 0;
	const char* disparity_filename2 = 0;
	const char* point_cloud_filename = 0;

	enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2};
	int alg = STEREO_SGBM;
	int SADWindowSize = 0, numberOfDisparities = 0, min_Disparities = 0;
	bool no_display = false;
	float scale = 1.f;



	StereoBM bm;
	StereoSGBM sgbm;
	StereoVar var;

	for( int i = 1; i < argc; i++ )
	{
		if( argv[i][0] != '-' )
		{
			if( !img1_filename )
				img1_filename = argv[i];
			else
				img2_filename = argv[i];
		}
		else if( strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0 )
		{
			char* _alg = argv[i] + strlen(algorithm_opt);
			alg = strcmp(_alg, "bm") == 0 ? STEREO_BM :
					strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
							strcmp(_alg, "hh") == 0 ? STEREO_HH : -1;
			if( alg < 0 )
			{
				printf("Command-line parameter error: Unknown stereo algorithm\n\n");
				print_help();
				return -1;
			}
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
		/*
		else if( strcmp(argv[i], "-i" ) == 0 )
			intrinsic_filename = argv[++i];
		else if( strcmp(argv[i], "-e" ) == 0 )
			extrinsic_filename = argv[++i];
		else if( strcmp(argv[i], "-o" ) == 0 )
			disparity_filename1 = argv[++i];
		else if( strcmp(argv[i], "-p" ) == 0 )
			point_cloud_filename = argv[++i]; */
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

		std::cout<<"R1 matrix:\n "<<R1<<endl<<endl;
		std::cout<<"R2 matrix:\n "<<R2<<endl<<endl;
		std::cout<<"P1 matrix:\n "<<P1<<endl<<endl;
		std::cout<<"P2 matrix:\n "<<P2<<endl<<endl;

		std::cout<<"P1 (0 3): "<<P1.at<double>(0,2)<<endl;

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
	}

	 //// 1) READ THE LIDAR POINT CLOUD AND EXTRACT THE POINTS
	pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud_org (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud_x (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	//// Loading the point clouds

	switch(option){
	case 1:
		if (pcl::io::loadPCDFile<pcl::PointXYZ> ("pointclouds/pointclouds_SAFE/lidar_440.pcd", *lidar_cloud_org) == -1) //* load the file
			{
				PCL_ERROR ("Couldn't read the LIDAR point cloud! \n");
				return (-1);
			}
		break;
	case 2:
		if (pcl::io::loadPCDFile<pcl::PointXYZ> ("pointclouds/pointclouds_COB/carmine_cob.pcd", *lidar_cloud_org) == -1) //* load the file
			{
				PCL_ERROR ("Couldn't read the Carmine point cloud. \n");
				return (-1);
			}
		break;
	case 3:
		if (pcl::io::loadPCDFile<pcl::PointXYZ> ("pointclouds/pointclouds_ACAT/carmine_acat.pcd", *lidar_cloud_org) == -1) //* load the file
			{
				PCL_ERROR ("Couldn't read the Carmine point cloud. \n");
				return (-1);
			}
		break;
	default:
		cout<<"Wrong option!"<<endl;
		break;
	}

	// Crop the unnecessary points from the LIDAR cloud
	// Create the filtering object
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud (lidar_cloud_org);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (-50,0);
	//pass.setFilterLimitsNegative (true);
	pass.filter (*lidar_cloud_x);

	pcl::io::savePCDFile("lidar_cloud_x.pcd", *lidar_cloud_x);


	pass.setInputCloud (lidar_cloud_x);
	pass.setFilterFieldName ("y");
	pass.setFilterLimits (-13,10);
	//pass.setFilterLimitsNegative (true);
	pass.filter (*lidar_cloud);

	pcl::io::savePCDFile("lidar_cloud.pcd", *lidar_cloud);

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

	 transform_lTs << -0.021274, 0.99979, -0.00414833, 0.76413,
	    0.24543, 0.00120029, -0.969395, 0.13227,
	    -0.969198, -0.0216423, -0.245399, 2.0549,
	    0, 0, 0, 1;

		/* transform_lTs << -0.0298508, 0.999563, -0.00252324, 0.752321,
				0.242289, 0.00478399, -0.970206, 0.128226,
				-0.969755, -0.0295702, -0.242321, 2.05342,
				0, 0, 0, 1;  //<--- ORIGIANL WORKING ONE

		transform_lTs << 0.998698, 0.048944, 0.0156136, 0.426832,
					0.0254905, -0.208193, -0.977776, 0.16635,
					-0.04461, 0.976888, -0.209161, -1.43821,
					0, 0, 0, 1; */  //<- FOR LIDAR WALL TEST

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
		/*
		transform_cobTs <<  1.00001, -0.00124211, 0.000743923, 0.0322781,
				0.00124286, 0.999932, -0.0126477, -0.0188326,
				-0.00073585, 0.0126586, 0.999934, -0.0153805,
				 0, 0, 0, 1; */

		transform_cobTs <<  1.00001, -0.00124211, 0.000743923, 0.0322781,
				0.00124286, 0.999932, -0.0126477, -0.0188326,
				-0.00073585, 0.0126586, 0.999934, -0.0153805,
				 0, 0, 0, 1;  // <- FOR COB PLATE TEST

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

	// Initialize global LIDAR matrix
	cv::Mat lidar_DISP=cv::Mat::ones(img1.rows, img1.cols, CV_16S);

	////////////////// FLANN SEARCH OPTION /////////////////////////////////////////////////////////////
	/*
	 *
	 *
	 *
	//// Begin FLANN search
	cv::Mat lidar_points_mat = cv::Mat::zeros(2072641, 2, CV_16S);
	std::vector<cv::Point_<int16_t> > lidar_points;
	std::vector<int> lidar_point_idx_to_global_idx;
	cv::Mat stereo_points = cv::Mat::zeros(2072641, 2, CV_16S);

	int c_l = 0;
	int c_s = 0;

	for(int w = 0; w < lidar_l.rows; ++w) {
		for(int v = 0; v < lidar_l.cols; ++v) {
			if(lidar_l.at<int16_t>(w,v)!=0)
				{
					lidar_points_mat.at<int16_t>(c_l,0)=v;
					lidar_points_mat.at<int16_t>(c_l,1)=w;
					lidar_points.push_back(cv::Point_<int16_t>(v,w)); //Stupid CV points need (x,y), NOT (row,col)
					lidar_point_idx_to_global_idx.push_back(c_l);
					c_l+=1;
				}
		}
	}

	for(int w = 0; w < img1.rows; ++w) {
			for(int v = 0; v < img1.cols; ++v) {
			if(img1.at<int16_t>(w,v)!=0) //STEREO.at<int16_t>(w,v)!=0
				{
				stereo_points.at<int16_t>(c_s,0)=v; //cols alias x
				stereo_points.at<int16_t>(c_s,1)=w; //rows alias y
				c_s+=1;
				}
			}
		}

	//std::cout<<"c_l: "<<c_l<<" c_s: "<<c_s<<endl;

	//// Convert your 2D lidar points to FLANN matrix
	::flann::Matrix<int16_t> lidar_points_flann(reinterpret_cast<int16_t*>(&lidar_points[0]), lidar_points.size(), 2);
	// Create single k-d tree
	::flann::KDTreeSingleIndex< ::flann::L2_Simple<int16_t> > kdtree_flann(lidar_points_flann);    //<-------------------------------- DISTANCE FOR FLANN SEARCH
	kdtree_flann.buildIndex();

	//// Convert the 2D stereo points to FLANN
	::flann::Matrix<int16_t> stereo_points_flann(stereo_points.ptr<int16_t>(), stereo_points.rows, stereo_points.cols);
	//// Do search
	vector<vector<size_t> > indices_flann;
	vector<vector<float> > dists_flann;

    struct timeval start_f, end_f;
    long mtime_f, seconds_f, useconds_f;
    gettimeofday(&start_f, NULL);

	kdtree_flann.knnSearch(stereo_points_flann, indices_flann, dists_flann, 1, ::flann::SearchParams());

	gettimeofday(&end_f, NULL);
    seconds_f  = end_f.tv_sec  - start_f.tv_sec;
    useconds_f = end_f.tv_usec - start_f.tv_usec;
    mtime_f = ((seconds_f) * 1000 + useconds_f/1000.0) + 0.5;
    printf("Elapsed time for FLANN: %f seconds\n", (float)mtime_f/1000);


	//// Counters for keeping track of position within the STEREO left image and the LIDAR image
	int s_i, s_j = 0;
	int l_i, l_j = 0;

	for(int ot=0;ot<indices_flann.size();++ot){
		for(int in=0;in<indices_flann.at(ot).size();in++){
			//cout<< stereo_points.row(ot) <<" ("<<indices_flann.at(ot).at(in)<<","<<dists_flann.at(ot).at(in)<<") " << lidar_points.row(indices_flann.at(ot).at(in))<<endl;
			//cout<<"Stereo: "<< stereo_points.row(ot) <<" -> Lidar:"<< lidar_points.row(indices_flann.at(ot).at(in))<<endl;
			s_i=stereo_points.row(ot).at<int16_t>(0,0); //row
			s_j=stereo_points.row(ot).at<int16_t>(0,1); //cols

			l_i=lidar_points_mat.row( lidar_point_idx_to_global_idx.at(indices_flann.at(ot).at(in)) ).at<int16_t>(0,0);
			l_j=lidar_points_mat.row( lidar_point_idx_to_global_idx.at(indices_flann.at(ot).at(in)) ).at<int16_t>(0,1);

			//cout<<"s_i: "<< s_i<<"  s_j: "<< s_j<<"  l_i: "<< l_i<<"  l_j: "<< l_j<<endl;
			if(abs(s_i-l_i)<15 && abs(s_j-l_j)<15)
				{
					lidar_DISP.at<int16_t>(s_j,s_i) = lidar_l.at<int16_t>(l_j,l_i);
				}
			////assign to each pixel in the STEREO left image the disparity of the corresponding nearest point from the LIDAR image
			//lidar_DISP.at<int16_t>(s_j,s_i) = lidar_l.at<int16_t>(l_j,l_i);
		}
	}

	DISP = lidar_DISP;


	*
	*
	*
	*/


	//////////////////   DILATION OPTION ///////////////////////////////////////////////////////////////
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
		cv::dilate(src,dst, kernel, cv::Point(-1, -1), 1, 1, 1); // 2 for COB!!
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

	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;
	bm.state->roi1 = roi1;
	bm.state->roi2 = roi2;
	bm.state->preFilterCap = 63;
	bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
	bm.state->minDisparity = min_Disparities;
	bm.state->numberOfDisparities = numberOfDisparities;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 100;
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 1;

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

	for(int d_l = numberOfDisparities;d_l>=0;d_l--)
	{
		gaussarr[numberOfDisparities-d_l]=1./exp(-((numberOfDisparities-d_l)*(numberOfDisparities-d_l))/(2*std_d*std_d)); //1./
	}

	struct timeval start, end;
	long mtime, seconds, useconds;
	gettimeofday(&start, NULL);

	//int64 t = getTickCount();
	if( alg == STEREO_BM )
		bm(img1, img2, disp);

	else if( alg == STEREO_SGBM || alg == STEREO_HH )
		sgbm(img1, img2, disp);

	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	printf("Elapsed time: %f seconds\n", (float)mtime/1000);

	//t = getTickCount() - t;
	//printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

	disp = disp/16;

	//disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));

	if(disparity_filename1)
		imwrite(disparity_filename1, disp);

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

	//// Remove the tractor from the disparity map /point cloud
	for(int w = 138; w < 434; ++w) {
		for(int v = 0; v < 400; ++v) {
			disp.at<int16_t>(w,v) = -1;
		}
	}

	for(int w = 434; w < 917; ++w) {
		for(int v = 0; v < 478; ++v) {
			disp.at<int16_t>(w,v) = -1;
		}
	}

	for(int w = 550; w < 810; ++w) {
		for(int v = 464; v < 630; ++v) {
			disp.at<int16_t>(w,v) = -1;
		}
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
		printf("\n");
	}

	//// Segmentation of the point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_croped (new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_filter (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sor_filter (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rad_filter (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obstacle1 (new pcl::PointCloud<pcl::PointXYZ>), cloud_obstacle2 (new pcl::PointCloud<pcl::PointXYZ>);

	if (pcl::io::loadPCDFile<pcl::PointXYZ> ("pointclouds/pointclouds_SAFE/stereo_out.pcd", *cloud) == -1) //* load the file
	{
		PCL_ERROR ("Couldn't read the point cloud. \n");
		return -1;
	}

	//// Filter out points not within the range of the tractor
	// Create the filtering object on the X axis
	pcl::PassThrough<pcl::PointXYZ> pass_x;
	pass_x.setInputCloud (cloud);
	pass_x.setFilterFieldName ("x");
	pass_x.setFilterLimits (-5.0, 5.0);
	pass_x.filter (*cloud_croped);
	*cloud_filtered = *cloud_croped;

	//pcl::io::savePCDFile("x_filter.pcd", *cloud_filtered);

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
	pcl::PCDWriter writer;
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (100);
	seg.setDistanceThreshold (0.25);

	int i=0, nr_points = (int) cloud_filtered->points.size ();
	while (cloud_filtered->points.size () > 0.3 * nr_points)
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud (cloud_filtered);
		seg.segment (*inliers, *coefficients);
		if (inliers->indices.size () == 0)
		{
		  std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
		  break;
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud (cloud_filtered);
		extract.setIndices (inliers);
		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_plane_filter);
		*cloud_filtered = *cloud_plane_filter;
		//pcl::io::savePCDFile("plane_filter.pcd", *cloud_filtered);

	}

	// Apply radius outlier filter
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
	outrem.setInputCloud(cloud_filtered);
	outrem.setRadiusSearch(0.5);
	outrem.setMinNeighborsInRadius (8000);
	// apply filter
	outrem.filter (*cloud_rad_filter);
	*cloud_filtered = *cloud_rad_filter;
	//pcl::io::savePCDFile("radius_filter.pcd", *cloud_filtered);

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (0.09); // 2cm
	ec.setMinClusterSize (8000);
	ec.setMaxClusterSize (55000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud_filtered);
	ec.extract (cluster_indices);

	int j = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
	for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	  cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*

	cloud_cluster->width = cloud_cluster->points.size ();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;
	std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
	//std::stringstream ss;
	//ss << "cluster_" << j << ".pcd";
	//writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*

	Lcloud=*cloud_cluster;
	//// Create empty disparity map for Lidar data
	cv::Mat obstacle_=cv::Mat::zeros(img1.rows, img1.cols, CV_16S);

	for (int i = 0; i < Lcloud.points.size (); ++i)  //try with cloud height and width
	{
		//// Scale position from m to mm;
		Lcloud.points[i].x = Lcloud.points[i].x/0.001;
		Lcloud.points[i].y = Lcloud.points[i].y/0.001;
		Lcloud.points[i].z = Lcloud.points[i].z/0.001;

		float d = (Lcloud.points[i].z*dcx-f*B)/Lcloud.points[i].z; // disparity
		float W = B/(-d+dcx); // Weighting
		int x = (Lcloud.points[i].x+cx*W)/W; // x value
		int y = (Lcloud.points[i].y+cy*W)/W; // y value

		//// Filter out all LIDAR points which are outside the camera view
		if(y>=0 && y<obstacle_.rows && x>=0 && x<obstacle_.cols)
			{
				obstacle_.at<int16_t>(y,x)=d;
			}
	}

	cv::Point corner_1, corner_2;
	corner_1.x = 99999;
	corner_1.y = 99999;
	corner_2.x = -99999;
	corner_2.y = -99999;

	for(int w = 0; w < obstacle_.rows; ++w) {
		for(int v = 0; v < obstacle_.cols; ++v) {
			//cout<<"disp:"<<disp.at<int16_t>(w,v)<<endl;
			//cout<<"Value:"<<obstacle_.at<int16_t>(w,v)<<endl;
			if(obstacle_.at<int16_t>(w,v)!=0)
				{
					if(corner_1.y > w)
						corner_1.y = w;
					if(corner_2.y < w)
						corner_2.y= w;
					if(corner_1.x > v)
						corner_1.x = v;
					if(corner_2.x < v)
						corner_2.x = v;
				}
			}
		}

	if(corner_1.x>50)
		corner_1.x = corner_1.x - 50;
	else corner_1.x = 0;
	if(corner_1.y>50)
		corner_1.y = corner_1.y - 50;
	else corner_1.y = 0;

	if(corner_2.x<1870)
		corner_2.x = corner_2.x + 50;
	else corner_2.x = 1920;
	if(corner_2.y<1030)
		corner_2.y = corner_2.y + 50;
	else corner_2.y = 1080;

	cout<<"Label is: ("<<corner_1.x<<","<<corner_1.y<<") and ("<<corner_2.x<<","<<corner_2.y<<")."<<endl;

	cv::rectangle(img1, corner_1, corner_2, cv::Scalar(0, 0, 255), 2);

	std::stringstream obss;
	//obss << "cluster_" << j <<".png";
	//imshow("Obstacle", img1);
	cv::waitKey(0);
	//// Save result of Lidar->image
	//cv::imwrite(obss.str (),obstacle_);
	j++;
	}

	cv::imwrite("Obstacles.png",img1);
	std::vector<cv::Point> Labels;




	printf("PROGRAM DONE!! \n");
	return 0;
}

