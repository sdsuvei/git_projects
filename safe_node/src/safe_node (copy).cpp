#include "safe_node.h"
using namespace std;



int main (int argc, char** argv)
{

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

	intrinsic_filename = "stereo_parameters_SAFE/int.yml";
	extrinsic_filename = "stereo_parameters_SAFE/ent.yml";
	disparity_filename1 = "DISP1.png";
	disparity_filename2 = "DISP2.png";
	point_cloud_filename = "stereo_out.pcd";

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
		//cv::imwrite("image_color.png",img1);
		cv::resize(img2, temp2, cv::Size(), scale, scale, method);
		img2 = temp2;
	}


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

		string ty =  type2str( img1.type() );
		//printf("Matrix: %s %dx%d \n", ty.c_str(), img1.cols, img1.rows );


		cv::imwrite("left_rect_L.png",img1);
		cv::imwrite("right_rect_L.png",img2);

		//cout<<"R1:"<<endl<<R1<<endl;
	}


	 //// 1) READ THE LIDAR POINT CLOUD AND EXTRACT THE POINTS
	pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	//// Loading the point clouds


	if (pcl::io::loadPCDFile<pcl::PointXYZ> ("lidar.pcd", *lidar_cloud) == -1) {
			PCL_ERROR ("Couldn't read the LIDAR point cloud! \n");
			return (-1);
		}


	Eigen::Matrix4f depthTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f depthTs_R = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_lTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_lTs_R = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_cobTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_cobTs_R = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_cTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_cTs_R = Eigen::Matrix4f::Identity();



	//// The Transform Matrix which moves the lidar points in the stereo camera frame
	//// This matrix comes from the ICP algorithm appliead before-hand

  //SAFE TRANSFORMATIONS
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


	/*
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
 */

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
	cv::Mat lidar_l=cv::Mat::zeros(img1.rows, img1.cols, CV_8U);

	float f = P1.at<double>(0,0); // Focal length
	float B = P2.at<double>(0,3)/f; // Baseline in the x direction
	float cx = P1.at<double>(0,2); // Center x coordinate
	float cy = P1.at<double>(1,2); // Center y coordinate

	float cx2 = P2.at<double>(0,2); // Center x coordinate of right image
	float cy2 = P2.at<double>(1,2); // Center y coordinate of right image
	float dcx = cx-cx2;
	cout<<"Camera parameters: "<<endl;
	cout<<"B: "<<B<<" f: "<<f<<" cx: "<< cx<<" cx2: "<<cx2<<endl;
	cout<<"P1:"<<endl<<P1<<endl;

	for (int i = 0; i < Lcloud.points.size (); ++i)  {
		//// Cross reference the 2D points by using the formula:
		/// [x_, y_, z_] = R*[X,Y,Z]+t   - where R rotation matrix from P1 and t is the translation from P1; P1 is the projective transformation
		double x_= 1131.377952471656*Lcloud.points[i].x+472.2575759887695*Lcloud.points[i].z;
		double y_= 1131.377952471656*Lcloud.points[i].y+517.5933837890625*Lcloud.points[i].z;
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
			//cout<<"z: "<<z_<<" lidar d: "<<d<<endl;
				lidar_l.at<int8_t>(y,x)=d;
				//std::cout<<"x_2d: "<<x_2d<<" y_2d: "<<y_2d<<endl;
				//std::cout<<"x: "<<x<<" y: "<<y<<" d: "<<d<<endl<<endl;
				saved_cloud.points.push_back(Lcloud.points[i]);
			}
	}

	saved_cloud.width = 1;
	saved_cloud.height = saved_cloud.points.size();

	pcl::io::savePCDFileASCII ("saved_lidar.pcd", saved_cloud);
	cv::imwrite("lidar_disp.png",lidar_l);

	//// 2) USE THE LIDAR POINTS TO INFLUENCE THE STEREO_BM METHOD
	//// Apply dilation on the LIDAR points

	cv::Mat src = lidar_l; // cv::imread("lidar_disp.png");
	cv::Mat dilated_lidar = cv::Mat(img1.rows, img1.cols, CV_8U);

	cv::Mat kernel = cv::Mat::ones(3,3,CV_8U);

	cv::dilate(src,dilated_lidar, kernel, cv::Point(-1, -1), 11, 1, 1); // 11 for SAFE!!
	// Apply the specified morphology operation
	cv::imwrite("dilated_lidar.png",dilated_lidar);

	//// Add own BM code here
	int blockSize = 7;
	int min_disp = 0;
	int max_disp = 240;
	int d_Stereo = 0;
	int d_Lidar = 0;
	int RANGE = 50;

	cv::Mat disp =cv::Mat(img1.rows, img1.cols, CV_8U);

	cv::Mat img1_pad, img2_pad;
	cv::copyMakeBorder(img1,img1_pad,2,2,2,2,cv::BORDER_CONSTANT,cv::Scalar(0));
	cv::copyMakeBorder(img2,img2_pad,2,2,2,2,cv::BORDER_CONSTANT,cv::Scalar(0));

	struct timeval start, end;
	long mtime, seconds, useconds;
	gettimeofday(&start, NULL);


    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.cols; ++j) {
           	//int minSAD = INT_MAX;
           	int minSAD = 500;
           	d_Stereo = 0;
           	d_Lidar = (int)dilated_lidar.at<int8_t>(i,j);
           	if(d_Lidar != 0){
           		min_disp = d_Lidar - RANGE;
           		if(min_disp<0)
           			min_disp=0;

           		max_disp = d_Lidar + RANGE;
           		if(max_disp > 240)
           			max_disp = 240;
           		//cout<<"d_Lidar: "<<d_Lidar<<" and [min_disp,max_disp]: ["<<min_disp<<","<<max_disp<<"]"<<endl;
           	}
            for (int d = min_disp; d <= max_disp; ++d) {
                int SAD = 0;
                if ((j+d-(blockSize-1)/2) < 0)
                		continue;
                if ((j+d+(blockSize-1)/2) > 1919)
                        break;
                for (int l = i-(blockSize-1)/2; l < i + (blockSize-1)/2; ++l) {
                    for (int m = j-(blockSize-1)/2; m <= j+(blockSize-1)/2; ++m) {
                       	if(SAD>minSAD)
                       		break;
                        SAD += abs((int)img1_pad.at<int8_t>(l, m)-(int)img2_pad.at<int8_t>(l, m - d));
                    }
                    if(SAD>minSAD)
                    	break;
                }
                if (SAD < minSAD && SAD<300) {
					minSAD = SAD;
					d_Stereo = d;
                }
            }
            cout<<"d_Lidar: "<<d_Lidar<<" and [min_disp,max_disp]: ["<<min_disp<<","<<max_disp<<"]"<<"  d:" <<(d_Stereo)<<endl;
            //cout<<"minSAD: "<<minSAD<<" and d: "<<d_Stereo<<endl<<endl;
            disp.at<int8_t>(i, j) = d_Stereo;
        }
    }


	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	printf("Elapsed time for BM: %f seconds\n", (float)mtime/1000);

	//disp = DISP; // disp/16 //because it is multiplied by 16 inside the BM method!!

	if(disparity_filename1)
		imwrite(disparity_filename1, disp);

	printf("PROGRAM DONE!! \n");
	return 0;
}
