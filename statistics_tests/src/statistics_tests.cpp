#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl-1.7/pcl/sample_consensus/sac_model_plane.h>
#include <fstream>

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

pcl::visualization::PCLVisualizer *p;
int vp_1;
int vp_2;

void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target");
  p->removePointCloud ("vp1_source");

  PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0);
   p->addPointCloud (cloud_target, tgt_h, "vp1_target");
  p->addPointCloud (cloud_source, src_h, "vp1_source");
  p-> spin();
}

int main (int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_BM (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_lidar(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_stereo (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fused (new pcl::PointCloud<pcl::PointXYZ>);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_BM (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_lidar (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_stereo (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_fused (new pcl::PointCloud<pcl::PointXYZ>);


	// Fill in the cloud data
	pcl::PCDReader reader;
	// Open the point clouds
	reader.read<pcl::PointXYZ> ("SAFE_TEST_RESULTS/BM_out_filtered.pcd", *cloud_BM);
	reader.read<pcl::PointXYZ> ("SAFE_TEST_RESULTS/lidar_out_filtered.pcd", *cloud_lidar);
	reader.read<pcl::PointXYZ> ("SAFE_TEST_RESULTS/stereo_out_filtered.pcd", *cloud_stereo);
	reader.read<pcl::PointXYZ> ("SAFE_TEST_RESULTS/fused_out_filtered.pcd", *cloud_fused);


    // LIDAR CLOUD - THIS IS THE REFFERENCE POINT FOR STEREO
	// Change the coordinate system of the LIDAR to the one of the STEREO
	Eigen::Matrix4f transform_lTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f transform_lTs_R = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f depthTs = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f depthTs_R = Eigen::Matrix4f::Identity();

	transform_lTs << 0.998698, 0.048944, 0.0156136, 0.426832,
				0.0254905, -0.208193, -0.977776, 0.16635,
				-0.04461, 0.976888, -0.209161, -1.43821,
				0, 0, 0, 1;

	// This is the R matrix from the calibration process - it will overlap the LIDAR onto the STEREO view


	 transform_lTs_R << 0.9999898012149877, -0.004498446772515268, -0.0004017992584439221, 0,
			0.004497161244118079, 0.9999849421569889, -0.003144996029905281, 0,
			0.0004159408054540794, 0.00314315699871394, 0.9999949737660324, 0,
			0, 0, 0, 1;   //<- FOR LIDAR WALL TEST

/*
	transform_lTs <<  1.00001, -0.00124211, 0.000743923, 0.0322781,
			0.00124286, 0.999932, -0.0126477, -0.0188326,
			-0.00073585, 0.0126586, 0.999934, -0.0153805,
			 0, 0, 0, 1; */ // <- FOR COB PLATE TEST

	depthTs = transform_lTs;
	depthTs_R = transform_lTs_R;

	//// Executing the transformation
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud2 (new pcl::PointCloud<pcl::PointXYZ> ());
	pcl::transformPointCloud (*cloud_lidar, *transformed_cloud, depthTs );
	pcl::transformPointCloud (*transformed_cloud, *transformed_cloud2, depthTs_R);

	*cloud_lidar=*transformed_cloud2;

	pcl::io::savePCDFile("lidar_out_transformed.pcd", *cloud_lidar);

	// Plane fitting and defining the model of the plane
	pcl::ModelCoefficients::Ptr coefficients_lidar (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers_lidar (new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg_lidar;
	// Optional
	seg_lidar.setOptimizeCoefficients (true);
	// Mandatory
	seg_lidar.setModelType (pcl::SACMODEL_PLANE);
	seg_lidar.setMethodType (pcl::SAC_RANSAC);
	double interval = 0.01;
	seg_lidar.setDistanceThreshold (interval);

	seg_lidar.setInputCloud (cloud_lidar);
	seg_lidar.segment (*inliers_lidar, *coefficients_lidar);

	if (inliers_lidar->indices.size () == 0)
	{
	  PCL_ERROR ("Could not estimate a planar model for the lidar dataset.");
	  return (-1);
	}

	std::cerr << "Lidar model coefficients: " << coefficients_lidar->values[0] << " "
										<< coefficients_lidar->values[1] << " "
										<< coefficients_lidar->values[2] << " "
										<< coefficients_lidar->values[3] << std::endl;

	// Extract the planar inliers from the input cloud
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	extract.setInputCloud (cloud_lidar);
	extract.setIndices (inliers_lidar);
	// Remove the planar inliers, extract the rest
	extract.setNegative (false);
	extract.filter (*cloud_plane_lidar);
	pcl::io::savePCDFile("lidar_plane.pcd", *cloud_plane_lidar);


	int counter_negative = 0;
	int counter_positive = 0;
	int counter_inliers = 0;

	double distance_negative = 0;
	double distance_positive = 0;
	double distance_inliers = 0;

	// Data files
	fstream distance_plus, distance_minus, distance_in;
	cout<<"Size of LIDAR pointcloud: "<<cloud_lidar->size()<<endl;
	//distance_plus.open ("LIDAR_plus_points.txt");
	//distance_minus.open ("LIDAR_minus_points.txt");
	distance_in.open ("LIDAR_inliers_points.txt");

	// Chek how many points belong to the plane
	for (size_t i = 0; i < cloud_lidar->points.size (); ++i)
	{
	  double rez= coefficients_lidar->values[0]*cloud_lidar->points[i].x+coefficients_lidar->values[1]*cloud_lidar->points[i].y+coefficients_lidar->values[2]*cloud_lidar->points[i].z+coefficients_lidar->values[3];
	  if(rez>interval){
			distance_positive = pcl::pointToPlaneDistance(cloud_lidar->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
			counter_positive+=1;
			distance_plus << distance_positive<<endl;
	  }

	  else if(rez < -interval){
		  distance_negative = pcl::pointToPlaneDistance(cloud_lidar->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
		  counter_negative+=1;
		  distance_minus <<"-"<< distance_negative<<endl;
	  }

	  else{
		  distance_inliers = pcl::pointToPlaneDistance(cloud_lidar->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
		  counter_inliers+=1;
		  distance_in << distance_inliers<<endl;
	  }

	}

	cout<<"Positive lidar points:"<<counter_positive<<endl;
	cout<<"Negative lidar points:"<<counter_negative<<endl;
	cout<<"Inliers lidar points:"<<counter_inliers<<endl;
	cout<<endl;

	distance_plus.close();
	distance_minus.close();
	distance_in.close();


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// BM cloud
	counter_negative = 0;
	counter_positive = 0;
	counter_inliers = 0;
	// Data files
	fstream d_plus, d_minus, d_in;
	//d_plus.open ("BM_plus_points.txt");
	//d_minus.open ("BM_minus_points.txt");
	d_in.open ("BM_inliers_points.txt");

	// Chek how many points belong to the plane
	for (size_t i = 0; i < cloud_BM->points.size (); ++i)
	{
	  double rez= coefficients_lidar->values[0]*cloud_BM->points[i].x+coefficients_lidar->values[1]*cloud_BM->points[i].y+coefficients_lidar->values[2]*cloud_BM->points[i].z+coefficients_lidar->values[3];
	  if(rez>interval){
			distance_positive = pcl::pointToPlaneDistance(cloud_BM->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
			counter_positive+=1;
			d_plus << distance_positive<<endl;
	  }

	  else if(rez < -interval){
		  distance_negative = pcl::pointToPlaneDistance(cloud_BM->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
		  counter_negative+=1;
		  d_minus <<"-"<< distance_negative<<endl;
	  }

	  else{
		  distance_inliers = pcl::pointToPlaneDistance(cloud_BM->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
		  counter_inliers+=1;
		  d_in << distance_inliers<< endl;
		  //bm_plane->points.push_back(cloud_BM->points[i]);
	  }
	}

	cout<<"Size of BM pointcloud: "<<cloud_BM->size()<<endl;
	cout<<"Positive BM points:"<<counter_positive<<endl;
	cout<<"Negative BM points:"<<counter_negative<<endl;
	cout<<"Inliers BM points:"<<counter_inliers<<endl;
	cout<<endl;


	//pcl::PointCloud<pcl::PointXYZ> bm_plane;
	pcl::PointCloud<pcl::PointXYZ>::Ptr bm_plane (new pcl::PointCloud<pcl::PointXYZ>);
	bm_plane->height=counter_inliers;//FOR COB: 42723; FOR LIDAR: 19898
	bm_plane->width=1;

	for(int s=0;s<bm_plane->points.size(); ++s)
	{
		bm_plane->points[s].x=0;
		bm_plane->points[s].y=0;
		bm_plane->points[s].z=0;
	}

	for (size_t i = 0; i < cloud_BM->points.size (); ++i)
	{
	  double rez= coefficients_lidar->values[0]*cloud_BM->points[i].x+coefficients_lidar->values[1]*cloud_BM->points[i].y+coefficients_lidar->values[2]*cloud_BM->points[i].z+coefficients_lidar->values[3];
	  if(rez>-interval && rez<interval){
		  bm_plane->points.push_back(cloud_BM->points[i]);
	  }
	}

	pcl::io::savePCDFileASCII ("bm_plane.pcd", *bm_plane);

	d_plus.close();
	d_minus.close();
	d_in.close();


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// STEREO cloud
	counter_negative = 0;
	counter_positive = 0;
	counter_inliers = 0;
	// Data files
	fstream di_plus, di_minus, di_in;
	//di_plus.open ("STEREO_plus_points.txt");
	//di_minus.open ("STEREO_minus_points.txt");
	di_in.open ("STEREO_inliers_points.txt");

	// Chek how many points belong to the plane
	for (size_t i = 0; i < cloud_stereo->points.size (); ++i)
	{
	  double rez= coefficients_lidar->values[0]*cloud_stereo->points[i].x+coefficients_lidar->values[1]*cloud_stereo->points[i].y+coefficients_lidar->values[2]*cloud_stereo->points[i].z+coefficients_lidar->values[3];
	  if(rez>interval){
			distance_positive = pcl::pointToPlaneDistance(cloud_stereo->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
			counter_positive+=1;
			di_plus << distance_positive<<endl;
	  }

	  else if(rez < -interval){
		  distance_negative = pcl::pointToPlaneDistance(cloud_stereo->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
		  counter_negative+=1;
		  di_minus <<"-"<< distance_negative<<endl;
	  }

	  else{
		  distance_inliers = pcl::pointToPlaneDistance(cloud_stereo->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
		  counter_inliers+=1;
		  di_in<<distance_inliers<<endl;
		  //stereo_plane->points.push_back(cloud_stereo->points[i]);
	  }
	}

	cout<<"Size of stereo pointcloud: "<<cloud_stereo->size()<<endl;
	cout<<"Positive stereo points:"<<counter_positive<<endl;
	cout<<"Negative stereo points:"<<counter_negative<<endl;
	cout<<"Inliers stereo points:"<<counter_inliers<<endl;
	cout<<endl;

	//pcl::PointCloud<pcl::PointXYZ> stereo_plane;
	pcl::PointCloud<pcl::PointXYZ>::Ptr stereo_plane (new pcl::PointCloud<pcl::PointXYZ>);
	stereo_plane->height=counter_inliers;//FOR COB: 233397; FOR LIDAR: 172060
	stereo_plane->width=1;

	for(int s=0;s<stereo_plane->points.size(); ++s)
	{
		stereo_plane->points[s].x=0;
		stereo_plane->points[s].y=0;
		stereo_plane->points[s].z=0;
	}

	for (size_t i = 0; i < cloud_stereo->points.size (); ++i)
	{
	  double rez= coefficients_lidar->values[0]*cloud_stereo->points[i].x+coefficients_lidar->values[1]*cloud_stereo->points[i].y+coefficients_lidar->values[2]*cloud_stereo->points[i].z+coefficients_lidar->values[3];
	  if(rez>-interval && rez<interval){
		  stereo_plane->points.push_back(cloud_stereo->points[i]);
	  }
	}

	pcl::io::savePCDFileASCII ("stereo_plane.pcd", *stereo_plane);

	di_plus.close();
	di_minus.close();
	di_in.close();


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// FUSED cloud
	counter_negative = 0;
	counter_positive = 0;
	counter_inliers = 0;
	// Data files
	fstream fused_file;
	fused_file.open ("FUSED_points.txt");


	// Chek how many points belong to the plane
	for (size_t i = 0; i < cloud_fused->points.size (); ++i)
	{
	  double rez= coefficients_lidar->values[0]*cloud_fused->points[i].x+coefficients_lidar->values[1]*cloud_fused->points[i].y+coefficients_lidar->values[2]*cloud_fused->points[i].z+coefficients_lidar->values[3];
	  if(rez>interval){
			distance_positive = pcl::pointToPlaneDistance(cloud_fused->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
			counter_positive+=1;
			//fused_file << distance_positive<<endl;
	  }

	  else if(rez < -interval){
		  distance_negative = pcl::pointToPlaneDistance(cloud_fused->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
		  counter_negative+=1;
		  //fused_file <<"-"<< distance_negative<<endl;
	  }

	  else{
		  distance_inliers = pcl::pointToPlaneDistance(cloud_fused->points[i], coefficients_lidar->values[0], coefficients_lidar->values[1],coefficients_lidar->values[2], coefficients_lidar->values[3]);
		  counter_inliers+=1;
		  fused_file << distance_inliers<< endl;
		  //bm_plane->points.push_back(cloud_BM->points[i]);
	  }
	}

	cout<<"Size of FUSED pointcloud: "<<cloud_fused->size()<<endl;
	cout<<"Positive FUSED points:"<<counter_positive<<endl;
	cout<<"Negative FUSED points:"<<counter_negative<<endl;
	cout<<"Inliers FUSED points:"<<counter_inliers<<endl;
	cout<<endl;


	//pcl::PointCloud<pcl::PointXYZ> bm_plane;
	pcl::PointCloud<pcl::PointXYZ>::Ptr fused_plane (new pcl::PointCloud<pcl::PointXYZ>);
	fused_plane->height=counter_inliers;//FOR COB: 42723; FOR LIDAR: 19898
	fused_plane->width=1;

	for(int s=0;s<fused_plane->points.size(); ++s)
	{
		fused_plane->points[s].x=0;
		fused_plane->points[s].y=0;
		fused_plane->points[s].z=0;
	}

	for (size_t i = 0; i < cloud_fused->points.size (); ++i)
	{
	  double rez= coefficients_lidar->values[0]*cloud_fused->points[i].x+coefficients_lidar->values[1]*cloud_fused->points[i].y+coefficients_lidar->values[2]*cloud_fused->points[i].z+coefficients_lidar->values[3];
	  if(rez>-interval && rez<interval){
		  fused_plane->points.push_back(cloud_fused->points[i]);
	  }
	}

	pcl::io::savePCDFileASCII ("fused_plane.pcd", *fused_plane);

	fused_file.close();



	p = new pcl::visualization::PCLVisualizer (argc, argv, "Plane fitting");

	//p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
	//p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
	showCloudsLeft(cloud_lidar, cloud_plane_lidar);
	showCloudsLeft(cloud_BM, bm_plane);
	showCloudsLeft(cloud_stereo, stereo_plane);

  return (0);
}
