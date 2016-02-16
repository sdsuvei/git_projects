#include <iostream>
#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <iostream>
#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

pcl::visualization::PCLVisualizer *p;
int vp_1;
int vp_2;
int vp_3;

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source, const PointCloud::Ptr cloud_source_2)
{
  p->removePointCloud ("vp1_target");
  p->removePointCloud ("vp1_source");
  p->removePointCloud ("vp1_source2");

  PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0);
  PointCloudColorHandlerCustom<PointT> src2_h (cloud_source, 255, 0, 0);
  p->addPointCloud (cloud_target, tgt_h, "vp1_target");
  p->addPointCloud (cloud_source, src_h, "vp1_source");
  p->addPointCloud (cloud_source_2, src_h, "vp1_source2");
  p-> spin();
}

int main (int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_croped (new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_filter (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sor_filter (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rad_filter (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obstacle1 (new pcl::PointCloud<pcl::PointXYZ>), cloud_obstacle2 (new pcl::PointCloud<pcl::PointXYZ>);

	if (pcl::io::loadPCDFile<pcl::PointXYZ> ("stereo_out.pcd", *cloud) == -1) //* load the file
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

	pcl::io::savePCDFile("x_filter.pcd", *cloud_filtered);

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
		pcl::io::savePCDFile("plane_filter.pcd", *cloud_filtered);

	}

	// Apply radius outlier filter
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
	outrem.setInputCloud(cloud_filtered);
	outrem.setRadiusSearch(0.5);
	outrem.setMinNeighborsInRadius (8000);
	// apply filter
	outrem.filter (*cloud_rad_filter);
	*cloud_filtered = *cloud_rad_filter;
	pcl::io::savePCDFile("radius_filter.pcd", *cloud_filtered);

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
	std::stringstream ss;
	ss << "cloud_cluster_" << j << ".pcd";
	writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
	j++;
	}

	/*
	if (j==0)
		cloud_obstacle1 = cloud_cluster;
	else if (j==1)
		cloud_obstacle2 = cloud_cluster; */

	//p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Incremental Registration example");

	//p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
	//p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
	//showCloudsLeft(cloud, cloud_obstacle1, cloud_obstacle2);
  return 0;
}
