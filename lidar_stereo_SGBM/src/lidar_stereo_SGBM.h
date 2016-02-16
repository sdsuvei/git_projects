#ifndef LIDAR_STEREO_SGBM_H_
#define LIDAR_STEREO_SGBM_H_

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include "precomp.hpp"

#include <iostream>
#include <stdio.h>
#include <ctime>

#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_representation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>

//convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

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

cv::Mat bm_disp, disp, disp8, disp2, disp2_8, DISP, LEFT, RIGHT;
int INTERVAL = 0;

static void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|var] [--blocksize=<block_size>]\n"
			"[--max-disparity=<max_disparity>] --min-disparity=<min_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
			"[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}

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

namespace cv
{

typedef uchar PixType;
typedef short CostType;
typedef short DispType;

enum { NR = 16, NR2 = NR/2 };

StereoSGBM::StereoSGBM()
{
	//cout<<"Check 0"<<endl;
    minDisparity = numberOfDisparities = 0;
    SADWindowSize = 0;
    P1 = P2 = 0;
    disp12MaxDiff = 0;
    preFilterCap = 0;
    uniquenessRatio = 0;
    speckleWindowSize = 0;
    speckleRange = 0;
    fullDP = false;
}


StereoSGBM::StereoSGBM( int _minDisparity, int _numDisparities, int _SADWindowSize,
                   int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
                   int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
                   bool _fullDP )
{
    minDisparity = _minDisparity;
    numberOfDisparities = _numDisparities;
    SADWindowSize = _SADWindowSize;
    P1 = _P1;
    P2 = _P2;
    disp12MaxDiff = _disp12MaxDiff;
    preFilterCap = _preFilterCap;
    uniquenessRatio = _uniquenessRatio;
    speckleWindowSize = _speckleWindowSize;
    speckleRange = _speckleRange;
    fullDP = _fullDP;
}


StereoSGBM::~StereoSGBM()
{
}

/*
 For each pixel row1[x], max(-maxD, 0) <= minX <= x < maxX <= width - max(0, -minD),
 and for each disparity minD<=d<maxD the function
 computes the cost (cost[(x-minX)*(maxD - minD) + (d - minD)]), depending on the difference between
 row1[x] and row2[x-d]. The subpixel algorithm from
 "Depth Discontinuities by Pixel-to-Pixel Stereo" by Stan Birchfield and C. Tomasi
 is used, hence the suffix BT.

 the temporary buffer should contain width2*2 elements
 */
static void calcPixelCostBT( const Mat& img1, const Mat& img2, int y,
                            int minD, int maxD, CostType* cost,
                            PixType* buffer, const PixType* tab,
                            int tabOfs, int )
{
	//cout<<"calcPixelCostBT"<<endl;
    int x, c, width = img1.cols, cn = img1.channels();
    int minX1 = max(-maxD, 0), maxX1 = width + min(minD, 0);
    int minX2 = max(minX1 - maxD, 0), maxX2 = min(maxX1 - minD, width);
    int D = maxD - minD, width1 = maxX1 - minX1, width2 = maxX2 - minX2;
    const PixType *row1 = img1.ptr<PixType>(y), *row2 = img2.ptr<PixType>(y);
    PixType *prow1 = buffer + width2*2;
    PixType *prow2 = prow1 + width*cn*2;

    tab += tabOfs;

    for( c = 0; c < cn*2; c++ )
    {
        prow1[width*c] = prow1[width*c + width-1] =
        prow2[width*c] = prow2[width*c + width-1] = tab[0];
    }

    int n1 = y > 0 ? -(int)img1.step : 0, s1 = y < img1.rows-1 ? (int)img1.step : 0;
    int n2 = y > 0 ? -(int)img2.step : 0, s2 = y < img2.rows-1 ? (int)img2.step : 0;

    if( cn == 1 )
    {
        for( x = 1; x < width-1; x++ )
        {
            prow1[x] = tab[(row1[x+1] - row1[x-1])*2 + row1[x+n1+1] - row1[x+n1-1] + row1[x+s1+1] - row1[x+s1-1]];
            prow2[width-1-x] = tab[(row2[x+1] - row2[x-1])*2 + row2[x+n2+1] - row2[x+n2-1] + row2[x+s2+1] - row2[x+s2-1]];

            prow1[x+width] = row1[x];
            prow2[width-1-x+width] = row2[x];
        }
    }
    else
    {
        for( x = 1; x < width-1; x++ )
        {
            prow1[x] = tab[(row1[x*3+3] - row1[x*3-3])*2 + row1[x*3+n1+3] - row1[x*3+n1-3] + row1[x*3+s1+3] - row1[x*3+s1-3]];
            prow1[x+width] = tab[(row1[x*3+4] - row1[x*3-2])*2 + row1[x*3+n1+4] - row1[x*3+n1-2] + row1[x*3+s1+4] - row1[x*3+s1-2]];
            prow1[x+width*2] = tab[(row1[x*3+5] - row1[x*3-1])*2 + row1[x*3+n1+5] - row1[x*3+n1-1] + row1[x*3+s1+5] - row1[x*3+s1-1]];

            prow2[width-1-x] = tab[(row2[x*3+3] - row2[x*3-3])*2 + row2[x*3+n2+3] - row2[x*3+n2-3] + row2[x*3+s2+3] - row2[x*3+s2-3]];
            prow2[width-1-x+width] = tab[(row2[x*3+4] - row2[x*3-2])*2 + row2[x*3+n2+4] - row2[x*3+n2-2] + row2[x*3+s2+4] - row2[x*3+s2-2]];
            prow2[width-1-x+width*2] = tab[(row2[x*3+5] - row2[x*3-1])*2 + row2[x*3+n2+5] - row2[x*3+n2-1] + row2[x*3+s2+5] - row2[x*3+s2-1]];

            prow1[x+width*3] = row1[x*3];
            prow1[x+width*4] = row1[x*3+1];
            prow1[x+width*5] = row1[x*3+2];

            prow2[width-1-x+width*3] = row2[x*3];
            prow2[width-1-x+width*4] = row2[x*3+1];
            prow2[width-1-x+width*5] = row2[x*3+2];
        }
    }

    memset( cost, 0, width1*D*sizeof(cost[0]) );

    buffer -= minX2;
    cost -= minX1*D + minD; // simplify the cost indices inside the loop



    for( c = 0; c < cn*2; c++, prow1 += width, prow2 += width )
    {
        int diff_scale = c < cn ? 0 : 2;

        // precompute
        //   v0 = min(row2[x-1/2], row2[x], row2[x+1/2]) and
        //   v1 = max(row2[x-1/2], row2[x], row2[x+1/2]) and
        for( x = minX2; x < maxX2; x++ )
        {
            int v = prow2[x];
            int vl = x > 0 ? (v + prow2[x-1])/2 : v;
            int vr = x < width-1 ? (v + prow2[x+1])/2 : v;
            int v0 = min(vl, vr); v0 = min(v0, v);
            int v1 = max(vl, vr); v1 = max(v1, v);
            buffer[x] = (PixType)v0;
            buffer[x + width2] = (PixType)v1;
        }

        for( x = minX1; x < maxX1; x++ )
        {
            int u = prow1[x];
            int ul = x > 0 ? (u + prow1[x-1])/2 : u;
            int ur = x < width-1 ? (u + prow1[x+1])/2 : u;
            int u0 = min(ul, ur); u0 = min(u0, u);
            int u1 = max(ul, ur); u1 = max(u1, u);

			for( int d = minD; d < maxD; d++ )
			{
				int v = prow2[width-x-1 + d];
				int v0 = buffer[width-x-1 + d];
				int v1 = buffer[width-x-1 + d + width2];
				int c0 = max(0, u - v1); c0 = max(c0, v0 - u);
				int c1 = max(0, v - u1); c1 = max(c1, u0 - v);

				cost[x*D + d] = (CostType)(cost[x*D+d] + (min(c0, c1) >> diff_scale));
			}
        }
    }
}


/*
 computes disparity for "roi" in img1 w.r.t. img2 and write it to disp1buf.
 that is, disp1buf(x, y)=d means that img1(x+roi.x, y+roi.y) ~ img2(x+roi.x-d, y+roi.y).
 minD <= d < maxD.
 disp2full is the reverse disparity map, that is:
 disp2full(x+roi.x,y+roi.y)=d means that img2(x+roi.x, y+roi.y) ~ img1(x+roi.x+d, y+roi.y)

 note that disp1buf will have the same size as the roi and
 disp2full will have the same size as img1 (or img2).
 On exit disp2buf is not the final disparity, it is an intermediate result that becomes
 final after all the tiles are processed.

 the disparity in disp1buf is written with sub-pixel accuracy
 (4 fractional bits, see CvStereoSGBM::DISP_SCALE),
 using quadratic interpolation, while the disparity in disp2buf
 is written as is, without interpolation.

 disp2cost also has the same size as img1 (or img2).
 It contains the minimum current cost, used to find the best disparity, corresponding to the minimal cost.
 */
static void computeDisparitySGBM( const Mat& img1, const Mat& img2,
                                 Mat& disp1, const StereoSGBM& params,
                                 Mat& buffer )
{
	//cout<<"static void computeDisparitySGBM"<<endl;
    const int ALIGN = 16;
    const int DISP_SHIFT = StereoSGBM::DISP_SHIFT;
    const int DISP_SCALE = StereoSGBM::DISP_SCALE;
    const CostType MAX_COST = SHRT_MAX;

    int minD = params.minDisparity, maxD = minD + params.numberOfDisparities;
    Size SADWindowSize;
    SADWindowSize.width = SADWindowSize.height = params.SADWindowSize > 0 ? params.SADWindowSize : 5;
    int ftzero = max(params.preFilterCap, 15) | 1;
    int uniquenessRatio = params.uniquenessRatio >= 0 ? params.uniquenessRatio : 10;
    int disp12MaxDiff = params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1;
    int P1 = params.P1 > 0 ? params.P1 : 2, P2 = max(params.P2 > 0 ? params.P2 : 5, P1+1);
    int k, width = disp1.cols, height = disp1.rows;
    int minX1 = max(-maxD, 0), maxX1 = width + min(minD, 0);
    int D = maxD - minD, width1 = maxX1 - minX1;
    int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;
    int SW2 = SADWindowSize.width/2, SH2 = SADWindowSize.height/2;
    int npasses = params.fullDP ? 2 : 1;
    const int TAB_OFS = 256*4, TAB_SIZE = 256 + TAB_OFS*2;
    PixType clipTab[TAB_SIZE];

    for( k = 0; k < TAB_SIZE; k++ )
        clipTab[k] = (PixType)(min(max(k - TAB_OFS, -ftzero), ftzero) + ftzero);

    if( minX1 >= maxX1 )
    {
        disp1 = Scalar::all(INVALID_DISP_SCALED);
        return;
    }

    CV_Assert( D % 16 == 0 );

    // NR - the number of directions. the loop on x below that computes Lr assumes that NR == 8.
    // if you change NR, please, modify the loop as well.
    int D2 = D+16, NRD2 = NR2*D2;

    // the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
    // for 8-way dynamic programming we need the current row and
    // the previous row, i.e. 2 rows in total
    const int NLR = 2;
    const int LrBorder = NLR - 1;

    // for each possible stereo match (img1(x,y) <=> img2(x-d,y))
    // we keep pixel difference cost (C) and the summary cost over NR directions (S).
    // we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
    size_t costBufSize = width1*D;
    size_t CSBufSize = costBufSize*(params.fullDP ? height : 1);
    size_t minLrSize = (width1 + LrBorder*2)*NR2, LrSize = minLrSize*D2;
    int hsumBufNRows = SH2*2 + 2;
    size_t totalBufSize = (LrSize + minLrSize)*NLR*sizeof(CostType) + // minLr[] and Lr[]
    costBufSize*(hsumBufNRows + 1)*sizeof(CostType) + // hsumBuf, pixdiff
    CSBufSize*2*sizeof(CostType) + // C, S
    width*16*img1.channels()*sizeof(PixType) + // temp buffer for computing per-pixel cost
    width*(sizeof(CostType) + sizeof(DispType)) + 1024; // disp2cost + disp2

    if( !buffer.data || !buffer.isContinuous() ||
        buffer.cols*buffer.rows*buffer.elemSize() < totalBufSize )
        buffer.create(1, (int)totalBufSize, CV_8U);

    // summary cost over different (nDirs) directions
    CostType* Cbuf = (CostType*)alignPtr(buffer.data, ALIGN);
    CostType* Sbuf = Cbuf + CSBufSize;
    CostType* hsumBuf = Sbuf + CSBufSize;
    CostType* pixDiff = hsumBuf + costBufSize*hsumBufNRows;

    CostType* disp2cost = pixDiff + costBufSize + (LrSize + minLrSize)*NLR;
    DispType* disp2ptr = (DispType*)(disp2cost + width);
    PixType* tempBuf = (PixType*)(disp2ptr + width);

    // add P2 to every C(x,y). it saves a few operations in the inner loops
    for( k = 0; k < width1*D; k++ )
        Cbuf[k] = (CostType)P2;

    for( int pass = 1; pass <= npasses; pass++ )
    {
        int x1, y1, x2, y2, dx, dy;

        if( pass == 1 )
        {
            y1 = 0; y2 = height; dy = 1;
            x1 = 0; x2 = width1; dx = 1;
        }
        else
        {
            y1 = height-1; y2 = -1; dy = -1;
            x1 = width1-1; x2 = -1; dx = -1;
        }

        CostType *Lr[NLR]={0}, *minLr[NLR]={0};

        for( k = 0; k < NLR; k++ )
        {
            // shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
            // and will occasionally use negative indices with the arrays
            // we need to shift Lr[k] pointers by 1, to give the space for d=-1.
            // however, then the alignment will be imperfect, i.e. bad for SSE,
            // thus we shift the pointers by 8 (8*sizeof(short) == 16 - ideal alignment)
            Lr[k] = pixDiff + costBufSize + LrSize*k + NRD2*LrBorder + 8;
            memset( Lr[k] - LrBorder*NRD2 - 8, 0, LrSize*sizeof(CostType) );
            minLr[k] = pixDiff + costBufSize + LrSize*NLR + minLrSize*k + NR2*LrBorder;
            memset( minLr[k] - LrBorder*NR2, 0, minLrSize*sizeof(CostType) );
        }

        for( int y = y1; y != y2; y += dy )    // <-------------------------------------------------------  START METHOD
        {
            int x, d;
            DispType* disp1ptr = disp1.ptr<DispType>(y);
            CostType* C = Cbuf + (!params.fullDP ? 0 : y*costBufSize);
            CostType* S = Sbuf + (!params.fullDP ? 0 : y*costBufSize);

            if( pass == 1 ) // compute C on the first pass, and reuse it on the second pass, if any.
            {
                int dy1 = y == 0 ? 0 : y + SH2, dy2 = y == 0 ? SH2 : dy1;

                for( k = dy1; k <= dy2; k++ )
                {
                    CostType* hsumAdd = hsumBuf + (min(k, height-1) % hsumBufNRows)*costBufSize;

                    if( k < height )
                    {
                        calcPixelCostBT( img1, img2, k, minD, maxD, pixDiff, tempBuf, clipTab, TAB_OFS, ftzero );

                        memset(hsumAdd, 0, D*sizeof(CostType));
                        for( x = 0; x <= SW2*D; x += D )
                        {
                            int scale = x == 0 ? SW2 + 1 : 1;
                            for( d = 0; d < D; d++ )
                                hsumAdd[d] = (CostType)(hsumAdd[d] + pixDiff[x + d]*scale);
                        }

                        if( y > 0 )
                        {
                            const CostType* hsumSub = hsumBuf + (max(y - SH2 - 1, 0) % hsumBufNRows)*costBufSize;
                            const CostType* Cprev = !params.fullDP || y == 0 ? C : C - costBufSize;

                            for( x = D; x < width1*D; x += D )
                            {
                                const CostType* pixAdd = pixDiff + min(x + SW2*D, (width1-1)*D);
                                const CostType* pixSub = pixDiff + max(x - (SW2+1)*D, 0);
								for( d = 0; d < D; d++ )
								{
									int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
									C[x + d] = (CostType)(Cprev[x + d] + hv - hsumSub[x + d]);
								}
                            }
                        }
                        else
                        {
                            for( x = D; x < width1*D; x += D )
                            {
                                const CostType* pixAdd = pixDiff + min(x + SW2*D, (width1-1)*D);
                                const CostType* pixSub = pixDiff + max(x - (SW2+1)*D, 0);
                                for( d = 0; d < D; d++ )
                                    hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                            }
                        }
                    }

                    if( y == 0 )
                    {
                        int scale = k == 0 ? SH2 + 1 : 1;
                        for( x = 0; x < width1*D; x++ )
                            C[x] = (CostType)(C[x] + hsumAdd[x]*scale);
                    }
                }

                // also, clear the S buffer
                for( k = 0; k < width1*D; k++ )
                    S[k] = 0;
            }

            // clear the left and the right borders
            memset( Lr[0] - NRD2*LrBorder - 8, 0, NRD2*LrBorder*sizeof(CostType) );
            memset( Lr[0] + width1*NRD2 - 8, 0, NRD2*LrBorder*sizeof(CostType) );
            memset( minLr[0] - NR2*LrBorder, 0, NR2*LrBorder*sizeof(CostType) );
            memset( minLr[0] + width1*NR2, 0, NR2*LrBorder*sizeof(CostType) );

            /*
             [formula 13 in the paper]
             compute L_r(p, d) = C(p, d) +
             min(L_r(p-r, d),
             L_r(p-r, d-1) + P1,
             L_r(p-r, d+1) + P1,
             min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
             where p = (x,y), r is one of the directions.
             we process all the directions at once:
             0: r=(-dx, 0)
             1: r=(-1, -dy)
             2: r=(0, -dy)
             3: r=(1, -dy)
             4: r=(-2, -dy)
             5: r=(-1, -dy*2)
             6: r=(1, -dy*2)
             7: r=(2, -dy)
             */
            for( x = x1; x != x2; x += dx )
            {

            	int xm = x*NR2, xd = xm*D2;

                int delta0 = minLr[0][xm - dx*NR2] + P2, delta1 = minLr[1][xm - NR2 + 1] + P2;
                int delta2 = minLr[1][xm + 2] + P2, delta3 = minLr[1][xm + NR2 + 3] + P2;

                CostType* Lr_p0 = Lr[0] + xd - dx*NRD2;
                CostType* Lr_p1 = Lr[1] + xd - NRD2 + D2;
                CostType* Lr_p2 = Lr[1] + xd + D2*2;
                CostType* Lr_p3 = Lr[1] + xd + NRD2 + D2*3;

                Lr_p0[-1] = Lr_p0[D] = Lr_p1[-1] = Lr_p1[D] =
                Lr_p2[-1] = Lr_p2[D] = Lr_p3[-1] = Lr_p3[D] = MAX_COST;

                CostType* Lr_p = Lr[0] + xd;
                const CostType* Cp = C + x*D;
                CostType* Sp = S + x*D;


				int minL0 = MAX_COST, minL1 = MAX_COST, minL2 = MAX_COST, minL3 = MAX_COST;

				for( d = 0; d < D; d++ )
				{
					int Cpd = Cp[d], L0, L1, L2, L3;

					L0 = Cpd + min((int)Lr_p0[d], min(Lr_p0[d-1] + P1, min(Lr_p0[d+1] + P1, delta0))) - delta0;
					L1 = Cpd + min((int)Lr_p1[d], min(Lr_p1[d-1] + P1, min(Lr_p1[d+1] + P1, delta1))) - delta1;
					L2 = Cpd + min((int)Lr_p2[d], min(Lr_p2[d-1] + P1, min(Lr_p2[d+1] + P1, delta2))) - delta2;
					L3 = Cpd + min((int)Lr_p3[d], min(Lr_p3[d-1] + P1, min(Lr_p3[d+1] + P1, delta3))) - delta3;

					Lr_p[d] = (CostType)L0;
					minL0 = min(minL0, L0);

					Lr_p[d + D2] = (CostType)L1;
					minL1 = min(minL1, L1);

					Lr_p[d + D2*2] = (CostType)L2;
					minL2 = min(minL2, L2);

					Lr_p[d + D2*3] = (CostType)L3;
					minL3 = min(minL3, L3);

					Sp[d] = saturate_cast<CostType>(Sp[d] + L0 + L1 + L2 + L3);
				}
				minLr[0][xm] = (CostType)minL0;
				minLr[0][xm+1] = (CostType)minL1;
				minLr[0][xm+2] = (CostType)minL2;
				minLr[0][xm+3] = (CostType)minL3;

            }

            if( pass == npasses )
            {
                for( x = 0; x < width; x++ )
                {
                    disp1ptr[x] = disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
                    disp2cost[x] = MAX_COST;
                }

                for( x = width1 - 1; x >= 0; x-- )
                {
                	//// ADDED THINGS FROM BM
                    int lidar_D = DISP.at<int16_t>(y, x);
                    int min_D = 0;
                    int max_D = D;
                    if(lidar_D>0)
                    {
                    	min_D = lidar_D-INTERVAL;
                    	max_D = lidar_D+INTERVAL;
                    }
                    CostType* Sp = S + x*D;
                    int minS = MAX_COST, bestDisp = -1;

                    if( npasses == 1 )
                    {
                        int xm = x*NR2, xd = xm*D2;

                        int minL0 = MAX_COST;
                        int delta0 = minLr[0][xm + NR2] + P2;
                        CostType* Lr_p0 = Lr[0] + xd + NRD2;
                        Lr_p0[-1] = Lr_p0[D] = MAX_COST;
                        CostType* Lr_p = Lr[0] + xd;

                        const CostType* Cp = C + x*D;

						for( d = 0; d < D; d++ )
						{
							int L0 = Cp[d] + min((int)Lr_p0[d], min(Lr_p0[d-1] + P1, min(Lr_p0[d+1] + P1, delta0))) - delta0;

							Lr_p[d] = (CostType)L0;
							minL0 = min(minL0, L0);

							int Sval = Sp[d] = saturate_cast<CostType>(Sp[d] + L0);
							if( Sval < minS && d>min_D && d<max_D )
							{
								minS = Sval;
								bestDisp = d;   // <-------------------------------------------------------------------------------------------
							}
						}
						minLr[0][xm] = (CostType)minL0;

                    }
                    else
                    {
                        for( d = 0; d < D; d++ )
                        {
                            int Sval = Sp[d];
                            if( Sval < minS )
                            {
                                minS = Sval;
                                bestDisp = d;
                            }
                        }
                    }

                    for( d = 0; d < D; d++ )
                    {
                        if( Sp[d]*(100 - uniquenessRatio) < minS*100 && std::abs(bestDisp - d) > 1 )
                            break;
                    }
                    if( d < D )
                        continue;
                    d = bestDisp;
                    int _x2 = x + minX1 - d - minD;
                    if( disp2cost[_x2] > minS )
                    {
                        disp2cost[_x2] = (CostType)minS;
                        disp2ptr[_x2] = (DispType)(d + minD);
                    }

                    if( 0 < d && d < D-1 )
                    {
                        // do subpixel quadratic interpolation:
                        //   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
                        //   then find minimum of the parabola.
                        int denom2 = max(Sp[d-1] + Sp[d+1] - 2*Sp[d], 1);
                        d = d*DISP_SCALE + ((Sp[d-1] - Sp[d+1])*DISP_SCALE + denom2)/(denom2*2);
                    }
                    else
                        d *= DISP_SCALE;
                    disp1ptr[x + minX1] = (DispType)(d + minD*DISP_SCALE);
                }

                for( x = minX1; x < maxX1; x++ )
                {
                    // we round the computed disparity both towards -inf and +inf and check
                    // if either of the corresponding disparities in disp2 is consistent.
                    // This is to give the computed disparity a chance to look valid if it is.
                    int d1 = disp1ptr[x];
                    if( d1 == INVALID_DISP_SCALED )
                        continue;
                    int _d = d1 >> DISP_SHIFT;
                    int d_ = (d1 + DISP_SCALE-1) >> DISP_SHIFT;
                    int _x = x - _d, x_ = x - d_;
                    if( 0 <= _x && _x < width && disp2ptr[_x] >= minD && std::abs(disp2ptr[_x] - _d) > disp12MaxDiff &&
                       0 <= x_ && x_ < width && disp2ptr[x_] >= minD && std::abs(disp2ptr[x_] - d_) > disp12MaxDiff )
                        disp1ptr[x] = (DispType)INVALID_DISP_SCALED;
                }
            }

            // now shift the cyclic buffers
            std::swap( Lr[0], Lr[1] );
            std::swap( minLr[0], minLr[1] );
        }
    }
	//imshow("After static void computeDisparitySGBM",disp1);waitKey();
}

typedef cv::Point_<short> Point2s;

void StereoSGBM::operator ()( InputArray _left, InputArray _right,
                             OutputArray _disp )
{
	//cout<<"void StereoSGBM::operator ()"<<endl;
    Mat left = _left.getMat(), right = _right.getMat();
    CV_Assert( left.size() == right.size() && left.type() == right.type() &&
              left.depth() == DataType<PixType>::depth );

    _disp.create( left.size(), CV_16S );
    Mat disp = _disp.getMat();

    computeDisparitySGBM( left, right, disp, *this, buffer );
    medianBlur(disp, disp, 3);

    if( speckleWindowSize > 0 )
        filterSpeckles(disp, (minDisparity - 1)*DISP_SCALE, speckleWindowSize, DISP_SCALE*speckleRange, buffer);
	//imshow("After static void computeDisparitySGBM",disp);waitKey();
}


Rect getValidDisparityROI( Rect roi1, Rect roi2,
                          int minDisparity,
                          int numberOfDisparities,
                          int SADWindowSize )
{
	//cout<<"Rect getValidDisparityROI"<<endl;
    int SW2 = SADWindowSize/2;
    int minD = minDisparity, maxD = minDisparity + numberOfDisparities - 1;

    int xmin = max(roi1.x, roi2.x + maxD) + SW2;
    int xmax = min(roi1.x + roi1.width, roi2.x + roi2.width - minD) - SW2;
    int ymin = max(roi1.y, roi2.y) + SW2;
    int ymax = min(roi1.y + roi1.height, roi2.y + roi2.height) - SW2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);

    return r.width > 0 && r.height > 0 ? r : Rect();
}

}

namespace
{
	//cout<<"Check 4"<<endl;
    template <typename T>
    void filterSpecklesImpl(cv::Mat& img, int newVal, int maxSpeckleSize, int maxDiff, cv::Mat& _buf)
    {
    	//cout<<"void filterSpecklesImpl"<<endl;
        using namespace cv;

        int width = img.cols, height = img.rows, npixels = width*height;
        size_t bufSize = npixels*(int)(sizeof(Point2s) + sizeof(int) + sizeof(uchar));
        if( !_buf.isContinuous() || !_buf.data || _buf.cols*_buf.rows*_buf.elemSize() < bufSize )
            _buf.create(1, (int)bufSize, CV_8U);

        uchar* buf = _buf.data;
        int i, j, dstep = (int)(img.step/sizeof(T));
        int* labels = (int*)buf;
        buf += npixels*sizeof(labels[0]);
        Point2s* wbuf = (Point2s*)buf;
        buf += npixels*sizeof(wbuf[0]);
        uchar* rtype = (uchar*)buf;
        int curlabel = 0;

        // clear out label assignments
        memset(labels, 0, npixels*sizeof(labels[0]));

        for( i = 0; i < height; i++ )
        {
            T* ds = img.ptr<T>(i);
            int* ls = labels + width*i;

            for( j = 0; j < width; j++ )
            {
                if( ds[j] != newVal )   // not a bad disparity
                {
                    if( ls[j] )     // has a label, check for bad label
                    {
                        if( rtype[ls[j]] ) // small region, zero out disparity
                            ds[j] = (T)newVal;
                    }
                    // no label, assign and propagate
                    else
                    {
                        Point2s* ws = wbuf; // initialize wavefront
                        Point2s p((short)j, (short)i);  // current pixel
                        curlabel++; // next label
                        int count = 0;  // current region size
                        ls[j] = curlabel;

                        // wavefront propagation
                        while( ws >= wbuf ) // wavefront not empty
                        {
                            count++;
                            // put neighbors onto wavefront
                            T* dpp = &img.at<T>(p.y, p.x);
                            T dp = *dpp;
                            int* lpp = labels + width*p.y + p.x;

                            if( p.y < height-1 && !lpp[+width] && dpp[+dstep] != newVal && std::abs(dp - dpp[+dstep]) <= maxDiff )
                            {
                                lpp[+width] = curlabel;
                                *ws++ = Point2s(p.x, p.y+1);
                            }

                            if( p.y > 0 && !lpp[-width] && dpp[-dstep] != newVal && std::abs(dp - dpp[-dstep]) <= maxDiff )
                            {
                                lpp[-width] = curlabel;
                                *ws++ = Point2s(p.x, p.y-1);
                            }

                            if( p.x < width-1 && !lpp[+1] && dpp[+1] != newVal && std::abs(dp - dpp[+1]) <= maxDiff )
                            {
                                lpp[+1] = curlabel;
                                *ws++ = Point2s(p.x+1, p.y);
                            }

                            if( p.x > 0 && !lpp[-1] && dpp[-1] != newVal && std::abs(dp - dpp[-1]) <= maxDiff )
                            {
                                lpp[-1] = curlabel;
                                *ws++ = Point2s(p.x-1, p.y);
                            }

                            // pop most recent and propagate
                            // NB: could try least recent, maybe better convergence
                            p = *--ws;
                        }

                        // assign label type
                        if( count <= maxSpeckleSize )   // speckle region
                        {
                            rtype[ls[j]] = 1;   // small region label
                            ds[j] = (T)newVal;
                        }
                        else
                            rtype[ls[j]] = 0;   // large region label
                    }
                }
            }
        }
    }
}

void cv::filterSpeckles( InputOutputArray _img, double _newval, int maxSpeckleSize,
                         double _maxDiff, InputOutputArray __buf )
{
	//cout<<"void cv::filterSpeckles"<<endl;
    Mat img = _img.getMat();
    Mat temp, &_buf = __buf.needed() ? __buf.getMatRef() : temp;
    CV_Assert( img.type() == CV_8UC1 || img.type() == CV_16SC1 );

    int newVal = cvRound(_newval);
    int maxDiff = cvRound(_maxDiff);

    if (img.type() == CV_8UC1)
        filterSpecklesImpl<uchar>(img, newVal, maxSpeckleSize, maxDiff, _buf);
    else
        filterSpecklesImpl<short>(img, newVal, maxSpeckleSize, maxDiff, _buf);
}

void cv::validateDisparity( InputOutputArray _disp, InputArray _cost, int minDisparity,
                            int numberOfDisparities, int disp12MaxDiff )
{
	//cout<<"Check 7"<<endl;
    Mat disp = _disp.getMat(), cost = _cost.getMat();
    int cols = disp.cols, rows = disp.rows;
    int minD = minDisparity, maxD = minDisparity + numberOfDisparities;
    int x, minX1 = max(maxD, 0), maxX1 = cols + min(minD, 0);
    AutoBuffer<int> _disp2buf(cols*2);
    int* disp2buf = _disp2buf;
    int* disp2cost = disp2buf + cols;
    const int DISP_SHIFT = 4, DISP_SCALE = 1 << DISP_SHIFT;
    int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;
    int costType = cost.type();

    disp12MaxDiff *= DISP_SCALE;

    CV_Assert( numberOfDisparities > 0 && disp.type() == CV_16S &&
              (costType == CV_16S || costType == CV_32S) &&
              disp.size() == cost.size() );

    for( int y = 0; y < rows; y++ )
    {
        short* dptr = disp.ptr<short>(y);

        for( x = 0; x < cols; x++ )
        {
            disp2buf[x] = INVALID_DISP_SCALED;
            disp2cost[x] = INT_MAX;
        }

        if( costType == CV_16S )
        {
            const short* cptr = cost.ptr<short>(y);

            for( x = minX1; x < maxX1; x++ )
            {
                int d = dptr[x], c = cptr[x];
                int x2 = x - ((d + DISP_SCALE/2) >> DISP_SHIFT);

                if( disp2cost[x2] > c )
                {
                    disp2cost[x2] = c;
                    disp2buf[x2] = d;
                }
            }
        }
        else
        {
            const int* cptr = cost.ptr<int>(y);

            for( x = minX1; x < maxX1; x++ )
            {
                int d = dptr[x], c = cptr[x];
                int x2 = x - ((d + DISP_SCALE/2) >> DISP_SHIFT);

                if( disp2cost[x2] < c )
                {
                    disp2cost[x2] = c;
                    disp2buf[x2] = d;
                }
            }
        }

        for( x = minX1; x < maxX1; x++ )
        {
            // we round the computed disparity both towards -inf and +inf and check
            // if either of the corresponding disparities in disp2 is consistent.
            // This is to give the computed disparity a chance to look valid if it is.
            int d = dptr[x];
            if( d == INVALID_DISP_SCALED )
                continue;
            int d0 = d >> DISP_SHIFT;
            int d1 = (d + DISP_SCALE-1) >> DISP_SHIFT;
            int x0 = x - d0, x1 = x - d1;
            if( (0 <= x0 && x0 < cols && disp2buf[x0] > INVALID_DISP_SCALED && std::abs(disp2buf[x0] - d) > disp12MaxDiff) &&
                (0 <= x1 && x1 < cols && disp2buf[x1] > INVALID_DISP_SCALED && std::abs(disp2buf[x1] - d) > disp12MaxDiff) )
                dptr[x] = (short)INVALID_DISP_SCALED;
        }
    }
}

CvRect cvGetValidDisparityROI( CvRect roi1, CvRect roi2, int minDisparity,
                               int numberOfDisparities, int SADWindowSize )
{
	//cout<<"Check 8"<<endl;
    return (CvRect)cv::getValidDisparityROI( roi1, roi2, minDisparity,
                                             numberOfDisparities, SADWindowSize );
}

void cvValidateDisparity( CvArr* _disp, const CvArr* _cost, int minDisparity,
                          int numberOfDisparities, int disp12MaxDiff )
{
	//cout<<"Check 9"<<endl;
    cv::Mat disp = cv::cvarrToMat(_disp), cost = cv::cvarrToMat(_cost);
    cv::validateDisparity( disp, cost, minDisparity, numberOfDisparities, disp12MaxDiff );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CV_IMPL CvStereoBMState* cvCreateStereoBMState( int /*preset*/, int numberOfDisparities )
{
    CvStereoBMState* state = (CvStereoBMState*)cvAlloc( sizeof(*state) );
    if( !state )
        return 0;

    state->preFilterType = CV_STEREO_BM_XSOBEL; //CV_STEREO_BM_NORMALIZED_RESPONSE;
    state->preFilterSize = 9;
    state->preFilterCap = 31;
    state->SADWindowSize = 15; //15
    state->minDisparity = 0;
    state->numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : 64;
    state->textureThreshold = 10;
    state->uniquenessRatio = 15;
    state->speckleRange = state->speckleWindowSize = 0;
    state->trySmallerWindows = 0;
    state->roi1 = state->roi2 = cvRect(0,0,0,0);
    state->disp12MaxDiff = -1;

    state->preFilteredImg0 = state->preFilteredImg1 = state->slidingSumBuf =
    state->disp = state->cost = 0;

    return state;
}

CV_IMPL void cvReleaseStereoBMState( CvStereoBMState** state )
{
    if( !state )
        CV_Error( CV_StsNullPtr, "" );

    if( !*state )
        return;

    cvReleaseMat( &(*state)->preFilteredImg0 );
    cvReleaseMat( &(*state)->preFilteredImg1 );
    cvReleaseMat( &(*state)->slidingSumBuf );
    cvReleaseMat( &(*state)->disp );
    cvReleaseMat( &(*state)->cost );
    cvFree( state );
}

namespace cv
{

static void prefilterNorm( const Mat& src, Mat& dst, int winsize, int ftzero, uchar* buf )
{
    int x, y, wsz2 = winsize/2;
    int* vsum = (int*)alignPtr(buf + (wsz2 + 1)*sizeof(vsum[0]), 32);
    int scale_g = winsize*winsize/8, scale_s = (1024 + scale_g)/(scale_g*2);
    const int OFS = 256*5, TABSZ = OFS*2 + 256;
    uchar tab[TABSZ];
    const uchar* sptr = src.data;
    int srcstep = (int)src.step;
    Size size = src.size();

    scale_g *= scale_s;

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero*2 : x - OFS + ftzero);

    for( x = 0; x < size.width; x++ )
        vsum[x] = (ushort)(sptr[x]*(wsz2 + 2));

    for( y = 1; y < wsz2; y++ )
    {
        for( x = 0; x < size.width; x++ )
            vsum[x] = (ushort)(vsum[x] + sptr[srcstep*y + x]);
    }

    for( y = 0; y < size.height; y++ )
    {
        const uchar* top = sptr + srcstep*MAX(y-wsz2-1,0);
        const uchar* bottom = sptr + srcstep*MIN(y+wsz2,size.height-1);
        const uchar* prev = sptr + srcstep*MAX(y-1,0);
        const uchar* curr = sptr + srcstep*y;
        const uchar* next = sptr + srcstep*MIN(y+1,size.height-1);
        uchar* dptr = dst.ptr<uchar>(y);
        x = 0;

        for( ; x < size.width; x++ )
            vsum[x] = (ushort)(vsum[x] + bottom[x] - top[x]);

        for( x = 0; x <= wsz2; x++ )
        {
            vsum[-x-1] = vsum[0];
            vsum[size.width+x] = vsum[size.width-1];
        }

        int sum = vsum[0]*(wsz2 + 1);
        for( x = 1; x <= wsz2; x++ )
            sum += vsum[x];

        int val = ((curr[0]*5 + curr[1] + prev[0] + next[0])*scale_g - sum*scale_s) >> 10;
        dptr[0] = tab[val + OFS];

        for( x = 1; x < size.width-1; x++ )
        {
            sum += vsum[x+wsz2] - vsum[x-wsz2-1];
            val = ((curr[x]*4 + curr[x-1] + curr[x+1] + prev[x] + next[x])*scale_g - sum*scale_s) >> 10;
            dptr[x] = tab[val + OFS];
        }

        sum += vsum[x+wsz2] - vsum[x-wsz2-1];
        val = ((curr[x]*5 + curr[x-1] + prev[x] + next[x])*scale_g - sum*scale_s) >> 10;
        dptr[x] = tab[val + OFS];
    }
}


static void
prefilterXSobel( const Mat& src, Mat& dst, int ftzero )
{
    int x, y;
    const int OFS = 256*4, TABSZ = OFS*2 + 256;
    uchar tab[TABSZ];
    Size size = src.size();

    for( x = 0; x < TABSZ; x++ )
        tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero*2 : x - OFS + ftzero);
    uchar val0 = tab[0 + OFS];

    for( y = 0; y < size.height-1; y += 2 )
    {
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
        const uchar* srow2 = y < size.height-1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
        const uchar* srow3 = y < size.height-2 ? srow1 + src.step*2 : srow1;
        uchar* dptr0 = dst.ptr<uchar>(y);
        uchar* dptr1 = dptr0 + dst.step;

        dptr0[0] = dptr0[size.width-1] = dptr1[0] = dptr1[size.width-1] = val0;
        x = 1;

        for( ; x < size.width-1; x++ )
        {
            int d0 = srow0[x+1] - srow0[x-1], d1 = srow1[x+1] - srow1[x-1],
                d2 = srow2[x+1] - srow2[x-1], d3 = srow3[x+1] - srow3[x-1];
            int v0 = tab[d0 + d1*2 + d2 + OFS];
            int v1 = tab[d1 + d2*2 + d3 + OFS];
            dptr0[x] = (uchar)v0;
            dptr1[x] = (uchar)v1;
        }
    }

    for( ; y < size.height; y++ )
    {
        uchar* dptr = dst.ptr<uchar>(y);
        x = 0;

        for(; x < size.width; x++ )
            dptr[x] = val0;
    }
}

static const int DISPARITY_SHIFT = 4;
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------------THE MAGIC HAPPENS HERE---------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
static void
findStereoCorrespondenceBM( const Mat& left, const Mat& right,
                            Mat& disp, Mat& cost, const CvStereoBMState& state,
                            uchar* buf, int _dy0, int _dy1 )
{
    const int ALIGN = 16;
    int x, y, d;
    int wsz = state.SADWindowSize, wsz2 = wsz/2;             // wsz = 9;   wsz2 = 4;    <-------------------------------------------------- SADWindow
    int dy0 = MIN(_dy0, wsz2+1), dy1 = MIN(_dy1, wsz2+1);    // dy0 = 5; dy1 = 5;
    int ndisp = state.numberOfDisparities;                   // ndisp = 240;
    int mindisp = state.minDisparity;                        // mindisp = 0;
    int lofs = MAX(ndisp - 1 + mindisp, 0);                  // lofs = 239;
    int rofs = -MIN(ndisp - 1 + mindisp, 0);                 // rofs = 0;
    int width = left.cols, height = left.rows;				 // width = 1920; height = 78;
    //cout<<"height: "<<height<<endl;
    int width1 = width - rofs - ndisp + 1;					 // width1 = 1681;
    int ftzero = state.preFilterCap;                         // ftzero = 31;
    int textureThreshold = state.textureThreshold;           // textureThreshold = 10;
    int uniquenessRatio = state.uniquenessRatio;             // uniquenessRation = 15;
    short FILTERED = (short)((mindisp - 1) << DISPARITY_SHIFT); // FILTERED = -16;

    int *sad, *hsad0, *hsad, *hsad_sub, *htext;
    uchar *cbuf0, *cbuf;
    const uchar* lptr0 = left.data + lofs;
    const uchar* rptr0 = right.data + rofs;
    const uchar *lptr, *lptr_sub, *rptr;
    //pointer to the disparity data
    short* dptr = (short*)disp.data;
    // the step used to compute each value of Mat
    int sstep = (int)left.step;                              // sstep = 1920;
    int dstep = (int)(disp.step/sizeof(dptr[0]));            // dstep = 1920
    int cstep = (height+dy0+dy1)*ndisp;                      // cstep = 19680;
    int costbuf = 0;
    int coststep = cost.data ? (int)(cost.step/sizeof(costbuf)) : 0; // coststep = 19680;
    const int TABSZ = 256;
    uchar tab[TABSZ];

    sad = (int*)alignPtr(buf + sizeof(sad[0]), ALIGN); // (12,ALIGN)
    hsad0 = (int*)alignPtr(sad + ndisp + 1 + dy0*ndisp, ALIGN); //(1445,ALIGN)
    htext = (int*)alignPtr((int*)(hsad0 + (height+dy1)*ndisp) + wsz2 + 2, ALIGN); //(19930,ALIGN)
    cbuf0 = (uchar*)alignPtr((uchar*)(htext + height + wsz2 + 2) + dy0*ndisp, ALIGN); //(88,ALIGN);


    for( x = 0; x < TABSZ; x++ ) //TABSZ=256
        tab[x] = (uchar)std::abs(x - ftzero); //ftzero=31 tab goes from 31 to 225

    memset( hsad0 - dy0*ndisp, 0, (height + dy0 + dy1)*ndisp*sizeof(hsad0[0]) ); //   ( ADDRESS, 0, 83520)
    memset( htext - wsz2 - 1, 0, (height + wsz + 1)*sizeof(htext[0]) );          //   ( ADDRESS, 0, 348)

    for( x = -wsz2-1; x < wsz2; x++ ) // from -5 to 4 <---- the S.A.D window size
    {
        hsad = hsad0 - dy0*ndisp; 												// hsad0+1200
        cbuf = cbuf0 + (x + wsz2 + 1)*cstep - dy0*ndisp;						// cbuf = cbuf0+(x+5)*19680-1200
        lptr = lptr0 + std::min(std::max(x, -lofs), width-lofs-1) - dy0*sstep;  // lptr0+x-9600=left.data+x+239-9600
        rptr = rptr0 + std::min(std::max(x, -rofs), width-rofs-1) - dy0*sstep;  // rptr0+x-9600=right.data+x-9600

        // from -4 to 82, hsad+240; cbuf+240; lptr+1920; rptr+1920
        for( y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep )
        {
            int lval = lptr[0]; //

            for( d = 0; d < ndisp; d++ )
            {
                int diff = std::abs(lval - rptr[d]); //difference between left pixel and right pixel
                cbuf[d] = (uchar)diff;  //this is the cost buffer
                hsad[d] = (int)(hsad[d] + diff);
            }
            htext[y] += tab[lval];

        }
    }

    for( y = 0; y < height; y++ )
    {
        for( x = 0; x < lofs; x++ )
            {
        	dptr[y*dstep + x] = FILTERED;

            }

        for( x = lofs + width1; x < width; x++ )
            dptr[y*dstep + x] = FILTERED;
    }
    dptr += lofs;

    for( x = 0; x < width1; x++, dptr++ )
    {
        int* costptr = cost.data ? (int*)cost.data + lofs + x : &costbuf;
        int x0 = x - wsz2 - 1, x1 = x + wsz2;

        const uchar* cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1))*cstep - dy0*ndisp;
        hsad = hsad0 - dy0*ndisp;
        lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width-1-lofs) - dy0*sstep;
        lptr = lptr0 + MIN(MAX(x1, -lofs), width-1-lofs) - dy0*sstep;
        rptr = rptr0 + MIN(MAX(x1, -rofs), width-1-rofs) - dy0*sstep;

        for( y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp, hsad += ndisp, lptr += sstep, lptr_sub += sstep, rptr += sstep )
        {
            int lval = lptr[0];
            for( d = 0; d < ndisp; d++ )
            {
                int diff = std::abs(lval - rptr[d]);
                cbuf[d] = (uchar)diff;
                hsad[d] = hsad[d] + diff - cbuf_sub[d];
            }
            htext[y] += tab[lval] - tab[lptr_sub[0]];
        }

        for( y = dy1; y <= wsz2; y++ )
        {
            htext[height+y] = htext[height+dy1-1];
        }

        for( y = -wsz2-1; y < -dy0; y++ )
        {
        	htext[y] = htext[-dy0];
        }

        for( d = 0; d < ndisp; d++ )
            sad[d] = (int)(hsad0[d-ndisp*dy0]*(wsz2 + 2 - dy0)); //(hsad0[d-1200])

        hsad = hsad0 + (1 - dy0)*ndisp; //  hsad0-960
        for( y = 1 - dy0; y < wsz2; y++, hsad += ndisp )
        {
            for( d = 0; d < ndisp; d++ )
                sad[d] = (int)(sad[d] + hsad[d]);
        }
        int tsum = 0;
        for( y = -wsz2-1; y < wsz2; y++ )
            tsum += htext[y];

        for( y = 0; y < height; y++ )
        {
            int minsad = INT_MAX, mind = -1;
            hsad = hsad0 + MIN(y + wsz2, height+dy1-1)*ndisp;
            hsad_sub = hsad0 + MAX(y - wsz2 - 1, -dy0)*ndisp;

            int lidar_D = DISP.at<int16_t>(y, x);

        	//// Draw a circle
            /*
        	cv::circle( LEFT, cv::Point( x,y ), 10.0, cv::Scalar( 0, 0, 255 ), 1, 8 );
        	imshow("Left",LEFT);

        	cv::circle( RIGHT, cv::Point( x,y ), 10.0, cv::Scalar(255, 56, 0 ), 1, 8 );
        	imshow("Right",RIGHT);

    		cv::Mat temp;
    		cv::normalize(DISP,temp,0,255,cv::NORM_MINMAX,CV_8U);

        	cv::circle( temp, cv::Point( x,y), 10.0, cv::Scalar( 255, 128, 0 ), 1, 8 );
        	imshow("Disp",temp);
        	cv::waitKey(0);*/
        	//cout<<"y:"<<y<<"  x:"<<x<<endl;


            int min_D = 0;
            int max_D = ndisp;
            if(lidar_D!=0)
            {
            	min_D = ndisp-lidar_D-INTERVAL;
            	max_D = ndisp-lidar_D+INTERVAL;
            }

            for( d = 0; d < ndisp; d++ )
            {
                int currsad = sad[d] + hsad[d] - hsad_sub[d];

                sad[d] = currsad;
                if(d>min_D && d<max_D && currsad < minsad ) //d>min_D && d<max_D &&
                {
                    minsad = currsad;
                    mind = d;
                }
            }

            min_D = 0;
            max_D = ndisp;
            //minsad=0;
            //mind=lidar_D;

            tsum += htext[y + wsz2] - htext[y - wsz2 - 1];
            if( tsum < textureThreshold )
            {
                dptr[y*dstep] = FILTERED;
                continue;
            }
            /*
            if( uniquenessRatio > 0 )
            {
                int thresh = minsad + (minsad * uniquenessRatio/100);
                for( d = 0; d < ndisp; d++ )
                {
                    if( sad[d] <= thresh && (d < mind-1 || d > mind+1))
                        break;
                }
                if( d < ndisp )
                {
                    dptr[y*dstep] = FILTERED;
                    continue;
                }
            }  */

            {

            sad[-1] = sad[1];
            sad[ndisp] = sad[ndisp-2];
            int p = sad[mind+1], n = sad[mind-1];
            d = p + n - 2*sad[mind] + std::abs(p - n); //934-816+114=232
            dptr[y*dstep] = (short)(((ndisp - mind - 1 + mindisp)*256 + (d != 0 ? (p-n)*256/d : 0) + 15) >> 4);
            int value=(short)((ndisp - mind - 1 + mindisp)*256 + (d != 0 ? (p-n)*256/d : 0) + 15);
            costptr[y*coststep] = sad[mind];
            //flag++;
            /*if(flag%2000==0)
            {
            	disp.convertTo(disp8, CV_32F);
				cv::normalize(disp8, disp8, 1, 0, CV_MINMAX);
				imshow("disp",disp8);waitKey();
            }*/
            }
        }
    }
    Mat temp;
	disp.convertTo(temp, CV_32F);
	cv::normalize(temp, temp, 1, 0, CV_MINMAX);
	//imshow("After BM",temp);waitKey();

}
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------THE MAGIC ENDS HERE---------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------------------------------------------//


struct PrefilterInvoker
{
    PrefilterInvoker(const Mat& left0, const Mat& right0, Mat& left, Mat& right,
                     uchar* buf0, uchar* buf1, CvStereoBMState* _state )
    {
        imgs0[0] = &left0; imgs0[1] = &right0;
        imgs[0] = &left; imgs[1] = &right;
        buf[0] = buf0; buf[1] = buf1;
        state = _state;
    }

    void operator()( int ind ) const
    {
        if( state->preFilterType == CV_STEREO_BM_NORMALIZED_RESPONSE )
        {
        	//printf("Before call for prefilterNorm \n");
            prefilterNorm( *imgs0[ind], *imgs[ind], state->preFilterSize, state->preFilterCap, buf[ind] );
            //printf("After call for prefilterNorm \n\n");
        }
        else
        {
        	//printf("Before call for prefilterXSobel \n");
            prefilterXSobel( *imgs0[ind], *imgs[ind], state->preFilterCap );
            //printf("After call for prefilterXSobel \n\n");
        }
    }

    const Mat* imgs0[2];
    Mat* imgs[2];
    uchar* buf[2];
    CvStereoBMState *state;
};


struct FindStereoCorrespInvoker : ParallelLoopBody
{
    FindStereoCorrespInvoker( const Mat& _left, const Mat& _right,
                              Mat& _disp, CvStereoBMState* _state,
                              int _nstripes, int _stripeBufSize,
                              bool _useShorts, Rect _validDisparityRect )
    {
        left = &_left; right = &_right;
        disp = &_disp; state = _state;
        nstripes = _nstripes; stripeBufSize = _stripeBufSize;
        useShorts = _useShorts;
        validDisparityRect = _validDisparityRect;
    }

    void operator()( const Range& range ) const
    {
        int cols = left->cols, rows = left->rows;
        int _row0 = min(cvRound(range.start * rows / nstripes), rows);
        int _row1 = min(cvRound(range.end * rows / nstripes), rows);
        uchar *ptr = state->slidingSumBuf->data.ptr + range.start * stripeBufSize;
        int FILTERED = (state->minDisparity - 1)*16;

        Rect roi = validDisparityRect & Rect(0, _row0, cols, _row1 - _row0);
        if( roi.height == 0 )
            return;
        int row0 = roi.y;
        int row1 = roi.y + roi.height;

        Mat part;
        if( row0 > _row0 )
        {
            part = disp->rowRange(_row0, row0);
            part = Scalar::all(FILTERED);
        }
        if( _row1 > row1 )
        {
            part = disp->rowRange(row1, _row1);
            part = Scalar::all(FILTERED);
        }

        Mat left_i = left->rowRange(row0, row1);
        Mat right_i = right->rowRange(row0, row1);
        Mat disp_i = disp->rowRange(row0, row1);
        Mat cost_i = state->disp12MaxDiff >= 0 ? Mat(state->cost).rowRange(row0, row1) : Mat();
        Mat temp;


        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //printf("Before FIRST \n");
        findStereoCorrespondenceBM( left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1 );
        // printf("After FIRST \n");
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    	/*disp_i.convertTo(temp, CV_32F);
		cv::normalize(temp, temp, 1, 0, CV_MINMAX);
		imshow("After CorrespondenceBM",temp);waitKey(); */

        if( state->disp12MaxDiff >= 0 )
            validateDisparity( disp_i, cost_i, state->minDisparity, state->numberOfDisparities, state->disp12MaxDiff );

        if( roi.x > 0 )
        {
            part = disp_i.colRange(0, roi.x);
            part = Scalar::all(FILTERED);
        }
        if( roi.x + roi.width < cols )
        {
            part = disp_i.colRange(roi.x + roi.width, cols);
            part = Scalar::all(FILTERED);
        }
    	/*disp_i.convertTo(temp, CV_32F);
		cv::normalize(temp, temp, 1, 0, CV_MINMAX);
		imshow("After validateDisparity",temp);waitKey(); */
    }

protected:
    const Mat *left, *right;
    Mat* disp;
    CvStereoBMState *state;

    int nstripes;
    int stripeBufSize;
    bool useShorts;
    Rect validDisparityRect;
};

static void findStereoCorrespondenceBM( const Mat& left0, const Mat& right0, Mat& disp0, CvStereoBMState* state)
{
    if (left0.size() != right0.size() || disp0.size() != left0.size())
        CV_Error( CV_StsUnmatchedSizes, "All the images must have the same size" );

    if (left0.type() != CV_8UC1 || right0.type() != CV_8UC1)
        CV_Error( CV_StsUnsupportedFormat, "Both input images must have CV_8UC1" );

    if (disp0.type() != CV_16SC1 && disp0.type() != CV_32FC1)
        CV_Error( CV_StsUnsupportedFormat, "Disparity image must have CV_16SC1 or CV_32FC1 format" );

    if( !state )
        CV_Error( CV_StsNullPtr, "Stereo BM state is NULL." );

    if( state->preFilterType != CV_STEREO_BM_NORMALIZED_RESPONSE && state->preFilterType != CV_STEREO_BM_XSOBEL )
        CV_Error( CV_StsOutOfRange, "preFilterType must be = CV_STEREO_BM_NORMALIZED_RESPONSE" );

    if( state->preFilterSize < 5 || state->preFilterSize > 255 || state->preFilterSize % 2 == 0 )
        CV_Error( CV_StsOutOfRange, "preFilterSize must be odd and be within 5..255" );

    if( state->preFilterCap < 1 || state->preFilterCap > 63 )
        CV_Error( CV_StsOutOfRange, "preFilterCap must be within 1..63" );

    if( state->SADWindowSize < 5 || state->SADWindowSize > 255 || state->SADWindowSize % 2 == 0 ||
        state->SADWindowSize >= min(left0.cols, left0.rows) )
        CV_Error( CV_StsOutOfRange, "SADWindowSize must be odd, be within 5..255 and be not larger than image width or height" );

    if( state->numberOfDisparities <= 0 || state->numberOfDisparities % 16 != 0 )
        CV_Error( CV_StsOutOfRange, "numberOfDisparities must be positive and divisble by 16" );

    if( state->textureThreshold < 0 )
        CV_Error( CV_StsOutOfRange, "texture threshold must be non-negative" );

    if( state->uniquenessRatio < 0 )
        CV_Error( CV_StsOutOfRange, "uniqueness ratio must be non-negative" );

    if( !state->preFilteredImg0 || state->preFilteredImg0->cols * state->preFilteredImg0->rows < left0.cols * left0.rows )
    {
        cvReleaseMat( &state->preFilteredImg0 );
        cvReleaseMat( &state->preFilteredImg1 );
        cvReleaseMat( &state->cost );

        state->preFilteredImg0 = cvCreateMat( left0.rows, left0.cols, CV_8U );
        state->preFilteredImg1 = cvCreateMat( left0.rows, left0.cols, CV_8U );
        state->cost = cvCreateMat( left0.rows, left0.cols, CV_16S );
    }
    Mat left(left0.size(), CV_8U, state->preFilteredImg0->data.ptr);
    Mat right(right0.size(), CV_8U, state->preFilteredImg1->data.ptr);

    int mindisp = state->minDisparity;
    int ndisp = state->numberOfDisparities;

    int width = left0.cols;
    int height = left0.rows;
    int lofs = max(ndisp - 1 + mindisp, 0);
    int rofs = -min(ndisp - 1 + mindisp, 0);
    int width1 = width - rofs - ndisp + 1;
    int FILTERED = (state->minDisparity - 1) << DISPARITY_SHIFT;

    if( lofs >= width || rofs >= width || width1 < 1 )
    {
        disp0 = Scalar::all( FILTERED * ( disp0.type() < CV_32F ? 1 : 1./(1 << DISPARITY_SHIFT) ) );
        return;
    }

    //cout<<"Disp0: \n"<<disp0<<endl;

    Mat disp = disp0;
    if( disp0.type() == CV_32F)
    {
        if( !state->disp || state->disp->rows != disp0.rows || state->disp->cols != disp0.cols )
        {
            cvReleaseMat( &state->disp );
            state->disp = cvCreateMat(disp0.rows, disp0.cols, CV_16S);
        }
        disp = cv::cvarrToMat(state->disp);
    }

    //cout<<"Disp: \n"<<disp<<endl;

    int wsz = state->SADWindowSize;
    int bufSize0 = (int)((ndisp + 2)*sizeof(int));
    bufSize0 += (int)((height+wsz+2)*ndisp*sizeof(int));
    bufSize0 += (int)((height + wsz + 2)*sizeof(int));
    bufSize0 += (int)((height+wsz+2)*ndisp*(wsz+2)*sizeof(uchar) + 256);

    int bufSize1 = (int)((width + state->preFilterSize + 2) * sizeof(int) + 256);
    int bufSize2 = 0;
    if( state->speckleRange >= 0 && state->speckleWindowSize > 0 )
        bufSize2 = width*height*(sizeof(cv::Point_<short>) + sizeof(int) + sizeof(uchar));

    bool useShorts = state->preFilterCap <= 31 && state->SADWindowSize <= 21 && checkHardwareSupport(CV_CPU_SSE2);


    const double SAD_overhead_coeff = 10.0;
    double N0 = 8000000 / (useShorts ? 1 : 4);  // approx tbb's min number instructions reasonable for one thread
    double maxStripeSize = min(max(N0 / (width * ndisp), (wsz-1) * SAD_overhead_coeff), (double)height);
    int nstripes = cvCeil(height / maxStripeSize);

    int bufSize = max(bufSize0 * nstripes, max(bufSize1 * 2, bufSize2));

    if( !state->slidingSumBuf || state->slidingSumBuf->cols < bufSize )
    {
        cvReleaseMat( &state->slidingSumBuf );
        state->slidingSumBuf = cvCreateMat( 1, bufSize, CV_8U );
    }

    uchar *_buf = state->slidingSumBuf->data.ptr;
    int idx[] = {0,1};
    parallel_do(idx, idx+2, PrefilterInvoker(left0, right0, left, right, _buf, _buf + bufSize1, state));
    // printf("After call for PrefilterInvoker \n\n ");
    Rect validDisparityRect(0, 0, width, height), R1 = state->roi1, R2 = state->roi2;
    validDisparityRect = getValidDisparityROI(R1.area() > 0 ? Rect(0, 0, width, height) : validDisparityRect,
                                              R2.area() > 0 ? Rect(0, 0, width, height) : validDisparityRect,
                                              state->minDisparity, state->numberOfDisparities,
                                              state->SADWindowSize);

    //printf("Before call for FindStereoCorrespInvoker \n ");
    parallel_for_(Range(0, nstripes),
                  FindStereoCorrespInvoker(left, right, disp, state, nstripes,
                                           bufSize0, useShorts, validDisparityRect));
    //printf("After call for FindStereoCorrespInvoker \n ");

    //-------------------------------------------------------------------------------------------------------------------------------------------------------

    Mat temp;
    /*
	disp.convertTo(temp, CV_32F);
	cv::normalize(temp, temp, 1, 0, CV_MINMAX);
	imshow("After Invoker",temp);waitKey();*/
    if( state->speckleRange >= 0 && state->speckleWindowSize > 0 )
    {
        Mat buf(state->slidingSumBuf);
        filterSpeckles(disp, FILTERED, state->speckleWindowSize, state->speckleRange, buf);
    }

    if (disp0.data != disp.data)
        disp.convertTo(disp0, disp0.type(), 1./(1 << DISPARITY_SHIFT), 0);

	disp.convertTo(temp, CV_32F);
	cv::normalize(temp, temp, 1, 0, CV_MINMAX);
	//imshow("After Spekle",temp);waitKey();
}

StereoBM::StereoBM()
{	state = cvCreateStereoBMState(); }

StereoBM::StereoBM(int _preset, int _ndisparities, int _SADWindowSize)
{ 	init(_preset, _ndisparities, _SADWindowSize); }

void StereoBM::init(int _preset, int _ndisparities, int _SADWindowSize)
{
    state = cvCreateStereoBMState(_preset, _ndisparities);
    state->SADWindowSize = _SADWindowSize;
}

void StereoBM::operator()( InputArray _left, InputArray _right,
                           OutputArray _disparity, int disptype )
{
	//printf("In operator. \n");
    Mat left = _left.getMat(), right = _right.getMat();
    CV_Assert( disptype == CV_16S || disptype == CV_32F );
    _disparity.create(left.size(), disptype);
    Mat disparity = _disparity.getMat();
	//printf("Before call for findStereoCorrespondenceBM - SECOND \n");
    findStereoCorrespondenceBM(left, right, disparity, state);
    //printf("After call for findStereoCorrespondenceBM - SECOND \n\n");
}

template<> void Ptr<CvStereoBMState>::delete_obj()
{ cvReleaseStereoBMState(&obj); }

}

CV_IMPL void cvFindStereoCorrespondenceBM( const CvArr* leftarr, const CvArr* rightarr,
                                           CvArr* disparr, CvStereoBMState* state )
{
    cv::Mat left = cv::cvarrToMat(leftarr),
        right = cv::cvarrToMat(rightarr),
        disp = cv::cvarrToMat(disparr);
    cv::findStereoCorrespondenceBM(left, right, disp, state);
}


#endif /* LIDAR_STEREO_SGBM_H_ */
