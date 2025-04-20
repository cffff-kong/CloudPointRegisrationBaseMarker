/*****************************************************************//**
 * @file   SSLReconstruction.h
 * @brief  Single Structured Light (SSL) Reconstruction
 *         用来实现基于多频外差编码的单目三维重建，生成点云
 * @author DC Kong
 * @date   January 2025
 *********************************************************************/
#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <boost/thread/thread.hpp>
#include <cmath>
using namespace std;
class SSLReconstruction
{
private:
	int m_ps_num;
	int m_width;
	int m_height;
	int* m_frequency;
	string m_path;
	vector<cv::Mat> m_imgs;
	cv::Mat m_wrapped_phase1;
	cv::Mat m_wrapped_phase2;
	cv::Mat m_wrapped_phase3;
	cv::Mat m_B; // 调制度
	cv::Mat m_phase;  //绝对相位
	int m_T13;
	int m_T23;
	int m_T123;
	cv::Mat m_phase13;
	cv::Mat m_phase23;
	cv::Mat m_phase123;
	cv::Mat m_phase_abs; // 相位绝对值
	cv::Mat m_B_mask;  //调制度掩码
	cv::Mat m_k_cam;  //相机内参
	cv::Mat m_k_dlp;  //DLP内参
	cv::Mat m_rt_cam;  //相机变换矩阵
	cv::Mat m_rt_dlp;  //DLP变换矩阵
	/*MRF参数*/
	float m_omega[2] = { 0.5664,0.4535 };

	float m_lambda1 = 0.3;
	float m_lambda2 = 0.05;
	float m_theta_phase = 1;
	int m_num_of_MRF = 10;
public:
	vector <cv::Point2f> m_centers; //标识点中心
public:
	/**
	 * @brief 构造函数
	 * @param ps_num       相移步数
	 * @param img_width    图像宽度
	 * @param img_height   图像高度
	 * @param T1           频率1
	 * @param T2           频率2
	 * @param T3           频率3
	 * @param path         图像路径
	 */
	SSLReconstruction(int ps_num, int img_width, int img_height, int T1, int T2, int T3, string path);


	/**
	 * @brief 重建接口
	 * @brief 重建接口
	 * @return 点云
	 */
	pcl::PointCloud<pcl::PointXYZ>::Ptr Reconstruction();

	/**
	* @brief 多频外差解码接口
	* filter 是否滤波
	* @return 解码后的图像
	*/
	cv::Mat Decode(bool filter = true);

	/**
	 * @brief 求解标识点中心
	 * @return 
	 */
	void FindCenters(cv::Mat& img);

	/**
	 * @brief 计算标识点三维坐标
	 */
	void Find3DPoints();
private:
	/**
	 * @bnrief 加载图像
	 */
	void loadImg();

	/**
	 * @brief 双频外差
	 * @param index  图像索引，用于选择频率
	 * @return       解码后的图像
	 */
	cv::Mat CalWrappedPhase(int index);

	/**
	 * @brief 双频外差
	 * @param phase1  频率1的相位
	 * @param T1      频率1
	 * @param phase2  频率2的相位
	 * @param T2      频率2
	 * @return        解码后的图像
	 */
	cv::Mat HeterodynePhase(cv::Mat phase1, int T1, cv::Mat phase2, int T2);

	/**
	 * @brief 最终的相位计算
	 */
	void CombinePhase();

	/**
	 * @brief 相位滤波,matlab那个，主要用的是每个像素的极值;适当调节max的阈值，效果极佳
	 */
	void FilterPhase();

	/**
	 * @brief 保存图像到csv文件
	 * @param mat        输入图像
	 * @param filename   文件名
	 */
	void saveToCSV(const cv::Mat& mat, const std::string& filename);

	/**
	 * @brief 单调性滤波
	 * @return 滤波掩码
	 */
	cv::Mat MonoFilter();

	/**
	 * @brief matlab里那个单调性和连续性滤波
	 */
	void Mono();



	/**
	 * @brief 滤波方法 目前只有离群点滤波
	 * @param cloud  输入点云
	 * @param cloud_filtered 滤波后点云
	 */
	void CloudPointFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);

	/**
	 * @brief 基于MRF的滤波
	 * @param img  输入图像
	 * @param img_filtered  输出mask
	 */
	void MRF(cv::Mat& img, cv::Mat& mask);

	/**
	 * @Kronecker 计算Kronecker积,相同返回1，不同返回0
	 * @param
	 * @param
	 * @return
	 */
	float Kronecker(float a, float b);

	float u1(float pha);

	float b1(int omega_index, float pha1, float pha2);

	float b2(int omega_index, float mask1, float mask2);

	void SavePointsToTXT(const std::vector<cv::Point3f>& points3D, const std::string& filename);
};


