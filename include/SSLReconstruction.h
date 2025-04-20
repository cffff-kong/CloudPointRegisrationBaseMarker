/*****************************************************************//**
 * @file   SSLReconstruction.h
 * @brief  Single Structured Light (SSL) Reconstruction
 *         ����ʵ�ֻ��ڶ�Ƶ������ĵ�Ŀ��ά�ؽ������ɵ���
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
	cv::Mat m_B; // ���ƶ�
	cv::Mat m_phase;  //������λ
	int m_T13;
	int m_T23;
	int m_T123;
	cv::Mat m_phase13;
	cv::Mat m_phase23;
	cv::Mat m_phase123;
	cv::Mat m_phase_abs; // ��λ����ֵ
	cv::Mat m_B_mask;  //���ƶ�����
	cv::Mat m_k_cam;  //����ڲ�
	cv::Mat m_k_dlp;  //DLP�ڲ�
	cv::Mat m_rt_cam;  //����任����
	cv::Mat m_rt_dlp;  //DLP�任����
	/*MRF����*/
	float m_omega[2] = { 0.5664,0.4535 };

	float m_lambda1 = 0.3;
	float m_lambda2 = 0.05;
	float m_theta_phase = 1;
	int m_num_of_MRF = 10;
public:
	vector <cv::Point2f> m_centers; //��ʶ������
public:
	/**
	 * @brief ���캯��
	 * @param ps_num       ���Ʋ���
	 * @param img_width    ͼ����
	 * @param img_height   ͼ��߶�
	 * @param T1           Ƶ��1
	 * @param T2           Ƶ��2
	 * @param T3           Ƶ��3
	 * @param path         ͼ��·��
	 */
	SSLReconstruction(int ps_num, int img_width, int img_height, int T1, int T2, int T3, string path);


	/**
	 * @brief �ؽ��ӿ�
	 * @brief �ؽ��ӿ�
	 * @return ����
	 */
	pcl::PointCloud<pcl::PointXYZ>::Ptr Reconstruction();

	/**
	* @brief ��Ƶ������ӿ�
	* filter �Ƿ��˲�
	* @return ������ͼ��
	*/
	cv::Mat Decode(bool filter = true);

	/**
	 * @brief ����ʶ������
	 * @return 
	 */
	void FindCenters(cv::Mat& img);

	/**
	 * @brief �����ʶ����ά����
	 */
	void Find3DPoints();
private:
	/**
	 * @bnrief ����ͼ��
	 */
	void loadImg();

	/**
	 * @brief ˫Ƶ���
	 * @param index  ͼ������������ѡ��Ƶ��
	 * @return       ������ͼ��
	 */
	cv::Mat CalWrappedPhase(int index);

	/**
	 * @brief ˫Ƶ���
	 * @param phase1  Ƶ��1����λ
	 * @param T1      Ƶ��1
	 * @param phase2  Ƶ��2����λ
	 * @param T2      Ƶ��2
	 * @return        ������ͼ��
	 */
	cv::Mat HeterodynePhase(cv::Mat phase1, int T1, cv::Mat phase2, int T2);

	/**
	 * @brief ���յ���λ����
	 */
	void CombinePhase();

	/**
	 * @brief ��λ�˲�,matlab�Ǹ�����Ҫ�õ���ÿ�����صļ�ֵ;�ʵ�����max����ֵ��Ч������
	 */
	void FilterPhase();

	/**
	 * @brief ����ͼ��csv�ļ�
	 * @param mat        ����ͼ��
	 * @param filename   �ļ���
	 */
	void saveToCSV(const cv::Mat& mat, const std::string& filename);

	/**
	 * @brief �������˲�
	 * @return �˲�����
	 */
	cv::Mat MonoFilter();

	/**
	 * @brief matlab���Ǹ������Ժ��������˲�
	 */
	void Mono();



	/**
	 * @brief �˲����� Ŀǰֻ����Ⱥ���˲�
	 * @param cloud  �������
	 * @param cloud_filtered �˲������
	 */
	void CloudPointFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);

	/**
	 * @brief ����MRF���˲�
	 * @param img  ����ͼ��
	 * @param img_filtered  ���mask
	 */
	void MRF(cv::Mat& img, cv::Mat& mask);

	/**
	 * @Kronecker ����Kronecker��,��ͬ����1����ͬ����0
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


