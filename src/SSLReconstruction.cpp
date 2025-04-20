#include "SSLReconstruction.h"

SSLReconstruction::SSLReconstruction(int ps_num, int img_width, int img_height, int T1, int T2, int T3, string path)
{
	this->m_ps_num = ps_num;
	this->m_width = img_width;
	this->m_height = img_height;
	this->m_frequency = new int [3] {T1, T2, T3};
	this->m_path = path;
	m_T13 = T1 - T3;
	m_T23 = T2 - T3;
	m_T123 = m_T13 - m_T23;
	m_k_cam = (cv::Mat_<float>(3, 3) <<
		4917.83770754812, 0, 1214.55428929112,
		0, 4923.82638685160, 1032.77712790899,
		0, 0, 1);
	m_k_dlp = (cv::Mat_<float>(3, 3) <<
		1257.86693007908, 0, 473.175940813014,
		0, 3154.36722942004, 810.775389044632,
		0, 0, 1);
	m_rt_cam = (cv::Mat_<float>(3, 4) <<
		0.0282843801595056, 0.999453513215491, 0.0171075644147791, -167.273600072281,
		-0.991514508545440, 0.0258791135103581, 0.127394076894544, 48.7112427545775,
		0.126881729113720, -0.0205656608240837, 0.991704633654588, 1030.39786715420);
	m_rt_dlp = (cv::Mat_<float>(3, 4) <<
		0.0156316443525661, 0.998161200177202, -0.0585650933205363, -96.8716349512309,
		-0.992163655512276, 0.0227461190972831, 0.122857212838845, 30.6138996645475,
		0.123963431605260, 0.0561856968170766, 0.990694824402463, 950.178896112994);
}
void saveMatToUcharCSV(const cv::Mat& mat, const string& filename)
{
	ofstream file(filename);
	if (!file.is_open())
	{
		cerr << "Error: Could not open file " << filename << endl;
		return;
	}

	for (int i = 0; i < mat.rows; i++)
	{
		for (int j = 0; j < mat.cols; j++)
		{
			file << (int)mat.at<uchar>(i, j); // ��ʽת��Ϊ int
			if (j < mat.cols - 1) file << ",";
		}
		file << endl;
	}

	file.close();
	cout << "Saved: " << filename << endl;
}
void saveMatToCSV(const cv::Mat& mat, const std::string& filename)
{
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
		return;
	}

	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			file << mat.at<float>(i, j);  // ������ CV_32F (float) ����
			if (j < mat.cols - 1) file << ","; // ���ŷָ�
		}
		file << "\n";  // ����
	}

	file.close();
	std::cout << "Saved matrix to " << filename << " successfully." << std::endl;
}

void SSLReconstruction::loadImg()
{

	for (int i = 0; i < this->m_ps_num * 3; i++)
	{
		string path = m_path + "/" + to_string(i + 1) + ".bmp";
		cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
		img.convertTo(img, CV_32FC1);
		m_imgs.push_back(img);
	}
	/*saveMatToUcharCSV(m_imgs[1], "imgs2.csv");
	saveMatToUcharCSV(m_imgs[2], "imgs3.csv");
	saveMatToUcharCSV(m_imgs[3], "imgs4.csv");
	saveMatToUcharCSV(m_imgs[4], "imgs5.csv");
	saveMatToUcharCSV(m_imgs[5], "imgs6.csv");
	saveMatToUcharCSV(m_imgs[6], "imgs7.csv");
	saveMatToUcharCSV(m_imgs[7], "imgs8.csv");*/
}


cv::Mat SSLReconstruction::CalWrappedPhase(int index)
{
	vector<cv::Mat> imgs;
	for (int i = index * m_ps_num; i < index * m_ps_num + m_ps_num; i++)
	{
		imgs.push_back(m_imgs[i]);
	}
	cv::Mat sin_sum = cv::Mat::zeros(imgs[0].size(), CV_32F);
	cv::Mat cos_sum = cv::Mat::zeros(imgs[0].size(), CV_32F);
	for (int i = 0; i < this->m_ps_num; i++)
	{
		//cv::Mat temp = imgs[i].clone(); // ʹ�� psImg[0] �����ݳ�ʼ�� temp
		//temp.convertTo(temp, CV_32FC1);
		float pk = 2 * i * CV_PI / m_ps_num;
		sin_sum += imgs[i] * sin(pk);
		cos_sum += imgs[i] * cos(pk);
		//cout<<"sin_sum:"<<sin_sum.at<float>(38,9)<<endl;
		//cout<<"cos_sum:"<<cos_sum.at<float>(38, 9) <<endl;

		//cout << i << endl;
		//if (i == 7)
		//{
		//	cout << "=============" << endl;
		//	cout << sin(pk) << endl;
		//	cout << imgs[i].at<float>(38, 9) << endl;
		//}
	}
	cv::Mat pha(sin_sum.size(), CV_32F);
	for (int row = 0; row < m_height; row++)
	{
		for (int col = 0; col < m_width; col++)
		{
			if (std::abs(sin_sum.at<float>(row, col)) < 1e-2)
			{
				sin_sum.at<float>(row, col) = 0;
			}
			if (std::abs(cos_sum.at<float>(row, col)) < 1e-2)
			{
				cos_sum.at<float>(row, col) = 0;
			}
			pha.at<float>(row, col) = -atan2(sin_sum.at<float>(row, col), cos_sum.at<float>(row, col));
		}
	}
	//pha = -pha;
	cv::Mat A = sin_sum.mul(sin_sum) + cos_sum.mul(cos_sum);
	sqrt(A, m_B);
	m_B = m_B * 2 / m_ps_num;
	m_B_mask = m_B > 10;
	m_B_mask.convertTo(m_B_mask, CV_32FC1);
	pha = pha.mul(m_B_mask / 255);

	/*cv::Mat pha_low_mask = pha <= 0;
	pha_low_mask.convertTo(pha_low_mask, CV_32FC1);
	pha = pha + pha_low_mask / 255 * CV_2PI;*/
	return pha;
}

cv::Mat SSLReconstruction::HeterodynePhase(cv::Mat phase1, int T1, cv::Mat phase2, int T2)
{
	int T12 = T1 - T2;
	float K = (float)T1 / (float)T12;
	cv::Mat phase_diff = phase1 - phase2;
	cv::Mat index = phase_diff < 0;
	index.convertTo(index, CV_32FC1);
	for (int row = 0; row < m_height; row++)
	{
		for (int col = 0; col < m_width; col++)
		{
			if (index.at<float>(row, col) > 0)
			{
				phase_diff.at<float>(row, col) += CV_2PI;
			}
		}
	}

	cv::Mat period = ((phase_diff * K - phase1) / (CV_2PI));
	for (int row = 0; row < m_height; row++)
	{
		for (int col = 0; col < m_width; col++)
		{
			period.at<float>(row, col) = round(period.at<float>(row, col));
		}

	}
	cv::Mat pha = (period * CV_2PI + phase1) / K;

	return pha;
}

//void SSLReconstruction::CombinePhase()
//{
//	cv::Mat phase1 = m_phase123 * float(m_frequency[0]);
//	cv::Mat phase2 = ((m_phase123 * (float)m_frequency[1] - m_wrapped_phase2) / CV_2PI) * CV_2PI + m_wrapped_phase2;
//	cv::Mat phase3 = ((m_phase123 * (float)m_frequency[2] - m_wrapped_phase3) / CV_2PI) * CV_2PI + m_wrapped_phase3;
//	for (int row = 0; row < m_height; row++)
//	{
//		for (int col = 0; col < m_width; col++)
//		{
//			phase2.at<float>(row, col) = round(phase2.at<float>(row, col));
//			phase3.at<float>(row, col) = round(phase3.at<float>(row, col));
//		}
//	}
//	m_phase_abs = (phase1 + phase2 + phase3) / (float)(m_frequency[0] + m_frequency[1] + m_frequency[2]);
//
//}
void SSLReconstruction::CombinePhase()
{
	// ���������λ
	cv::Mat phase1 = m_phase123 * (float)(m_frequency[0]);

	// ���� phase2 �� phase3��ȷ�� round �߼��� Matlab һ��
	cv::Mat temp2 = (m_phase123 * (float)(m_frequency[1]) - m_wrapped_phase2) / CV_2PI;
	cv::Mat temp3 = (m_phase123 * (float)(m_frequency[2]) - m_wrapped_phase3) / CV_2PI;

	// Ԥ����� m_phase123 ��ͬ��С�ľ���
	cv::Mat phase2 = cv::Mat::zeros(m_height, m_width, CV_32F);
	cv::Mat phase3 = cv::Mat::zeros(m_height, m_width, CV_32F);

	for (int row = 0; row < m_height; row++)
	{
		for (int col = 0; col < m_width; col++)
		{
			phase2.at<float>(row, col) = round(temp2.at<float>(row, col))*CV_2PI+m_wrapped_phase2.at<float>(row, col);
			phase3.at<float>(row, col) = round(temp3.at<float>(row, col))*CV_2PI+m_wrapped_phase3.at<float>(row, col);
		}
	}

	// �������յľ�����λ
	m_phase_abs = (phase1 + phase2 + phase3) / static_cast<float>(m_frequency[0] + m_frequency[1] + m_frequency[2]);
}

void SSLReconstruction::FilterPhase()
{
	//�˲���������ÿ�����ص㣺1.���ֵ����245��2.��СֵС��10��3.�����Сֵ��ֵС��10
	for (int row = 0; row < m_height; row++)
	{
		for (int col = 0; col < m_width; col++)
		{
			int max_val = 0, min_val = 255;
			for (int i = 0; i < m_imgs.size(); i++)
			{

				if (m_imgs[i].at<uchar>(row, col) > max_val)
				{
					max_val = m_imgs[i].at<uchar>(row, col);
				}
				if (m_imgs[i].at<uchar>(row, col) < min_val)
				{
					min_val = m_imgs[i].at<uchar>(row, col);
				}
			}
			if (max_val > 253 || abs(max_val - min_val) <= 2 || min_val < 2)
			{
				m_phase_abs.at<float>(row, col) = 0;
			}
		}
	}
	//��Ϊ��Ч�㶼���õ�-100�����ֱ�ӹ�һ�����÷�ĸ��úܴ󣬽�������ÿ��Ԫ�ض���࣬������ʾ��ʱ��������Ǵ��ס�
	//Ϊ��������ʾ��Ҫ��������ͼ������ֵ

}
void SSLReconstruction::saveToCSV(const cv::Mat& mat, const std::string& filename)
{
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{
			file << mat.at<float>(i, j);
			if (j < mat.cols - 1)
				file << ",";
		}
		file << "\n";
	}

	file.close();
	std::cout << "Saved to " << filename << " successfully." << std::endl;
}

cv::Mat SSLReconstruction::MonoFilter()
{
	cv::Mat mask = cv::Mat::zeros(m_phase_abs.size(), CV_32FC1);
	for (int row = 0; row < m_height; row++)
	{
		for (int col = 1; col < m_width - 1; col++)
		{
			if (m_phase_abs.at<float>(row, col) > m_phase_abs.at<float>(row, col - 1)
				&& m_phase_abs.at<float>(row, col) < m_phase_abs.at<float>(row, col + 1))
			{
				mask.at<float>(row, col) = 1;
			}
		}
	}
	return mask;

}

void SSLReconstruction::Mono()
{
	for (int row = 0; row < m_height; row++)
	{
		for (int col = 1; col < m_width - 1; col++)
		{
			if (col<1 || col>m_width - 2 || row<1 || row>m_height - 2)
			{
				m_phase_abs.at<float>(row, col) = -100;
			}
			else if (m_phase_abs.at<float>(row, col) != -100)
			{
				if (m_phase_abs.at<float>(row, col - 1) != -100)
				{
					if (
						m_phase_abs.at<float>(row, col) < m_phase_abs.at<float>(row, col - 1))
					{
						m_phase_abs.at<float>(row, col) = -100;
					}
				}
				if (m_phase_abs.at<float>(row, col + 1) != -100)
				{
					if (m_phase_abs.at<float>(row, col + 1) < m_phase_abs.at<float>(row, col))
					{
						m_phase_abs.at<float>(row, col) = -100;
					}
				}
				if (m_phase_abs.at<float>(row, col - 1) == 0 && m_phase_abs.at<float>(row, col + 1) == -100)
				{
					m_phase_abs.at<float>(row, col) = -100;
				}
				/*if (m_phase_abs.at<float>(row - 1, col) != -100)
				{
					if (m_phase_abs.at<float>(row, col) - m_phase_abs.at<float>(row - 1, col) > (CV_PI / m_width / 4))
					{
						m_phase_abs.at<float>(row, col) = -100;
					}
				}
				if (m_phase_abs.at<float>(row + 1, col) != -100)
				{
					if (m_phase_abs.at<float>(row + 1, col) - m_phase_abs.at<float>(row, col) > (CV_PI / m_width / 4))
					{
						m_phase_abs.at<float>(row + 1, col) = -100;
					}
				}*/
				if (m_phase_abs.at<float>(row - 1, col) == -100 && m_phase_abs.at<float>(row + 1, col) == -100)
				{
					m_phase_abs.at<float>(row, col) = -100;
				}
			}
		}
	}
}

cv::Mat SSLReconstruction::Decode(bool filter)
{
	loadImg();
	m_wrapped_phase1 = CalWrappedPhase(0);
	m_wrapped_phase2 = CalWrappedPhase(1);
	m_wrapped_phase3 = CalWrappedPhase(2);
	m_phase13 = HeterodynePhase(m_wrapped_phase1, m_frequency[0], m_wrapped_phase3, m_frequency[2]);
	m_phase23 = HeterodynePhase(m_wrapped_phase2, m_frequency[1], m_wrapped_phase3, m_frequency[2]);
	m_phase123 = HeterodynePhase(m_phase13, m_T13, m_phase23, m_T23);
	/*saveMatToCSV(m_phase13, "phase13.csv");
	saveMatToCSV(m_phase23, "phase23.csv");
	saveMatToCSV(m_phase123, "phase123.csv");*/
	CombinePhase();
	//saveMatToCSV(m_phase_abs, "phase_abs.csv");
	//FilterPhase();
	//���ƶ��˲�
	if (filter)
	{
		m_B_mask.convertTo(m_B_mask, CV_32FC1);
		for (int col = 0; col < m_width; col++)
		{
			for (int row = 0; row < m_height; row++)
			{
				if (m_B_mask.at<float>(row, col) == 0)
				{
					m_phase_abs.at<float>(row, col) = 0;
				}
			}
		}
	}
	

	//MRF�˲�	
	//cv::Mat MRF_mask = cv::Mat::zeros(m_phase_abs.size(), CV_32FC1);
	//MRF(m_phase_abs, MRF_mask);

	//for (int col = 0; col < m_width; col++)
	//{
	//	for (int row = 0; row < m_height; row++)
	//	{
	//		if (MRF_mask.at<float>(row, col) == 0)
	//		{
	//			m_phase_abs.at<float>(row, col) = 0;
	//		}
	//	}
	//}
	//�������˲�
	//cv::Mat mask = MonoFilter();



	return m_phase_abs;
}

void SSLReconstruction::FindCenters(cv::Mat& image)
{
	// ��ֵ��ͼ�������δ��ֵ����
	cv::Mat binaryImage;
	threshold(image, binaryImage, 158, 255, cv::THRESH_BINARY);

	// ������ͨ��
	cv::Mat labels, stats, centroids;
	int numLabels = connectedComponentsWithStats(binaryImage, labels, stats, centroids);

	// �趨ɸѡ����
	int minArea = 100;       // ��С���
	int maxArea = 1000;      // ������
	double minCircularity = 0.8; // ��СԲ��

	// ����һ����ɫͼ�����ڿ��ӻ�
	cv::Mat outputImage;
	cvtColor(binaryImage, outputImage, cv::COLOR_GRAY2BGR);

	vector<cv::Point2f> centers;
	// ����������ͨ��
	for (int i = 1; i < numLabels; i++)
	{ // ������������ǩΪ0��
		int area = stats.at<int>(i, cv::CC_STAT_AREA);

		// ������ͨ�������
		cv::Mat mask = (labels == i);
		vector<vector<cv::Point>> contours;
		findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		if (contours.empty()) continue;

		// �����ܳ�
		double perimeter = arcLength(contours[0], true);

		// ����Բ��
		double circularity = (4 * CV_PI * area) / (perimeter * perimeter);

		// ɸѡ������������ͨ��
		if (area > minArea && area < maxArea && circularity > minCircularity)
		{
			cout << "��ͨ�� " << i << ": ��� = " << area << ", Բ�� = " << circularity << endl;

			// �����Բ
			cv::RotatedRect ellipse = fitEllipse(contours[0]);
			cv::Point2f center = ellipse.center; // ��Բ���ĵ�
			m_centers.push_back(center);
			// ��ͼ���ϱ����ͨ��
			cv::Rect boundingBox(stats.at<int>(i, cv::CC_STAT_LEFT),
				stats.at<int>(i, cv::CC_STAT_TOP),
				stats.at<int>(i, cv::CC_STAT_WIDTH),
				stats.at<int>(i, cv::CC_STAT_HEIGHT));
			rectangle(outputImage, boundingBox, cv::Scalar(0, 255, 0), 2); // ����ɫ���α��

			// ��ͼ���ϱ����Բ���ĵ�
			circle(outputImage, center, 5, cv::Scalar(255, 0, 0), -1); // ����ɫԲ��������
			putText(outputImage, "Center", cv::Point(center.x + 10, center.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
		}
	}

	// ��ʾ���
	/*cv::namedWindow("ɸѡ���", cv::WINDOW_NORMAL);
	cv::imshow("ɸѡ���", outputImage);
	cv::waitKey(0);*/
}

void SSLReconstruction::Find3DPoints()
{
	Decode(false);
	cv::Mat phase_temp;
	medianBlur(m_phase_abs, phase_temp, 5); // ʹ�� 5��5 ����ֵ�˲�
	cout<<phase_temp.type()<<endl;

	cv::Mat A(3, 3, CV_32FC1);
	cv::Mat b = cv::Mat(3, 1, CV_32FC1);
	cv::Mat xyz;
	cv::Mat m_q_cam = m_k_cam * m_rt_cam;
	cv::Mat m_q_dlp = m_k_dlp * m_rt_dlp;
	//�ضϱ�ʶ���ά����
	std::vector<cv::Point> points2D; // �洢�ضϺ�� int ����

	for (const auto& pt : m_centers)
	{
		points2D.emplace_back(
			static_cast<int>(pt.x), // ֱ�ӽض�С������
			static_cast<int>(pt.y)
		);
	}

	std::vector<cv::Point3f> points3D; // �洢���� 3D ��

	for (const auto& pt : points2D)
	{
		int row = pt.y; // �����꣨��Ӧͼ��߶ȣ�
		int col = pt.x; // �����꣨��Ӧͼ����ȣ�

		if (phase_temp.at<float>(row, col) > 0)
		{
			float up = phase_temp.at<float>(row, col) * 912 / (2 * CV_PI);
			float uc = col;
			float vc = row;

			// ���� A �� b
			A.at<float>(0, 0) = m_q_cam.at<float>(0, 0) - m_q_cam.at<float>(2, 0) * uc;
			A.at<float>(0, 1) = m_q_cam.at<float>(0, 1) - m_q_cam.at<float>(2, 1) * uc;
			A.at<float>(0, 2) = m_q_cam.at<float>(0, 2) - m_q_cam.at<float>(2, 2) * uc;

			A.at<float>(1, 0) = m_q_cam.at<float>(1, 0) - m_q_cam.at<float>(2, 0) * vc;
			A.at<float>(1, 1) = m_q_cam.at<float>(1, 1) - m_q_cam.at<float>(2, 1) * vc;
			A.at<float>(1, 2) = m_q_cam.at<float>(1, 2) - m_q_cam.at<float>(2, 2) * vc;

			A.at<float>(2, 0) = m_q_dlp.at<float>(0, 0) - m_q_dlp.at<float>(2, 0) * up;
			A.at<float>(2, 1) = m_q_dlp.at<float>(0, 1) - m_q_dlp.at<float>(2, 1) * up;
			A.at<float>(2, 2) = m_q_dlp.at<float>(0, 2) - m_q_dlp.at<float>(2, 2) * up;

			b.at<float>(0, 0) = m_q_cam.at<float>(2, 3) * uc - m_q_cam.at<float>(0, 3);
			b.at<float>(1, 0) = m_q_cam.at<float>(2, 3) * vc - m_q_cam.at<float>(1, 3);
			b.at<float>(2, 0) = m_q_dlp.at<float>(2, 3) * up - m_q_dlp.at<float>(0, 3);

			cv::solve(A, b, xyz);

			// �洢 3D ��
			points3D.emplace_back(
				xyz.at<float>(0, 0),
				xyz.at<float>(1, 0),
				xyz.at<float>(2, 0)
			);
		}
	}
	SavePointsToTXT(points3D, "output_3d_points.txt");
}

void SSLReconstruction::CloudPointFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered)
{
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(50);
	sor.setStddevMulThresh(1.0);
	sor.filter(*cloud_filtered);
}

float SSLReconstruction::Kronecker(float a, float b)
{
	if (a == b)
	{
		return 1.0f;
	}
	return 0.0f;
}

float SSLReconstruction::u1(float pha)
{
	return 0.5 * (1 - std::erf(pha - m_theta_phase));
}

float SSLReconstruction::b1(int omega_index, float pha1, float pha2)
{
	return m_lambda1 * m_omega[omega_index] * std::abs(pha2 - pha1);
}

float SSLReconstruction::b2(int omega_index, float mask1, float mask2)
{
	return m_lambda2 * m_omega[omega_index] * (1 - Kronecker(mask1, mask2));
}

void SSLReconstruction::SavePointsToTXT(const std::vector<cv::Point3f>& points3D, const std::string& filename)
{
	std::ofstream outFile(filename);
	if (!outFile.is_open())
	{
		std::cerr << "Error: Could not open file for writing!" << std::endl;
		return;
	}

	// д���ʽ��x y z��ÿ��һ���㣩
	for (const auto& pt : points3D)
	{
		outFile << pt.x << " " << pt.y << " " << pt.z << "\n";
	}

	outFile.close();
	std::cout << "3D points saved to " << filename << std::endl;
}

void SSLReconstruction::MRF(cv::Mat& phase, cv::Mat& mask)
{

	//��ʼ�����룬������λ������
	for (int row = 0; row < m_height; row++)
	{
		for (int col = 1; col < m_width - 1; col++)
		{
			if (m_phase_abs.at<float>(row, col) > m_phase_abs.at<float>(row, col - 1)
				&& m_phase_abs.at<float>(row, col) < m_phase_abs.at<float>(row, col + 1))
			{
				mask.at<float>(row, col) = 1;
			}
		}
	}
	//���� u2
	cv::Mat u2 = cv::Mat::zeros(phase.size(), CV_32FC1);
	for (int row = 1; row < m_height - 2; row++)
	{
		for (int col = 1; col < m_width - 2; col++)
		{
			//if (row == 0 || row == m_width - 1 || col == 0 || col == m_height - 1)
			//{
			//	//���ñ߽��
			//	u2.at<float>(row, col) = 1;
			//	mask.at<float>(row, col) = 1;
			//	continue;
			//}
			u2.at<float>(row, col) = u1(phase.at<float>(row, col)) +
				b1(0, phase.at<float>(row, col), phase.at<float>(row, col + 1)) + b1(0, phase.at<float>(row, col), phase.at<float>(row, col - 1)) +
				b1(0, phase.at<float>(row, col), phase.at<float>(row + 1, col)) + b1(0, phase.at<float>(row, col), phase.at<float>(row - 1, col)) +
				b1(1, phase.at<float>(row, col), phase.at<float>(row + 1, col + 1)) + b1(1, phase.at<float>(row, col), phase.at<float>(row - 1, col - 1)) +
				b1(1, phase.at<float>(row, col), phase.at<float>(row + 1, col - 1)) + b1(1, phase.at<float>(row, col), phase.at<float>(row - 1, col + 1));
		}
	}

	int count_of_changes = 0;
	for (int round = 0; round < m_num_of_MRF; round++)
	{
		for (int row = 1; row < m_height - 2; row++)
		{
			for (int col = 1; col < m_width - 2; col++)
			{
				float e0 = u2.at<float>(row, col) +
					b2(0, 0, mask.at<float>(row, col + 1)) + b2(0, 0, mask.at<float>(row, col - 1)) +
					b2(0, 0, mask.at<float>(row + 1, col)) + b2(0, 0, mask.at<float>(row - 1, col)) +
					b2(1, 0, mask.at<float>(row + 1, col + 1)) + b2(1, 0, mask.at<float>(row - 1, col - 1)) +
					b2(1, 0, mask.at<float>(row + 1, col - 1)) + b2(1, 0, mask.at<float>(row - 1, col + 1));
				float e1 = u2.at<float>(row, col) +
					b2(0, 1, mask.at<float>(row, col + 1)) + b2(0, 1, mask.at<float>(row, col - 1)) +
					b2(0, 1, mask.at<float>(row + 1, col)) + b2(0, 1, mask.at<float>(row - 1, col)) +
					b2(1, 1, mask.at<float>(row + 1, col + 1)) + b2(1, 1, mask.at<float>(row - 1, col - 1)) +
					b2(1, 1, mask.at<float>(row + 1, col - 1)) + b2(1, 1, mask.at<float>(row - 1, col + 1));
				if (m_theta_phase > e1 && e0 >= e1)
				{
					if (mask.at<float>(row, col) == 0)
					{
						count_of_changes++;
					}
					mask.at<float>(row, col) = 1;
				}
				else
				{
					if (mask.at<float>(row, col) == 1)
					{
						count_of_changes++;
					}
					mask.at<float>(row, col) = 0;
				}
			}
		}
	}
	cv::namedWindow("mask", cv::WINDOW_NORMAL);
	cv::imshow("mask", mask);
	cv::waitKey(0);
	cout << "count_of_changes:" << count_of_changes << endl;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr SSLReconstruction::Reconstruction()
{
	cv::Mat phase = Decode(true);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	phase.convertTo(phase, CV_32FC1);
	cv::Mat A(3, 3, CV_32FC1);
	cv::Mat b = cv::Mat(3, 1, CV_32FC1);
	cv::Mat xyz;
	cv::Mat m_q_cam = m_k_cam * m_rt_cam;
	cv::Mat m_q_dlp = m_k_dlp * m_rt_dlp;

	//�ؽ�
	for (int row = 1; row < m_height; row++)
	{
		for (int col = 1; col < m_width; col++)
		{
			if (phase.at<float>(row, col) > 0)
			{
				float up = phase.at<float>(row, col) * 912 / (2 * CV_PI);
				float uc = col;
				float vc = row;

				// A �ļ���
				A.at<float>(0, 0) = m_q_cam.at<float>(0, 0) - m_q_cam.at<float>(2, 0) * uc;
				A.at<float>(0, 1) = m_q_cam.at<float>(0, 1) - m_q_cam.at<float>(2, 1) * uc;
				A.at<float>(0, 2) = m_q_cam.at<float>(0, 2) - m_q_cam.at<float>(2, 2) * uc;

				A.at<float>(1, 0) = m_q_cam.at<float>(1, 0) - m_q_cam.at<float>(2, 0) * vc;
				A.at<float>(1, 1) = m_q_cam.at<float>(1, 1) - m_q_cam.at<float>(2, 1) * vc;
				A.at<float>(1, 2) = m_q_cam.at<float>(1, 2) - m_q_cam.at<float>(2, 2) * vc;

				A.at<float>(2, 0) = m_q_dlp.at<float>(0, 0) - m_q_dlp.at<float>(2, 0) * up;
				A.at<float>(2, 1) = m_q_dlp.at<float>(0, 1) - m_q_dlp.at<float>(2, 1) * up;
				A.at<float>(2, 2) = m_q_dlp.at<float>(0, 2) - m_q_dlp.at<float>(2, 2) * up;

				// b �ļ���
				b.at<float>(0, 0) = m_q_cam.at<float>(2, 3) * uc - m_q_cam.at<float>(0, 3);
				b.at<float>(1, 0) = m_q_cam.at<float>(2, 3) * vc - m_q_cam.at<float>(1, 3);
				b.at<float>(2, 0) = m_q_dlp.at<float>(2, 3) * up - m_q_dlp.at<float>(0, 3);

				cv::solve(A, b, xyz);

				pcl::PointXYZ points;
				points.x = xyz.at<float>(0, 0);
				points.y = xyz.at<float>(1, 0);
				points.z = xyz.at<float>(2, 0);
				cloud->push_back(points);
			}
		}
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	CloudPointFilter(cloud, cloud_filtered);
	return cloud_filtered;
}


//void SSLReconstruction::FindCircleCenter(cv::Mat image)
//{
//	//��ֵ��
//	cv::Mat img_binary;
//	cv::threshold(image, img_binary, 128, 255, cv::THRESH_BINARY);
//	cv::Mat labels, stats, centroids;
//	int numLabels = connectedComponentsWithStats(img_binary, labels, stats, centroids);
//	// ����һ����ɫͼ�����ڿ��ӻ�
//	cv::Mat outputImage;
//	cvtColor(img_binary, outputImage, cv::COLOR_GRAY2BGR);
//	// ����������ͨ��
//	for (int i = 1; i < numLabels; i++)
//	{ // ������������ǩΪ0��
//		int area = stats.at<int>(i, cv::CC_STAT_AREA);
//
//		// ������ͨ�������
//		cv::Mat mask = (labels == i);
//		vector<vector<cv::Point>> contours;
//		findContours(mask, contours,cv:: RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//		if (contours.empty()) continue;
//
//		// �����ܳ�
//		float perimeter = arcLength(contours[0], true);
//
//		// ����Բ��
//		float circularity = (4 * CV_PI * area) / (perimeter * perimeter);
//		// ɸѡ������������ͨ��
//		if (area > minArea && area < maxArea && circularity > minCircularity)
//		{
//			cout << "��ͨ�� " << i << ": ��� = " << area << ", Բ�� = " << circularity << endl;
//			//�ŵ���Ա������
//			// �����Բ
//			cv::RotatedRect ellipse = fitEllipse(contours[0]);
//			cv::Point2f center = ellipse.center; // ��Բ���ĵ�
//			m_circle_centers.push_back(center);
//
//			// ��ͼ���ϱ����ͨ��
//			cv::Rect boundingBox(stats.at<int>(i, cv::CC_STAT_LEFT),
//				stats.at<int>(i,cv::CC_STAT_TOP),
//				stats.at<int>(i,cv::CC_STAT_WIDTH),
//				stats.at<int>(i,cv::CC_STAT_HEIGHT));
//			rectangle(outputImage, boundingBox, cv::Scalar(0, 255, 0), 2); // ����ɫ���α��
//
//			// ��ͼ���ϱ����Բ���ĵ�
//			circle(outputImage, center, 5,cv::Scalar(255, 0, 0), -1); // ����ɫԲ��������
//			putText(outputImage, "Center",cv::Point(center.x + 10, center.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
//			
//		}
//	}
//	cv::namedWindow("ɸѡ���", cv::WINDOW_NORMAL);
//	cv::imshow("ɸѡ���", outputImage);
//	cv::waitKey(0);
//}
