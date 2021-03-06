#include <Windows.h>
#include <Ole2.h>
#include <Kinect.h>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/calib3d.hpp"
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

#define cColorWidth 1920
#define cColorHeight 1080

// aruco consts
const float calibrationSquareDimensions = 0.023f; // meters
const float arucoSquareDimensions = 0.066f; // meters
const Size chessboardDimensions = Size(6, 9);

// Kinect Vars
IKinectSensor* sensor;
IColorFrameReader* reader;

bool initKinect()
{
	if (FAILED(GetDefaultKinectSensor(&sensor)))
	{
		cout << "Failed to find Kinect\n";
		return false;
	}
	if (sensor)
	{
		sensor->Open();

		IColorFrameSource* framesource = NULL;
		sensor->get_ColorFrameSource(&framesource);
		framesource->OpenReader(&reader);
		if (framesource)
		{
			cout << "Reader opened successfully\n";
			framesource->Release();
			framesource = NULL;
		}
		return true;
	}
	else
	{
		cout << "Failed to open color source\n";
		return false;
	}
}

bool getKinectData(Mat& colorMat)
{
	IColorFrame* frame = NULL;
	ColorImageFormat imageFormat = ColorImageFormat_None;
	HRESULT hr;
	UINT bufferSize = 0;
	BYTE *pBuffer = nullptr;
	hr = reader->AcquireLatestFrame(&frame);

	if (SUCCEEDED(hr))
	{
		hr = frame->AccessRawUnderlyingBuffer(&bufferSize, &pBuffer); // YUY2
	}

	if (SUCCEEDED(hr))
	{
		//cout << "Frame read\n";
		Mat bufferMat(cColorHeight, cColorWidth, CV_8UC2, pBuffer);
		cvtColor(bufferMat, colorMat, COLOR_YUV2BGR_YUYV);
		if (frame) frame->Release();
		return true;
	}

	if (frame) frame->Release();
	cout << "Failed to get frame\n";
	return false;
}

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	cout << "create known board\n" << endl;
	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
		}
	}
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(9, 6), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			allFoundCorners.push_back(pointBuf);
		}

		if (showResults)
		{
			drawChessboardCorners(*iter, Size(9, 6), pointBuf, found);
			imshow("Looking for corners\n", *iter);
			waitKey(0);
		}
	}
}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	vector<vector<Point2f>> checkerboardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

	vector<Mat> rVectors, tVectors;
	distanceCoefficients = Mat::zeros(8, 1, CV_64F);

	cout << "calibrating" << endl;
	calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
	ofstream outStream(name);
	if (outStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		cout << "saving cam matrix\n" << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}
		}

		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		cout << "saving dist coeff\n" << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = distanceCoefficients.at<double>(r, c);
				outStream << value << endl;
			}
		}

		outStream.close();
		return true;
	}

	return false;
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	ifstream inStream(name);
	if (inStream)
	{
		uint16_t rows;
		uint16_t columns;

		inStream >> rows;
		inStream >> columns;

		cameraMatrix = Mat(Size(columns, rows), CV_64F);

		// cam matrix
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double read = 0.0f;
				inStream >> read;
				cameraMatrix.at<double>(r, c) = read;
				cout << cameraMatrix.at<double>(r, c) << "\n";
			}
		}

		// distortion coeff
		inStream >> rows;
		inStream >> columns;

		distanceCoefficients = Mat::zeros(rows, columns, CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double read = 0.0f;
				inStream >> read;
				distanceCoefficients.at<double>(r, c) = read;
				cout << distanceCoefficients.at<double>(r, c) << "\n";
			}
		}

		inStream.close();
		return true;
	}

	return false;
}


int startMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimensions)
{
	cout << "Monitoring started\n";
	Mat frame(cColorHeight, cColorWidth, CV_8UC3);
	Mat flipped(cColorHeight, cColorWidth, CV_8UC3);
	//Mat frame_gray(cColorHeight, cColorWidth, CV_8UC1);
	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	aruco::DetectorParameters parameters;

	// aruco size 250
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

	namedWindow("Kinect", WINDOW_AUTOSIZE);

	vector<Vec3d> rotationVectors, translationVectors;

	while (true)
	{
		if (!getKinectData(frame))
		{
			cout << "failed to read\n";
		}

		flip(frame, flipped, 1);
		aruco::detectMarkers(flipped, markerDictionary, markerCorners, markerIds);
		aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimensions, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);
		//aruco::drawDetectedMarkers(flipped, markerCorners, markerIds);
		for (int i = 0; i < markerIds.size(); i++)
		{
			cout << "marker detected\n";
			aruco::drawAxis(flipped, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f);
		}
		
		imshow("Kinect", flipped);
		if (waitKey(1000/60) == 27) break;
	}

	return 1;

}

void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients)
{
	Mat frame;
	Mat drawToFrame;

	//Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

	//Mat distanceCoefficients;

	vector<Mat> savedImages;

	vector<vector<Point2f>> markerCorners, rejectedCandidates;

	int framesPerSecond = 60;

	namedWindow("Kinect", WINDOW_AUTOSIZE);

	while (true)
	{
		if (!getKinectData(frame))
		{
			cout << "frame not read\n";
			break;
		}

		vector<Vec2f> foundPoints;
		bool found = false;

		found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);

		if (found)
			imshow("Kinect", drawToFrame);
		else
			imshow("Kinect", frame);
		char character = waitKey(1000 / framesPerSecond);

		switch (character)
		{
		case ' ':
			// space key: save image
			if (found)
			{
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
				cout << "Image Pushed" << endl;
			}
			break;
		case 13:
			// enter key: start calibration
			if (savedImages.size() > 15)
			{
				cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimensions, cameraMatrix, distanceCoefficients);
				saveCameraCalibration("Calibration", cameraMatrix, distanceCoefficients);
				cout << "Calibration Saved" << endl;
			}
			break;
		case 27:
			// esc key: exit
			return;
			break;
		}
	}
}

int main(int argc, char** argv)
{
	//setUseOptimized(true);
	// it takes the kinect a while to be ready
	initKinect();
	Sleep(6000);

	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	Mat distanceCoefficients;

	//cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
	loadCameraCalibration("Calibration", cameraMatrix, distanceCoefficients);
	startMonitoring(cameraMatrix, distanceCoefficients, arucoSquareDimensions);

	waitKey(0);

	if (reader) {
		sensor->Release();
		sensor = NULL;
	}

	if (sensor) {
		sensor->Close();
		sensor->Release();
		sensor = NULL;
	}
	return 0;
}