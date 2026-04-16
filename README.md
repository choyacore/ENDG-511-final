# ENDG-511-Project

# Project Proposal

                   
     ENDG 511:Industrial IoT Systems & Artificial Intelligence

 Project Proposal—Team 14

Darren Taylor (UCID: 30119273)
Naishah Adetunji (UCID: 30122754)
Sehba Samman (UCID: 30160908)
















Problem:


Selecting the best hairstyles or hair treatments relies on the subjective visual judgment by stylists or customers, which is often very inconsistent and can be quite time consuming.  This is especially the case in fast paced environments with lots of people, like barbershops, salons, and self service kiosks. Many customers also lack knowledge on which styles best suit their hair length and color, leading to unideal outcomes.


Our proposed solution for this problem is an edge-based IoT vision system that detects a person's hair color and hair length category using a standard camera, in our case a computer camera. Based on these detected attributes, the system will provide basic, non-personalized styling recommendations such as haircut suggestions, or maintenance tips.  The system is designed to run locally on edge hardware, to help preserve privacy while also enabling fast real time operation.


Potential Use cases:


Smart kiosks in barbershops or salons
A cosmetic app
Training or decision support tool for junior hair stylists


Datasets & Hardware:


Datasets:
We will train the system using a combo of:
Publicly available face/hair datasets containing labeled hair attributes (Hair color and length)
A custom data set collected by our team using our webcams under varied lighting conditions so that we can improve the robustness of our system.

As a side note, we are going to discretize our hair length into categories (short, medium, long) rather than attempting to do some form of continuous measurement.  This will allow us to maintain a reasonable scope ensuring feasibility.
Hardware:
A standard webcam acting as the IoT sensor
The NVIDIA Jetson Orin Nano Developer Kit (8GB) for the edge interface
Development and testing will start on our desktops then once our code is refined we will move to the Jetson. For testing and to evaluate real time performance.

Goals & Machine Learning Methods:
Machine Learning Methods:
We will use light weight CNNs for the hair region detection and classification, because it's the simplest tool that learns visual patterns and it also works on Jetson. Image processing and segmentation will be done using OpenCv.  Color classification will be done using either HSV or LAB color space.  Hair length classification will be done using geometry to compare the hair length to relative detected facial landmarks.
System Architecture:

Camera Capture live image
Head/hair region is detected
Hair color and length category are inferred
Rule based logic outputs basic styling tips
Goals:
Minimal Goal:
The minimal achievable goal is to develop a system that classifies a person's hair length and category and displays the results in real time giving practical suggestions.  This version will operate on still images or low frame rate videos as a proof of concept.


Advanced Goal:
The advanced goal is to deploy our system on the NVIDIA Jetson to enable real time interface while optimizing model size, latency and efficiency.  Performance trade-offs related to accuracy, speed and reliability under different lighting conditions will be analyzed.


Performance Metrics:

Metrics 
Justification
Classification Accuracy (Hair colour and length)
To ensure the visual attributes are measured correctly, we will evaluate the model using labelled datasets like the CelebA dataset. [1]
Measure model robustness
We will be looking at precision, recall and F1-scores to make sure there is no class imbalance, like overrepresentation of dark hair. We will also be looking into different attributes like lighting, exposure and camera angles before preprocessing the data. [2]
Inference Latency 
Would be tested on NVIDIA Jetson to make sure that the model can be processed in real-time with webcam. 
Frames Per Second (FPS)
Our minimum target FPS ≥ 15 [3]


Expected Analysis & Trade-offs:

Analysis and Trade-offs 
Justification
Accuracy vs Latency 
Larger CNN models have better performance but higher inference time, whereas lighter models would process faster with slightly reduced accuracy [4]
Model size vs Edge Deployability
Pruning and quantization would make the model size smaller and process speed faster but will impact accuracy. However, to implement Jetson we would need an optimized model. So we need to prune our model in a way that doesn’t compromise the output’s validity.  [5]
Colour Space Selection 
Instead of RGB,using HSV or LAB colour spacing would improve the model’s sensitivity to light exposure but would introduce latency[6] 
Resolution vs Processing Speed 
High-resolution pictures would increase accuracy, but would be heavier on computational load and lower FPS



Potential Challenges:

Challenge
Impact
Mitigation Strategy
Lighting differences
Changes in illumination affect hair colour accuracy
Use HSV/LAB colour spaces and train/validate under different lighting conditions
Hair obstruction and style diversity
Some hairstyles, accessories, or clothing can make the hair harder to detect or measure accurately
Use coarse length categories and relative geometry via facial landmark
Dataset bias and limited diversity
Public datasets may not generalize across hair types
Expand the dataset using additional webcam images collected by the team
Limited processing power on the Jetson device
Complex models may slow down real-time performance
Use lightweight CNNs and efficient OpenCV 


Model accuracy vs. processing speed
Higher accuracy can slow real-time performance
Tune models to balance speed and accuracy


Team Roles and Collaboration Methods:

Team Member
Primary Technical Responsibility
Supporting Contributions
Darren Taylor
Hair region detection and CNN model training/optimization
System integration, performance evaluation on Jetson device
Naishah Adetunji
Hair length classification using geometric analysis and facial landmarks
Model deployment and performance optimization
Sehba Samman
Hair color classification using OpenCV and HSV/LAB color spaces
Dataset labeling, validation, and logic


Collaboration Approach:

The team will coordinate primarily through Microsoft Teams, with regular check-ins to discuss progress, challenges, and next steps. Work will be planned to ensure all members contribute to core technical components of the project, including model development, image processing, and system integration. Progress will be reviewed frequently to maintain steady development and shared understanding of the full system.

References:
[1] Y. Zhang, “CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations,” GitHub, May 19, 2023. Available: https://github.com/ZhangYuanhan-AI/CelebA-Spoof

[2] D. Borza, Tudor Ileni, and A. Darabant, “A Deep Learning Approach to Hair Segmentation and Color Extraction from Facial Images,” Lecture notes in computer science, pp. 438–449, Jan. 2018, doi: https://doi.org/10.1007/978-3-030-01449-0_37
[3]“Benchmarking Deep Learning Models on NVIDIA Jetson Nano for Real-Time Systems: An Empirical Investigation,” Arxiv.org, 2015. Available: https://arxiv.org/html/2406.17749v1

[4] A. McConnon, “Are bigger language models better?,” Ibm.com, Jul. 15, 2024. Available: https://www.ibm.com/think/insights/are-bigger-language-models-better

[5] S. Han, H. Mao, and W. J. Dally, “Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding,” arXiv:1510.00149 [cs], Feb. 2016, Available: https://arxiv.org/abs/1510.00149

[6] D. J. Bora, A. K. Gupta, and F. A. Khan, “Comparing the Performance of L*A*B* and HSV Color Spaces with Respect to Color Image Segmentation,” arXiv:1506.01472 [cs], Jun. 2015, Available: https://arxiv.org/abs/1506.01472


