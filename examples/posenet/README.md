{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# WebML PoseNet Sample   \n",
    "This sample loads a pretrained posenet model, constructs and infers it by WebML API.    \n",
    "## Screen Shoot\n",
    "<p float=\"left\">\n",
    "  <img align=\"left\" src=\"https://i.imgur.com/8brxB77.png\" width=\"400px\" height=\"400px\"/> \n",
    "  <img src=\"https://i.imgur.com/EG1TR20.png\" width=\"300px\" height=\"400px\"/>   \n",
    "</p>   \n",
    "## Usage\n",
    "PoseNet can be used to estimate either a single pose or multiple poses, so there are two versions, one is for single pose detection with an assumption that is there exists only one persion in image or video, another one is for multiple poses detection used to detect several person poses. At the same time, there are various parameters shown in above control pad could affect PoseNet model's accuracy and cost time.    \n",
    "     \n",
    "***Model: 1.01/1.0/0.75/0.5***   \n",
    "The larger the value, the larger the size of the layers, and more accurate the model at the cost of speed. Set this to a smaller value to increase speed at the cost of accuracy.   \n",
    "`1.01`/`1.0`: Computers with powerful GPUs   \n",
    "`0.75`: Computers with mid-range/lower-end GPUs   \n",
    "`0.5`: Mobile     \n",
    "\n",
    "***OutputStride: 8/16/32***     \n",
    "The desired stride for the output, it decides output dimension of model. The higher the number, the faster the performance but slower the accuracy.  \n",
    "     \n",
    "***Scale Factor: (0.1, 1)***    \n",
    "Scale down the image size before feed it through model, set this number lower to scale down the image and increase the speed when feeding through the network at the cost of accuracy.     \n",
    "    \n",
    "***Score Threshold:***    \n",
    "Score is the probability of keypoint and pose, set score threshold higher to reduce the number of poses to draw on image and visa versa.    \n",
    "\n",
    "***nmsRadius: ***    \n",
    "The minimal distance value between two poses under multiple poses situation. The smaller this value, the poses in image are more concentrated.     \n",
    "    \n",
    "***maxDetection:***    \n",
    "The maximul number of poses to be detected in multiple poses situation.     \n",
    "\n",
    "## Algorithm      \n",
    "Single Pose Detection Reference: [**Blog**](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5)    \n",
    "Multiple Pose Detection Reference: [Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model](https://arxiv.org/abs/1803.08225)    \n",
    "\n",
    "## Reference    \n",
    "https://github.com/tensorflow/tfjs-models/tree/master/posenet"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
