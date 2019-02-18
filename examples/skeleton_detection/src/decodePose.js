/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
 
const partNames = [
  'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
  'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
  'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
];

const NUM_KEYPOINTS = partNames.length;
var partIds = partNames.reduce(function (result, jointName, i) {
  result[jointName] = i;
  return result;
}, {});

const connectedPartNames = [
  ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
  ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
  ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
  ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
  ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
];

var connectedPartIndeces = connectedPartNames.map(function (_a) {
  var jointNameA = _a[0], jointNameB = _a[1];
  return ([partIds[jointNameA], partIds[jointNameB]]);
});

const poseChain = [
  ['nose', 'leftEye'], ['leftEye', 'leftEar'], ['nose', 'rightEye'],
  ['rightEye', 'rightEar'], ['nose', 'leftShoulder'],
  ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
  ['leftShoulder', 'leftHip'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['nose', 'rightShoulder'],
  ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
  ['rightShoulder', 'rightHip'], ['rightHip', 'rightKnee'],
  ['rightKnee', 'rightAnkle']
];

var parentChildrenTuples = poseChain.map(function (_a) {
  var parentJoinName = _a[0], childJoinName = _a[1];
  return ([partIds[parentJoinName], partIds[childJoinName]]);
});

var parentToChildEdges = parentChildrenTuples.map(function (_a) {
  var childJointId = _a[1];
  return childJointId;
});

var childToParentEdges = parentChildrenTuples.map(function (_a) {
  var parentJointId = _a[0];
  return parentJointId;
});

function clamp(a, min, max) {
  if (a < min) {
    return min;
  }
  if (a > max) {
    return max;
  }
  return a;
}

function decode(point, outputStride, height, width) {
  return {
    y: clamp(Math.round(point.y / outputStride), 0, height - 1),
    x: clamp(Math.round(point.x / outputStride), 0, width - 1)
  };
}

function addVectors(a, b) {
  return { x: a.x + b.x, y: a.y + b.y };
}

function getDisplacement(index, displacements) {
  var displacementY = displacements[index];
  var displacementX = displacements[index + 16];
  return {
    y: displacementY,
    x: displacementX
  };
}

function getOffset(index, offsets) {
  var offsetY = offsets[index];
  var offsetX = offsets[index + 17];
  return {
    y: offsetY,
    x: offsetX
  };
}

function traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scores, offsets, outputStride, displacements, dimension) {
  var dimensionDisplace = [dimension[0], dimension[1], 32];
  var dimensionOffset = [dimension[0], dimension[1], 34];
  var height = dimension[0], width = dimension[1];
  var sourceKeypointIndeces = decode(sourceKeypoint.position, outputStride, height, width);
  var indexDisp = convertCoortoIndex(sourceKeypointIndeces.x, sourceKeypointIndeces.y, edgeId, dimensionDisplace);
  var displacement = getDisplacement(indexDisp, displacements);
  var displacedPoint = addVectors(sourceKeypoint.position, displacement);
  var displacedPointIndeces = decode(displacedPoint, outputStride, height, width);
  var indexOffset = convertCoortoIndex(displacedPointIndeces.x, displacedPointIndeces.y, targetKeypointId, dimensionOffset);
  var offsetPoint = getOffset(indexOffset, offsets);
  var targetKeypoint = addVectors(
    {
      x: displacedPointIndeces.x * outputStride,
      y: displacedPointIndeces.y * outputStride
    }, {
      x: offsetPoint.x, 
      y: offsetPoint.y
    }); 
  var indexHeatmap = convertCoortoIndex(displacedPointIndeces.x, displacedPointIndeces.y, targetKeypointId, dimension);
  var score = scores[indexHeatmap];
  return {position: targetKeypoint, part: partNames[targetKeypointId], score: score};
}

function decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd, dimension) {
  var numParts = dimension[2];
  var numEdges = parentToChildEdges.length;
  var instanceKeypoints = new Array(numParts);
  var rootPart = root.part, rootScore = root.score;
  var rootPoint = getImageCoords(rootPart, outputStride, offsets, dimension);
  instanceKeypoints[rootPart.id] = {
    score: rootScore,
    part: partNames[rootPart.id],
    position: rootPoint
  };
  for(var edge = numEdges-1; edge >=0; --edge) {
    var sourceKeypointId = parentToChildEdges[edge];
    var targetKeypointId = childToParentEdges[edge];
    if(instanceKeypoints[sourceKeypointId] && !instanceKeypoints[targetKeypointId]) {
      instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, 
                                                                     scores, offsets, outputStride, displacementsBwd, dimension);
    }
  }

  for(var edge = 0; edge<numEdges; ++edge) {
    var sourceKeypointId = childToParentEdges[edge];
    var targetKeypointId = parentToChildEdges[edge];
    if(instanceKeypoints[sourceKeypointId] && !instanceKeypoints[targetKeypointId]) {
      instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, 
                                                                     scores, offsets, outputStride, displacementsFwd, dimension);
    }
  }
    return instanceKeypoints;
}
