var partNames = [
    'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
    'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
    'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
];

var NUM_KEYPOINTS = partNames.length;
var partIds = partNames.reduce(function (result, jointName, i) {
    result[jointName] = i;
    return result;
}, {});

var connectedPartNames = [
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

var poseChain = [
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

function getDisplacement(index, displacements){
	var displacement_y = displacements[index];
	var displacement_x = displacements[index+16];
	return{
		y: displacement_y,
		x: displacement_x
	};
}

function getOffset(index, offsets){
	var offset_y = offsets[index];
	var offset_x = offsets[index+17];
	return{
		y: offset_y,
		x: offset_x
	};
}

function traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scores, offsets, outputStride, displacements, dimension){
	var dimension_displace = [dimension[0], dimension[1], 32];
	var dimension_offset = [dimension[0], dimension[1], 34];
	var height = dimension[0], width = dimension[1];
	var sourceKeypointIndeces = decode(sourceKeypoint.position, outputStride, height, width);
	var index_disp = convertCoortoIndex(sourceKeypointIndeces.x, sourceKeypointIndeces.y, edgeId, dimension_displace);
	var displacement = getDisplacement(index_disp, displacements);
	var displacedPoint = addVectors(sourceKeypoint.position, displacement);
	var displacedPointIndeces = decode(displacedPoint, outputStride, height, width);
	var index_off = convertCoortoIndex(displacedPointIndeces.x, displacedPointIndeces.y, targetKeypointId, dimension_offset);
	var offsetPoint = getOffset(index_off, offsets);
	var targetKeypoint = addVectors(displacedPoint, {x: offsetPoint.x, y: offsetPoint.y});
	var targetKeypointIndeces = decode(targetKeypoint, outputStride, height, width);
	var index_heatmap = convertCoortoIndex(targetKeypointIndeces.x, targetKeypointIndeces.y, targetKeypointId, dimension);
	var score = scores[index_heatmap];
	return{position: targetKeypoint, part: partNames[targetKeypointId], score: score};
}

function decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd, dimension){
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
	for(var edge = numEdges-1; edge >=0; --edge){
		var sourceKeypointId = parentToChildEdges[edge];
		var targetKeypointId = childToParentEdges[edge];
		if(instanceKeypoints[sourceKeypointId] && 
			!instanceKeypoints[targetKeypointId]){
			instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge,
													instanceKeypoints[sourceKeypointId], targetKeypointId, 
													scores, offsets, outputStride, displacementsBwd, dimension);
		}
	}

	for(var edge = 0; edge<numEdges; ++edge){
		var sourceKeypointId = childToParentEdges[edge];
		var targetKeypointId = parentToChildEdges[edge];
		if(instanceKeypoints[sourceKeypointId] && 
			!instanceKeypoints[targetKeypointId]){
			instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, 
													instanceKeypoints[sourceKeypointId], targetKeypointId, 
													scores, offsets, outputStride, displacementsFwd, dimension);
		}
	}
	return instanceKeypoints;
}