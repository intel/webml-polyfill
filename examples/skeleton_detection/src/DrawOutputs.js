/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licnses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const boundingBoxColor = 'rgba(255, 111, 97, 1.0)';
const color = 'aqua';
const lineWidth = 2;

function eitherPointDoesntMeetConfidence(a, b, minConfidence) {
  return (a < minConfidence || b < minConfidence);
}

function getAdjacentKeyPoints(keypoints, minConfidence) {
  return connectedPartIndeces.reduce(function (result, _a) {
    let leftJoint = _a[0], rightJoint = _a[1];
    if (eitherPointDoesntMeetConfidence(keypoints[leftJoint].score, keypoints[rightJoint].score, minConfidence)) {
      return result;
    }
    result.push([keypoints[leftJoint], keypoints[rightJoint]]);
    return result;
  }, []);
}

function toTuple({ y, x }) {
  return [y, x];
}

function drawSegment([ay, ax], [by, bx], color, ctx, scaleX = 1, scaleY = 1) {
  ctx.beginPath();
  ctx.moveTo(ax * scaleX, ay * scaleY);
  ctx.lineTo(bx * scaleX, by * scaleY);
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawKeypoints(keypoints, minConfidence, ctx, scaleX = 1, scaleY = 1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];
    if (keypoint.score < minConfidence) {
      continue;
    }
    const { y, x } = keypoint.position;
    ctx.beginPath();
    ctx.arc(x * scaleX, y * scaleY, 3, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

function drawSkeleton(keypoints, minConfidence, ctx, scaleX = 1, scaleY = 1) {
  const adjacentKeyPoints = getAdjacentKeyPoints(keypoints, minConfidence);
  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(toTuple(keypoints[0].position),toTuple(keypoints[1].position), color, ctx, scaleX, scaleY);
  });
}

function drawBoundingBox(keypoints, ctx, scaleX = 1, scaleY = 1) {
  const boundingBox = getBoundingBox(keypoints);
  ctx.beginPath();
  ctx.rect(boundingBox.minX * scaleX, boundingBox.minY * scaleY,
           (boundingBox.maxX - boundingBox.minX) * scaleX, 
           (boundingBox.maxY - boundingBox.minY) * scaleY);
  ctx.strokeStyle = boundingBoxColor;
  ctx.stroke();
}
