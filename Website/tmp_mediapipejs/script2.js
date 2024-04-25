const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');

// Load the MediaPipe pose model.
const pose = new Pose({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
}});

// Set up the model configurations.
pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: false,
  smoothSegmentation: false,
  minDetectionConfidence: 0.1,
  minTrackingConfidence: 0.1
});

// Connect the camera to the model.
pose.onResults(onResults);

// Define a function to handle the results from the model.
function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//   canvasCtx.drawImage(results.segmentationMask, 0, 0,
//                       canvasElement.width, canvasElement.height);

  // Only overwrite existing pixels.
  canvasCtx.globalCompositeOperation = 'source-in';
  canvasCtx.fillStyle = '#00FF00';
  canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

  // Only overwrite missing pixels.
  canvasCtx.globalCompositeOperation = 'destination-atop';
  canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

  canvasCtx.globalCompositeOperation = 'source-over';
  drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
                 {color: '#00FF00', lineWidth: 4});
  drawLandmarks(canvasCtx, results.poseLandmarks,
                {color: '#FF0000', lineWidth: 2});
  canvasCtx.restore();
}

// Start the video and when it's ready, start the model.
// videoElement.onloadeddata = () => {
//   pose.send({image: videoElement});
// };



videoElement.onpause = async function() {
    await pose.send({image: videoElement});

    video.play()
    console.log("play")  

  };
videoElement.onplay = function() {
     setTimeout(()=>{
     console.log("pause")
     video.pause()
     },100)
   };  
   
// videoElement.onended=function() {
//     videoElement.onplay = null
//     videoElement.onpause = null

// };
