// ISTEduca - Detecci\u00f3n de Poses con MediaPipe
// Sistema de detecci\u00f3n de poses mejorado con informaci\u00f3n educativa

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

// Variables globales
let poseLandmarker = undefined;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastVideoTime = -1;
let detectionStats = {
  poseCount: 0,
  confidence: 0,
  status: 'Esperando...'
};

// Elementos del DOM
const webcamButton = document.getElementById("webcamButton");
const videoContainer = document.getElementById("videoContainer");
const challengesSection = document.getElementById("challenges");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

// Elementos de estad\u00edsticas
const poseCountElement = document.getElementById("poseCount");
const confidenceElement = document.getElementById("confidence");
const statusElement = document.getElementById("status");

// Inicializar MediaPipe
const createPoseLandmarker = async () => {
  updateStatus('Cargando modelo de IA...');
  
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
        delegate: "GPU"
      },
      runningMode: runningMode,
      numPoses: 2
    });
    
    updateStatus('Modelo cargado \u2713');
    console.log("MediaPipe PoseLandmarker cargado correctamente");
  } catch (error) {
    console.error("Error al cargar MediaPipe:", error);
    updateStatus('Error al cargar el modelo');
  }
};

// Actualizar estado en la UI
function updateStatus(status) {
  detectionStats.status = status;
  if (statusElement) {
    statusElement.textContent = status;
  }
}

// Actualizar estad\u00edsticas en la UI
function updateStats() {
  if (poseCountElement) {
    poseCountElement.textContent = detectionStats.poseCount;
  }
  if (confidenceElement) {
    const conf = detectionStats.confidence > 0 
      ? `${(detectionStats.confidence * 100).toFixed(1)}%` 
      : '-';
    confidenceElement.textContent = conf;
  }
}

// Verificar soporte de c\u00e1mara
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// Configurar bot\u00f3n de c\u00e1mara
if (hasGetUserMedia()) {
  webcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() no es soportado por tu navegador");
  updateStatus('C\u00e1mara no disponible');
  webcamButton.disabled = true;
}

// Habilitar/deshabilitar c\u00e1mara
async function enableCam(event) {
  if (!poseLandmarker) {
    updateStatus('Espera, cargando modelo...');
    return;
  }

  if (webcamRunning === true) {
    // Detener c\u00e1mara
    webcamRunning = false;
    webcamButton.querySelector('.button-text').textContent = "Activar C\u00e1mara";
    webcamButton.classList.remove('active');
    videoContainer.classList.add('hidden');
    challengesSection.classList.add('hidden');
    updateStatus('C\u00e1mara desactivada');
    
    // Detener el stream
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null;
    }
  } else {
    // Iniciar c\u00e1mara
    webcamRunning = true;
    webcamButton.querySelector('.button-text').textContent = "Desactivar C\u00e1mara";
    webcamButton.classList.add('active');
    videoContainer.classList.remove('hidden');
    challengesSection.classList.remove('hidden');
    updateStatus('Iniciando c\u00e1mara...');

    // Configuraci\u00f3n de video
    const constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      video.addEventListener("loadeddata", () => {
        updateStatus('C\u00e1mara activa - Detectando poses...');
        predictWebcam();
      });
    } catch (error) {
      console.error("Error al acceder a la c\u00e1mara:", error);
      updateStatus('Error: No se pudo acceder a la c\u00e1mara');
      webcamRunning = false;
      webcamButton.querySelector('.button-text').textContent = "Activar C\u00e1mara";
      webcamButton.classList.remove('active');
    }
  }
}

// Predecir poses desde webcam
async function predictWebcam() {
  // Ajustar tama\u00f1o del canvas
  if (video.videoWidth > 0) {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }

  // Cambiar a modo VIDEO si es necesario
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  let startTimeMs = performance.now();
  
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    
    try {
      poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
        // Limpiar canvas
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        // Actualizar contador de poses
        detectionStats.poseCount = result.landmarks.length;
        
        // Dibujar landmarks
        const drawingUtils = new DrawingUtils(canvasCtx);
        
        for (const landmark of result.landmarks) {
          // Calcular confianza promedio
          const avgConfidence = landmark.reduce((sum, point) => 
            sum + (point.visibility || 0), 0) / landmark.length;
          detectionStats.confidence = avgConfidence;
          
          // Dibujar puntos con colores personalizados
          drawingUtils.drawLandmarks(landmark, {
            color: '#C74398',
            fillColor: '#4A3168',
            radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 8, 2)
          });
          
          // Dibujar conexiones
          drawingUtils.drawConnectors(
            landmark, 
            PoseLandmarker.POSE_CONNECTIONS,
            { color: '#C74398', lineWidth: 3 }
          );
        }
        
        canvasCtx.restore();
        updateStats();
        
        if (detectionStats.poseCount > 0) {
          updateStatus(`\u2705 Detectando ${detectionStats.poseCount} pose(s)`);
        } else {
          updateStatus('\ud83d\udc40 Esperando persona en cuadro...');
        }
      });
    } catch (error) {
      console.error("Error en detecci\u00f3n:", error);
    }
  }

  // Continuar predicci\u00f3n si la c\u00e1mara est\u00e1 activa
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// Inicializar la aplicaci\u00f3n
createPoseLandmarker();

// Mensaje de bienvenida en consola
console.log('%c\ud83c\udf93 ISTEduca - Detecci\u00f3n de Poses con IA ', 
  'background: linear-gradient(135deg, #C74398, #4A3168); color: white; padding: 10px 20px; font-size: 16px; font-weight: bold; border-radius: 5px;');
console.log('%cPowered by MediaPipe \ud83e\udd16', 
  'color: #4A3168; font-size: 14px; font-weight: bold;');
    const canvasCtx = canvas.getContext("2d");
    const drawingUtils = new DrawingUtils(canvasCtx);
    for (const landmark of result.landmarks) {
      drawingUtils.drawLandmarks(landmark, {
        radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1)
      });
      drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
    }
  });
}

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam") as HTMLVideoElement;
const canvasElement = document.getElementById(
  "output_canvas"
) as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!poseLandmarker) {
    console.log("Wait! poseLandmaker not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  }

  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastVideoTime = -1;
async function predictWebcam() {
  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      for (const landmark of result.landmarks) {
        drawingUtils.drawLandmarks(landmark, {
          radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1)
        });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
      }
      canvasCtx.restore();
    });
  }

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
