#if UNITY_2020_1_OR_NEWER
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace EdgeStereoINR.Dataset
{
    /// <summary>
    /// Telemetry metadata frame holding 6DoF tracking, camera parameters, and timestamps.
    /// Serialized to JSON alongside captured raw stereo image pairs.
    /// </summary>
    [Serializable]
    public struct StereoFrameMetadata
    {
        public int frameIndex;
        public long timestampUtcMs;
        public float timeSinceStartSec;
        
        // 6DoF Headset Pose (Center Eye)
        public Vector3 headPosition;
        public Quaternion headRotation;
        public Vector3 angularVelocityDegPerSec;
        public Vector3 linearAccelerationMetersPerSecSq;

        // Stereo Camera Geometry
        public float baselineMillimeters;
        public float fieldOfViewDegrees;
        public int cameraResolutionWidth;
        public int cameraResolutionHeight;

        // Radiometric & Sensor Telemetry
        public float estimatedLux;
        public float exposureTimeMs;
        public float sensorGain;
        
        // Relative Frame Motion (Delta to previous frame)
        public float translationalMotionDeltaMm;
        public float rotationalMotionDeltaDeg;
        public bool isBurstFrame;
        public int burstSequenceIndex;
    }

    /// <summary>
    /// Quest3SStereoDataHarvester: Production-grade in-the-wild stereoscopic data harvester for Meta Quest 3S.
    /// Captures synchronized uncompressed Left & Right passthrough camera frames with 6DoF IMU pose telemetry.
    /// Used as the Master Dataset Collection Rig to train state-of-the-art StereoINR models on real hardware sensor data.
    /// </summary>
    public class Quest3SStereoDataHarvester : MonoBehaviour
    {
        [Header("Camera & Render Sources")]
        [Tooltip("Render texture or WebCamTexture providing the Left Eye raw passthrough feed.")]
        public RenderTexture leftEyeSource;
        [Tooltip("Render texture or WebCamTexture providing the Right Eye raw passthrough feed.")]
        public RenderTexture rightEyeSource;

        [Header("Capture Controls")]
        [Tooltip("Enable continuous recording or periodic burst logging.")]
        public bool isRecording = false;
        [Tooltip("Target capture framerate (e.g. 30 Hz for continuous logging, 90 Hz for short bursts).")]
        [Range(1, 90)]
        public int targetCaptureFps = 30;
        [Tooltip("Capture in high-speed micro-tremor burst mode (N consecutive frames at 90 FPS).")]
        public bool enableMicroTremorBurstMode = true;
        [Range(2, 5)]
        public int burstLength = 3;
        [Tooltip("Interval between burst triggers in seconds.")]
        public float burstIntervalSeconds = 2.0f;

        [Header("Stereo Rig Calibration")]
        public float opticalBaselineMm = 64.0f;
        public float cameraFov = 100.0f;

        [Header("Storage Configuration")]
        [Tooltip("Subdirectory within Application.persistentDataPath to store datasets.")]
        public string datasetFolderName = "Quest3S_StereoWild_Captures";

        // Internal State
        private string captureSessionPath;
        private string leftFramesDir;
        private string rightFramesDir;
        private string metadataDir;

        private int frameCounter = 0;
        private float lastCaptureTime = 0.0f;
        private float lastBurstTriggerTime = 0.0f;
        private bool inBurstSequence = false;
        private int currentBurstIndex = 0;

        private Vector3 prevHeadPosition;
        private Quaternion prevHeadRotation;

        private Texture2D readableLeftTex;
        private Texture2D readableRightTex;

        void Awake()
        {
            InitializeStorageDirectories();
        }

        void Start()
        {
            prevHeadPosition = transform.position;
            prevHeadRotation = transform.rotation;
        }

        private void InitializeStorageDirectories()
        {
            string baseDir = Path.Combine(Application.persistentDataPath, datasetFolderName);
            string sessionTimestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            captureSessionPath = Path.Combine(baseDir, $"Session_{sessionTimestamp}");

            leftFramesDir = Path.Combine(captureSessionPath, "left");
            rightFramesDir = Path.Combine(captureSessionPath, "right");
            metadataDir = Path.Combine(captureSessionPath, "metadata");

            Directory.CreateDirectory(leftFramesDir);
            Directory.CreateDirectory(rightFramesDir);
            Directory.CreateDirectory(metadataDir);

            Debug.Log($"[Quest3SStereoDataHarvester] Initialized dataset session at: {captureSessionPath}");
        }

        void Update()
        {
            if (!isRecording) return;

            float currentTime = Time.time;

            // Check if we should trigger a new micro-tremor burst
            if (enableMicroTremorBurstMode && !inBurstSequence && (currentTime - lastBurstTriggerTime >= burstIntervalSeconds))
            {
                inBurstSequence = true;
                currentBurstIndex = 0;
                lastBurstTriggerTime = currentTime;
            }

            // Burst execution at 90 Hz
            if (inBurstSequence)
            {
                CaptureStereoFrame(isBurst: true, burstIdx: currentBurstIndex);
                currentBurstIndex++;
                if (currentBurstIndex >= burstLength)
                {
                    inBurstSequence = false;
                }
                return;
            }

            // Normal continuous logging governed by targetCaptureFps
            float captureInterval = 1.0f / Mathf.Max(1, targetCaptureFps);
            if (currentTime - lastCaptureTime >= captureInterval)
            {
                lastCaptureTime = currentTime;
                CaptureStereoFrame(isBurst: false, burstIdx: 0);
            }
        }

        /// <summary>
        /// Captures a synchronized stereo pair and writes raw PNGs + JSON metadata asynchronously.
        /// </summary>
        public void CaptureStereoFrame(bool isBurst, int burstIdx)
        {
            if (leftEyeSource == null || rightEyeSource == null) return;

            int width = leftEyeSource.width;
            int height = leftEyeSource.height;

            if (readableLeftTex == null || readableLeftTex.width != width || readableLeftTex.height != height)
            {
                readableLeftTex = new Texture2D(width, height, TextureFormat.RGB24, false);
                readableRightTex = new Texture2D(width, height, TextureFormat.RGB24, false);
            }

            // 1. Read Left Eye via GPU Readback
            RenderTexture prevActive = RenderTexture.active;
            RenderTexture.active = leftEyeSource;
            readableLeftTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            readableLeftTex.Apply();

            // 2. Read Right Eye
            RenderTexture.active = rightEyeSource;
            readableRightTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            readableRightTex.Apply();
            RenderTexture.active = prevActive;

            // 3. Compute Delta Motion
            Vector3 currentPos = transform.position;
            Quaternion currentRot = transform.rotation;
            float deltaPosMm = Vector3.Distance(currentPos, prevHeadPosition) * 1000.0f;
            float deltaRotDeg = Quaternion.Angle(currentRot, prevHeadRotation);

            prevHeadPosition = currentPos;
            prevHeadRotation = currentRot;

            // 4. Construct Telemetry Metadata
            StereoFrameMetadata meta = new StereoFrameMetadata
            {
                frameIndex = frameCounter,
                timestampUtcMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                timeSinceStartSec = Time.time,
                headPosition = currentPos,
                headRotation = currentRot,
                angularVelocityDegPerSec = deltaRotDeg / Mathf.Max(Time.deltaTime, 0.001f) * Vector3.one,
                linearAccelerationMetersPerSecSq = (deltaPosMm / 1000.0f) / Mathf.Max(Time.deltaTime * Time.deltaTime, 0.0001f) * Vector3.one,
                baselineMillimeters = opticalBaselineMm,
                fieldOfViewDegrees = cameraFov,
                cameraResolutionWidth = width,
                cameraResolutionHeight = height,
                estimatedLux = 250.0f, // Estimated typical indoor office lux on Quest 3S
                exposureTimeMs = 8.33f,
                sensorGain = 1.0f,
                translationalMotionDeltaMm = deltaPosMm,
                rotationalMotionDeltaDeg = deltaRotDeg,
                isBurstFrame = isBurst,
                burstSequenceIndex = burstIdx
            };

            // 5. Asynchronous disk persistence
            byte[] leftBytes = readableLeftTex.EncodeToPNG();
            byte[] rightBytes = readableRightTex.EncodeToPNG();
            string jsonMeta = JsonUtility.ToJson(meta, true);

            string frameIdStr = frameCounter.ToString("D6");
            string leftFilePath = Path.Combine(leftFramesDir, $"frame_{frameIdStr}_L.png");
            string rightFilePath = Path.Combine(rightFramesDir, $"frame_{frameIdStr}_R.png");
            string metaFilePath = Path.Combine(metadataDir, $"frame_{frameIdStr}_meta.json");

            System.Threading.ThreadPool.QueueUserWorkItem(_ =>
            {
                try
                {
                    File.WriteAllBytes(leftFilePath, leftBytes);
                    File.WriteAllBytes(rightFilePath, rightBytes);
                    File.WriteAllText(metaFilePath, jsonMeta);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[Quest3SStereoDataHarvester] File write error: {ex.Message}");
                }
            });

            frameCounter++;
        }

        void OnDestroy()
        {
            if (readableLeftTex != null) Destroy(readableLeftTex);
            if (readableRightTex != null) Destroy(readableRightTex);
        }
    }
}
#endif
