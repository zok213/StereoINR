using System;
using UnityEngine;

namespace ActiveStereo.EdgeStereoINR
{
    /// <summary>
    /// Temporal Stabilization & Anti-Flicker Extension for EdgeStereoINR.
    /// Runs a zero-copy GPU temporal reprojection pass on Meta Quest 3S (Adreno 740).
    /// Eliminates high-frequency shimmering / pixel scintillation between consecutive VR frames
    /// while rejecting ghosting artifacts in rapid head movement.
    /// </summary>
    [RequireComponent(typeof(TiledStereoSentisStreamer))]
    public class TemporalStereoSentisStreamer : MonoBehaviour
    {
        [Header("Temporal Configuration")]
        [Tooltip("Enables temporal hysteresis accumulation")]
        public bool enableTemporalStabilization = true;

        [Range(0.0f, 0.95f)]
        [Tooltip("Temporal blend weight in static regions (0.85 = 85% history, 15% new)")]
        public float blendFactor = 0.85f;

        [Range(0.02f, 0.5f)]
        [Tooltip("Color discrepancy threshold triggering disocclusion reset (ghosting rejection)")]
        public float disocclusionThreshold = 0.15f;

        [Header("6DoF Rotational Warp (Ghosting Prevention)")]
        [Tooltip("Applies IMU/Head rotation warp to history texture before blending")]
        public bool enableRotationalWarp = true;
        [Tooltip("Optional reference to CenterEyeAnchor transform (auto-finds Camera.main if null)")]
        public Transform headTrackingTransform;
        private Quaternion _lastHeadRotation = Quaternion.identity;

        [Header("Shader References")]
        public Material temporalBlendMaterial;

        // References to base streamer
        private TiledStereoSentisStreamer _baseStreamer;

        // Ping-pong history textures (allocated 100% on GPU)
        private RenderTexture _historyRTLeft;
        private RenderTexture _historyRTRight;
        private RenderTexture _tempBlitRTLeft;
        private RenderTexture _tempBlitRTRight;
        private bool _isFirstFrame = true;

        void Awake()
        {
            _baseStreamer = GetComponent<TiledStereoSentisStreamer>();
        }

        void Start()
        {
            InitializeHistoryBuffers();
        }

        void OnDestroy()
        {
            ReleaseHistoryBuffers();
        }

        private void InitializeHistoryBuffers()
        {
            if (_baseStreamer == null || _baseStreamer.leftOutputEyeTarget == null) return;

            int w = _baseStreamer.leftOutputEyeTarget.width;
            int h = _baseStreamer.leftOutputEyeTarget.height;

            _historyRTLeft = new RenderTexture(w, h, 0, RenderTextureFormat.ARGBHalf);
            _historyRTRight = new RenderTexture(w, h, 0, RenderTextureFormat.ARGBHalf);
            _tempBlitRTLeft = new RenderTexture(w, h, 0, RenderTextureFormat.ARGBHalf);
            _tempBlitRTRight = new RenderTexture(w, h, 0, RenderTextureFormat.ARGBHalf);

            _historyRTLeft.Create();
            _historyRTRight.Create();
            _tempBlitRTLeft.Create();
            _tempBlitRTRight.Create();

            _isFirstFrame = true;
            Debug.Log($"[TemporalStereoStreamer] History buffers allocated ({w}x{h}).");
        }

        /// <summary>
        /// Post-processes the resolved stereo eye targets with temporal anti-flicker blending.
        /// Call immediately after TiledStereoSentisStreamer.ProcessStereoFrame().
        /// </summary>
        public void ApplyTemporalFilter()
        {
            if (!enableTemporalStabilization || temporalBlendMaterial == null) return;
            if (_baseStreamer == null || _baseStreamer.leftOutputEyeTarget == null) return;

            RenderTexture outL = _baseStreamer.leftOutputEyeTarget;
            RenderTexture outR = _baseStreamer.rightOutputEyeTarget;

            if (_isFirstFrame)
            {
                // First frame bootstrap: copy direct output to history
                Graphics.Blit(outL, _historyRTLeft);
                Graphics.Blit(outR, _historyRTRight);
                _isFirstFrame = false;
                return;
            }

            temporalBlendMaterial.SetFloat("_BlendFactor", blendFactor);
            temporalBlendMaterial.SetFloat("_DisocclusionThreshold", disocclusionThreshold);

            Vector2 deltaRotUV = Vector2.zero;
            if (enableRotationalWarp)
            {
                Transform trackingT = headTrackingTransform != null ? headTrackingTransform : (Camera.main != null ? Camera.main.transform : null);
                if (trackingT != null)
                {
                    Quaternion currentRot = trackingT.rotation;
                    if (!_isFirstFrame)
                    {
                        Quaternion deltaRot = Quaternion.Inverse(_lastHeadRotation) * currentRot;
                        Vector3 euler = deltaRot.eulerAngles;
                        float yaw = (euler.y > 180f) ? euler.y - 360f : euler.y;
                        float pitch = (euler.x > 180f) ? euler.x - 360f : euler.x;
                        deltaRotUV = new Vector2(yaw / 110.0f, -pitch / 96.0f);
                    }
                    _lastHeadRotation = currentRot;
                }
            }
            temporalBlendMaterial.SetVector("_DeltaRotation", new Vector4(deltaRotUV.x, deltaRotUV.y, 0, 0));

            // Left Eye Temporal Pass
            temporalBlendMaterial.SetTexture("_HistoryTex", _historyRTLeft);
            Graphics.Blit(outL, _tempBlitRTLeft, temporalBlendMaterial);
            Graphics.Blit(_tempBlitRTLeft, outL);
            Graphics.Blit(outL, _historyRTLeft);

            // Right Eye Temporal Pass
            temporalBlendMaterial.SetTexture("_HistoryTex", _historyRTRight);
            Graphics.Blit(outR, _tempBlitRTRight, temporalBlendMaterial);
            Graphics.Blit(_tempBlitRTRight, outR);
            Graphics.Blit(outR, _historyRTRight);
        }

        private void ReleaseHistoryBuffers()
        {
            if (_historyRTLeft != null) { _historyRTLeft.Release(); _historyRTLeft = null; }
            if (_historyRTRight != null) { _historyRTRight.Release(); _historyRTRight = null; }
            if (_tempBlitRTLeft != null) { _tempBlitRTLeft.Release(); _tempBlitRTLeft = null; }
            if (_tempBlitRTRight != null) { _tempBlitRTRight.Release(); _tempBlitRTRight = null; }
        }
    }
}
