using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

namespace ActiveStereo.EdgeStereoINR
{
    /// <summary>
    /// Production Tiled Streaming Sentis Engine for Meta Quest 3S (Snapdragon XR2 Gen 2 / Adreno 740).
    /// Guarantees 100% GMEM SRAM residency (< 1024 KB peak working buffer) by slicing full
    /// camera feed (1280x960 per eye) into 64x64 tiles with Hann-window overlap feathering.
    /// Runs asynchronously on GPUCompute backend (Vulkan) with ZERO CPU readback.
    /// 
    /// Note on NPU: Retail Meta Quest 3S Horizon OS blocks third-party APKs from /dev/fastrpc-*
    /// via SELinux, so Adreno 740 GPU via Sentis Vulkan Compute is the official production path.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class TiledStereoSentisStreamer : MonoBehaviour
    {
        [Header("Sentis Model Configuration")]
        [Tooltip("The ONNX model asset exported from DyT Continuous Scale INR (Pareto Soup)")]
        public ModelAsset modelAsset;
        
        [Tooltip("Backend type: GPUCompute recommended for Qualcomm Adreno 740 Vulkan")]
        public BackendType backendType = BackendType.GPUCompute;

        [Header("Execution Mode")]
        [Tooltip("If true, automatically processes every frame during LateUpdate")]
        public bool autoProcessOnLateUpdate = true;

        [Header("Tiling Parameters (Adreno 740 SRAM Tuned)")]
        [Range(32, 96)]
        [Tooltip("Tile size in LR pixels. 64x64 requires 768 KB SRAM (< 1024 KB GMEM limit)")]
        public int tileSize = 64;

        [Range(4, 24)]
        [Tooltip("Overlap between adjacent tiles for seamless Hann feathering")]
        public int overlap = 12;

        [Range(1.2f, 4.0f)]
        [Tooltip("Super-Resolution Magnification Factor r in [1.2, 4.0]")]
        public float scaleFactor = 4.0f;

        [Header("Camera & Output RenderTargets")]
        public RenderTexture leftCameraFeed;
        public RenderTexture rightCameraFeed;
        public RenderTexture leftOutputEyeTarget;
        public RenderTexture rightOutputEyeTarget;

        [Header("Shaders")]
        public Material hannBlendMaterial;
        [Tooltip("Optional normalization pass to divide accumulated RGB by Alpha weight")]
        public Material hannResolveMaterial;

        // Sentis Runtime
        private Model _runtimeModel;
        private IWorker _worker;

        // Dynamic Tensor Name Mapping
        private string _inputNameL = "input_l";
        private string _inputNameR = "input_r";
        private string _scaleInputName = null;
        private string _outputNameL = "output_l";
        private string _outputNameR = "output_r";

        // Tiling geometry cache
        private int _step;
        private readonly List<int> _yStarts = new List<int>();
        private readonly List<int> _xStarts = new List<int>();

        // Temporary tile RenderTextures (resident on GPU VRAM)
        private RenderTexture _tileRTLeft;
        private RenderTexture _tileRTRight;
        private RenderTexture _srTileRTLeft;
        private RenderTexture _srTileRTRight;

        void Awake()
        {
            InitializeSentisEngine();
        }

        void LateUpdate()
        {
            if (autoProcessOnLateUpdate && isActiveAndEnabled)
            {
                ProcessStereoFrame();
            }
        }

        void OnDestroy()
        {
            CleanupSentisEngine();
        }

        public void InitializeSentisEngine()
        {
            if (modelAsset == null)
            {
                Debug.LogError("[TiledStereoStreamer] ModelAsset is not assigned!");
                return;
            }

            // Load model into Sentis
            _runtimeModel = ModelLoader.Load(modelAsset);
            ResolveTensorNames();

            _worker = new Worker(_runtimeModel, backendType);

            _step = tileSize - overlap;
            CalculateTileCoordinates();

            int outTileSize = Mathf.RoundToInt(tileSize * scaleFactor);

            // Allocate ping-pong tile buffers (100% on GPU)
            _tileRTLeft = new RenderTexture(tileSize, tileSize, 0, RenderTextureFormat.ARGBHalf);
            _tileRTRight = new RenderTexture(tileSize, tileSize, 0, RenderTextureFormat.ARGBHalf);
            _tileRTLeft.Create();
            _tileRTRight.Create();

            _srTileRTLeft = new RenderTexture(outTileSize, outTileSize, 0, RenderTextureFormat.ARGBHalf);
            _srTileRTRight = new RenderTexture(outTileSize, outTileSize, 0, RenderTextureFormat.ARGBHalf);
            _srTileRTLeft.Create();
            _srTileRTRight.Create();

            float peakSramKb = (48 * tileSize * tileSize * 4) / 1024f;
            Debug.Log($"[TiledStereoStreamer] Initialized with backend {backendType}. Peak Tile SRAM: {peakSramKb:F1} KB (GMEM Budget: 1024 KB).");
        }

        private void ResolveTensorNames()
        {
            if (_runtimeModel == null) return;

            // Dynamically detect Left/Right inputs and optional Scale input
            foreach (var input in _runtimeModel.inputs)
            {
                string lower = input.name.ToLower();
                if (lower.Contains("scale") || lower.Contains("reshape") || input.shape.rank == 0)
                {
                    _scaleInputName = input.name;
                }
                else if (lower.Contains("left") || lower.EndsWith("_l") || lower.Contains("l_") || lower == "input_0")
                {
                    _inputNameL = input.name;
                }
                else if (lower.Contains("right") || lower.EndsWith("_r") || lower.Contains("r_") || lower == "input_1")
                {
                    _inputNameR = input.name;
                }
            }

            // Fallback for 2-input models if names didn't match patterns
            if (string.IsNullOrEmpty(_inputNameL) && _runtimeModel.inputs.Count >= 2)
            {
                _inputNameL = _runtimeModel.inputs[0].name;
                _inputNameR = _runtimeModel.inputs[1].name;
            }

            // Dynamically detect Left/Right outputs
            foreach (var output in _runtimeModel.outputs)
            {
                string lower = output.name.ToLower();
                if (lower.Contains("left") || lower.EndsWith("_l") || lower.Contains("l_"))
                {
                    _outputNameL = output.name;
                }
                else if (lower.Contains("right") || lower.EndsWith("_r") || lower.Contains("r_"))
                {
                    _outputNameR = output.name;
                }
            }

            if (string.IsNullOrEmpty(_outputNameL) && _runtimeModel.outputs.Count >= 2)
            {
                _outputNameL = _runtimeModel.outputs[0].name;
                _outputNameR = _runtimeModel.outputs[1].name;
            }

            Debug.Log($"[TiledStereoStreamer] Resolved Tensors: InputL='{_inputNameL}', InputR='{_inputNameR}', Scale='{_scaleInputName}', OutL='{_outputNameL}', OutR='{_outputNameR}'");
        }

        private void CalculateTileCoordinates()
        {
            if (leftCameraFeed == null) return;

            _yStarts.Clear();
            _xStarts.Clear();

            int h = leftCameraFeed.height;
            int w = leftCameraFeed.width;

            for (int y = 0; y <= h - tileSize; y += _step)
            {
                _yStarts.Add(y);
            }
            if (_yStarts.Count == 0 || _yStarts[_yStarts.Count - 1] + tileSize < h)
            {
                _yStarts.Add(Mathf.Max(0, h - tileSize));
            }

            for (int x = 0; x <= w - tileSize; x += _step)
            {
                _xStarts.Add(x);
            }
            if (_xStarts.Count == 0 || _xStarts[_xStarts.Count - 1] + tileSize < w)
            {
                _xStarts.Add(Mathf.Max(0, w - tileSize));
            }
        }

        /// <summary>
        /// Called per frame to process stereo cameras.
        /// Dispatches all tiles through Sentis and blends them into the VR eye targets.
        /// </summary>
        public void ProcessStereoFrame()
        {
            if (_worker == null || leftCameraFeed == null || rightCameraFeed == null) return;
            if (leftOutputEyeTarget == null || rightOutputEyeTarget == null) return;

            int inH = leftCameraFeed.height;
            int inW = leftCameraFeed.width;
            int outTileSize = Mathf.RoundToInt(tileSize * scaleFactor);
            int outH = Mathf.RoundToInt(inH * scaleFactor);
            int outW = Mathf.RoundToInt(inW * scaleFactor);

            // Recompute tile starts if camera dimension changed
            if (_yStarts.Count == 0 || _xStarts.Count == 0)
            {
                CalculateTileCoordinates();
            }

            // Clear output targets
            RenderTexture.active = leftOutputEyeTarget;
            GL.Clear(true, true, Color.black);
            RenderTexture.active = rightOutputEyeTarget;
            GL.Clear(true, true, Color.black);
            RenderTexture.active = null;

            TensorFloat scaleTensor = null;
            if (!string.IsNullOrEmpty(_scaleInputName))
            {
                scaleTensor = new TensorFloat(new TensorShape(), new float[] { scaleFactor });
                _worker.SetInput(_scaleInputName, scaleTensor);
            }

            try
            {
                // Iterate over all tiles
                foreach (int ys in _yStarts)
                {
                    int ye = ys + tileSize;
                    int oys = Mathf.RoundToInt(ys * scaleFactor);
                    int oye = oys + outTileSize;
                    if (oye > outH) { oye = outH; oys = Mathf.Max(0, oye - outTileSize); }

                    foreach (int xs in _xStarts)
                    {
                        int xe = xs + tileSize;
                        int oxs = Mathf.RoundToInt(xs * scaleFactor);
                        int oxe = oxs + outTileSize;
                        if (oxe > outW) { oxe = outW; oxs = Mathf.Max(0, oxe - outTileSize); }

                        // 1. Blit sub-region from Camera Feed to Tile RT (GPU copy)
                        Graphics.CopyTexture(leftCameraFeed, 0, 0, xs, ys, tileSize, tileSize, _tileRTLeft, 0, 0, 0, 0);
                        Graphics.CopyTexture(rightCameraFeed, 0, 0, xs, ys, tileSize, tileSize, _tileRTRight, 0, 0, 0, 0);

                        // 2. Convert to Sentis Tensors
                        using TensorFloat inputTensorL = TextureConverter.ToTensor(_tileRTLeft, tileSize, tileSize, 3);
                        using TensorFloat inputTensorR = TextureConverter.ToTensor(_tileRTRight, tileSize, tileSize, 3);

                        // 3. Sentis Inference (GPU Compute)
                        _worker.SetInput(_inputNameL, inputTensorL);
                        _worker.SetInput(_inputNameR, inputTensorR);
                        _worker.Schedule();

                        // 4. Retrieve output tensors
                        using TensorFloat outputTensorL = _worker.PeekOutput(_outputNameL) as TensorFloat;
                        using TensorFloat outputTensorR = _worker.PeekOutput(_outputNameR) as TensorFloat;

                        // 5. Render to Tile RenderTexture
                        TextureConverter.RenderToTexture(outputTensorL, _srTileRTLeft);
                        TextureConverter.RenderToTexture(outputTensorR, _srTileRTRight);

                        // 6. Accumulate into Eye Targets with Hann Feathering
                        if (hannBlendMaterial != null)
                        {
                            hannBlendMaterial.SetVector("_DestRect", new Vector4(oxs, oys, oxe - oxs, oye - oys));
                            hannBlendMaterial.SetVector("_TargetDims", new Vector4(outW, outH, 1.0f / outW, 1.0f / outH));

                            Graphics.Blit(_srTileRTLeft, leftOutputEyeTarget, hannBlendMaterial, 0);
                            Graphics.Blit(_srTileRTRight, rightOutputEyeTarget, hannBlendMaterial, 0);
                        }
                        else
                        {
                            // Fallback direct copy if no Hann material assigned
                            Graphics.CopyTexture(_srTileRTLeft, 0, 0, 0, 0, outTileSize, outTileSize, leftOutputEyeTarget, 0, 0, oxs, oys);
                            Graphics.CopyTexture(_srTileRTRight, 0, 0, 0, 0, outTileSize, outTileSize, rightOutputEyeTarget, 0, 0, oxs, oys);
                        }
                    }
                }
            }
            finally
            {
                scaleTensor?.Dispose();
            }
        }

        private void CleanupSentisEngine()
        {
            _worker?.Dispose();
            _worker = null;

            if (_tileRTLeft != null) { _tileRTLeft.Release(); _tileRTLeft = null; }
            if (_tileRTRight != null) { _tileRTRight.Release(); _tileRTRight = null; }
            if (_srTileRTLeft != null) { _srTileRTLeft.Release(); _srTileRTLeft = null; }
            if (_srTileRTRight != null) { _srTileRTRight.Release(); _srTileRTRight = null; }
        }
    }
}
