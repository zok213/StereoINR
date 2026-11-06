Shader "EdgeStereoINR/TemporalBlendShader"
{
    Properties
    {
        _MainTex ("Current Super-Resolved Eye Texture", 2D) = "black" {}
        _HistoryTex ("Previous Frame History Texture", 2D) = "black" {}
        _MotionTex ("Motion Vectors / Disparity Offset", 2D) = "gray" {}
        _BlendFactor ("Base Temporal Hysteresis (0.7 - 0.9)", Range(0.0, 0.95)) = 0.85
        _DisocclusionThreshold ("Disocclusion Color Delta", Range(0.01, 0.5)) = 0.15
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 100

        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _HistoryTex;
            sampler2D _MotionTex;
            float4 _DeltaRotation;
            float _BlendFactor;
            float _DisocclusionThreshold;

            fixed4 frag (v2f_img i) : SV_Target
            {
                // Current sharp neural super-resolution tile
                fixed4 current = tex2D(_MainTex, i.uv);

                // Sample motion displacement + 6DoF head rotational warp
                float2 motion = (tex2D(_MotionTex, i.uv).rg - 0.5f) * 0.1f + _DeltaRotation.xy;
                float2 historyUV = clamp(i.uv - motion, 0.001f, 0.999f);

                // Sample reprojected history
                fixed4 history = tex2D(_HistoryTex, historyUV);

                // Measure photometric disocclusion error
                float colorDiff = length(current.rgb - history.rgb);

                // Dynamic hysteresis: in disoccluded areas, blend = 0 (reject ghosting); in static areas, blend = _BlendFactor
                float dynamicBlend = _BlendFactor * saturate(1.0f - (colorDiff / _DisocclusionThreshold));

                // Anti-flicker temporal accumulation
                float3 finalRGB = lerp(current.rgb, history.rgb, dynamicBlend);

                return fixed4(finalRGB, 1.0f);
            }
            ENDCG
        }
    }
}
