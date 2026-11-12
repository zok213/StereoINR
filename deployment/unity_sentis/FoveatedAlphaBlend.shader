Shader "QuestStereoINR/FoveatedAlphaBlend"
{
    Properties
    {
        _MainTex ("Peripheral Bilinear Feed", 2D) = "black" {}
        _FoveaSRTex ("Neural Super-Resolved Fovea (RGB=W*Col, A=W)", 2D) = "black" {}
        _GazeCenterUV ("Gaze Center in UV [0,1]", Vector) = (0.5, 0.5, 0, 0)
        _DeltaReprojectUV ("Late-Stage Reprojection Delta UV", Vector) = (0, 0, 0, 0)
        _InnerRadiusUV ("Inner Fovea Radius (Psychophysical 7.5 deg)", Float) = 0.0781 // 100/1280
        _OuterRadiusUV ("Outer Parafovea Radius (Psychophysical 13.5 deg)", Float) = 0.1406 // 180/1280
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Overlay" }
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
            sampler2D _FoveaSRTex;
            float4 _GazeCenterUV;
            float4 _DeltaReprojectUV;
            float _InnerRadiusUV;
            float _OuterRadiusUV;

            #define PI 3.14159265359
            #define EPSILON 1e-4

            fixed4 frag (v2f_img i) : SV_Target
            {
                // 1. Sample Background Periphery (Hardware Bilinear / Bicubic)
                fixed4 periphColor = tex2D(_MainTex, i.uv);

                // 2. Compute Radial Distance from Gaze Center (Aspect-ratio corrected for 4:3)
                float2 deltaCoord = i.uv - _GazeCenterUV.xy;
                deltaCoord.y *= (960.0 / 1280.0); // Correct for 4:3 camera aspect ratio
                float dist = length(deltaCoord);

                // Early exit for distant periphery (> outer radius)
                if (dist >= _OuterRadiusUV)
                {
                    return periphColor;
                }

                // 3. Sample Reprojected Neural Fovea Patch
                // Apply 6DoF late-stage rotational delta to eliminate swimming
                float2 reprojectedFoveaUV = i.uv - _DeltaReprojectUV.xy;
                fixed4 neuralSample = tex2D(_FoveaSRTex, reprojectedFoveaUV);

                // 4. Exact Partition-of-Unity Normalization (Shepard Normalization)
                // Eliminates -83% luminance dips at tile seams and radial dark banding
                float weightSum = neuralSample.a;
                fixed3 foveaCol = (weightSum > EPSILON) ? (neuralSample.rgb / weightSum) : periphColor.rgb;

                // 5. Compute C1 Continuous Cosine Alpha Feathering across [7.5 deg, 13.5 deg]
                float alpha = 1.0;
                if (dist > _InnerRadiusUV)
                {
                    float t = (dist - _InnerRadiusUV) / (_OuterRadiusUV - _InnerRadiusUV);
                    alpha = 0.5 * (1.0 + cos(PI * t)); // Smoothstep cosine in [0, 1]
                }

                // If accumulated tile coverage terminates before outer radius, taper alpha safely
                float coverageFade = saturate(weightSum / 0.1);
                alpha *= coverageFade;

                // 6. Alpha Blend Neural Fovea over Bilinear Background
                float3 finalColor = lerp(periphColor.rgb, foveaCol, alpha);
                return fixed4(finalColor, 1.0);
            }
            ENDCG
        }
    }
}
