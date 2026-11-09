Shader "EdgeStereoINR/WallisEqualizerShader"
{
    Properties
    {
        _MainTex ("Camera Feed", 2D) = "white" {}
        _TargetMean ("Desired Mean Luminance", Range(0.1, 0.9)) = 0.5
        _TargetStd ("Desired Standard Deviation", Range(0.05, 0.5)) = 0.22
        _BrightnessFactor ("Brightness Enforcement (b)", Range(0.0, 1.0)) = 0.75
        _ContrastFactor ("Contrast Enforcement (c)", Range(0.0, 1.0)) = 0.85
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
            float4 _MainTex_TexelSize;
            float _TargetMean;
            float _TargetStd;
            float _BrightnessFactor;
            float _ContrastFactor;

            // Approximates local 5x5 mean luminance
            float computeLocalMean(float2 uv)
            {
                float sum = 0.0;
                float2 step = _MainTex_TexelSize.xy * 2.0;

                for (int y = -2; y <= 2; y++)
                {
                    for (int x = -2; x <= 2; x++)
                    {
                        float3 c = tex2D(_MainTex, uv + float2(x, y) * step).rgb;
                        float lum = dot(c, float3(0.299, 0.587, 0.114));
                        sum += lum;
                    }
                }
                return sum / 25.0;
            }

            // Approximates local 5x5 standard deviation
            float computeLocalStd(float2 uv, float mean)
            {
                float varSum = 0.0;
                float2 step = _MainTex_TexelSize.xy * 2.0;

                for (int y = -2; y <= 2; y++)
                {
                    for (int x = -2; x <= 2; x++)
                    {
                        float3 c = tex2D(_MainTex, uv + float2(x, y) * step).rgb;
                        float lum = dot(c, float3(0.299, 0.587, 0.114));
                        varSum += (lum - mean) * (lum - mean);
                    }
                }
                return sqrt(varSum / 25.0);
            }

            fixed4 frag (v2f_img i) : SV_Target
            {
                fixed4 original = tex2D(_MainTex, i.uv);
                float mean = computeLocalMean(i.uv);
                float std = computeLocalStd(i.uv, mean);

                // Wallis transformation gain and offset
                float contrastGain = (_ContrastFactor * _TargetStd) / (_ContrastFactor * std + (1.0 - _ContrastFactor) * _TargetStd + 1e-4);
                float brightnessOffset = _BrightnessFactor * _TargetMean + (1.0 - _BrightnessFactor) * mean;

                // Apply Wallis contrast/brightness equalization
                float3 equalized = (original.rgb - mean) * contrastGain + brightnessOffset;
                equalized = saturate(equalized);

                return fixed4(equalized, original.a);
            }
            ENDCG
        }
    }
}
