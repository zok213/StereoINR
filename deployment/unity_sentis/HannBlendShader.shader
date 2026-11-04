Shader "EdgeStereoINR/HannBlendShader"
{
    Properties
    {
        _MainTex ("Tile SR Texture", 2D) = "black" {}
        _DestRect ("Destination Rect (X, Y, W, H)", Vector) = (0, 0, 256, 256)
        _TargetDims ("Target Dims (W, H, 1/W, 1/H)", Vector) = (1280, 960, 0.00078125, 0.00104166)
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Transparent" }
        LOD 100

        // Hardware Additive Blending
        Blend One One
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _DestRect;   // (oxs, oys, tileW, tileH)
            float4 _TargetDims; // (targetW, targetH, invW, invH)

            #define PI 3.14159265359

            v2f vert (appdata v)
            {
                v2f o;
                // Map tile vertices to destination rectangle in normalized device coordinates (NDC)
                float2 pixelPos = _DestRect.xy + v.uv * _DestRect.zw;
                float2 ndc = (pixelPos * _TargetDims.zw) * 2.0 - 1.0;
                #if UNITY_UV_STARTS_AT_TOP
                ndc.y = -ndc.y;
                #endif
                o.vertex = float4(ndc, 0.0, 1.0);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);

                // Compute 2D Hann window weight in [0, 1]
                float wy = 0.5 * (1.0 - cos(2.0 * PI * i.uv.y));
                float wx = 0.5 * (1.0 - cos(2.0 * PI * i.uv.x));
                float weight = max(wy * wx, 0.0001);

                return fixed4(col.rgb * weight, weight);
            }
            ENDCG
        }
    }
}
