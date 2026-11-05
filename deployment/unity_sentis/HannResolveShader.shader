Shader "EdgeStereoINR/HannResolveShader"
{
    Properties
    {
        _MainTex ("Accumulated Eye Texture", 2D) = "black" {}
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

            fixed4 frag (v2f_img i) : SV_Target
            {
                fixed4 acc = tex2D(_MainTex, i.uv);
                float weight = max(acc.a, 0.0001);
                // Normalized luminance/chroma: sum(W * I) / sum(W)
                return fixed4(acc.rgb / weight, 1.0);
            }
            ENDCG
        }
    }
}
