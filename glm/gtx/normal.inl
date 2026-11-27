/// @ref gtx_normal

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> triangleNormal
	(
		vec<3, T, Q> const& p1,
		vec<3, T, Q> const& p2,
		vec<3, T, Q> const& p3
	)
	{
		return normalize(cross(p1 - p2, p1 - p3));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> triangleTangent
	(
		vec<3, T, Q> const& p1,
		vec<3, T, Q> const& p2,
		vec<3, T, Q> const& p3,
		vec<2, T, Q> const& uv1,
		vec<2, T, Q> const& uv2,
		vec<2, T, Q> const& uv3
	)
	{
		vec<3, T, Q> const edge1 = p2 - p1;
		vec<3, T, Q> const edge2 = p3 - p1;
		vec<2, T, Q> const deltaUV1 = uv2 - uv1;
		vec<2, T, Q> const deltaUV2 = uv3 - uv1;

		T const r = T(1) / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
		return normalize((edge1 * deltaUV2.y - edge2 * deltaUV1.y) * r);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> triangleBitangent
	(
		vec<3, T, Q> const& p1,
		vec<3, T, Q> const& p2,
		vec<3, T, Q> const& p3,
		vec<2, T, Q> const& uv1,
		vec<2, T, Q> const& uv2,
		vec<2, T, Q> const& uv3
	)
	{
		vec<3, T, Q> const edge1 = p2 - p1;
		vec<3, T, Q> const edge2 = p3 - p1;
		vec<2, T, Q> const deltaUV1 = uv2 - uv1;
		vec<2, T, Q> const deltaUV2 = uv3 - uv1;

		T const r = T(1) / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
		return normalize((edge2 * deltaUV1.x - edge1 * deltaUV2.x) * r);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T bitangentSign
	(
		vec<3, T, Q> const& normal,
		vec<3, T, Q> const& tangent,
		vec<3, T, Q> const& bitangent
	)
	{
		return dot(cross(normal, tangent), bitangent) < T(0) ? T(-1) : T(1);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER void convertTangentFrame
	(
		mat<3, 3, T, Q> const& basisChange,
		bool flipV,
		vec<3, T, Q> const& normalIn,
		vec<3, T, Q> const& tangentIn,
		T bitangentSignIn,
		vec<3, T, Q>& normalOut,
		vec<3, T, Q>& tangentOut,
		T& bitangentSignOut
	)
	{
		// A reflection (det < 0) reverses cross products, so the bitangent
		// reconstructed as sign * cross(normal, tangent) flips side unless the
		// sign is adjusted. The texture V axis reversal is an independent
		// convention change and contributes its own sign flip.
		T const detSign = determinant(basisChange) < T(0) ? T(-1) : T(1);
		T const vSign = flipV ? T(-1) : T(1);

		normalOut = normalize(basisChange * normalIn);
		tangentOut = normalize(basisChange * tangentIn);
		bitangentSignOut = bitangentSignIn * detSign * vSign;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> convertNormalMapTexel
	(
		vec<3, T, Q> const& texel,
		bool flipV
	)
	{
		return vec<3, T, Q>(texel.x, flipV ? -texel.y : texel.y, texel.z);
	}
}//namespace glm
