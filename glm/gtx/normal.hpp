/// @ref gtx_normal
/// @file glm/gtx/normal.hpp
///
/// @see core (dependence)
/// @see gtx_extented_min_max (dependence)
///
/// @defgroup gtx_normal GLM_GTX_normal
/// @ingroup gtx
///
/// Include <glm/gtx/normal.hpp> to use the features of this extension.
///
/// Compute the normal of a triangle, as well as per-triangle tangent frames
/// for normal mapping, including conversion of tangent frames between
/// authoring-tool and renderer conventions (geometric change of basis and
/// texture V axis direction are controlled independently).

#pragma once

// Dependency:
#include "../glm.hpp"

#ifndef GLM_ENABLE_EXPERIMENTAL
#	error "GLM: GLM_GTX_normal is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it."
#elif GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTX_normal extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_normal
	/// @{

	/// Computes triangle normal from triangle points.
	///
	/// @see gtx_normal
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> triangleNormal(vec<3, T, Q> const& p1, vec<3, T, Q> const& p2, vec<3, T, Q> const& p3);

	/// Computes the tangent of a triangle from triangle points and texture coordinates.
	/// The tangent points along the texture U axis in object space.
	/// The frame is computed per triangle, so texture bases on both sides of a UV
	/// seam are preserved instead of being averaged together.
	///
	/// @see gtx_normal
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> triangleTangent(
		vec<3, T, Q> const& p1, vec<3, T, Q> const& p2, vec<3, T, Q> const& p3,
		vec<2, T, Q> const& uv1, vec<2, T, Q> const& uv2, vec<2, T, Q> const& uv3);

	/// Computes the bitangent of a triangle from triangle points and texture coordinates.
	/// The bitangent points along the texture V axis in object space.
	/// The frame is computed per triangle, so texture bases on both sides of a UV
	/// seam are preserved instead of being averaged together.
	///
	/// @see gtx_normal
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> triangleBitangent(
		vec<3, T, Q> const& p1, vec<3, T, Q> const& p2, vec<3, T, Q> const& p3,
		vec<2, T, Q> const& uv1, vec<2, T, Q> const& uv2, vec<2, T, Q> const& uv3);

	/// Returns the bitangent direction sign (+1 or -1) of a tangent frame, such that
	/// bitangent == bitangentSign * cross(normal, tangent).
	/// Store it in the w component of the tangent attribute so that the renderer can
	/// reconstruct the bitangent and the frame handedness is preserved.
	///
	/// @see gtx_normal
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T bitangentSign(
		vec<3, T, Q> const& normal,
		vec<3, T, Q> const& tangent,
		vec<3, T, Q> const& bitangent);

	/// Converts a tangent frame between a normal-map authoring tool convention and a
	/// renderer convention, fixing the flipped bump lighting that appears when a frame
	/// is carried across conventions without adjusting the bitangent direction.
	///
	/// The geometric change of basis (e.g. a mirror used to switch space handedness)
	/// and the texture V axis direction are specified independently:
	/// - basisChange: geometric change of basis applied to normal and tangent. If it
	///   is a reflection (determinant < 0), the bitangent sign is flipped accordingly.
	/// - flipV: set when the texture V axis points the other way in the target
	///   convention; flips the bitangent sign independently of the geometric basis.
	///
	/// Outputs a render-ready normal, tangent and bitangent sign. The perturbed normal
	/// stored in a texel, reconstructed with bitangent == bitangentSignOut *
	/// cross(normalOut, tangentOut), points to the same side in the common world space
	/// as before the conversion (use convertNormalMapTexel for the texel itself).
	///
	/// @see gtx_normal
	template<typename T, qualifier Q>
	GLM_FUNC_DECL void convertTangentFrame(
		mat<3, 3, T, Q> const& basisChange,
		bool flipV,
		vec<3, T, Q> const& normalIn,
		vec<3, T, Q> const& tangentIn,
		T bitangentSignIn,
		vec<3, T, Q>& normalOut,
		vec<3, T, Q>& tangentOut,
		T& bitangentSignOut);

	/// Converts a tangent-space normal-map texel between conventions.
	/// The geometric change of basis leaves tangent-space texels untouched; only a
	/// reversal of the texture V axis negates the green (V) channel.
	///
	/// @see gtx_normal
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> convertNormalMapTexel(
		vec<3, T, Q> const& texel,
		bool flipV);

	/// @}
}//namespace glm

#include "normal.inl"
