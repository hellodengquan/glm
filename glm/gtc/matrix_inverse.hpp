/// @ref gtc_matrix_inverse
/// @file glm/gtc/matrix_inverse.hpp
///
/// @see core (dependence)
///
/// @defgroup gtc_matrix_inverse GLM_GTC_matrix_inverse
/// @ingroup gtc
///
/// Include <glm/gtc/matrix_inverse.hpp> to use the features of this extension.
///
/// Defines additional matrix inverting functions.

#pragma once

// Dependencies
#include "../detail/setup.hpp"
#include "../matrix.hpp"
#include "../mat2x2.hpp"
#include "../mat3x3.hpp"
#include "../mat4x4.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_matrix_inverse extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_matrix_inverse
	/// @{

	/// Fast matrix inverse for affine matrix.
	///
	/// @param m Input matrix to invert.
	/// @tparam genType Squared floating-point matrix: half, float or double. Inverse of matrix based of half-qualifier floating point value is highly inaccurate.
	/// @see gtc_matrix_inverse
	template<typename genType>
	GLM_FUNC_DECL genType affineInverse(genType const& m);

	/// Compute the inverse transpose of a matrix.
	///
	/// @param m Input matrix to invert transpose.
	/// @tparam genType Squared floating-point matrix: half, float or double. Inverse of matrix based of half-qualifier floating point value is highly inaccurate.
	/// @see gtc_matrix_inverse
	template<typename genType>
	GLM_FUNC_DECL genType inverseTranspose(genType const& m);

	/// Try to compute the inverse of a matrix and explicitly report failure.
	///
	/// Unlike @ref inverse, which silently produces infinities or NaNs when the
	/// input is singular or ill-conditioned, this function only writes the
	/// result when the inverse can be validated. On success, the forward
	/// transform followed by the inverse round-trips to identity within the
	/// caller supplied relative tolerance.
	///
	/// The invertibility decision is scale invariant: uniformly changing the
	/// units of a transform (multiplying the matrix by a scalar) does not change
	/// whether this function succeeds or fails.
	///
	/// @param m Input matrix to invert.
	/// @param result Receives the inverse only when the function returns true; left untouched on failure so the caller keeps its last valid inverse.
	/// @param tolerance Maximum relative error accepted for the m * result round-trip; defaults to a small multiple of the machine epsilon.
	/// @tparam T Floating-point scalar type: half, float or double.
	/// @tparam Q Value qualifier, e.g. glm::packed_highp.
	/// @return true if a valid inverse was written to result, false otherwise.
	/// @see gtc_matrix_inverse
	template<typename T, qualifier Q>
	GLM_FUNC_DECL bool tryInverse(mat<2, 2, T, Q> const& m, mat<2, 2, T, Q>& result, T const& tolerance);

	/// @copydoc tryInverse(mat<2, 2, T, Q> const&, mat<2, 2, T, Q>&, T const&)
	template<typename T, qualifier Q>
	GLM_FUNC_DECL bool tryInverse(mat<2, 2, T, Q> const& m, mat<2, 2, T, Q>& result);

	/// @copydoc tryInverse(mat<2, 2, T, Q> const&, mat<2, 2, T, Q>&, T const&)
	template<typename T, qualifier Q>
	GLM_FUNC_DECL bool tryInverse(mat<3, 3, T, Q> const& m, mat<3, 3, T, Q>& result, T const& tolerance);

	/// @copydoc tryInverse(mat<2, 2, T, Q> const&, mat<2, 2, T, Q>&, T const&)
	template<typename T, qualifier Q>
	GLM_FUNC_DECL bool tryInverse(mat<3, 3, T, Q> const& m, mat<3, 3, T, Q>& result);

	/// @copydoc tryInverse(mat<2, 2, T, Q> const&, mat<2, 2, T, Q>&, T const&)
	template<typename T, qualifier Q>
	GLM_FUNC_DECL bool tryInverse(mat<4, 4, T, Q> const& m, mat<4, 4, T, Q>& result, T const& tolerance);

	/// @copydoc tryInverse(mat<2, 2, T, Q> const&, mat<2, 2, T, Q>&, T const&)
	template<typename T, qualifier Q>
	GLM_FUNC_DECL bool tryInverse(mat<4, 4, T, Q> const& m, mat<4, 4, T, Q>& result);

	/// @}
}//namespace glm

#include "matrix_inverse.inl"
