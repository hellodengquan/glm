/// @ref gtc_matrix_inverse

#include "../common.hpp"
#include "../geometric.hpp"
#include "../vector_relational.hpp"
#include <limits>

namespace glm
{
	namespace detail
	{
		// Smallest value such that 1 + epsilon is distinguishable from 1 for T.
		// Computed portably (including for the half storage type, whose
		// std::numeric_limits specialization is integer based). The loop uses a
		// difference based check rather than a floating point == to stay clear
		// of -Wfloat-equal.
		template<typename T>
		GLM_FUNC_QUALIFIER T machine_epsilon()
		{
			T epsilon = static_cast<T>(1);
			T const one = static_cast<T>(1);
			T const half = static_cast<T>(0.5);
			while(true)
			{
				T const next = epsilon * half;
				if(glm::abs((one + next) - one) <= static_cast<T>(0))
					break;
				epsilon = next;
			}
			return epsilon;
		}

		// Default relative round-trip error accepted by tryInverse.
		template<typename T>
		GLM_FUNC_QUALIFIER T default_inverse_tolerance()
		{
			return machine_epsilon<T>() * static_cast<T>(16);
		}

		// Squared Euclidean norm of a matrix column.
		template<length_t C, length_t R, typename T, qualifier Q>
		GLM_FUNC_QUALIFIER T column_norm_squared(mat<C, R, T, Q> const& m, length_t column)
		{
			return dot(m[column], m[column]);
		}

		// Scale invariant measure of how close a matrix is to singular.
		// Returns |det(m)| / product(||column_i||), which equals 1 for an
		// orthogonal matrix, 0 for a singular one and is unchanged when m is
		// multiplied by any non zero scalar. Changing scene length units
		// therefore cannot change the invertibility decision.
		template<length_t C, length_t R, typename T, qualifier Q>
		GLM_FUNC_QUALIFIER T normalized_determinant(mat<C, R, T, Q> const& m)
		{
			T column_norm_product = static_cast<T>(1);
			for(length_t i = 0; i < C; ++i)
			{
				T const norm_squared = column_norm_squared(m, i);
				if(!(norm_squared > static_cast<T>(0)))
					return static_cast<T>(0);
				column_norm_product *= sqrt(norm_squared);
			}

			T const det = determinant(m);
			if(!(glm::abs(det) > static_cast<T>(0)))
				return static_cast<T>(0);

			return glm::abs(det) / column_norm_product;
		}

		// Relative Frobenius norm of m * inverse - identity and
		// inverse * m - identity; the maximum of both round-trips.
		template<length_t C, length_t R, typename T, qualifier Q>
		GLM_FUNC_QUALIFIER T inverse_relative_error(mat<C, R, T, Q> const& m, mat<C, R, T, Q> const& inverse)
		{
			mat<C, R, T, Q> const identity(static_cast<T>(1));
			mat<C, R, T, Q> const forward = m * inverse - identity;
			mat<C, R, T, Q> const backward = inverse * m - identity;

			T residual_squared = static_cast<T>(0);
			T identity_squared = static_cast<T>(0);
			for(length_t c = 0; c < C; ++c)
			{
				residual_squared += max(dot(forward[c], forward[c]), dot(backward[c], backward[c]));
				identity_squared += dot(identity[c], identity[c]);
			}

			return sqrt(residual_squared) / sqrt(identity_squared);
		}

		// Every entry must be a finite floating point value; NaN or inf
		// produced while inverting means the result is unusable.
		template<length_t C, length_t R, typename T, qualifier Q>
		GLM_FUNC_QUALIFIER bool matrix_is_finite(mat<C, R, T, Q> const& m)
		{
			for(length_t c = 0; c < C; ++c)
			{
				if(any(isnan(m[c])) || any(isinf(m[c])))
					return false;
			}
			return true;
		}

		template<length_t C, length_t R, typename T, qualifier Q>
		GLM_FUNC_QUALIFIER bool try_inverse_impl(mat<C, R, T, Q> const& m, mat<C, R, T, Q>& result, T const& tolerance)
		{
			static_assert(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_GENTYPE,
				"'tryInverse' only accept floating-point inputs");

			if(!(tolerance > static_cast<T>(0)))
				return false;

			// Reject singular or ill-conditioned matrices before inverting.
			// The threshold and the measure itself are scale invariant.
			T const singular_limit = default_inverse_tolerance<T>();
			if(normalized_determinant(m) <= singular_limit)
				return false;

			mat<C, R, T, Q> candidate = inverse(m);
			if(!matrix_is_finite(candidate))
				return false;

			if(inverse_relative_error(m, candidate) > tolerance)
				return false;

			// Only commit on success: the caller keeps its previous result.
			result = candidate;
			return true;
		}
	}//namespace detail

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool tryInverse(mat<2, 2, T, Q> const& m, mat<2, 2, T, Q>& result, T const& tolerance)
	{
		return detail::try_inverse_impl(m, result, tolerance);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool tryInverse(mat<2, 2, T, Q> const& m, mat<2, 2, T, Q>& result)
	{
		return detail::try_inverse_impl(m, result, detail::default_inverse_tolerance<T>());
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool tryInverse(mat<3, 3, T, Q> const& m, mat<3, 3, T, Q>& result, T const& tolerance)
	{
		return detail::try_inverse_impl(m, result, tolerance);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool tryInverse(mat<3, 3, T, Q> const& m, mat<3, 3, T, Q>& result)
	{
		return detail::try_inverse_impl(m, result, detail::default_inverse_tolerance<T>());
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool tryInverse(mat<4, 4, T, Q> const& m, mat<4, 4, T, Q>& result, T const& tolerance)
	{
		return detail::try_inverse_impl(m, result, tolerance);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool tryInverse(mat<4, 4, T, Q> const& m, mat<4, 4, T, Q>& result)
	{
		return detail::try_inverse_impl(m, result, detail::default_inverse_tolerance<T>());
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> affineInverse(mat<3, 3, T, Q> const& m)
	{
		mat<2, 2, T, Q> const Inv(inverse(mat<2, 2, T, Q>(m)));

		return mat<3, 3, T, Q>(
			vec<3, T, Q>(Inv[0], static_cast<T>(0)),
			vec<3, T, Q>(Inv[1], static_cast<T>(0)),
			vec<3, T, Q>(-Inv * vec<2, T, Q>(m[2]), static_cast<T>(1)));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> affineInverse(mat<4, 4, T, Q> const& m)
	{
		mat<3, 3, T, Q> const Inv(inverse(mat<3, 3, T, Q>(m)));

		return mat<4, 4, T, Q>(
			vec<4, T, Q>(Inv[0], static_cast<T>(0)),
			vec<4, T, Q>(Inv[1], static_cast<T>(0)),
			vec<4, T, Q>(Inv[2], static_cast<T>(0)),
			vec<4, T, Q>(-Inv * vec<3, T, Q>(m[3]), static_cast<T>(1)));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> inverseTranspose(mat<2, 2, T, Q> const& m)
	{
		T Determinant = m[0][0] * m[1][1] - m[1][0] * m[0][1];

		mat<2, 2, T, Q> Inverse(
			+ m[1][1] / Determinant,
			- m[0][1] / Determinant,
			- m[1][0] / Determinant,
			+ m[0][0] / Determinant);

		return Inverse;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> inverseTranspose(mat<3, 3, T, Q> const& m)
	{
		T Determinant =
			+ m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
			- m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
			+ m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

		mat<3, 3, T, Q> Inverse;
		Inverse[0][0] = + (m[1][1] * m[2][2] - m[2][1] * m[1][2]);
		Inverse[0][1] = - (m[1][0] * m[2][2] - m[2][0] * m[1][2]);
		Inverse[0][2] = + (m[1][0] * m[2][1] - m[2][0] * m[1][1]);
		Inverse[1][0] = - (m[0][1] * m[2][2] - m[2][1] * m[0][2]);
		Inverse[1][1] = + (m[0][0] * m[2][2] - m[2][0] * m[0][2]);
		Inverse[1][2] = - (m[0][0] * m[2][1] - m[2][0] * m[0][1]);
		Inverse[2][0] = + (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
		Inverse[2][1] = - (m[0][0] * m[1][2] - m[1][0] * m[0][2]);
		Inverse[2][2] = + (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
		Inverse /= Determinant;

		return Inverse;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> inverseTranspose(mat<4, 4, T, Q> const& m)
	{
		T SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		T SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		T SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		T SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		T SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		T SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		T SubFactor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
		T SubFactor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
		T SubFactor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
		T SubFactor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
		T SubFactor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
		T SubFactor11 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
		T SubFactor12 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
		T SubFactor13 = m[1][1] * m[2][3] - m[2][1] * m[1][3];
		T SubFactor14 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
		T SubFactor15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];
		T SubFactor16 = m[1][0] * m[2][2] - m[2][0] * m[1][2];
		T SubFactor17 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

		mat<4, 4, T, Q> Inverse;
		Inverse[0][0] = + (m[1][1] * SubFactor00 - m[1][2] * SubFactor01 + m[1][3] * SubFactor02);
		Inverse[0][1] = - (m[1][0] * SubFactor00 - m[1][2] * SubFactor03 + m[1][3] * SubFactor04);
		Inverse[0][2] = + (m[1][0] * SubFactor01 - m[1][1] * SubFactor03 + m[1][3] * SubFactor05);
		Inverse[0][3] = - (m[1][0] * SubFactor02 - m[1][1] * SubFactor04 + m[1][2] * SubFactor05);

		Inverse[1][0] = - (m[0][1] * SubFactor00 - m[0][2] * SubFactor01 + m[0][3] * SubFactor02);
		Inverse[1][1] = + (m[0][0] * SubFactor00 - m[0][2] * SubFactor03 + m[0][3] * SubFactor04);
		Inverse[1][2] = - (m[0][0] * SubFactor01 - m[0][1] * SubFactor03 + m[0][3] * SubFactor05);
		Inverse[1][3] = + (m[0][0] * SubFactor02 - m[0][1] * SubFactor04 + m[0][2] * SubFactor05);

		Inverse[2][0] = + (m[0][1] * SubFactor06 - m[0][2] * SubFactor07 + m[0][3] * SubFactor08);
		Inverse[2][1] = - (m[0][0] * SubFactor06 - m[0][2] * SubFactor09 + m[0][3] * SubFactor10);
		Inverse[2][2] = + (m[0][0] * SubFactor07 - m[0][1] * SubFactor09 + m[0][3] * SubFactor11);
		Inverse[2][3] = - (m[0][0] * SubFactor08 - m[0][1] * SubFactor10 + m[0][2] * SubFactor11);

		Inverse[3][0] = - (m[0][1] * SubFactor12 - m[0][2] * SubFactor13 + m[0][3] * SubFactor14);
		Inverse[3][1] = + (m[0][0] * SubFactor12 - m[0][2] * SubFactor15 + m[0][3] * SubFactor16);
		Inverse[3][2] = - (m[0][0] * SubFactor13 - m[0][1] * SubFactor15 + m[0][3] * SubFactor17);
		Inverse[3][3] = + (m[0][0] * SubFactor14 - m[0][1] * SubFactor16 + m[0][2] * SubFactor17);

		T Determinant =
			+ m[0][0] * Inverse[0][0]
			+ m[0][1] * Inverse[0][1]
			+ m[0][2] * Inverse[0][2]
			+ m[0][3] * Inverse[0][3];

		Inverse /= Determinant;

		return Inverse;
	}
}//namespace glm
