/// @ref gtx_wrap_integer

#pragma once

#include <limits>
#include <type_traits>

namespace glm
{
	namespace detail
	{
		// Reinterpret a signed value as the unsigned integer of the same width.
		// The most negative value is handled through (max / 2 + 1) so that signed
		// negation never overflows; the result is usable in a constant expression.
		template<typename S>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename std::make_unsigned<S>::type signed_to_unsigned(S v)
		{
			typedef typename std::make_unsigned<S>::type U;

			if(v >= S(0))
				return static_cast<U>(v);
			if(v == (std::numeric_limits<S>::min)())
				return static_cast<U>((std::numeric_limits<U>::max)() / U(2) + U(1));
			return static_cast<U>(U(0) - static_cast<U>(-v));
		}

		// Expose the unsigned bit pattern of an integer scalar.
		template<typename U>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR U to_unsigned_bits(U v, std::false_type /*is_signed*/)
		{
			return v;
		}

		template<typename S>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename std::make_unsigned<S>::type to_unsigned_bits(S v, std::true_type /*is_signed*/)
		{
			return signed_to_unsigned(v);
		}

		// Build a signed integer of width T from its unsigned-equivalent bits.
		// All signed arithmetic stays inside the representable range, so nothing
		// overflows and the function remains valid in a constant expression.
		template<typename T>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR T from_unsigned_bits(typename std::make_unsigned<T>::type bits, std::true_type /*is_signed*/)
		{
			typedef typename std::make_unsigned<T>::type U;

			U const signBit = static_cast<U>(U(1) << (sizeof(U) * 8 - 1));
			if((bits & signBit) == U(0))
				return static_cast<T>(bits);
			if(bits == signBit)
				return (std::numeric_limits<T>::min)();
			U const magnitude = static_cast<U>(U(0) - bits);
			return static_cast<T>(-static_cast<T>(magnitude));
		}

		// Build an unsigned integer of width T from its bit pattern.
		template<typename T>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR T from_unsigned_bits(T bits, std::false_type /*is_signed*/)
		{
			return bits;
		}

		// Modular product. The operands are read through their unsigned bit
		// patterns and multiplied in the unsigned domain, where overflow wraps
		// modulo 2^N. The target only decides how the retained bits are read back.
		template<typename T>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR T wrapping_mul(T x, T y)
		{
			typedef typename std::make_unsigned<T>::type U;
			U const product =
				to_unsigned_bits(x, std::is_signed<T>()) * to_unsigned_bits(y, std::is_signed<T>());
			return from_unsigned_bits<T>(product, std::is_signed<T>());
		}

		// Reduce the bit pattern of v to the low bits that fit the target width.
		template<typename T, typename U>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR T wrapping_cast_scalar(U v)
		{
			typedef typename std::make_unsigned<T>::type target_unsigned;

			target_unsigned const bits = static_cast<target_unsigned>(
				to_unsigned_bits(v, std::is_signed<U>()));

			return from_unsigned_bits<T>(bits, std::is_signed<T>());
		}
	}//namespace detail

	// Forward declarations of the per-dimension helpers.
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> wrappingMul_vec(vec<1, T, Q> const& x, vec<1, T, Q> const& y);
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> wrappingMul_vec(vec<2, T, Q> const& x, vec<2, T, Q> const& y);
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> wrappingMul_vec(vec<3, T, Q> const& x, vec<3, T, Q> const& y);
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, T, Q> wrappingMul_vec(vec<4, T, Q> const& x, vec<4, T, Q> const& y);

	template<typename T, typename U, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> wrapping_cast_vec(vec<1, U, Q> const& v);
	template<typename T, typename U, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> wrapping_cast_vec(vec<2, U, Q> const& v);
	template<typename T, typename U, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> wrapping_cast_vec(vec<3, U, Q> const& v);
	template<typename T, typename U, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, T, Q> wrapping_cast_vec(vec<4, U, Q> const& v);

	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, typename std::make_signed<T>::type, Q> asSigned_vec(vec<1, T, Q> const& v);
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, typename std::make_signed<T>::type, Q> asSigned_vec(vec<2, T, Q> const& v);
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, typename std::make_signed<T>::type, Q> asSigned_vec(vec<3, T, Q> const& v);
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, typename std::make_signed<T>::type, Q> asSigned_vec(vec<4, T, Q> const& v);

	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, typename std::make_unsigned<T>::type, Q> asUnsigned_vec(vec<1, T, Q> const& v);
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, typename std::make_unsigned<T>::type, Q> asUnsigned_vec(vec<2, T, Q> const& v);
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, typename std::make_unsigned<T>::type, Q> asUnsigned_vec(vec<3, T, Q> const& v);
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, typename std::make_unsigned<T>::type, Q> asUnsigned_vec(vec<4, T, Q> const& v);

	// wrappingMul: scalars

	template<typename T,
		typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, int>::type>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T wrappingMul(T x, T y)
	{
		return detail::wrapping_mul(x, y);
	}

	// wrappingMul: vectors

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, T, Q> wrappingMul(vec<L, T, Q> const& x, vec<L, T, Q> const& y)
	{
		static_assert(std::is_integral<T>::value && !std::is_same<T, bool>::value,
			"glm::wrappingMul only supports fixed-width integer vector types");
		return wrappingMul_vec(x, y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> wrappingMul_vec(vec<1, T, Q> const& x, vec<1, T, Q> const& y)
	{
		return vec<1, T, Q>(detail::wrapping_mul(x.x, y.x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> wrappingMul_vec(vec<2, T, Q> const& x, vec<2, T, Q> const& y)
	{
		return vec<2, T, Q>(detail::wrapping_mul(x.x, y.x), detail::wrapping_mul(x.y, y.y));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> wrappingMul_vec(vec<3, T, Q> const& x, vec<3, T, Q> const& y)
	{
		return vec<3, T, Q>(detail::wrapping_mul(x.x, y.x), detail::wrapping_mul(x.y, y.y), detail::wrapping_mul(x.z, y.z));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, T, Q> wrappingMul_vec(vec<4, T, Q> const& x, vec<4, T, Q> const& y)
	{
		return vec<4, T, Q>(detail::wrapping_mul(x.x, y.x), detail::wrapping_mul(x.y, y.y),
			detail::wrapping_mul(x.z, y.z), detail::wrapping_mul(x.w, y.w));
	}

	// wrapping_cast: scalars

	template<typename T, typename U,
		typename std::enable_if<
			std::is_integral<T>::value && !std::is_same<T, bool>::value
			&& std::is_integral<U>::value && !std::is_same<U, bool>::value, int>::type>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T wrapping_cast(U v)
	{
		return detail::wrapping_cast_scalar<T>(v);
	}

	// wrapping_cast: vectors

	template<typename T, length_t L, typename U, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, T, Q> wrapping_cast(vec<L, U, Q> const& v)
	{
		static_assert(std::is_integral<T>::value && !std::is_same<T, bool>::value,
			"glm::wrapping_cast target must be a fixed-width integer type");
		static_assert(std::is_integral<U>::value && !std::is_same<U, bool>::value,
			"glm::wrapping_cast source must be a fixed-width integer type");
		return wrapping_cast_vec<T>(v);
	}

	template<typename T, typename U, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> wrapping_cast_vec(vec<1, U, Q> const& v)
	{
		return vec<1, T, Q>(detail::wrapping_cast_scalar<T>(v.x));
	}

	template<typename T, typename U, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> wrapping_cast_vec(vec<2, U, Q> const& v)
	{
		return vec<2, T, Q>(detail::wrapping_cast_scalar<T>(v.x), detail::wrapping_cast_scalar<T>(v.y));
	}

	template<typename T, typename U, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> wrapping_cast_vec(vec<3, U, Q> const& v)
	{
		return vec<3, T, Q>(detail::wrapping_cast_scalar<T>(v.x), detail::wrapping_cast_scalar<T>(v.y),
			detail::wrapping_cast_scalar<T>(v.z));
	}

	template<typename T, typename U, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, T, Q> wrapping_cast_vec(vec<4, U, Q> const& v)
	{
		return vec<4, T, Q>(detail::wrapping_cast_scalar<T>(v.x), detail::wrapping_cast_scalar<T>(v.y),
			detail::wrapping_cast_scalar<T>(v.z), detail::wrapping_cast_scalar<T>(v.w));
	}

	// asSigned: scalars and vectors

	template<typename T,
		typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value
			&& !std::is_same<T, bool>::value, int>::type>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename std::make_signed<T>::type asSigned(T v)
	{
		return detail::from_unsigned_bits<typename std::make_signed<T>::type>(v, std::true_type());
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, typename std::make_signed<T>::type, Q> asSigned(vec<L, T, Q> const& v)
	{
		static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value && !std::is_same<T, bool>::value,
			"glm::asSigned expects an unsigned fixed-width integer vector type");
		return asSigned_vec(v);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, typename std::make_signed<T>::type, Q> asSigned_vec(vec<1, T, Q> const& v)
	{
		typedef typename std::make_signed<T>::type S;
		return vec<1, S, Q>(detail::from_unsigned_bits<S>(v.x, std::true_type()));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, typename std::make_signed<T>::type, Q> asSigned_vec(vec<2, T, Q> const& v)
	{
		typedef typename std::make_signed<T>::type S;
		return vec<2, S, Q>(detail::from_unsigned_bits<S>(v.x, std::true_type()),
			detail::from_unsigned_bits<S>(v.y, std::true_type()));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, typename std::make_signed<T>::type, Q> asSigned_vec(vec<3, T, Q> const& v)
	{
		typedef typename std::make_signed<T>::type S;
		return vec<3, S, Q>(detail::from_unsigned_bits<S>(v.x, std::true_type()),
			detail::from_unsigned_bits<S>(v.y, std::true_type()),
			detail::from_unsigned_bits<S>(v.z, std::true_type()));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, typename std::make_signed<T>::type, Q> asSigned_vec(vec<4, T, Q> const& v)
	{
		typedef typename std::make_signed<T>::type S;
		return vec<4, S, Q>(detail::from_unsigned_bits<S>(v.x, std::true_type()),
			detail::from_unsigned_bits<S>(v.y, std::true_type()),
			detail::from_unsigned_bits<S>(v.z, std::true_type()),
			detail::from_unsigned_bits<S>(v.w, std::true_type()));
	}

	// asUnsigned: scalars and vectors

	template<typename T,
		typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value
			&& !std::is_same<T, bool>::value, int>::type>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename std::make_unsigned<T>::type asUnsigned(T v)
	{
		return detail::signed_to_unsigned(v);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, typename std::make_unsigned<T>::type, Q> asUnsigned(vec<L, T, Q> const& v)
	{
		static_assert(std::is_integral<T>::value && std::is_signed<T>::value && !std::is_same<T, bool>::value,
			"glm::asUnsigned expects a signed fixed-width integer vector type");
		return asUnsigned_vec(v);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, typename std::make_unsigned<T>::type, Q> asUnsigned_vec(vec<1, T, Q> const& v)
	{
		typedef typename std::make_unsigned<T>::type U;
		return vec<1, U, Q>(detail::signed_to_unsigned(v.x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, typename std::make_unsigned<T>::type, Q> asUnsigned_vec(vec<2, T, Q> const& v)
	{
		typedef typename std::make_unsigned<T>::type U;
		return vec<2, U, Q>(detail::signed_to_unsigned(v.x), detail::signed_to_unsigned(v.y));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, typename std::make_unsigned<T>::type, Q> asUnsigned_vec(vec<3, T, Q> const& v)
	{
		typedef typename std::make_unsigned<T>::type U;
		return vec<3, U, Q>(detail::signed_to_unsigned(v.x), detail::signed_to_unsigned(v.y),
			detail::signed_to_unsigned(v.z));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, typename std::make_unsigned<T>::type, Q> asUnsigned_vec(vec<4, T, Q> const& v)
	{
		typedef typename std::make_unsigned<T>::type U;
		return vec<4, U, Q>(detail::signed_to_unsigned(v.x), detail::signed_to_unsigned(v.y),
			detail::signed_to_unsigned(v.z), detail::signed_to_unsigned(v.w));
	}
}//namespace glm
