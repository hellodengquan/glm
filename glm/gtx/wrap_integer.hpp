/// @ref gtx_wrap_integer
/// @file glm/gtx/wrap_integer.hpp
///
/// @see core (dependence)
///
/// @defgroup gtx_wrap_integer GLM_GTX_wrap_integer
/// @ingroup gtx
///
/// Include <glm/gtx/wrap_integer.hpp> to use the features of this extension.
///
/// @brief Fixed-width integer arithmetic with explicit low-bit (modular)
/// semantics.
///
/// Voxel renderers commonly compute cyclic texture page indices by multiplying
/// fixed-width integer brick coordinates component by component. The ordinary
/// multiplication operator relies on signed integer overflow, whose behaviour is
/// undefined in C++: a page table generated at compile time (constant context)
/// and one generated at runtime could therefore disagree.
///
/// This extension gives every operation an explicit, well-defined meaning in
/// terms of the low bits of the mathematical result:
/// - wrappingMul() returns the product reduced modulo 2 to the power of the
///   component bit width, independently of signedness.
/// - wrapping_cast() selects a target component width explicitly and keeps only
///   the low bits that fit that width.
/// - asSigned() / asUnsigned() reinterpret the very same bit pattern between the
///   signed and unsigned integer of the same width, so texture coordinates can be
///   expressed either way without changing the page number.

#pragma once

// Dependency:
#include "../glm.hpp"
#include <type_traits>

#ifndef GLM_ENABLE_EXPERIMENTAL
#	error "GLM: GLM_GTX_wrap_integer is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it."
#elif GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTX_wrap_integer extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_wrap_integer
	/// @{

	/// Multiply two fixed-width integers keeping only the low bits of the product.
	///
	/// The result is the mathematical product reduced modulo 2^N where N is the
	/// bit width of T, computed through unsigned arithmetic so that no signed
	/// integer overflow (which is undefined behaviour) ever takes place. Signed and
	/// unsigned inputs of the same width produce the same bit pattern.
	///
	/// @tparam T A signed or unsigned fixed-width integer scalar type.
	///
	/// @see gtx_wrap_integer
	template<typename T,
		typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, int>::type = 0>
	GLM_FUNC_DECL GLM_CONSTEXPR T wrappingMul(T x, T y);

	/// Multiply two fixed-width integer vectors component by component, keeping
	/// only the low bits of each product (modulo 2 to the component bit width).
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T A signed or unsigned fixed-width integer scalar type.
	/// @tparam Q Value from qualifier enum.
	///
	/// @see gtx_wrap_integer
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<L, T, Q> wrappingMul(vec<L, T, Q> const& x, vec<L, T, Q> const& y);

	/// Convert a fixed-width integer to an integer of another width using explicit
	/// low-bit (modular) semantics.
	///
	/// The caller selects the target width through T. The source value is reduced
	/// modulo 2 to the power of the target bit width; widening therefore keeps the
	/// (non-negative) bit pattern rather than sign extending, and narrowing drops
	/// the high bits. Signedness of the target only changes how the retained bit
	/// pattern is interpreted.
	///
	/// @tparam T A signed or unsigned fixed-width integer scalar type (target).
	/// @tparam U A signed or unsigned fixed-width integer scalar type (source).
	///
	/// @see gtx_wrap_integer
	template<typename T, typename U,
		typename std::enable_if<
			std::is_integral<T>::value && !std::is_same<T, bool>::value
			&& std::is_integral<U>::value && !std::is_same<U, bool>::value, int>::type = 0>
	GLM_FUNC_DECL GLM_CONSTEXPR T wrapping_cast(U v);

	/// Convert each component of a fixed-width integer vector to the integer width
	/// T using explicit low-bit (modular) semantics. The target width is selected
	/// by the caller through T.
	///
	/// @tparam T A signed or unsigned fixed-width integer scalar type (target).
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam U A signed or unsigned fixed-width integer scalar type (source).
	/// @tparam Q Value from qualifier enum.
	///
	/// @see gtx_wrap_integer
	template<typename T, length_t L, typename U, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<L, T, Q> wrapping_cast(vec<L, U, Q> const& v);

	/// Reinterpret the bit pattern of a fixed-width unsigned integer as the signed
	/// integer of the same width. No bits are added, removed or reordered.
	///
	/// @tparam T An unsigned fixed-width integer scalar type.
	///
	/// @see gtx_wrap_integer
	template<typename T,
		typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value
			&& !std::is_same<T, bool>::value, int>::type = 0>
	GLM_FUNC_DECL GLM_CONSTEXPR typename std::make_signed<T>::type asSigned(T v);

	/// Reinterpret component by component the bit pattern of a fixed-width unsigned
	/// integer vector as the signed integer vector of the same width.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T An unsigned fixed-width integer scalar type.
	/// @tparam Q Value from qualifier enum.
	///
	/// @see gtx_wrap_integer
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<L, typename std::make_signed<T>::type, Q> asSigned(vec<L, T, Q> const& v);

	/// Reinterpret the bit pattern of a fixed-width signed integer as the unsigned
	/// integer of the same width. No bits are added, removed or reordered.
	///
	/// @tparam T A signed fixed-width integer scalar type.
	///
	/// @see gtx_wrap_integer
	template<typename T,
		typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value
			&& !std::is_same<T, bool>::value, int>::type = 0>
	GLM_FUNC_DECL GLM_CONSTEXPR typename std::make_unsigned<T>::type asUnsigned(T v);

	/// Reinterpret component by component the bit pattern of a fixed-width signed
	/// integer vector as the unsigned integer vector of the same width.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T A signed fixed-width integer scalar type.
	/// @tparam Q Value from qualifier enum.
	///
	/// @see gtx_wrap_integer
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<L, typename std::make_unsigned<T>::type, Q> asUnsigned(vec<L, T, Q> const& v);

	/// @}
}//namespace glm

#include "wrap_integer.inl"
