#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/wrap_integer.hpp>
#include <glm/ext/vector_int2_sized.hpp>
#include <glm/ext/vector_int3_sized.hpp>
#include <glm/ext/vector_int4_sized.hpp>
#include <glm/ext/vector_uint2_sized.hpp>
#include <glm/ext/vector_uint3_sized.hpp>
#include <glm/ext/vector_uint4_sized.hpp>
#include <cstdint>

// Compile-time guarantees: every operation must be a constant expression, so a
// page table built at compile time and one built at runtime use the exact same
// well-defined arithmetic rather than signed overflow (undefined behaviour).
namespace compile_time_checks
{
	// wrappingMul keeps only the low bits of the product, for every width.
	static_assert(glm::wrappingMul(glm::int32(100000), glm::int32(100000))
		== glm::int32(static_cast<glm::uint32>(100000u * 100000u)), "wrappingMul int32 low bits");
	static_assert(glm::wrappingMul(glm::uint32(100000u), glm::uint32(100000u))
		== glm::uint32(100000u * 100000u), "wrappingMul uint32 low bits");
	static_assert(glm::wrappingMul(glm::int16(200), glm::int16(200)) == glm::int16(-25536), "wrappingMul int16");
	static_assert(glm::wrappingMul(glm::int8(12), glm::int8(12)) == glm::int8(static_cast<glm::int8>(144)), "wrappingMul int8");
	static_assert(glm::wrappingMul(glm::int32(-7), glm::int32(13)) == glm::int32(-91), "wrappingMul signed negative");
	static_assert(glm::wrappingMul((std::numeric_limits<glm::int32>::min)(), glm::int32(2)) == glm::int32(0), "wrappingMul INT_MIN*2");

	// Signed and unsigned operands share the same bit pattern.
	static_assert(static_cast<glm::uint32>(glm::wrappingMul(glm::int32(100000), glm::int32(100000)))
		== glm::wrappingMul(glm::uint32(100000u), glm::uint32(100000u)), "signed/unsigned bit pattern parity");

	// Vector low-bit product is itself constant-evaluable.
	constexpr glm::i32vec3 ConstVec = glm::wrappingMul(
		glm::i32vec3(99991, -424242, 1234567), glm::i32vec3(77777, 31337, 98765));
	static_assert(ConstVec.x == glm::wrappingMul(glm::int32(99991), glm::int32(77777)), "constexpr vec3 x");
	static_assert(ConstVec.y == glm::wrappingMul(glm::int32(-424242), glm::int32(31337)), "constexpr vec3 y");
	static_assert(ConstVec.z == glm::wrappingMul(glm::int32(1234567), glm::int32(98765)), "constexpr vec3 z");

	// asSigned / asUnsigned preserve the bit pattern, including the extremes.
	static_assert(glm::asUnsigned(glm::int32(-1)) == glm::uint32(0xFFFFFFFFu), "asUnsigned -1");
	static_assert(glm::asUnsigned((std::numeric_limits<glm::int32>::min)()) == glm::uint32(0x80000000u), "asUnsigned INT_MIN");
	static_assert(glm::asSigned(glm::uint32(0x80000000u)) == (std::numeric_limits<glm::int32>::min)(), "asSigned sign bit");
	static_assert(glm::asSigned(glm::uint8(200)) == glm::int8(-56), "asSigned int8");
	static_assert(glm::asUnsigned(glm::asSigned(glm::uint16(40000))) == glm::uint16(40000), "signed<->unsigned round trip");
	constexpr glm::u32vec3 UnsignedVec = glm::asUnsigned(glm::i32vec3(-1, 0, -2));
	static_assert(UnsignedVec.x == glm::uint32(0xFFFFFFFFu) && UnsignedVec.z == glm::uint32(0xFFFFFFFEu), "constexpr asUnsigned vec3");
	static_assert(glm::asSigned(UnsignedVec) == glm::i32vec3(-1, 0, -2), "constexpr asSigned vec3 round trip");

	// wrapping_cast selects the target width explicitly and keeps its low bits.
	static_assert(glm::wrapping_cast<glm::uint8>(glm::uint32(0x12345678u)) == glm::uint8(0x78), "wrapping_cast narrowing");
	static_assert(glm::wrapping_cast<glm::int16>(glm::int32(0x00008000)) == glm::int16(-32768), "wrapping_cast to signed narrow");
	static_assert(glm::wrapping_cast<glm::int16>(glm::int8(-1)) == glm::int16(255), "wrapping_cast widening keeps bits");
	static_assert(glm::wrapping_cast<glm::int32>(glm::uint32(0xFFFFFFFFu)) == glm::int32(-1), "wrapping_cast same width reinterpret");
	constexpr glm::i8vec3 NarrowVec = glm::wrapping_cast<glm::int8>(glm::i32vec3(0x12345678, -1, 0x00008000));
	static_assert(NarrowVec == glm::i8vec3(0x78, -1, 0), "constexpr wrapping_cast vec3");
}//namespace compile_time_checks

namespace
{
	// Accumulate a check without a compile-time-folded ternary, which would
	// otherwise trip -Wunreachable-code when the comparison is provably constant.
	int check(bool condition)
	{
		return condition ? 0 : 1;
	}

	// Reference low-bit product computed with a wider unsigned accumulator, so the
	// expected value never relies on the overflow behaviour under test.
	template<typename T>
	T reference_mul(T x, T y)
	{
		typedef typename std::make_unsigned<T>::type U;
		U const product = static_cast<U>(static_cast<U>(x) * static_cast<U>(y));
		return static_cast<T>(product);
	}
}

namespace test_scalar_wrapping_mul
{
	// Compile-time (constant) page table entries.
	constexpr glm::int32 c_s32 = glm::wrappingMul(glm::int32(100000), glm::int32(100000));
	constexpr glm::uint32 c_u32 = glm::wrappingMul(glm::uint32(100000u), glm::uint32(100000u));
	constexpr glm::int16 c_s16 = glm::wrappingMul(glm::int16(200), glm::int16(200));
	constexpr glm::int8 c_s8 = glm::wrappingMul(glm::int8(12), glm::int8(12));

	static int test()
	{
		int Error = 0;

		// 32 bit overflow: signed and unsigned share the same low bits.
		glm::int32 const s32 = glm::wrappingMul(glm::int32(100000), glm::int32(100000));
		glm::uint32 const u32 = glm::wrappingMul(glm::uint32(100000u), glm::uint32(100000u));
		Error += check(s32 == c_s32 ); // runtime == constant
		Error += check(u32 == c_u32 );
		Error += check(static_cast<glm::uint32>(s32) == u32 ); // same bit pattern
		Error += check(u32 == reference_mul(glm::uint32(100000u), glm::uint32(100000u)) );

		// 16 bit overflow.
		glm::int16 const s16 = glm::wrappingMul(glm::int16(200), glm::int16(200));
		Error += check(s16 == c_s16 );
		Error += check(static_cast<glm::uint16>(s16)
			== reference_mul(glm::uint16(200u), glm::uint16(200u)) );

		// 8 bit overflow.
		glm::int8 const s8 = glm::wrappingMul(glm::int8(12), glm::int8(12));
		Error += check(s8 == c_s8 );
		Error += check(static_cast<glm::uint8>(s8)
			== reference_mul(glm::uint8(12u), glm::uint8(12u)) );

		// Negative operands and the most negative value stay well defined.
		Error += check(glm::wrappingMul(glm::int32(-7), glm::int32(13)) == glm::int32(-91) );
		Error += check(glm::wrappingMul(glm::int32(-7), glm::int32(-13)) == glm::int32(91) );
		glm::int32 const minv = (std::numeric_limits<glm::int32>::min)();
		Error += check(glm::wrappingMul(minv, glm::int32(2)) == glm::int32(0) );
		Error += check(glm::wrappingMul(minv, minv) == glm::int32(0) );

		// 64 bit wrapping keeps only the low 64 bits.
		glm::uint64 const wide = glm::wrappingMul(glm::uint64(0x100000000ULL), glm::uint64(0x100000000ULL));
		Error += check(wide == glm::uint64(0) );

		return Error;
	}
}//namespace test_scalar_wrapping_mul

namespace test_vector_wrapping_mul
{
	static int test()
	{
		int Error = 0;

		// A 3D brick coordinate mapped to cyclic texture pages.
		glm::i32vec3 const Coords(100000, -123456, 777);
		glm::i32vec3 const Stride(100000, 65536, 999);

		constexpr glm::i32vec3 ConstPage(
			glm::wrappingMul(glm::int32(100000), glm::int32(100000)),
			glm::wrappingMul(glm::int32(-123456), glm::int32(65536)),
			glm::wrappingMul(glm::int32(777), glm::int32(999)));
		glm::i32vec3 const RuntimePage = glm::wrappingMul(Coords, Stride);

		// Constant-generated and runtime-generated page tables must be identical.
		Error += check(glm::all(glm::equal(RuntimePage, ConstPage)) );

		// Each component matches the independently computed low-bit product.
		for(glm::length_t i = 0; i < 3; ++i)
			Error += check(static_cast<glm::uint32>(RuntimePage[i])
				== reference_mul(static_cast<glm::uint32>(Coords[i]),
					static_cast<glm::uint32>(Stride[i])) );

		// Signed and unsigned page coordinates carry the same bit pattern.
		glm::u32vec3 const UnsignedPage = glm::wrappingMul(glm::asUnsigned(Coords), glm::asUnsigned(Stride));
		Error += check(glm::all(glm::equal(UnsignedPage, glm::asUnsigned(RuntimePage))) );

		// Other dimensions and widths behave the same.
		glm::u8vec4 const A8(200, 12, 255, 7);
		glm::u8vec4 const B8(3, 14, 2, 100);
		glm::u8vec4 const P8 = glm::wrappingMul(A8, B8);
		for(glm::length_t i = 0; i < 4; ++i)
			Error += check(P8[i] == reference_mul(A8[i], B8[i]) );

		glm::i16vec2 const A16(30000, -1234);
		glm::i16vec2 const B16(2, 5678);
		glm::i16vec2 const P16 = glm::wrappingMul(A16, B16);
		for(glm::length_t i = 0; i < 2; ++i)
			Error += check(static_cast<glm::uint16>(P16[i])
				== reference_mul(static_cast<glm::uint16>(A16[i]),
					static_cast<glm::uint16>(B16[i])) );

		return Error;
	}
}//namespace test_vector_wrapping_mul

namespace test_as_signed_unsigned
{
	static int test()
	{
		int Error = 0;

		// Scalar round trips preserve the bit pattern for every width.
		Error += check(glm::asUnsigned(glm::asSigned(glm::uint8(200))) == glm::uint8(200) );
		Error += check(glm::asSigned(glm::uint8(200)) == glm::int8(-56) );
		Error += check(glm::asUnsigned(glm::int8(-1)) == glm::uint8(255) );
		Error += check(glm::asUnsigned(glm::int16(-1)) == glm::uint16(65535) );
		Error += check(glm::asSigned(glm::uint16(65535)) == glm::int16(-1) );
		Error += check(glm::asUnsigned((std::numeric_limits<glm::int32>::min)())
			== glm::uint32(0x80000000u) );
		Error += check(glm::asSigned(glm::uint32(0x80000000u))
			== (std::numeric_limits<glm::int32>::min)() );
		Error += check(glm::asUnsigned(glm::int64(-1)) == glm::uint64(0xFFFFFFFFFFFFFFFFULL) );
		Error += check(glm::asSigned(glm::uint64(0x8000000000000000ULL))
			== (std::numeric_limits<glm::int64>::min)() );

		// Vector reinterpretation keeps the same bit pattern component by component.
		glm::i32vec3 const Signed(-1, 0, (std::numeric_limits<glm::int32>::min)());
		glm::u32vec3 const Unsigned = glm::asUnsigned(Signed);
		Error += check(Unsigned == glm::u32vec3(0xFFFFFFFFu, 0u, 0x80000000u) );
		Error += check(glm::asSigned(Unsigned) == Signed );

		// The same page number is usable with either signed or unsigned texture coords.
		glm::i32vec3 const Page = glm::wrappingMul(glm::i32vec3(100000, 5, -9), glm::i32vec3(100000, 7, 11));
		glm::u32vec3 const PageFromUnsigned =
			glm::wrappingMul(glm::asUnsigned(glm::i32vec3(100000, 5, -9)),
				glm::asUnsigned(glm::i32vec3(100000, 7, 11)));
		Error += check(glm::asUnsigned(Page) == PageFromUnsigned );

		return Error;
	}
}//namespace test_as_signed_unsigned

namespace test_wrapping_cast
{
	static int test()
	{
		int Error = 0;

		// Narrowing keeps the low bits and drops the high bits.
		Error += check(glm::wrapping_cast<glm::uint8>(glm::uint32(0x12345678u)) == glm::uint8(0x78) );
		Error += check(glm::wrapping_cast<glm::int16>(glm::int32(0x00007FFF)) == glm::int16(32767) );
		Error += check(glm::wrapping_cast<glm::int16>(glm::int32(0x00008000)) == glm::int16(-32768) );
		Error += check(glm::wrapping_cast<glm::int8>(glm::int32(-1)) == glm::int8(-1) );

		// Widening keeps the (non-negative) bit pattern rather than sign extending.
		Error += check(glm::wrapping_cast<glm::int16>(glm::int8(-1)) == glm::int16(255) );
		Error += check(glm::wrapping_cast<glm::uint32>(glm::uint8(255)) == glm::uint32(255) );
		Error += check(glm::wrapping_cast<glm::int32>(glm::uint16(65535)) == glm::int32(65535) );

		// Same-width cast only reinterprets the bits, matching asSigned/asUnsigned.
		Error += check(glm::wrapping_cast<glm::int32>(glm::uint32(0xFFFFFFFFu)) == glm::int32(-1) );
		Error += check(glm::wrapping_cast<glm::uint32>(glm::int32(-1)) == glm::uint32(0xFFFFFFFFu) );

		// Vector form selects the target width explicitly per component.
		glm::i32vec3 const Wide(0x12345678, -1, 0x00008000);
		glm::i8vec3 const Narrow = glm::wrapping_cast<glm::int8>(Wide);
		Error += check(Narrow == glm::i8vec3(0x78, -1, 0) );
		glm::u16vec3 const ToU16 = glm::wrapping_cast<glm::uint16>(Wide);
		Error += check(ToU16 == glm::u16vec3(0x5678, 0xFFFF, 0x8000) );

		// Explicit target width makes the page index stable regardless of source width.
		glm::i16vec3 const Coarse(-1, 1000, -200);
		glm::i8vec3 const Fine = glm::wrapping_cast<glm::int8>(Coarse);
		for(glm::length_t i = 0; i < 3; ++i)
			Error += check(static_cast<glm::uint8>(Fine[i])
				== static_cast<glm::uint8>(static_cast<glm::uint16>(Coarse[i])) );

		return Error;
	}
}//namespace test_wrapping_cast

namespace test_constant_runtime_equality
{
	// A purely compile-time page table.
	static constexpr glm::i32vec3 build_const_page()
	{
		return glm::wrappingMul(glm::i32vec3(99991, -424242, 1234567),
			glm::i32vec3(77777, 31337, 98765));
	}

	static int test()
	{
		int Error = 0;

		constexpr glm::i32vec3 ConstPage = build_const_page();
		glm::i32vec3 const RuntimePage = glm::wrappingMul(
			glm::i32vec3(99991, -424242, 1234567),
			glm::i32vec3(77777, 31337, 98765));

		Error += check(glm::all(glm::equal(ConstPage, RuntimePage)) );

		return Error;
	}
}//namespace test_constant_runtime_equality

int main()
{
	int Error = 0;

	Error += test_scalar_wrapping_mul::test();
	Error += test_vector_wrapping_mul::test();
	Error += test_as_signed_unsigned::test();
	Error += test_wrapping_cast::test();
	Error += test_constant_runtime_equality::test();

	return Error;
}
