#include <glm/gtc/type_precision.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/vector_relational.hpp>
#include <glm/packing.hpp>
#include <cmath>
#include <limits>
#include <vector>

static int test_packUnorm2x16()
{
	int Error = 0;

	std::vector<glm::vec2> A;
	A.push_back(glm::vec2(1.0f, 0.0f));
	A.push_back(glm::vec2(0.5f, 0.7f));
	A.push_back(glm::vec2(0.1f, 0.2f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::uint32 C = glm::packUnorm2x16(B);
		glm::vec2 D = glm::unpackUnorm2x16(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 65535.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

static int test_packSnorm2x16()
{
	int Error = 0;

	std::vector<glm::vec2> A;
	A.push_back(glm::vec2( 1.0f, 0.0f));
	A.push_back(glm::vec2(-0.5f,-0.7f));
	A.push_back(glm::vec2(-0.1f, 0.1f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::uint32 C = glm::packSnorm2x16(B);
		glm::vec2 D = glm::unpackSnorm2x16(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 32767.0f * 2.0f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

static int test_packUnorm4x8()
{
	int Error = 0;

	glm::uint32 Packed = glm::packUnorm4x8(glm::vec4(1.0f, 0.5f, 0.0f, 1.0f));
	glm::u8vec4 Vec(255, 128, 0, 255);
	glm::uint32 & Ref = *reinterpret_cast<glm::uint32*>(&Vec[0]);

	Error += Packed == Ref ? 0 : 1;

	std::vector<glm::vec4> A;
	A.push_back(glm::vec4(1.0f, 0.7f, 0.3f, 0.0f));
	A.push_back(glm::vec4(0.5f, 0.1f, 0.2f, 0.3f));
	
	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec4 B(A[i]);
		glm::uint32 C = glm::packUnorm4x8(B);
		glm::vec4 D = glm::unpackUnorm4x8(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 255.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

static int test_packSnorm4x8()
{
	int Error = 0;
	
	std::vector<glm::vec4> A;
	A.push_back(glm::vec4( 1.0f, 0.0f,-0.5f,-1.0f));
	A.push_back(glm::vec4(-0.7f,-0.1f, 0.1f, 0.7f));
	
	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec4 B(A[i]);
		glm::uint32 C = glm::packSnorm4x8(B);
		glm::vec4 D = glm::unpackSnorm4x8(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 127.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

static int test_packHalf2x16()
{
	int Error = 0;
/*
	std::vector<glm::hvec2> A;
	A.push_back(glm::hvec2(glm::half( 1.0f), glm::half( 2.0f)));
	A.push_back(glm::hvec2(glm::half(-1.0f), glm::half(-2.0f)));
	A.push_back(glm::hvec2(glm::half(-1.1f), glm::half( 1.1f)));
*/
	std::vector<glm::vec2> A;
	A.push_back(glm::vec2( 1.0f, 2.0f));
	A.push_back(glm::vec2(-1.0f,-2.0f));
	A.push_back(glm::vec2(-1.1f, 1.1f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::uint C = glm::packHalf2x16(B);
		glm::vec2 D = glm::unpackHalf2x16(C);
		//Error += B == D ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 127.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

//
// packHalf2x16 / unpackHalf2x16 contract under test (glm/packing.hpp):
//  - packHalf2x16 converts both components to 16-bit half-precision floats
//    and packs them into a 32-bit word: the FIRST component occupies the 16
//    LEAST significant bits, the SECOND component the 16 MOST significant
//    bits. unpackHalf2x16 reads the same layout back.
//  - The half conversion itself (glm/detail/type_half.inl) follows binary16:
//    1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits; round to
//    nearest with ties ("0.5") rounded up in magnitude; magnitudes below
//    2^-24 become signed zero; magnitudes above the half range become signed
//    infinity; infinities stay infinities; NaNs stay NaNs with sign and the
//    top significand bits preserved.
//

// Independent reference conversion of a 16-bit half-precision bit pattern to
// float, written directly from the binary16 format. This is the numerical
// basis for the expectations in this file; it deliberately does NOT call
// glm::detail::toFloat32 so the tests do not grade the implementation with
// its own answer key.
static float referenceHalfToFloat(glm::uint16 Bits)
{
	glm::uint16 const Sign = (Bits >> 15) & 0x1;
	glm::uint16 const Exponent = (Bits >> 10) & 0x1F;
	glm::uint16 const Mantissa = Bits & 0x03FF;

	float Magnitude = 0.0f;
	if(Exponent == 0)
		Magnitude = std::ldexp(static_cast<float>(Mantissa), -24); // zero and subnormals: m * 2^-24
	else if(Exponent == 31)
		Magnitude = Mantissa ? std::numeric_limits<float>::quiet_NaN()
		                     : std::numeric_limits<float>::infinity();
	else
		Magnitude = std::ldexp(1024.0f + static_cast<float>(Mantissa), Exponent - 25); // (1 + m/1024) * 2^(e-15)

	return Sign ? -Magnitude : Magnitude;
}

// Category-aware comparison: NaN matches NaN, infinities match by sign,
// zeros match by sign, everything else must be bit-exact.
static bool halfValueMatches(float Actual, float Expected)
{
	if(std::isnan(Expected))
		return std::isnan(Actual) != 0;
	if(std::isinf(Expected))
		return std::isinf(Actual) && (std::signbit(Actual) == std::signbit(Expected));
	if(Expected == 0.0f)
		return Actual == 0.0f && (std::signbit(Actual) == std::signbit(Expected));
	return Actual == Expected;
}

// Unpacking: every one of the 65536 half bit patterns, in both component
// slots, checked against the independent reference conversion.
static int test_unpackHalf2x16_allBitPatterns()
{
	int Error = 0;

	for(glm::uint32 h = 0; h <= 0xFFFFu; ++h)
	{
		float const Ref = referenceHalfToFloat(static_cast<glm::uint16>(h));

		// Pattern in the 16 low bits -> first component; high bits are zero -> +0.0.
		glm::vec2 const Lo = glm::unpackHalf2x16(h);
		Error += halfValueMatches(Lo.x, Ref) ? 0 : 1;
		Error += (Lo.y == 0.0f && !std::signbit(Lo.y)) ? 0 : 1;

		// Pattern in the 16 high bits -> second component; low bits are zero -> +0.0.
		glm::vec2 const Hi = glm::unpackHalf2x16(h << 16);
		Error += halfValueMatches(Hi.y, Ref) ? 0 : 1;
		Error += (Hi.x == 0.0f && !std::signbit(Hi.x)) ? 0 : 1;

		assert(!Error);
	}

	return Error;
}

// Packing: every finite half value is exactly representable as a float, so
// packing that float must reproduce the original 16-bit pattern exactly, in
// the component slot dictated by the layout. Non-finite inputs are checked
// by category.
static int test_packHalf2x16_allHalfValues()
{
	int Error = 0;

	for(glm::uint32 h = 0; h <= 0xFFFFu; ++h)
	{
		float const f = referenceHalfToFloat(static_cast<glm::uint16>(h));

		glm::uint32 const PackedLo = glm::packHalf2x16(glm::vec2(f, 0.0f));
		glm::uint32 const PackedHi = glm::packHalf2x16(glm::vec2(0.0f, f));
		glm::uint32 const Lo = PackedLo & 0xFFFFu;
		glm::uint32 const Hi = PackedHi >> 16;

		if(std::isnan(f))
		{
			// NaN: exponent all ones, nonzero mantissa, sign preserved.
			Error += ((Lo & 0x7C00u) == 0x7C00u && (Lo & 0x03FFu) != 0 && ((Lo >> 15) == (h >> 15))) ? 0 : 1;
			Error += ((Hi & 0x7C00u) == 0x7C00u && (Hi & 0x03FFu) != 0 && ((Hi >> 15) == (h >> 15))) ? 0 : 1;
		}
		else if(std::isinf(f))
		{
			glm::uint32 const Expected = (h & 0x8000u) ? 0xFC00u : 0x7C00u;
			Error += Lo == Expected ? 0 : 1;
			Error += Hi == Expected ? 0 : 1;
		}
		else
		{
			// Finite, exactly representable: bit-exact round trip.
			Error += Lo == h ? 0 : 1;
			Error += Hi == h ? 0 : 1;
		}

		// The component not under test must be +0.0 (0x0000).
		Error += (PackedLo >> 16) == 0u ? 0 : 1;
		Error += (PackedHi & 0xFFFFu) == 0u ? 0 : 1;

		assert(!Error);
	}

	return Error;
}

// Quantization of finite floats that are NOT exactly representable in half
// precision. Every expected pattern below was computed by hand from the
// float bit pattern, independently of the GLM implementation, following the
// conversion contract quoted above.
static int test_packHalf2x16_quantization()
{
	int Error = 0;

	struct Case
	{
		float In;
		glm::uint32 Half; // expected low 16 bits of packHalf2x16(vec2(In, 0))
	};

	Case const Cases[] =
	{
		// 1.1f = 0x3F8CCCCD: kept mantissa 0x066, discarded bits 0x0CCD are
		// less than half a ulp -> rounds down to 1.099609375.
		{ 1.1f,                  0x3C66u },
		{ -1.1f,                 0xBC66u },
		// 1.00048828125f = 1 + 2^-11: exactly halfway between 0x3C00 and
		// 0x3C01; the contract rounds ties up in magnitude.
		{ 1.00048828125f,        0x3C01u },
		{ -1.00048828125f,       0xBC01u },
		// 1.99951171875f: rounding carries out of the mantissa and increments
		// the exponent, landing exactly on 2.0.
		{ 1.99951171875f,        0x4000u },
		// 2^-16 and 2^-20: normal floats that become subnormal halves.
		{ 1.52587890625e-5f,     0x0100u },
		{ 9.5367431640625e-7f,   0x0010u },
		// 2^-25: exactly halfway to the smallest subnormal; tie rounds up.
		{ 2.9802322387695312e-8f, 0x0001u },
		// 2^-26: below half of the smallest subnormal -> signed zero,
		// with the sign preserved for negative inputs.
		{ 1.4901161193847656e-8f, 0x0000u },
		{ -1.4901161193847656e-8f, 0x8000u },
		// 65504 = 0x7BFF is the largest finite half; 65520 rounds up to
		// infinity; 65536 overflows straight to infinity.
		{ 65504.0f,              0x7BFFu },
		{ 65520.0f,              0x7C00u },
		{ 65536.0f,              0x7C00u },
		{ -65536.0f,             0xFC00u },
		// Signed zeros keep their sign bit.
		{ 0.0f,                  0x0000u },
		{ -0.0f,                 0x8000u },
	};

	for(std::size_t i = 0; i < sizeof(Cases) / sizeof(Cases[0]); ++i)
	{
		glm::uint32 const Packed = glm::packHalf2x16(glm::vec2(Cases[i].In, 0.0f));
		Error += (Packed & 0xFFFFu) == Cases[i].Half ? 0 : 1;

		// What a shader reads back after the round trip must be the value of
		// the expected half pattern, not the original float.
		glm::vec2 const ReadBack = glm::unpackHalf2x16(Packed);
		Error += halfValueMatches(ReadBack.x, referenceHalfToFloat(static_cast<glm::uint16>(Cases[i].Half))) ? 0 : 1;

		assert(!Error);
	}

	return Error;
}

// Signed zeros and non-finite values, interpreted by category.
static int test_half2x16_specialValues()
{
	int Error = 0;

	// Signed zeros: -0.0 packs to 0x8000, +0.0 to 0x0000, and the signs
	// survive unpacking.
	glm::uint32 const PackedZero = glm::packHalf2x16(glm::vec2(-0.0f, 0.0f));
	Error += PackedZero == 0x00008000u ? 0 : 1;

	glm::vec2 const Zero = glm::unpackHalf2x16(0x00008000u);
	Error += (Zero.x == 0.0f && std::signbit(Zero.x)) ? 0 : 1;
	Error += (Zero.y == 0.0f && !std::signbit(Zero.y)) ? 0 : 1;

	// Infinities: sign preserved in both directions.
	float const Inf = std::numeric_limits<float>::infinity();
	glm::uint32 const PackedInf = glm::packHalf2x16(glm::vec2(Inf, -Inf));
	Error += PackedInf == 0xFC007C00u ? 0 : 1;

	glm::vec2 const Infs = glm::unpackHalf2x16(0xFC007C00u);
	Error += (std::isinf(Infs.x) && !std::signbit(Infs.x)) ? 0 : 1;
	Error += (std::isinf(Infs.y) && std::signbit(Infs.y)) ? 0 : 1;

	// NaN: the packed half must stay in the NaN category (exponent all ones,
	// nonzero mantissa) with the sign preserved; unpacking must yield NaN.
	float const NaN = std::numeric_limits<float>::quiet_NaN();
	glm::uint32 const PackedNaN = glm::packHalf2x16(glm::vec2(NaN, -NaN));
	glm::uint32 const LoNaN = PackedNaN & 0xFFFFu;
	glm::uint32 const HiNaN = PackedNaN >> 16;
	Error += ((LoNaN & 0x7C00u) == 0x7C00u && (LoNaN & 0x03FFu) != 0 && !(LoNaN & 0x8000u)) ? 0 : 1;
	Error += ((HiNaN & 0x7C00u) == 0x7C00u && (HiNaN & 0x03FFu) != 0 && (HiNaN & 0x8000u)) ? 0 : 1;

	glm::vec2 const NaNs = glm::unpackHalf2x16(PackedNaN);
	Error += std::isnan(NaNs.x) ? 0 : 1;
	Error += std::isnan(NaNs.y) ? 0 : 1;

	assert(!Error);
	return Error;
}

// Round trip through the same endpoint, with asymmetric component values so
// that the error a shader would see — components swapped or read from the
// wrong half of the word — is detected. The packed word is also dissected
// manually, exactly the way a shader extracts the components per the
// documented convention.
static int test_half2x16_roundtripShaderView()
{
	int Error = 0;

	// Distinct x/y, all exactly representable in half precision.
	std::vector<glm::vec2> A;
	A.push_back(glm::vec2( 1.0f,  2.0f));
	A.push_back(glm::vec2(-1.0f,  0.5f));
	A.push_back(glm::vec2( 0.5f, -2.0f));
	A.push_back(glm::vec2( 1.5f, -1.75f));
	A.push_back(glm::vec2( 65504.0f, 9.5367431640625e-7f)); // max finite half, 2^-20
	A.push_back(glm::vec2(-0.0f,  0.0f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::uint32 const Packed = glm::packHalf2x16(A[i]);

		// Shader-side extraction per the documented convention: x from the
		// 16 least significant bits, y from the 16 most significant bits.
		glm::uint16 const LoBits = static_cast<glm::uint16>(Packed & 0xFFFFu);
		glm::uint16 const HiBits = static_cast<glm::uint16>(Packed >> 16);
		Error += halfValueMatches(referenceHalfToFloat(LoBits), A[i].x) ? 0 : 1;
		Error += halfValueMatches(referenceHalfToFloat(HiBits), A[i].y) ? 0 : 1;

		// Library read-back must agree exactly (values are representable).
		glm::vec2 const D = glm::unpackHalf2x16(Packed);
		Error += (D.x == A[i].x && D.y == A[i].y) ? 0 : 1;
		Error += (std::signbit(D.x) == std::signbit(A[i].x) && std::signbit(D.y) == std::signbit(A[i].y)) ? 0 : 1;

		assert(!Error);
	}

	// Concrete spot check of the word layout: 0.75 -> 0x3A00 in the low
	// half, -1.25 -> 0xBD00 in the high half.
	glm::uint32 const Packed = glm::packHalf2x16(glm::vec2(0.75f, -1.25f));
	Error += Packed == 0xBD003A00u ? 0 : 1;
	glm::vec2 const Fetched = glm::unpackHalf2x16(0xBD003A00u);
	Error += (Fetched.x == 0.75f && Fetched.y == -1.25f) ? 0 : 1;

	assert(!Error);
	return Error;
}

static int test_packDouble2x32()
{
	int Error = 0;
	
	std::vector<glm::uvec2> A;
	A.push_back(glm::uvec2( 1, 2));
	A.push_back(glm::uvec2(-1,-2));
	A.push_back(glm::uvec2(-1000, 1100));
	
	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::uvec2 B(A[i]);
		double C = glm::packDouble2x32(B);
		glm::uvec2 D = glm::unpackDouble2x32(C);
		Error += B == D ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int main()
{
	int Error = 0;
	
	Error += test_packSnorm4x8();
	Error += test_packUnorm4x8();
	Error += test_packSnorm2x16();
	Error += test_packUnorm2x16();
	Error += test_packHalf2x16();
	Error += test_unpackHalf2x16_allBitPatterns();
	Error += test_packHalf2x16_allHalfValues();
	Error += test_packHalf2x16_quantization();
	Error += test_half2x16_specialValues();
	Error += test_half2x16_roundtripShaderView();
	Error += test_packDouble2x32();

	return Error;
}

