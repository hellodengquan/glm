#define GLM_FORCE_SWIZZLE
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/glm.hpp>

static int test_ivec2_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::ivec2 A(1, 2);
		glm::ivec2 B = A.yx();
		glm::ivec2 C = B.yx();

		Error += A != B ? 0 : 1;
		Error += A == C ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::ivec2 A(1, 2);
		glm::ivec2 B = A.yx;
		glm::ivec2 C = A.yx;

		Error += A != B ? 0 : 1;
		Error += B == C ? 0 : 1;

		B.xy = B.yx;
		C.xy = C.yx;

		Error += B == C ? 0 : 1;

		glm::ivec2 D(0, 0);
		D.yx = A.xy;
		Error += A.yx() == D ? 0 : 1;

		glm::ivec2 E = A.yx;
		Error += E == D ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE

	return Error;
}

static int test_ivec3_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::ivec3 A(1, 2, 3);
		glm::ivec3 B = A.zyx();
		glm::ivec3 C = B.zyx();

		Error += A != B ? 0 : 1;
		Error += A == C ? 0 : 1;
	}
#	endif

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::ivec3 const A(1, 2, 3);
		glm::ivec2 B = A.yx;
		glm::ivec2 C = A.yx;

		Error += A.yx() == B ? 0 : 1;
		Error += B == C ? 0 : 1;

		B.xy = B.yx;
		C.xy = C.yx;

		Error += B == C ? 0 : 1;

		glm::ivec2 D(0, 0);
		D.yx = A.xy;

		Error += A.yx() == D ? 0 : 1;

		glm::ivec2 E(0, 0);
		E.xy = A.xy();

		Error += E == A.xy() ? 0 : 1;
		Error += E.xy() == A.xy() ? 0 : 1;

		glm::ivec3 const F = A.xxx + A.xxx;
		Error += F == glm::ivec3(2) ? 0 : 1;

		glm::ivec3 const G = A.xxx - A.xxx;
		Error += G == glm::ivec3(0) ? 0 : 1;

		glm::ivec3 const H = A.xxx * A.xxx;
		Error += H == glm::ivec3(1) ? 0 : 1;

		glm::ivec3 const I = A.xxx / A.xxx;
		Error += I == glm::ivec3(1) ? 0 : 1;

		glm::ivec3 J(1, 2, 3);
		J.xyz += glm::ivec3(1);
		Error += J == glm::ivec3(2, 3, 4) ? 0 : 1;

		glm::ivec3 K(1, 2, 3);
		K.xyz += A.xyz;
		Error += K == glm::ivec3(2, 4, 6) ? 0 : 1;
	}
#	endif

	return Error;
}

static int test_ivec4_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::ivec4 A(1, 2, 3, 4);
		glm::ivec4 B = A.wzyx();
		glm::ivec4 C = B.wzyx();

		Error += A != B ? 0 : 1;
		Error += A == C ? 0 : 1;
	}
#	endif

	return Error;
}

static int test_vec4_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::vec4 A(1, 2, 3, 4);
		glm::vec4 B = A.wzyx();
		glm::vec4 C = B.wzyx();

		Error += glm::any(glm::notEqual(A, B, 0.0001f)) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;

		float D = glm::dot(C.wzyx(), C.xyzw());
		Error += glm::equal(D, 20.f, 0.001f) ? 0 : 1;
	}
#	endif

	return Error;
}

#if GLM_HAS_CONSTEXPR
#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
// The operator swizzle storage is typed and sized after the host vector so its
// accessors can participate in constant expressions. These checks exercise the
// writable component reordering logic at compile time.

// Every target component of one assignment sees the same source color.
constexpr int test_constexpr_self_swap()
{
	glm::detail::_swizzle<2, int, glm::defaultp, 0, 1, -1, -2> xy{};
	xy = glm::ivec2(1, 2);

	// Snapshot the source color once before applying the reversed layout.
	int const s0 = xy[0];
	int const s1 = xy[1];

	glm::detail::_swizzle<2, int, glm::defaultp, 1, 0, -1, -2> yx{};
	yx = glm::ivec2(s0, s1);

	return yx[0] + 10 * yx[1];
}

// A partial write only touches the selected components and leaves the others
// available in the host storage untouched.
constexpr int test_constexpr_partial_write()
{
	glm::detail::_swizzle<4, int, glm::defaultp, 0, 1, 2, 3> v{};
	v = glm::ivec4(1, 2, 3, 4);

	glm::detail::_swizzle<2, int, glm::defaultp, 0, 1, -1, -2> xy{};
	xy = glm::ivec2(9, 8);

	return xy[0] + 10 * xy[1];
}

// Compile time baking and a runtime update run the very same assignment code,
// so they must agree on the resulting values.
constexpr int test_constexpr_baked_value()
{
	glm::detail::_swizzle<4, int, glm::defaultp, 0, 1, 2, 3> v{};
	v = glm::ivec4(1, 2, 3, 4);
	v += glm::ivec4(10, 10, 10, 10);
	return v[0] + 10 * v[1] + 100 * v[2] + 1000 * v[3];
}

// Repeated components stay readable through the typed storage; they are not
// promoted to writable targets (writability is rejected by the Stub overload).
constexpr int test_constexpr_duplicate_read()
{
	glm::detail::_swizzle<4, int, glm::defaultp, 0, 1, 2, 3> v{};
	v = glm::ivec4(1, 2, 7, 4);
	return v[2] + v[2] + v[2] + v[2];
}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR

static int test_constexpr_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	static_assert(test_constexpr_self_swap() == 21, "GLM: constexpr self swap");
	static_assert(test_constexpr_partial_write() == 89, "GLM: constexpr partial write");
	static_assert(test_constexpr_baked_value() == 15431, "GLM: constexpr baked value");
	static_assert(test_constexpr_duplicate_read() == 28, "GLM: constexpr duplicate read");

	// Mirror the baked computations at runtime and require identical results.
	{
		glm::detail::_swizzle<4, int, glm::defaultp, 0, 1, 2, 3> v{};
		v = glm::ivec4(1, 2, 3, 4);
		v += glm::ivec4(10, 10, 10, 10);
		Error += (v[0] + 10 * v[1] + 100 * v[2] + 1000 * v[3] == test_constexpr_baked_value()) ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	// Function swizzles return a value through the active scalar members and
	// are therefore usable directly in constant expressions.
	constexpr glm::ivec4 A(1, 2, 3, 4);
	constexpr glm::ivec4 B = A.wzyx();
	static_assert(B == glm::ivec4(4, 3, 2, 1), "GLM: constexpr function swizzle");
#	endif

	return Error;
}
#endif//GLM_HAS_CONSTEXPR

int main()
{
	int Error = 0;

	Error += test_ivec2_swizzle();
	Error += test_ivec3_swizzle();
	Error += test_ivec4_swizzle();
	Error += test_vec4_swizzle();

#if GLM_HAS_CONSTEXPR
	Error += test_constexpr_swizzle();
#endif

	return Error;
}



