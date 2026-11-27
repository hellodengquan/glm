#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/epsilon.hpp>
#include <cmath>
#include <limits>

static int test_affine()
{
	int Error = 0;

	{
		glm::mat3 const M(
			2.f, 0.f, 0.f,
			0.f, 2.f, 0.f,
			0.f, 0.f, 1.f);
		glm::mat3 const A = glm::affineInverse(M);
		glm::mat3 const I = glm::inverse(M);
		glm::mat3 const R = glm::affineInverse(A);

		for(glm::length_t i = 0; i < A.length(); ++i)
		{
			Error += glm::all(glm::epsilonEqual(M[i], R[i], 0.01f)) ? 0 : 1;
			Error += glm::all(glm::epsilonEqual(A[i], I[i], 0.01f)) ? 0 : 1;
		}
	}

	{
		glm::mat4 const M(
			2.f, 0.f, 0.f, 0.f,
			0.f, 2.f, 0.f, 0.f,
			0.f, 0.f, 2.f, 0.f,
			0.f, 0.f, 0.f, 1.f);
		glm::mat4 const A = glm::affineInverse(M);
		glm::mat4 const I = glm::inverse(M);
		glm::mat4 const R = glm::affineInverse(A);

		for(glm::length_t i = 0; i < A.length(); ++i)
		{
			Error += glm::all(glm::epsilonEqual(M[i], R[i], 0.01f)) ? 0 : 1;
			Error += glm::all(glm::epsilonEqual(A[i], I[i], 0.01f)) ? 0 : 1;
		}
	}

	return Error;
}

// Relative Frobenius distance between m * inverse (and inverse * m) and identity.
template<glm::length_t C, typename T, glm::qualifier Q>
static T round_trip_error(glm::mat<C, C, T, Q> const& m, glm::mat<C, C, T, Q> const& inverse)
{
	glm::mat<C, C, T, Q> const Identity(static_cast<T>(1));
	glm::mat<C, C, T, Q> const Forward = m * inverse - Identity;
	glm::mat<C, C, T, Q> const Backward = inverse * m - Identity;

	T Residual(0);
	T IdentityNorm(0);
	for(glm::length_t c = 0; c < C; ++c)
	{
		Residual += glm::max(glm::dot(Forward[c], Forward[c]), glm::dot(Backward[c], Backward[c]));
		IdentityNorm += glm::dot(Identity[c], Identity[c]);
	}
	return std::sqrt(Residual) / std::sqrt(IdentityNorm);
}

// Exact bit preservation check done without floating point operator==,
// which -Wfloat-equal rejects. The sentinel is a uniform constant matrix, so
// an epsilon comparison is equivalent here.
template<glm::length_t C, typename T, glm::qualifier Q>
static bool matrix_unchanged(glm::mat<C, C, T, Q> const& value, glm::mat<C, C, T, Q> const& expected)
{
	for(glm::length_t c = 0; c < C; ++c)
	{
		if(!glm::all(glm::epsilonEqual(value[c], expected[c], std::numeric_limits<T>::epsilon())))
			return false;
	}
	return true;
}

// On success tryInverse must round-trip within the caller supplied tolerance.
template<glm::length_t C, typename T, glm::qualifier Q>
static int test_try_success(glm::mat<C, C, T, Q> const& m, T tolerance)
{
	int Error = 0;

	glm::mat<C, C, T, Q> Result;
	bool const Success = glm::tryInverse(m, Result, tolerance);
	Error += Success ? 0 : 1;
	Error += round_trip_error(m, Result) <= tolerance ? 0 : 1;

	// Default tolerance overload must also accept a well-conditioned matrix.
	glm::mat<C, C, T, Q> DefaultResult;
	Error += glm::tryInverse(m, DefaultResult) ? 0 : 1;

	return Error;
}

// tryInverse reports failure and never touches the caller's previous result.
template<glm::length_t C, typename T, glm::qualifier Q>
static int test_try_failure(glm::mat<C, C, T, Q> const& m)
{
	int Error = 0;

	glm::mat<C, C, T, Q> const Sentinel(static_cast<T>(7));
	glm::mat<C, C, T, Q> Result = Sentinel;
	bool const Success = glm::tryInverse(m, Result);

	Error += Success ? 1 : 0;
	Error += matrix_unchanged(Result, Sentinel) ? 0 : 1;

	return Error;
}

static int test_try_inverse_success()
{
	int Error = 0;

	{
		glm::mat2 const M(glm::vec2(4.f, 7.f), glm::vec2(2.f, 6.f));
		Error += test_try_success(M, 1e-5f);
	}
	{
		glm::mat3 const M(
			1.f, 2.f, 0.f,
			0.f, 1.f, 3.f,
			4.f, 0.f, 1.f);
		Error += test_try_success(M, 1e-5f);
	}
	{
		glm::mat4 const M(
			2.f, 0.f, 0.f, 0.f,
			0.f, 3.f, 0.f, 0.f,
			0.f, 0.f, 4.f, 0.f,
			1.f, 2.f, 3.f, 1.f);
		Error += test_try_success(M, 1e-5f);
	}
	{
		glm::dmat4 const M(
			2.0, 0.0, 0.0, 0.0,
			0.0, 3.0, 0.0, 0.0,
			0.0, 0.0, 4.0, 0.0,
			1.0, 2.0, 3.0, 1.0);
		Error += test_try_success(M, 1e-12);
	}

	return Error;
}

static int test_try_inverse_singular()
{
	int Error = 0;

	// Exactly singular matrices of every supported size.
	Error += test_try_failure(glm::mat2(glm::vec2(1.f, 2.f), glm::vec2(2.f, 4.f)));

	Error += test_try_failure(glm::mat3(
		glm::vec3(1.f, 0.f, 0.f),
		glm::vec3(0.f, 1.f, 0.f),
		glm::vec3(1.f, 1.f, 0.f)));

	Error += test_try_failure(glm::mat4(
		glm::vec4(1.f, 0.f, 0.f, 0.f),
		glm::vec4(0.f, 1.f, 0.f, 0.f),
		glm::vec4(1.f, 1.f, 0.f, 0.f),
		glm::vec4(0.f, 0.f, 0.f, 1.f)));

	// A zero matrix has no inverse and must be rejected.
	Error += test_try_failure(glm::mat4(0.f));

	return Error;
}

static int test_try_inverse_ill_conditioned()
{
	int Error = 0;

	// 4x4 Hilbert matrix: invertible but ill-conditioned, the float
	// round-trip error is several orders of magnitude above the default
	// tolerance so it must be rejected rather than handed to the caller.
	glm::mat4 Hilbert;
	for(glm::length_t c = 0; c < 4; ++c)
		for(glm::length_t r = 0; r < 4; ++r)
			Hilbert[c][r] = 1.f / static_cast<float>(c + r + 1);

	Error += test_try_failure(Hilbert);

	// The legacy, unchecked inverse still produces a (poor) result: existing
	// callers that take the inverse directly keep their behaviour.
	glm::mat4 const Legacy = glm::inverse(Hilbert);
	Error += round_trip_error(Hilbert, Legacy) > 1e-4f ? 0 : 1;

	return Error;
}

// Changing scene length units (a uniform scale) must never change whether a
// transform is judged invertible.
static int test_try_inverse_scale_invariance()
{
	int Error = 0;

	{
		glm::mat3 const M(
			1.f, 2.f, 0.f,
			0.f, 1.f, 3.f,
			4.f, 0.f, 1.f);
		glm::mat3 Inverse;
		bool const Reference = glm::tryInverse(M, Inverse);

		float const Scales[] = {1000.f, 100.f, 0.01f, 1e-4f};
		for(std::size_t s = 0; s < sizeof(Scales) / sizeof(Scales[0]); ++s)
		{
			glm::mat3 ScaledInverse;
			bool const Scaled = glm::tryInverse(M * Scales[s], ScaledInverse);
			Error += Scaled == Reference ? 0 : 1;
		}
	}

	{
		// A singular transform stays singular at every scale.
		glm::mat2 const Singular(glm::vec2(1.f, 2.f), glm::vec2(2.f, 4.f));
		float const Scales[] = {1000.f, 0.001f};
		for(std::size_t s = 0; s < sizeof(Scales) / sizeof(Scales[0]); ++s)
			Error += test_try_failure(Singular * Scales[s]);
	}

	{
		// The ill-conditioned Hilbert matrix is rejected at every scale too.
		glm::mat4 Hilbert;
		for(glm::length_t c = 0; c < 4; ++c)
			for(glm::length_t r = 0; r < 4; ++r)
				Hilbert[c][r] = 1.f / static_cast<float>(c + r + 1);

		Error += test_try_failure(Hilbert * 1000.f);
		Error += test_try_failure(Hilbert * 1e-4f);
	}

	return Error;
}

// The caller supplied tolerance really controls acceptance: a tolerance above
// the observed round-trip error accepts, below it rejects.
static int test_try_inverse_tolerance()
{
	int Error = 0;

	glm::mat4 const M(
		1.234567f, 0.245f,   0.113f,   0.f,
		0.245f,   1.345678f, 0.412f,   0.f,
		0.113f,   0.412f,    1.56789f, 0.f,
		0.789f,   0.321f,    0.654f,   1.f);

	glm::mat4 const LegacyInverse = glm::inverse(M);
	float const ActualError = round_trip_error(M, LegacyInverse);

	glm::mat4 Result;
	Error += glm::tryInverse(M, Result, ActualError * 10.f) ? 0 : 1;

	glm::mat4 const Sentinel = Result;
	Error += glm::tryInverse(M, Result, ActualError * 0.1f) ? 1 : 0;
	Error += matrix_unchanged(Result, Sentinel) ? 0 : 1;

	// Non positive tolerances are rejected up front without writing.
	Error += glm::tryInverse(M, Result, 0.f) ? 1 : 0;
	Error += glm::tryInverse(M, Result, -1.f) ? 1 : 0;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_affine();
	Error += test_try_inverse_success();
	Error += test_try_inverse_singular();
	Error += test_try_inverse_ill_conditioned();
	Error += test_try_inverse_scale_invariance();
	Error += test_try_inverse_tolerance();

	return Error;
}
