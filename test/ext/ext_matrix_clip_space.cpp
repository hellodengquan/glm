#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_float3.hpp>

#include <cmath>

namespace
{
	const float Near = 1.0f;
	const float Far = 9.0f;
	const float Eps = 1.0e-4f;
	const float Aspect = 4.0f / 3.0f;
	const float Fovy = 0.927295218f; // 2 * atan(0.5), about 53 degrees

	// Depth in normalized device coordinates after the perspective divide.
	float ndcDepth(glm::mat4 const& Projection, float ViewZ)
	{
		glm::vec4 const Clip = Projection * glm::vec4(0.0f, 0.0f, ViewZ, 1.0f);
		return Clip.z / Clip.w;
	}

	// Horizontal screen position in normalized device coordinates after the perspective divide.
	float ndcX(glm::mat4 const& Projection, float ViewX, float ViewZ)
	{
		glm::vec4 const Clip = Projection * glm::vec4(ViewX, 0.0f, ViewZ, 1.0f);
		return Clip.x / Clip.w;
	}

	int checkReversePerspective(
		glm::mat4 const& Projection,
		glm::mat4 const& ForwardProjection,
		float Sign, // +1 for left-handed view space (front is positive z), -1 for right-handed
		float NearDepth,
		float FarDepth,
		float MiddleDepth)
	{
		int Error = 0;

		float const NearZ = ndcDepth(Projection, Sign * Near);
		float const FarZ = ndcDepth(Projection, Sign * Far);
		float const MiddleZ = ndcDepth(Projection, Sign * std::sqrt(Near * Far));

		Error += std::abs(NearZ - NearDepth) < Eps ? 0 : 1;
		Error += std::abs(FarZ - FarDepth) < Eps ? 0 : 1;
		Error += std::abs(MiddleZ - MiddleDepth) < Eps ? 0 : 1;

		// The near plane receives the larger depth value than the far plane.
		Error += NearZ > FarZ ? 0 : 1;

		// The projected composition is unchanged: the same view position lands on the same screen column.
		float const ViewX = 0.3f;
		float const ViewZ = Sign * 3.0f;
		Error += std::abs(ndcX(Projection, ViewX, ViewZ) - ndcX(ForwardProjection, ViewX, ViewZ)) < Eps ? 0 : 1;

		return Error;
	}
}

static int test_perspective()
{
	int Error = 0;

	// Right-handed, zero to one clip space: near maps to 1, far to 0.
	Error += checkReversePerspective(
		glm::perspectiveRH_ZO_REV(Fovy, Aspect, Near, Far),
		glm::perspectiveRH_ZO(Fovy, Aspect, Near, Far),
		-1.0f, 1.0f, 0.0f, 0.25f);

	// Right-handed, negative one to one clip space: near maps to 1, far to -1.
	Error += checkReversePerspective(
		glm::perspectiveRH_NO_REV(Fovy, Aspect, Near, Far),
		glm::perspectiveRH_NO(Fovy, Aspect, Near, Far),
		-1.0f, 1.0f, -1.0f, -0.5f);

	// Left-handed, zero to one clip space: near maps to 1, far to 0.
	Error += checkReversePerspective(
		glm::perspectiveLH_ZO_REV(Fovy, Aspect, Near, Far),
		glm::perspectiveLH_ZO(Fovy, Aspect, Near, Far),
		1.0f, 1.0f, 0.0f, 0.25f);

	// Left-handed, negative one to one clip space: near maps to 1, far to -1.
	Error += checkReversePerspective(
		glm::perspectiveLH_NO_REV(Fovy, Aspect, Near, Far),
		glm::perspectiveLH_NO(Fovy, Aspect, Near, Far),
		1.0f, 1.0f, -1.0f, -0.5f);

	return Error;
}

static int test_explicitHandedness()
{
	int Error = 0;

	// The explicit entries must honour their named handedness regardless of the global default.
	glm::vec4 const RightHanded = glm::perspectiveRH_ZO_REV(Fovy, Aspect, Near, Far) * glm::vec4(0.0f, 0.0f, -3.0f, 1.0f);
	glm::vec4 const LeftHanded = glm::perspectiveLH_ZO_REV(Fovy, Aspect, Near, Far) * glm::vec4(0.0f, 0.0f, 3.0f, 1.0f);

	// In front of a right-handed camera w equals -z, in front of a left-handed camera w equals +z: both positive.
	Error += std::abs(RightHanded.w - 3.0f) < Eps ? 0 : 1;
	Error += std::abs(LeftHanded.w - 3.0f) < Eps ? 0 : 1;

	// The same physical point in front maps to the same screen column for either handedness.
	Error += std::abs(RightHanded.x / RightHanded.w - LeftHanded.x / LeftHanded.w) < Eps ? 0 : 1;

	return Error;
}

static int test_frustum()
{
	int Error = 0;

	glm::mat4 const RightHandedZO = glm::frustumRH_ZO_REV(-1.0f, 1.0f, -1.0f, 1.0f, Near, Far);
	glm::mat4 const RightHandedNO = glm::frustumRH_NO_REV(-1.0f, 1.0f, -1.0f, 1.0f, Near, Far);
	glm::mat4 const LeftHandedZO = glm::frustumLH_ZO_REV(-1.0f, 1.0f, -1.0f, 1.0f, Near, Far);
	glm::mat4 const LeftHandedNO = glm::frustumLH_NO_REV(-1.0f, 1.0f, -1.0f, 1.0f, Near, Far);

	Error += std::abs(ndcDepth(RightHandedZO, -Near) - 1.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(RightHandedZO, -Far) - 0.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(RightHandedNO, -Near) - 1.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(RightHandedNO, -Far) + 1.0f) < Eps ? 0 : 1;

	Error += std::abs(ndcDepth(LeftHandedZO, Near) - 1.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(LeftHandedZO, Far) - 0.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(LeftHandedNO, Near) - 1.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(LeftHandedNO, Far) + 1.0f) < Eps ? 0 : 1;

	// An off-axis frustum matches the symmetric perspective matrix when the extents match the field of view at the near plane.
	glm::mat4 const Symmetric = glm::perspectiveRH_ZO_REV(Fovy, Aspect, Near, Far);
	float const HalfHeight = Near * std::tan(Fovy * 0.5f);
	float const HalfWidth = HalfHeight * Aspect;
	glm::mat4 const OffAxis = glm::frustumRH_ZO_REV(-HalfWidth, HalfWidth, -HalfHeight, HalfHeight, Near, Far);
	Error += glm::all(glm::equal(Symmetric, OffAxis, Eps)) ? 0 : 1;

	return Error;
}

static int test_perspectiveFov()
{
	int Error = 0;

	float const Width = 800.0f;
	float const Height = 600.0f;

	// A square viewport makes perspectiveFov reduce to perspective.
	Error += glm::all(glm::equal(
		glm::perspectiveFovRH_ZO_REV(Fovy, Height, Height, Near, Far),
		glm::perspectiveRH_ZO_REV(Fovy, 1.0f, Near, Far), Eps)) ? 0 : 1;
	Error += glm::all(glm::equal(
		glm::perspectiveFovLH_NO_REV(Fovy, Height, Height, Near, Far),
		glm::perspectiveLH_NO_REV(Fovy, 1.0f, Near, Far), Eps)) ? 0 : 1;

	// A non-square viewport keeps the horizontal to vertical ratio from the field of view.
	glm::mat4 const Projection = glm::perspectiveFovRH_ZO_REV(Fovy, Width, Height, Near, Far);
	Error += std::abs(ndcDepth(Projection, -Near) - 1.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(Projection, -Far) - 0.0f) < Eps ? 0 : 1;
	Error += std::abs(Projection[0][0] * Width - Projection[1][1] * Height) < Eps ? 0 : 1;

	return Error;
}

static int test_forwardUnchanged()
{
	int Error = 0;

	// The original entry points keep their existing conventions.
	Error += std::abs(ndcDepth(glm::perspectiveRH_ZO(Fovy, Aspect, Near, Far), -Near) - 0.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(glm::perspectiveRH_ZO(Fovy, Aspect, Near, Far), -Far) - 1.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(glm::perspectiveRH_NO(Fovy, Aspect, Near, Far), -Near) + 1.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(glm::perspectiveRH_NO(Fovy, Aspect, Near, Far), -Far) - 1.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(glm::perspectiveLH_ZO(Fovy, Aspect, Near, Far), Near) - 0.0f) < Eps ? 0 : 1;
	Error += std::abs(ndcDepth(glm::perspectiveLH_ZO(Fovy, Aspect, Near, Far), Far) - 1.0f) < Eps ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_perspective();
	Error += test_explicitHandedness();
	Error += test_frustum();
	Error += test_perspectiveFov();
	Error += test_forwardUnchanged();

	return Error;
}
