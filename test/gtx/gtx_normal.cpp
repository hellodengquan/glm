#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/normal.hpp>
#include <glm/gtc/epsilon.hpp>

namespace
{
	// Reconstructs the world-space perturbed normal of a normal-map texel,
	// the way a renderer does: bitangent = sign * cross(normal, tangent).
	glm::vec3 perturbedNormal(
		glm::vec3 const& normal,
		glm::vec3 const& tangent,
		float bitangentSign,
		glm::vec3 const& texel)
	{
		glm::vec3 const bitangent = bitangentSign * glm::cross(normal, tangent);
		return glm::normalize(tangent * texel.x + bitangent * texel.y + normal * texel.z);
	}

	bool nearEqual(glm::vec3 const& a, glm::vec3 const& b, float epsilon = 1e-5f)
	{
		return glm::all(glm::epsilonEqual(a, b, epsilon));
	}

	bool nearEqual(float a, float b, float epsilon = 1e-5f)
	{
		return glm::epsilonEqual(a, b, epsilon);
	}
}//namespace

int test_triangleTangentFrame()
{
	int Error(0);

	glm::vec3 const p1(0.0f, 0.0f, 0.0f);
	glm::vec3 const p2(2.0f, 0.0f, 0.0f);
	glm::vec3 const p3(0.0f, 3.0f, 1.0f);
	glm::vec2 const uv1(0.25f, 0.5f);
	glm::vec2 const uv2(0.75f, 0.5f);
	glm::vec2 const uv3(0.25f, 1.0f);

	glm::vec3 const normal = glm::triangleNormal(p1, p2, p3);
	glm::vec3 const tangent = glm::triangleTangent(p1, p2, p3, uv1, uv2, uv3);
	glm::vec3 const bitangent = glm::triangleBitangent(p1, p2, p3, uv1, uv2, uv3);

	// U varies only along p1->p2, so the tangent must follow that edge.
	Error += nearEqual(tangent, glm::vec3(1.0f, 0.0f, 0.0f)) ? 0 : 1;
	// V varies only along p1->p3.
	Error += nearEqual(bitangent, glm::normalize(p3 - p1)) ? 0 : 1;
	// The frame is orthogonal to the geometric normal.
	Error += nearEqual(glm::dot(normal, tangent), 0.0f) ? 0 : 1;
	Error += nearEqual(glm::dot(normal, bitangent), 0.0f) ? 0 : 1;

	return Error;
}

int test_bitangentSign()
{
	int Error(0);

	glm::vec3 const normal(0.0f, 0.0f, 1.0f);
	glm::vec3 const tangent(1.0f, 0.0f, 0.0f);

	// Right-handed frame: bitangent == cross(normal, tangent).
	Error += glm::bitangentSign(normal, tangent, glm::vec3(0.0f, 1.0f, 0.0f)) == 1.0f ? 0 : 1;
	// Left-handed frame: bitangent == -cross(normal, tangent).
	Error += glm::bitangentSign(normal, tangent, glm::vec3(0.0f, -1.0f, 0.0f)) == -1.0f ? 0 : 1;

	return Error;
}

int test_convertTangentFrame()
{
	int Error(0);

	// Source tangent frame from the authoring tool.
	glm::vec3 const p1(0.0f, 0.0f, 0.0f);
	glm::vec3 const p2(2.0f, 0.0f, 0.0f);
	glm::vec3 const p3(0.0f, 3.0f, 1.0f);
	glm::vec2 const uv1(0.25f, 0.5f);
	glm::vec2 const uv2(0.75f, 0.5f);
	glm::vec2 const uv3(0.25f, 1.0f);

	glm::vec3 const normalIn = glm::triangleNormal(p1, p2, p3);
	glm::vec3 const tangentIn = glm::triangleTangent(p1, p2, p3, uv1, uv2, uv3);
	glm::vec3 const bitangentIn = glm::triangleBitangent(p1, p2, p3, uv1, uv2, uv3);
	float const signIn = glm::bitangentSign(normalIn, tangentIn, bitangentIn);

	// A tangent-space normal-map texel (pointing mostly up, slightly tilted).
	glm::vec3 const texelIn = glm::normalize(glm::vec3(0.25f, -0.5f, 1.0f));

	// The world-space perturbation the texel represents in the source convention.
	glm::vec3 const worldIn = perturbedNormal(normalIn, tangentIn, signIn, texelIn);

	// Geometric change of basis between tool and renderer: mirror on Z
	// (typical left-handed <-> right-handed conversion), determinant -1.
	glm::mat3 const mirrorZ = glm::mat3(
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, -1.0f);

	// 1) Geometry mirrored, texture V axis unchanged.
	{
		glm::vec3 normalOut, tangentOut;
		float signOut(0.0f);
		glm::convertTangentFrame(mirrorZ, false, normalIn, tangentIn, signIn, normalOut, tangentOut, signOut);

		Error += nearEqual(normalOut, mirrorZ * normalIn) ? 0 : 1;
		Error += nearEqual(tangentOut, mirrorZ * tangentIn) ? 0 : 1;
		// Reflection must flip the bitangent sign.
		Error += nearEqual(signOut, -signIn) ? 0 : 1;

		glm::vec3 const texelOut = glm::convertNormalMapTexel(texelIn, false);
		glm::vec3 const worldOut = perturbedNormal(normalOut, tangentOut, signOut, texelOut);
		// The same texel must perturb the normal to the same side in the
		// common world space: worldOut == basisChange * worldIn.
		Error += nearEqual(worldOut, glm::normalize(mirrorZ * worldIn)) ? 0 : 1;
	}

	// 2) Geometry unchanged, texture V axis reversed.
	{
		glm::mat3 const identity(1.0f);
		glm::vec3 normalOut, tangentOut;
		float signOut(0.0f);
		glm::convertTangentFrame(identity, true, normalIn, tangentIn, signIn, normalOut, tangentOut, signOut);

		Error += nearEqual(normalOut, normalIn) ? 0 : 1;
		Error += nearEqual(tangentOut, tangentIn) ? 0 : 1;
		// V reversal flips the bitangent sign independently of the geometry.
		Error += nearEqual(signOut, -signIn) ? 0 : 1;

		glm::vec3 const texelOut = glm::convertNormalMapTexel(texelIn, true);
		glm::vec3 const worldOut = perturbedNormal(normalOut, tangentOut, signOut, texelOut);
		Error += nearEqual(worldOut, worldIn) ? 0 : 1;
	}

	// 3) Geometry mirrored and texture V axis reversed: the two sign flips cancel.
	{
		glm::vec3 normalOut, tangentOut;
		float signOut(0.0f);
		glm::convertTangentFrame(mirrorZ, true, normalIn, tangentIn, signIn, normalOut, tangentOut, signOut);

		Error += nearEqual(signOut, signIn) ? 0 : 1;

		glm::vec3 const texelOut = glm::convertNormalMapTexel(texelIn, true);
		glm::vec3 const worldOut = perturbedNormal(normalOut, tangentOut, signOut, texelOut);
		Error += nearEqual(worldOut, glm::normalize(mirrorZ * worldIn)) ? 0 : 1;
	}

	return Error;
}

int test_seamFramesPreserved()
{
	int Error(0);

	// Two triangles sharing the edge p1-p2 in world space (a folded quad),
	// textured with independent UV islands: a UV seam runs along the edge.
	glm::vec3 const p1(0.0f, 0.0f, 0.0f);
	glm::vec3 const p2(1.0f, 0.0f, 0.0f);
	glm::vec3 const p3a(0.0f, 1.0f, 0.0f);
	glm::vec3 const p3b(0.0f, 0.0f, 1.0f);

	// Side A: V axis maps to +Y.
	glm::vec3 const tangentA = glm::triangleTangent(p1, p2, p3a,
		glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 0.0f), glm::vec2(0.0f, 1.0f));
	glm::vec3 const bitangentA = glm::triangleBitangent(p1, p2, p3a,
		glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 0.0f), glm::vec2(0.0f, 1.0f));

	// Side B: V axis maps to -Z (its own island, V pointing the other way).
	glm::vec3 const tangentB = glm::triangleTangent(p1, p2, p3b,
		glm::vec2(0.0f, 1.0f), glm::vec2(1.0f, 1.0f), glm::vec2(0.0f, 0.0f));
	glm::vec3 const bitangentB = glm::triangleBitangent(p1, p2, p3b,
		glm::vec2(0.0f, 1.0f), glm::vec2(1.0f, 1.0f), glm::vec2(0.0f, 0.0f));

	// Each side of the seam keeps its own texture basis.
	Error += nearEqual(tangentA, glm::vec3(1.0f, 0.0f, 0.0f)) ? 0 : 1;
	Error += nearEqual(bitangentA, glm::vec3(0.0f, 1.0f, 0.0f)) ? 0 : 1;
	Error += nearEqual(tangentB, glm::vec3(1.0f, 0.0f, 0.0f)) ? 0 : 1;
	Error += nearEqual(bitangentB, glm::vec3(0.0f, 0.0f, -1.0f)) ? 0 : 1;

	// The handedness flags of the two islands differ and are both preserved.
	glm::vec3 const normalA = glm::triangleNormal(p1, p2, p3a);
	glm::vec3 const normalB = glm::triangleNormal(p1, p2, p3b);
	float const signA = glm::bitangentSign(normalA, tangentA, bitangentA);
	float const signB = glm::bitangentSign(normalB, tangentB, bitangentB);
	Error += nearEqual(glm::abs(signA), 1.0f) ? 0 : 1;
	Error += nearEqual(glm::abs(signB), 1.0f) ? 0 : 1;
	Error += signA != signB ? 0 : 1;

	return Error;
}

int main()
{
	int Error(0);

	Error += test_triangleTangentFrame();
	Error += test_bitangentSign();
	Error += test_convertTangentFrame();
	Error += test_seamFramesPreserved();

	return Error;
}
