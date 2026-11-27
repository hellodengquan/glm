#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/quaternion.hpp>

static int test_identity() {
	int Error = 0;

	glm::mat4 Matrix(1);

	glm::vec3 Scale;
	glm::quat Orientation;
	glm::vec3 Translation;
	glm::vec3 Skew(1);
	glm::vec4 Perspective(1);

	glm::decompose(Matrix, Scale, Orientation, Translation, Skew, Perspective);

	glm::mat4 Out = glm::recompose(Scale, Orientation, Translation, Skew, Perspective);

	Error += glm::all(glm::equal(Matrix, Out, glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

static int test_scale_translate() {
	int Error = 0;

	glm::vec3 const T(2.0f);
	glm::vec3 const S(2.0f);

	glm::mat4 Matrix = glm::translate(glm::scale(glm::mat4(1), S), T);

	glm::vec3 Scale(2);
	glm::quat Orientation;
	glm::vec3 Translation(2);
	glm::vec3 Skew(1);
	glm::vec4 Perspective(1);

	glm::decompose(Matrix, Scale, Orientation, Translation, Skew, Perspective);

	glm::mat4 Out = glm::recompose(Scale, Orientation, Translation, Skew, Perspective);

	Error += glm::all(glm::equal(Matrix, Out, glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

static glm::mat4 makeShear(glm::vec3 const& Shear)
{
	glm::mat4 Lz(1);
	Lz[1][0] = Shear.z;

	glm::mat4 Ly(1);
	Ly[2][0] = Shear.y;

	glm::mat4 Lx(1);
	Lx[2][1] = Shear.x;

	return Lz * Ly * Lx;
}

// Decompose and recompose go through a shear factorization, so the round trip
// of a composed transform is accurate to a small relative tolerance rather than
// to machine epsilon.
static bool nearEqual(glm::mat4 const& A, glm::mat4 const& B)
{
	return glm::all(glm::equal(A, B, 0.0001f));
}

static int test_mirrored_reference_continuity()
{
	int Error = 0;

	// A mirrored transform with an offset and a shear: both reference frames
	// below must produce the same geometry while matching their own signs.
	glm::mat4 const Matrix =
		glm::translate(glm::mat4(1), glm::vec3(1, -2, 3)) *
		glm::mat4_cast(glm::angleAxis(glm::radians(25.0f), glm::normalize(glm::vec3(1, 2, 3)))) *
		makeShear(glm::vec3(0.2f, 0.3f, 0.1f)) *
		glm::scale(glm::mat4(1), glm::vec3(-2, 3, 4));

	glm::vec3 const References[2] =
	{
		glm::vec3(1, 1, -1),
		glm::vec3(-1, -1, -1)
	};

	glm::quat const ReferenceOrientation = glm::angleAxis(glm::radians(20.0f), glm::normalize(glm::vec3(1, 2, 3)));

	glm::mat4 Recomposed[2];

	for(int i = 0; i < 2; ++i)
	{
		glm::vec3 Scale;
		glm::quat Orientation;
		glm::vec3 Translation;
		glm::vec3 Skew;
		glm::vec4 Perspective;

		bool const Decomposed = glm::decompose(Matrix, Scale, Orientation, Translation, Skew, Perspective, References[i], ReferenceOrientation);
		Error += Decomposed ? 0 : 1;

		// Each axis keeps the sign of its own reference.
		Error += (Scale.x < 0.0f) == (References[i].x < 0.0f) ? 0 : 1;
		Error += (Scale.y < 0.0f) == (References[i].y < 0.0f) ? 0 : 1;
		Error += (Scale.z < 0.0f) == (References[i].z < 0.0f) ? 0 : 1;

		// The rotation stays in the reference hemisphere.
		Error += glm::dot(Orientation, ReferenceOrientation) >= 0.0f ? 0 : 1;

		Recomposed[i] = glm::recompose(Scale, Orientation, Translation, Skew, Perspective);
		Error += nearEqual(Matrix, Recomposed[i]) ? 0 : 1;
	}

	// Both reference frames must reconstruct the original geometry.
	Error += nearEqual(Recomposed[0], Recomposed[1]) ? 0 : 1;

	return Error;
}

static int test_rotation_hemisphere_across_trace_branch()
{
	int Error = 0;

	// Rotating through 120 degrees around an axis with mixed signs pushes the
	// rotation matrix through the trace branch boundary of the quaternion
	// extraction, where the raw result flips hemisphere. A non-mirrored matrix
	// isolates that branch from the scale sign handling.
	glm::vec3 const Axis = glm::normalize(glm::vec3(1, 2, -3));
	glm::quat ReferenceOrientation(1, 0, 0, 0);

	for(float Degrees = 100.0f; Degrees <= 140.0f; Degrees += 5.0f)
	{
		glm::mat4 const Matrix = glm::mat4_cast(glm::angleAxis(glm::radians(Degrees), Axis));

		glm::vec3 Scale;
		glm::quat Orientation;
		glm::vec3 Translation;
		glm::vec3 Skew;
		glm::vec4 Perspective;

		bool const Decomposed = glm::decompose(Matrix, Scale, Orientation, Translation, Skew, Perspective, glm::vec3(1, 1, 1), ReferenceOrientation);
		Error += Decomposed ? 0 : 1;

		// Same hemisphere as the previous frame: no representation jump.
		Error += glm::dot(Orientation, ReferenceOrientation) >= 0.0f ? 0 : 1;

		glm::mat4 const Recomposed = glm::recompose(Scale, Orientation, Translation, Skew, Perspective);
		Error += nearEqual(Matrix, Recomposed) ? 0 : 1;

		ReferenceOrientation = Orientation;
	}

	return Error;
}

static int test_mirrored_rotation_hemisphere_across_trace_branch()
{
	int Error = 0;

	// Same trace branch crossing, now on a mirrored matrix with an offset and a
	// shear: the scale signs and the shear factors must be adjusted together to
	// keep the geometry while the rotation stays in the reference hemisphere.
	glm::vec3 const Axis = glm::normalize(glm::vec3(1, 2, -3));
	glm::vec3 const ReferenceScale(1, 1, -1);
	glm::quat ReferenceOrientation(1, 0, 0, 0);

	for(float Degrees = 100.0f; Degrees <= 140.0f; Degrees += 5.0f)
	{
		glm::mat4 const Matrix =
			glm::translate(glm::mat4(1), glm::vec3(1, 2, 3)) *
			glm::mat4_cast(glm::angleAxis(glm::radians(Degrees), Axis)) *
			makeShear(glm::vec3(0.2f, -0.3f, 0.15f)) *
			glm::scale(glm::mat4(1), glm::vec3(-2, 3, 4));

		glm::vec3 Scale;
		glm::quat Orientation;
		glm::vec3 Translation;
		glm::vec3 Skew;
		glm::vec4 Perspective;

		bool const Decomposed = glm::decompose(Matrix, Scale, Orientation, Translation, Skew, Perspective, ReferenceScale, ReferenceOrientation);
		Error += Decomposed ? 0 : 1;

		Error += (Scale.x < 0.0f) == (ReferenceScale.x < 0.0f) ? 0 : 1;
		Error += (Scale.y < 0.0f) == (ReferenceScale.y < 0.0f) ? 0 : 1;
		Error += (Scale.z < 0.0f) == (ReferenceScale.z < 0.0f) ? 0 : 1;

		Error += glm::dot(Orientation, ReferenceOrientation) >= 0.0f ? 0 : 1;

		glm::mat4 const Recomposed = glm::recompose(Scale, Orientation, Translation, Skew, Perspective);
		Error += nearEqual(Matrix, Recomposed) ? 0 : 1;

		ReferenceOrientation = Orientation;
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_identity();
	Error += test_scale_translate();
	Error += test_mirrored_reference_continuity();
	Error += test_rotation_hemisphere_across_trace_branch();
	Error += test_mirrored_rotation_hemisphere_across_trace_branch();

	return Error;
}
