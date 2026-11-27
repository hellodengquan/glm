// Vertex baking data exchange unit tests.
//
// A vertex baker reads tightly packed three-component positions out of an
// interleaved vertex stream, feeds them to a geometry transform routine that
// relies on vector (SIMD) instructions, and writes the result back into the
// stream. Two storage representations are involved:
//
//   * packed  - the raw interleaved float stream, three floats per position,
//               whose promised layout is 12 bytes per object (no padding);
//   * aligned - the SIMD friendly storage, whose promised layout is a 16 byte
//               aligned 16 byte object carrying a fourth padding slot (a full
//               vector register) even though only three components are math.
//
// The tests distinguish three quantities that are easy to conflate:
//   1. the mathematical component count (always 3);
//   2. the object stride (sizeof the representation, 12 vs 16 bytes);
//   3. the actual distance between two consecutive vertices in an array
//      (the element stride, which includes any per-element padding).
//
// For every representation the same triangle transformed by the same matrix
// must yield identical coordinates, while the bytes belonging to the next
// vertex and the other attributes of the same vertex must be preserved
// bit for bit.

#ifndef GLM_FORCE_INTRINSICS
#	define GLM_FORCE_INTRINSICS
#endif//GLM_FORCE_INTRINSICS

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic push
#	pragma clang diagnostic ignored "-Wpadded"
#	pragma clang diagnostic ignored "-Wcovered-switch-default"
#	pragma clang diagnostic ignored "-Wexit-time-destructors"
#	pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

namespace
{
	// Explicit qualifier types make the two promised layouts independent of
	// the platform default (on NEON based targets the default vec3 stays
	// packed, while the explicitly aligned type always reserves a register).
	using PackedVec3 = glm::vec<3, float, glm::packed_highp>;
#	if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
	using AlignedVec3 = glm::vec<3, float, glm::aligned_highp>;
#	endif

	// A position followed by other attributes, as found in an interleaved
	// vertex buffer. The position is the only member the baker rewrites.
	struct PackedVertex
	{
		PackedVec3 Position;
		float Color[4];
	};

#	if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
	// SIMD friendly representation: an aligned 16 byte vec3 (three math
	// components plus one padding slot) followed by the same color attributes.
	struct AlignedVertex
	{
		AlignedVec3 Position;
		float Color[4];
	};

	struct AlignedStream
	{
		AlignedVertex Vertices[3];
	};
#	endif

	glm::length_t const ComponentCount = glm::vec3::length(); // always 3
	std::size_t const VertexCount = 3;                        // one triangle
	std::size_t const FloatPerVertex = 7;                    // xyz + rgba
	std::size_t const StreamFloats = VertexCount * FloatPerVertex;

	// Distinct sentinel values make any stray byte overwrite observable.
	float const CanaryColor[4] = {70.0f, 71.0f, 72.0f, 73.0f};
#	if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
	float const PaddingSentinel = 99.0f;
#	endif

	// Geometry transform routine exercised by the baker. Positions are points
	// (w = 1) so that the translation part of the matrix is applied.
	glm::vec3 transformPoint(glm::mat4 const & m, glm::vec3 const & p)
	{
		return glm::vec3(m * glm::vec4(p, 1.0f));
	}

	// Deterministic affine matrix shared by both representations. It is an
	// exact integer coefficient replica of a 90 degree rotation around Z
	// (mapping (x, y) to (-y, x)) followed by a translation. Avoiding
	// trigonometry keeps the transformed coordinates exact, so the test can
	// compare against glm::epsilon without depending on sin/cos rounding.
	glm::mat4 makeBakeMatrix()
	{
		glm::mat4 Rotation(1.0f);
		Rotation[0] = glm::vec4( 0.0f,  1.0f, 0.0f, 0.0f);
		Rotation[1] = glm::vec4(-1.0f,  0.0f, 0.0f, 0.0f);
		Rotation[2] = glm::vec4( 0.0f,  0.0f, 1.0f, 0.0f);
		Rotation[3] = glm::vec4( 0.0f,  0.0f, 0.0f, 1.0f);

		glm::mat4 Translation(1.0f);
		Translation[3] = glm::vec4(10.0f, 20.0f, 30.0f, 1.0f);

		return Translation * Rotation;
	}

	bool bitEqual(float a, float b)
	{
		// Bitwise comparison: the contract is byte preservation, not merely
		// numeric equivalence.
		std::uint32_t ba = 0;
		std::uint32_t bb = 0;
		std::memcpy(&ba, &a, sizeof(float));
		std::memcpy(&bb, &b, sizeof(float));
		return ba == bb;
	}

	bool identical(float const * a, float const * b, std::size_t count)
	{
		for(std::size_t i = 0; i < count; ++i)
		{
			if(!bitEqual(a[i], b[i]))
				return false;
		}
		return true;
	}

	std::ptrdiff_t byteDistance(void const * a, void const * b)
	{
		return static_cast<char const *>(b) - static_cast<char const *>(a);
	}

	// Layout promises of each representation: math component count, object
	// stride and actual in-array vertex spacing are asserted separately.
	int test_layout()
	{
		int Error = 0;

		// Mathematical component count: a position has three components no
		// matter how much storage backs it.
		Error += ComponentCount == 3 ? 0 : 1;
		Error += PackedVec3::length() == 3 ? 0 : 1;

		// Packed representation: three tightly packed floats, no padding.
		Error += sizeof(PackedVec3) == 3 * sizeof(float) ? 0 : 1;
		Error += alignof(PackedVec3) == alignof(float) ? 0 : 1;
		Error += PackedVec3::is_aligned::value == false ? 0 : 1;

		// In the interleaved stream the position occupies exactly three floats
		// and the next float already belongs to another attribute.
		Error += sizeof(PackedVertex) == 7 * sizeof(float) ? 0 : 1;
		Error += alignof(PackedVertex) == alignof(float) ? 0 : 1;

		// Actual distance between consecutive vertices of the packed stream.
		PackedVertex PackedArray[2] = {};
		std::ptrdiff_t const PackedVertexSpacing = byteDistance(&PackedArray[0], &PackedArray[1]);
		Error += PackedVertexSpacing == static_cast<std::ptrdiff_t>(sizeof(PackedVertex)) ? 0 : 1;
		Error += PackedVertexSpacing == static_cast<std::ptrdiff_t>(7 * sizeof(float)) ? 0 : 1;

#		if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
		Error += AlignedVec3::length() == 3 ? 0 : 1;

		// Aligned representation: the object stride grows to a full vector
		// register even though the mathematical component count stays at 3.
		Error += sizeof(AlignedVec3) == 4 * sizeof(float) ? 0 : 1;
		Error += alignof(AlignedVec3) == 4 * sizeof(float) ? 0 : 1;
		Error += AlignedVec3::is_aligned::value == true ? 0 : 1;

		// Array of aligned vectors: the padding slot is part of the per-vertex
		// spacing, hence vertices sit 16 bytes apart, not 12.
		AlignedVec3 AlignedVecArray[2] = {};
		std::ptrdiff_t const AlignedVecSpacing = byteDistance(&AlignedVecArray[0], &AlignedVecArray[1]);
		Error += AlignedVecSpacing == static_cast<std::ptrdiff_t>(4 * sizeof(float)) ? 0 : 1;

		// 16 byte position plus 16 bytes of color, with the overall object
		// aligned to the register boundary.
		Error += sizeof(AlignedVertex) == 8 * sizeof(float) ? 0 : 1;
		Error += alignof(AlignedVertex) == 4 * sizeof(float) ? 0 : 1;

		AlignedVertex AlignedVertexArray[2] = {};
		std::ptrdiff_t const AlignedVertexSpacing = byteDistance(&AlignedVertexArray[0], &AlignedVertexArray[1]);
		Error += AlignedVertexSpacing == static_cast<std::ptrdiff_t>(sizeof(AlignedVertex)) ? 0 : 1;
		Error += AlignedVertexSpacing == static_cast<std::ptrdiff_t>(8 * sizeof(float)) ? 0 : 1;

		Error += sizeof(AlignedStream) == VertexCount * sizeof(AlignedVertex) ? 0 : 1;
#		endif

		return Error;
	}

	// Bake through the packed interleaved stream. The three position floats
	// are read into a vec3, transformed with vector instructions and written
	// back; every surrounding byte must survive.
	int test_bake_packed()
	{
		int Error = 0;

		float Stream[StreamFloats];
		float Reference[StreamFloats];

		float const SourcePositions[VertexCount][3] =
		{
			{1.0f, 0.0f, 2.0f},
			{0.0f, 1.0f, 4.0f},
			{-1.0f, -1.0f, 8.0f}
		};

		for(std::size_t v = 0; v < VertexCount; ++v)
		{
			std::size_t const Base = v * FloatPerVertex;
			Stream[Base + 0] = SourcePositions[v][0];
			Stream[Base + 1] = SourcePositions[v][1];
			Stream[Base + 2] = SourcePositions[v][2];
			Stream[Base + 3] = CanaryColor[0];
			Stream[Base + 4] = CanaryColor[1];
			Stream[Base + 5] = CanaryColor[2];
			Stream[Base + 6] = CanaryColor[3];
		}
		std::memcpy(Reference, Stream, sizeof(Stream));

		// Same matrix used for every representation.
		glm::mat4 const Matrix = makeBakeMatrix();

		PackedVertex * Vertices = reinterpret_cast<PackedVertex *>(Stream);

		for(std::size_t v = 0; v < VertexCount; ++v)
		{
			// Read the tightly packed three-component position.
			glm::vec3 const In(Vertices[v].Position[0], Vertices[v].Position[1], Vertices[v].Position[2]);
			glm::vec3 const Out = transformPoint(Matrix, In);
			// Write the transformed position back, touching three floats only.
			Vertices[v].Position[0] = Out.x;
			Vertices[v].Position[1] = Out.y;
			Vertices[v].Position[2] = Out.z;
		}

		// Expected coordinates: the column-major 90 degree rotation around Z
		// maps (x, y) to (-y, x), then the translation is added. z only
		// translates, and all values are exact integers.
		glm::vec3 const Expected[VertexCount] =
		{
			glm::vec3(10.0f, 21.0f, 32.0f),
			glm::vec3(9.0f, 20.0f, 34.0f),
			glm::vec3(11.0f, 19.0f, 38.0f)
		};

		for(std::size_t v = 0; v < VertexCount; ++v)
		{
			glm::vec3 const Baked(Vertices[v].Position[0], Vertices[v].Position[1], Vertices[v].Position[2]);
			Error += glm::all(glm::equal(Baked, Expected[v], glm::epsilon<float>())) ? 0 : 1;

			// The other attributes of this vertex are untouched.
			Error += identical(Vertices[v].Color, CanaryColor, 4) ? 0 : 1;
		}

		// Bytes outside the rewritten position floats (all colors and the
		// remaining vertices) must be preserved bit for bit.
		for(std::size_t v = 0; v < VertexCount; ++v)
		{
			std::size_t const Base = v * FloatPerVertex;
			Error += identical(&Stream[Base + 3], &Reference[Base + 3], 4) ? 0 : 1;
		}

		// The very last position is followed by no padding: the next float is
		// live data and must still hold its canary value.
		Error += bitEqual(Stream[(VertexCount - 1) * FloatPerVertex + 3], CanaryColor[0]) ? 0 : 1;

		return Error;
	}

#	if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
	// Bake through the aligned representation. The vec3 position occupies a
	// full vector register; its fourth slot, the following color attributes
	// and the neighboring vertices must all survive the round trip.
	int test_bake_aligned()
	{
		int Error = 0;

		AlignedStream * Stream = new AlignedStream();
		// The storage honors the 16 byte register alignment of the element.
		Error += (reinterpret_cast<std::uintptr_t>(Stream) & 0xFu) == 0u ? 0 : 1;

		float const SourcePositions[VertexCount][3] =
		{
			{1.0f, 0.0f, 2.0f},
			{0.0f, 1.0f, 4.0f},
			{-1.0f, -1.0f, 8.0f}
		};

		for(std::size_t v = 0; v < VertexCount; ++v)
		{
			Stream->Vertices[v].Position = AlignedVec3(SourcePositions[v][0], SourcePositions[v][1], SourcePositions[v][2]);
			// The fourth storage slot of the register-sized vec3 is not a math
			// component: seed it so an out-of-bounds SIMD write becomes visible.
			glm::value_ptr(Stream->Vertices[v].Position)[3] = PaddingSentinel;
			Stream->Vertices[v].Color[0] = CanaryColor[0];
			Stream->Vertices[v].Color[1] = CanaryColor[1];
			Stream->Vertices[v].Color[2] = CanaryColor[2];
			Stream->Vertices[v].Color[3] = CanaryColor[3];
		}

		glm::mat4 const Matrix = makeBakeMatrix();

		for(std::size_t v = 0; v < VertexCount; ++v)
		{
			// value_ptr hands the SIMD transform routine the address of the
			// aligned register backing the vec3.
			glm::vec3 const In = glm::vec3(Stream->Vertices[v].Position);
			glm::vec3 const Out = transformPoint(Matrix, In);
			float * Dst = glm::value_ptr(Stream->Vertices[v].Position);
			// Only the three math components are rewritten; the padding slot
			// must keep its sentinel.
			Dst[0] = Out.x;
			Dst[1] = Out.y;
			Dst[2] = Out.z;
		}

		glm::vec3 const Expected[VertexCount] =
		{
			glm::vec3(10.0f, 21.0f, 32.0f),
			glm::vec3(9.0f, 20.0f, 34.0f),
			glm::vec3(11.0f, 19.0f, 38.0f)
		};

		for(std::size_t v = 0; v < VertexCount; ++v)
		{
			AlignedVertex & Vertex = Stream->Vertices[v];

			// Same triangle, same matrix: the aligned coordinates must match
			// the reference values.
			Error += glm::all(glm::equal(glm::vec3(Vertex.Position), Expected[v], glm::epsilon<float>())) ? 0 : 1;

			// The padding slot of the register-sized vec3 is preserved.
			Error += bitEqual(glm::value_ptr(Vertex.Position)[3], PaddingSentinel) ? 0 : 1;

			// The other attributes of this vertex are untouched.
			Error += identical(Vertex.Color, CanaryColor, 4) ? 0 : 1;
		}

		// Cross representation agreement: both storage layouts promise the same
		// transformed coordinates for the same input triangle.
		{
			float PackedRaw[StreamFloats];
			for(std::size_t v = 0; v < VertexCount; ++v)
			{
				std::size_t const Base = v * FloatPerVertex;
				PackedRaw[Base + 0] = SourcePositions[v][0];
				PackedRaw[Base + 1] = SourcePositions[v][1];
				PackedRaw[Base + 2] = SourcePositions[v][2];
				PackedRaw[Base + 3] = CanaryColor[0];
				PackedRaw[Base + 4] = CanaryColor[1];
				PackedRaw[Base + 5] = CanaryColor[2];
				PackedRaw[Base + 6] = CanaryColor[3];
			}

			PackedVertex * PackedVertices = reinterpret_cast<PackedVertex *>(PackedRaw);
			for(std::size_t v = 0; v < VertexCount; ++v)
			{
				glm::vec3 const In(PackedVertices[v].Position[0], PackedVertices[v].Position[1], PackedVertices[v].Position[2]);
				glm::vec3 const Out = transformPoint(Matrix, In);
				PackedVertices[v].Position[0] = Out.x;
				PackedVertices[v].Position[1] = Out.y;
				PackedVertices[v].Position[2] = Out.z;
			}

			for(std::size_t v = 0; v < VertexCount; ++v)
			{
				glm::vec3 const PackedResult(
					PackedVertices[v].Position[0],
					PackedVertices[v].Position[1],
					PackedVertices[v].Position[2]);
				Error += glm::all(glm::equal(PackedResult, glm::vec3(Stream->Vertices[v].Position), glm::epsilon<float>())) ? 0 : 1;
			}
		}

		delete Stream;
		return Error;
	}
#	endif//GLM_CONFIG_ALIGNED_GENTYPES
}//namespace

#if GLM_COMPILER & GLM_COMPILER_CLANG
#	pragma clang diagnostic pop
#endif

int main()
{
	int Error = 0;

	Error += test_layout();
	Error += test_bake_packed();

#	if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
		Error += test_bake_aligned();
#	endif

	return Error;
}
