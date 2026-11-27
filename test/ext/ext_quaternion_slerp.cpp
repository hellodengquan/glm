// Unit tests for the two spherical linear interpolation overloads of quaternions:
//
//   * slerp(x, y, a)    -- short path spherical linear interpolation
//   * slerp(x, y, a, k) -- spherical linear interpolation with k additional spins
//
// The tests model an animator recording the start/end poses of a turntable that
// rotates around a fixed axis (the local Z axis). Every reference orientation is
// built independently with angleAxis(), from the public contracts documented in
// glm/ext/quaternion_common.hpp:
//
//   * slerp(x, y, a) "always take the short path": when dot(x, y) < 0 the second
//     key is negated internally, so the interpolation covers the shortest great
//     circle arc and is driven solely by the spatial orientation of the keys.
//     For a relative short arc alpha (in [0, pi]) the accumulated physical
//     rotation at parameter a is a * alpha, hence the mid orientation is
//     angleAxis(theta0 + a * alpha, axis).
//
//   * slerp(x, y, a, k) uses phi = alpha + k * pi in the quaternion half-angle
//     parameterization, which corresponds to an accumulated physical rotation of
//     2 * a * phi = a * (alpha + 2 * pi * k). A positive k adds |k| full turns
//     along the short path direction, a negative k drives the animation the other
//     way. At a = 1 the extra 2 * pi * k whole turns bring the turntable back to
//     the very same end orientation, so animations stopping at the same pose are
//     told apart through the motion itself (direction and number of turns).
//
// Because q and -q encode the same orientation, recording a rotation key with
// either equivalent sign must not change the resulting spatial motion.

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/scalar_constants.hpp>

namespace
{
	// Orientation of the turntable is read back spatially: a radial marker lying
	// on the X axis is rotated by the interpolated quaternion. Its signed Z
	// rotation angle is atan2(marker.y, marker.x), in (-pi, pi].
	glm::vec3 const AXIS(0.0f, 0.0f, 1.0f);
	glm::vec3 const MARKER(1.0f, 0.0f, 0.0f);
	float const EPS = 0.001f;

	float markerAngle(glm::quat const& q)
	{
		glm::vec3 const r = q * MARKER;
		return glm::atan(r.y, r.x);
	}

	// Unwrap a sequence of signed angles sampled in (-pi, pi] into a continuous
	// accumulated angle, preserving the number and the direction of the turns.
	float unwrap(float previous, float current)
	{
		float delta = current - previous;
		while(delta > glm::pi<float>())
			delta -= glm::two_pi<float>();
		while(delta < -glm::pi<float>())
			delta += glm::two_pi<float>();
		return previous + delta;
	}

	// Wrap an angle into the (-pi, pi] range returned by markerAngle, since the
	// orientation of a rigid body is defined modulo 2*pi.
	float wrapAngle(float angle)
	{
		while(angle > glm::pi<float>())
			angle -= glm::two_pi<float>();
		while(angle <= -glm::pi<float>())
			angle += glm::two_pi<float>();
		return angle;
	}

	int expectAngle(float actual, float expected, float epsilon)
	{
		return glm::equal(actual, expected, epsilon) ? 0 : 1;
	}

	int expectQuat(glm::quat const& actual, glm::quat const& expected, float epsilon)
	{
		return glm::all(glm::equal(
			glm::vec4(actual.w, actual.x, actual.y, actual.z),
			glm::vec4(expected.w, expected.x, expected.y, expected.z),
			glm::vec4(epsilon))) ? 0 : 1;
	}

	// Two quaternions describe the same spatial orientation when they rotate the
	// marker onto the same direction (q == -q as rotations). This is the judgment
	// that must follow spatial motion rather than recorded quaternion signs.
	int expectOrientation(glm::quat const& actual, float expectedAngle, float epsilon)
	{
		return expectAngle(markerAngle(actual), wrapAngle(expectedAngle), epsilon);
	}
} // namespace

// Short path slerp: the turntable goes from 0 to 90 degrees, halfway it must
// face 45 degrees, and endpoints must be exact.
static int test_slerp_short_path()
{
	int Error = 0;

	float const alpha = glm::radians(90.0f);
	glm::quat const x = glm::angleAxis(0.0f, AXIS);
	glm::quat const y = glm::angleAxis(alpha, AXIS);

	// Endpoints
	Error += expectQuat(glm::slerp(x, y, 0.0f), x, EPS);
	{
		glm::quat const q = glm::slerp(x, y, 1.0f);
		Error += expectQuat(q, y, EPS);
		Error += expectAngle(markerAngle(q), alpha, EPS);
	}

	// Mid orientation: half of the 90 degree short arc, 45 degrees.
	{
		glm::quat const q = glm::slerp(x, y, 0.5f);
		Error += expectOrientation(q, glm::radians(45.0f), EPS);
	}

	// Quarter point: 22.5 degrees.
	{
		glm::quat const q = glm::slerp(x, y, 0.25f);
		Error += expectOrientation(q, glm::radians(22.5f), EPS);
	}

	return Error;
}

// A 270 degree key is the same spatial orientation as a -90 degree key. Short
// path slerp must rotate through the 90 degree arc, never the 270 degree one,
// regardless of which equivalent sign was used to record the key.
static int test_slerp_short_path_equivalent_key_sign()
{
	int Error = 0;

	glm::quat const x = glm::angleAxis(0.0f, AXIS);
	glm::quat const yLong = glm::angleAxis(glm::radians(270.0f), AXIS); // +270 deg
	glm::quat const yShort = -yLong;                                    // == -90 deg

	// Both keys describe the same end orientation.
	Error += expectQuat(yShort, glm::angleAxis(glm::radians(-90.0f), AXIS), EPS);

	// At the midpoint the turntable must be at -45 degrees (short way),
	// not at +135 degrees (the long way around). Judgment is spatial.
	Error += expectOrientation(glm::slerp(x, yLong, 0.5f), glm::radians(-45.0f), EPS);
	Error += expectOrientation(glm::slerp(x, yShort, 0.5f), glm::radians(-45.0f), EPS);

	// The two calls are not merely equivalent in orientation: the short path
	// contract makes them produce the very same quaternion at every parameter.
	for(float a = 0.0f; a <= 1.0f; a += 0.125f)
		Error += expectQuat(glm::slerp(x, yLong, a), glm::slerp(x, yShort, a), EPS);

	// End orientation is identical for both recorded keys.
	Error += expectQuat(glm::slerp(x, yLong, 1.0f), glm::slerp(x, yShort, 1.0f), EPS);
	Error += expectAngle(markerAngle(glm::slerp(x, yLong, 1.0f)), glm::radians(-90.0f), EPS);

	// Contrast with mix(), which is oriented without the short path correction:
	// fed the +270 degree key it keeps the long way, so its midpoint faces
	// +135 degrees. This pins down the "short path" guarantee of slerp.
	{
		glm::quat const q = glm::mix(x, yLong, 0.5f);
		Error += expectOrientation(q, glm::radians(135.0f), EPS);
	}

	return Error;
}

// Short path slerp between two non-identity recorded poses: the mid orientation
// is the average of the two poses along the short arc.
static int test_slerp_short_path_between_poses()
{
	int Error = 0;

	glm::quat const x = glm::angleAxis(glm::radians(30.0f), AXIS);
	glm::quat const y = glm::angleAxis(glm::radians(120.0f), AXIS);

	// Relative short arc: 90 degrees. Halfway: 30 + 45 = 75 degrees.
	glm::quat const mid = glm::slerp(x, y, 0.5f);
	Error += expectOrientation(mid, glm::radians(75.0f), EPS);

	// Recording the end key with the opposite sign must give identical motion.
	Error += expectQuat(glm::slerp(x, -y, 0.5f), mid, EPS);
	Error += expectQuat(glm::slerp(x, -y, 1.0f), glm::slerp(x, y, 1.0f), EPS);

	return Error;
}

// Spin slerp with k == 0 must reduce to the short path interpolation.
static int test_slerp_spin_zero_matches_short_path()
{
	int Error = 0;

	glm::quat const x = glm::angleAxis(glm::radians(20.0f), AXIS);
	glm::quat const y = glm::angleAxis(glm::radians(110.0f), AXIS);

	for(float a = 0.0f; a <= 1.0f; a += 0.125f)
		Error += expectQuat(glm::slerp(x, y, a, 0), glm::slerp(x, y, a), EPS);

	return Error;
}

// Spin slerp with one additional turn in each direction. Both animations stop at
// the same end pose, but at the midpoint the positive spin has accumulated
// +225 degrees of travel while the negative spin has accumulated -135 degrees:
// same stop, different number of turns and opposite winding direction.
static int test_slerp_spin_opposite_directions()
{
	int Error = 0;

	float const alpha = glm::radians(90.0f); // relative short arc
	glm::quat const x = glm::angleAxis(0.0f, AXIS);
	glm::quat const y = glm::angleAxis(alpha, AXIS);

	glm::quat const positive = glm::slerp(x, y, 0.5f, 1);
	glm::quat const negative = glm::slerp(x, y, 0.5f, -1);

	// Derived mid orientations from accumulated = a * (alpha + 2*pi*k).
	// k = +1: 0.5 * (90 + 360) = +225 degrees of accumulated rotation.
	Error += expectOrientation(positive, glm::radians(225.0f), EPS);
	// k = -1: 0.5 * (90 - 360) = -135 degrees of accumulated rotation.
	Error += expectOrientation(negative, glm::radians(-135.0f), EPS);

	// +225 and -135 are the same spatial orientation modulo 360...
	Error += expectAngle(markerAngle(positive), markerAngle(negative), EPS);
	// ...but the two animations wind in opposite directions, so the quaternions
	// themselves (and hence the motions) are not equal at the midpoint.
	Error += (expectQuat(positive, negative, EPS) == 1) ? 0 : 1;

	// At a = 1 both have returned to the recorded end orientation (the extra
	// whole turns cancel out spatially).
	Error += expectOrientation(glm::slerp(x, y, 1.0f, 1), alpha, EPS);
	Error += expectOrientation(glm::slerp(x, y, 1.0f, -1), alpha, EPS);

	// At a = 0.25 the marker already points to different spatial directions:
	// +112.5 degrees versus -67.5 degrees (not congruent modulo 360).
	Error += expectAngle(
		markerAngle(glm::slerp(x, y, 0.25f, 1)), glm::radians(112.5f), EPS);
	Error += expectAngle(
		markerAngle(glm::slerp(x, y, 0.25f, -1)), glm::radians(-67.5f), EPS);

	return Error;
}

// Recover the number of turns and the winding direction from the motion itself
// by unwrapping the marker angle along a fine sampling of each animation.
static int test_slerp_spin_accumulated_turns()
{
	int Error = 0;

	float const alpha = glm::radians(90.0f);
	glm::quat const x = glm::angleAxis(0.0f, AXIS);
	glm::quat const y = glm::angleAxis(alpha, AXIS);

	int const steps = 8;

	// k = +1: one extra positive turn, total accumulation alpha + 2*pi = 450 deg.
	{
		float accumulated = markerAngle(x);
		for(int i = 1; i <= steps; ++i)
		{
			float const a = static_cast<float>(i) / static_cast<float>(steps);
			accumulated = unwrap(accumulated, markerAngle(glm::slerp(x, y, a, 1)));

			// Expected accumulated rotation at parameter a.
			float const expected = a * (alpha + glm::two_pi<float>());
			Error += expectAngle(accumulated, expected, 0.01f);
		}
		Error += expectAngle(accumulated, glm::radians(450.0f), 0.01f);
	}

	// k = -1: one extra negative turn, total accumulation alpha - 2*pi = -270 deg.
	{
		float accumulated = markerAngle(x);
		for(int i = 1; i <= steps; ++i)
		{
			float const a = static_cast<float>(i) / static_cast<float>(steps);
			accumulated = unwrap(accumulated, markerAngle(glm::slerp(x, y, a, -1)));

			float const expected = a * (alpha - glm::two_pi<float>());
			Error += expectAngle(accumulated, expected, 0.01f);
		}
		Error += expectAngle(accumulated, glm::radians(-270.0f), 0.01f);
	}

	return Error;
}

// Same accumulated-turn check with a non-identity starting pose, proving the
// spin count is relative to the short arc between the two recorded poses.
static int test_slerp_spin_from_non_identity_pose()
{
	int Error = 0;

	float const theta0 = glm::radians(60.0f);
	float const alpha = glm::radians(90.0f);
	glm::quat const x = glm::angleAxis(theta0, AXIS);
	glm::quat const y = glm::angleAxis(theta0 + alpha, AXIS); // 150 degrees

	int const steps = 8;

	float accumulatedPos = markerAngle(x);
	float accumulatedNeg = markerAngle(x);
	for(int i = 1; i <= steps; ++i)
	{
		float const a = static_cast<float>(i) / static_cast<float>(steps);
		accumulatedPos = unwrap(accumulatedPos, markerAngle(glm::slerp(x, y, a, 1)));
		accumulatedNeg = unwrap(accumulatedNeg, markerAngle(glm::slerp(x, y, a, -1)));

		Error += expectAngle(accumulatedPos, theta0 + a * (alpha + glm::two_pi<float>()), 0.01f);
		Error += expectAngle(accumulatedNeg, theta0 + a * (alpha - glm::two_pi<float>()), 0.01f);
	}

	// Same stop pose (150 degrees), reached after +450 or -270 degrees of travel.
	Error += expectAngle(accumulatedPos, glm::radians(510.0f), 0.01f);
	Error += expectAngle(accumulatedNeg, glm::radians(-210.0f), 0.01f);
	Error += expectOrientation(glm::slerp(x, y, 1.0f, 1), theta0 + alpha, EPS);
	Error += expectOrientation(glm::slerp(x, y, 1.0f, -1), theta0 + alpha, EPS);

	return Error;
}

// Replacing an end key by its opposite sign must not alter the spatial motion,
// including the extra spins: the judgment always follows spatial orientation.
static int test_slerp_spin_equivalent_key_sign()
{
	int Error = 0;

	glm::quat const x = glm::angleAxis(0.0f, AXIS);
	glm::quat const y = glm::angleAxis(glm::radians(90.0f), AXIS);
	glm::quat const yEquivalent = -y;

	for(int k = -2; k <= 2; ++k)
	{
		for(float a = 0.0f; a <= 1.0f; a += 0.125f)
		{
			Error += expectQuat(
				glm::slerp(x, y, a, k),
				glm::slerp(x, yEquivalent, a, k),
				EPS);
		}
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_slerp_short_path();
	Error += test_slerp_short_path_equivalent_key_sign();
	Error += test_slerp_short_path_between_poses();
	Error += test_slerp_spin_zero_matches_short_path();
	Error += test_slerp_spin_opposite_directions();
	Error += test_slerp_spin_accumulated_turns();
	Error += test_slerp_spin_from_non_identity_pose();
	Error += test_slerp_spin_equivalent_key_sign();

	return Error;
}
