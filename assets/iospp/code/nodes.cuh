#include <helper_math.h> 

#define M_PI 3.1415926535
#define PI_2 (M_PI * 2.0)

struct AddIn {
  float value1;
  float value2;
};
struct AddOut {
  float value;
};
__device__ __forceinline__ AddOut AddImpl(AddIn input_) {
  return AddOut{input_.value1 + input_.value2};
}

struct Add2fIn {
  float2 value1;
  float2 value2;
};
struct Add2fOut {
  float2 value;
};
__device__ __forceinline__ Add2fOut Add2fImpl(Add2fIn input_) {
  return Add2fOut{float2{input_.value1.x + input_.value2.x, input_.value1.y + input_.value2.y}};
}

struct MulIn {
  float value1;
  float value2;
};
struct MulOut {
  float value;
};
__device__ __forceinline__ MulOut MulImpl(MulIn input_) {
  return MulOut{input_.value1 * input_.value2};
}

struct Mul2fIn {
  float2 value1;
  float2 value2;
};
struct Mul2fOut {
  float2 value;
};
__device__ __forceinline__ Mul2fOut Mul2fImpl(Mul2fIn input_) {
  return Mul2fOut{float2{input_.value1.x * input_.value2.x, input_.value1.y * input_.value2.y}};
}

struct DivIn {
  float value1;
  float value2;
};
struct DivOut {
  float value;
};
__device__ __forceinline__ DivOut DivImpl(DivIn input_) {
  return DivOut{input_.value1 / input_.value2};
}

struct Div2fIn {
  float2 value1;
  float2 value2;
};
struct Div2fOut {
  float2 value;
};
__device__ __forceinline__ Div2fOut Div2fImpl(Div2fIn input_) {
  return Div2fOut{float2{input_.value1.x / input_.value2.x, input_.value1.y / input_.value2.y}};
}

struct SubIn {
  float value1;
  float value2;
};
struct SubOut {
  float value;
};
__device__ __forceinline__ SubOut SubImpl(SubIn input_) {
  return SubOut{input_.value1 - input_.value2};
}

struct Sub2fIn {
  float2 value1;
  float2 value2;
};
struct Sub2fOut {
  float2 value;
};
__device__ __forceinline__ Sub2fOut Sub2fImpl(Sub2fIn input_) {
  return Sub2fOut{float2{input_.value1.x - input_.value2.x, input_.value1.y - input_.value2.y}};
}

struct NormalizeIn {
  float value1;
  float value2;
  float value3;
};
struct NormalizeOut {
  float value;
};
__device__ __forceinline__ NormalizeOut NormalizeImpl(NormalizeIn input_) {
  return NormalizeOut{input_.value1 * input_.value2 - input_.value3};
}

struct Normalize2fIn {
  float2 value1;
  float2 value2;
  float2 value3;
};
struct Normalize2fOut {
  float2 value;
};
__device__ __forceinline__ Normalize2fOut Normalize2fImpl(Normalize2fIn input_) {
  return Normalize2fOut{float2{input_.value1.x * input_.value2.x - input_.value3.x, input_.value1.y * input_.value2.y - input_.value3.y}};
}

struct NegIn {
  float x;
};
struct NegOut {
  float value;
};
__device__ __forceinline__ NegOut NegImpl(NegIn input_) {
  return NegOut{-input_.x};
}

struct ToVec2In {
  float x;
  float y;
};
struct ToVec2Out {
  float2 value;
};
__device__ __forceinline__ ToVec2Out ToVec2Impl(ToVec2In input_) {
  return ToVec2Out{float2{input_.x, input_.y}};
}

struct FromVec2In {
  float2 value;
};
struct FromVec2Out {
  float x;
  float y;
};
__device__ __forceinline__ FromVec2Out FromVec2Impl(FromVec2In input_) {
  return FromVec2Out{input_.value.x, input_.value.y};
}

struct RoundIn {
  float value;
};
struct RoundOut {
  float value;
};
__device__ __forceinline__ RoundOut RoundImpl(RoundIn input_) {
  return RoundOut{round(input_.value)};
}

struct CircleIn {
  float2 position;
  float radius;
  float inner;
};
struct CircleOut {
  float distance;
};
__device__ __forceinline__ CircleOut CircleImpl(CircleIn input_) {
  float2 p = input_.position;
  float inner = input_.inner > input_.radius * 0.1 ? 1.0 : input_.inner;
  float circle = sqrt(length(p)) - input_.radius;
  return CircleOut{abs(circle + input_.inner) - input_.inner};
}

struct BoxIn {
  float2 position;
  float radius;
};
struct BoxOut {
  float distance;
};
__device__ __forceinline__ BoxOut BoxImpl(BoxIn input_) {
  float2 p = input_.position;
  float radius = input_.radius;
  return BoxOut{abs(p.x) + abs(p.y) - radius};
}

struct JointIn {
  float2 position;
  float len;
  float curvature;
  float thickness;
};
struct JointOut {
  float distance;
};
__device__ __forceinline__ JointOut JointImpl(JointIn input_) {
  float2 p = float2{input_.position.y, input_.position.x};
  float len = input_.len;
  float curvature = input_.curvature;
  float thickness = input_.thickness;

  if (abs(curvature) < 0.001) {
    p.y -= clamp(p.y, 0.0 , len);
    return JointOut{length(p)};
  }

  float2 sc = make_float2(sin(curvature), cos(curvature));
  float ra = 0.5 * len / curvature;
  p.x -= ra;
  float2 q = p - 2.0 * sc * max(0.0, dot(sc, p));
  float d = (q.y < 0.0) ? length(q + make_float2(ra, 0.0)) : abs(abs(ra) - length(q));
  return JointOut{d - thickness};
}

struct StripeIn {
  float2 position;
  float thickness;
};
struct StripeOut {
  float distance;
};
__device__ __forceinline__ StripeOut StripeImpl(StripeIn input_) {
  float2 p = input_.position;
  float sdfStripeY = abs(p.y) - input_.thickness;
  return StripeOut{sdfStripeY};
}

struct TriangleWaveIn {
  float2 position;
  float frequence;
  float amplitude;
  float thickness;
};
struct TriangleWaveOut {
  float distance;
};
__device__ __forceinline__ float TriangleWaveMod(float a, float b) {
  int fl = static_cast<int>(floor(a / b));
  return a - static_cast<float>(fl) * b;
}
__device__ __forceinline__ TriangleWaveOut TriangleWaveImpl(TriangleWaveIn input_) {
  float2 p = input_.position;
  float frequence = input_.frequence;
  float amplitude = input_.amplitude;
  float thickness = input_.thickness;

  float pw = 2.0 / frequence;
  float qw = 0.25 * pw;
  float2 sc = make_float2(4.0 * amplitude, pw);
  float l = length(sc);

  float posX = abs(TriangleWaveMod(p.x + qw, pw) - 0.5 * pw) - qw;
  float posX1 = (posX * sc.x + p.y * sc.y) / l;

  float posY = (-posX * sc.y + p.y * sc.x) / l;

  float2 pos = make_float2(posX1, max(0.0, abs(posY) - 0.25 * l));
  return TriangleWaveOut{length(pos) - thickness};
}

struct SineWaveIn {
  float2 position;
  float frequence;
  float amplitude;
  float thickness;
};
struct SineWaveOut {
  float distance;
};
__device__ __forceinline__ float SineWaveMod(float a, float b) {
  int fl = int(floor(a / b));
  return a - float(fl) * b;
}
__device__ __forceinline__ float Sign(float a) {
  return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0);
}
__device__ __forceinline__ SineWaveOut SineWaveImpl(SineWaveIn input_) {
  float2 p = input_.position;
  float frequence = input_.frequence;
  float amplitude = input_.amplitude;
  float thickness = input_.thickness;

  float f = frequence * M_PI * amplitude;
  float px = p.x / amplitude;
  float py = p.y / amplitude;

  float r = M_PI / f;
  float h = 0.5 * r;
  float ff = f * f;

  float2 pos = make_float2(SineWaveMod(px + h, r) - h, py * Sign(r - SineWaveMod(px + h, 2.0 * r)));
  float t = fminf(fmaxf((0.818309886184 * f * pos.y + pos.x) / (0.669631069826 * ff + 1.0), -h), h);

  for (int n = 0; n < 3; n++) {
    float k = t * f;
    float c = cosf(k);
    float s = sinf(k);
    t -= ((s - pos.y) * c * f + t - pos.x) / ((c * c - s * s + s * pos.y) * ff + 1.0);
  }

  float sine_wave = sqrtf((pos.x - t) * (pos.x - t) + (sinf(t * f) - pos.y) * (sinf(t * f) - pos.y)) * amplitude;
  return SineWaveOut{abs(sine_wave) - thickness};
}

struct InterpolationIn {
  float distance1;
  float distance2;
  float interpolation;
};
struct InterpolationOut {
  float distance;
};
__device__ __forceinline__ InterpolationOut InterpolationImpl(InterpolationIn input_) {
  return InterpolationOut{input_.distance1 + input_.interpolation * (input_.distance2 - input_.distance1)};
}

struct GridIn {
  float2 position;
  float thickness;
};
struct GridOut {
  float distance;
};
__device__ __forceinline__ GridOut GridImpl(GridIn input_) {
  float2 p = input_.position;
  float sdfStripeX = abs(p.x) - input_.thickness;
  float sdfStripeY = abs(p.y) - input_.thickness;
  return GridOut{min(sdfStripeX, sdfStripeY)};
}

struct HoneycombGridIn {
  float2 position;
  float thickness;
};
struct HoneycombGridOut {
  float distance;
};
__device__ __forceinline__ float Fract(float x) {
  return x - floor(x);
} 
__device__ __forceinline__ HoneycombGridOut HoneycombGridImpl(HoneycombGridIn input_) {
  float2 p = input_.position;
  float py = p.y / 1.7320508076 - 0.5;
  float px = p.x / 2.0 - Fract(floor(py) * 0.5);
  float px_ = abs(Fract(px) - 0.5);
  float py_ = abs(Fract(py) - 0.5);
  float sdf = abs(1.0 - max(px_ + py_ * 1.5, px_ * 2.0));
  return HoneycombGridOut{sdf - input_.thickness};
}

struct RadialRepetitionIn {
  float2 position;
  float theta;
  float angle;
  float distance;
};
struct RadialRepetitionOut {
  float2 position1;
  float2 position2;
};
__device__ __forceinline__ RadialRepetitionOut RadialRepetitionImpl(RadialRepetitionIn input_) {
  float2 position = input_.position;
  float angle = input_.angle;
  float theta = input_.theta;
  float distance = input_.distance;

  float a = atan2(position.y, position.x);
  float i = theta + floor(a / angle);

  float c1 = angle * (i + 0.0);
  float p1X = position.x * cos(c1) + position.y * sin(c1);
  float p1Y = position.y * cos(c1) - position.x * sin(c1);

  float c2 = angle * (i + 1.0);
  float p2X = position.x * cos(c2) + position.y * sin(c2);
  float p2Y = position.y * cos(c2) - position.x * sin(c2);

  float p1XTra = p1X - distance;
  float p2XTra = p2X - distance;

  return RadialRepetitionOut{float2{p1XTra, p1Y}, float2{p2XTra, p2Y}};
}

struct RadialLineIn {
  float2 position;
  float theta;
  float angle;
  float thickness;
};
struct RadialLineOut {
  float distance;
};
__device__ __forceinline__ float RadialLineDist(float2 position, float2 ba_, float r) {
  float2 pa = position - ba_ * r;
  float2 ba = ba_ * r * -2.0;
  return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}
__device__ __forceinline__ RadialLineOut RadialLineImpl(RadialLineIn input_) {
  float2 position = input_.position;
  float theta = input_.theta;
  float angle = input_.angle;
  float thickness = input_.thickness;
  float r = length(position);
  float alpha = atan2(position.y, position.x);
  float edge = round(alpha / angle);
  float step = (theta + edge) * angle;
  float2 end = float2{cos(step), sin(step)};
  float line = RadialLineDist(position, end, 3.0);
  return RadialLineOut{abs(line) - thickness};
}

struct TruchetIn {
  float2 position;
  float thickness;
  float rotation;
};
struct TruchetOut {
  float distance;
};
__device__ __forceinline__ TruchetOut TruchetImpl(TruchetIn input_) {
  float2 p = input_.position;
  float2 ip = floorf(p);
  p -= ip + 0.5;
  p.y *= (Fract(sin(dot(ip, make_float2(141.213, 289.867))) * 43758.5453 + input_.rotation) > 0.5) ? 1.0 : -1.0;
  p *= Sign(p.x + p.y);
  p -= 0.5;
  float d = length(p) - 0.5;
  d = abs(d) - input_.thickness / 2.0;
  return TruchetOut{d};
}

struct RotationIn {
  float2 position;
  float angle;
};
struct RotationOut {
  float2 position;
};
__device__ __forceinline__ RotationOut RotationImpl(RotationIn input_) {
  float2 p = input_.position;
  float theta = input_.angle;
  float xRot = p.x * cos(theta) + p.y * sin(theta);
  float yRot = p.y * cos(theta) - p.x * sin(theta);
  return RotationOut{float2{xRot, yRot}};
}

struct TranslationIn {
  float2 position;
  float2 translation;
};
struct TranslationOut {
  float2 position;
};
__device__ __forceinline__ TranslationOut TranslationImpl(TranslationIn input_) {
  float2 p = input_.position;
  float2 translation = input_.translation;
  return TranslationOut{p - translation};
}

struct ScaleIn {
  float2 position;
  float scale;
};
struct ScaleOut {
  float2 position;
};
__device__ __forceinline__ ScaleOut ScaleImpl(ScaleIn input_) {
  float2 p = input_.position;
  float scale = input_.scale;
  return ScaleOut{p / scale};
}

struct RepetitionIn {
  float2 position;
  float2 space;
};
struct RepetitionOut {
  float2 position;
};
__device__ __forceinline__ float RepetitionMod(float a, float b) {
  int fl = int(floor(a / b));
  return a - float(fl) * b;
}
__device__ __forceinline__ RepetitionOut RepetitionImpl(RepetitionIn input_) {
  float2 p = input_.position;
  float2 space = input_.space;
  float repPosX = RepetitionMod((p.x + space.x/2.0), space.x) - space.x / 2.0;
  float repPosY = RepetitionMod((p.y + space.y/2.0), space.y) - space.y / 2.0;
  return RepetitionOut{float2{repPosX, repPosY}};
}

struct Combine2In {
  float2 value1;
  float2 scale1;
  float2 offset1;
  float2 value2;
  float2 scale2;
  float2 offset2;
};
struct Combine2Out {
  float2 value;
};
__device__ __forceinline__ Combine2Out Combine2Impl(Combine2In input_) {
  return Combine2Out{input_.value1 * input_.scale1 + input_.offset1 + 
                        input_.value2 * input_.scale2 + input_.offset2};
}

struct CircleOnionIn {
  float2 position;
  float radius;
  float inner;
  float step;
};
struct CircleOnionOut {
  float distance;
};
__device__ __forceinline__ CircleOnionOut CircleOnionImpl(CircleOnionIn input_) {
  float2 p = input_.position;
  float radius = input_.radius;
  float inner = input_.inner > input_.radius * 0.1 ? 1.0 : input_.inner;
  float step = max(0.2, input_.step);
  float circle = sqrt(length(p)) - radius;
  float distance = abs(circle + inner) - inner;
  
  float r = radius - step * radius;
  while (r > 0.025) {
    float sdf_circle = sqrt(length(p)) - r;
    float i = inner > r * 0.1 ? 1.0 : inner;
    float sdf_circle_inner = abs(sdf_circle + i) - i;
    distance = min(distance, sdf_circle_inner);
    r = r - step * radius;
  }
  return CircleOnionOut{distance};
}

struct OverIdIn {
  float distance1;
  float levelid1;
  float distance2;
  float levelid2;
};
struct OverIdOut {
  float distance;
  float levelid;
};
#define OVERID_THICKNESS 0.0025
__device__ __forceinline__ float OverIdBorderSdf(float sdf) {
  return abs(sdf + OVERID_THICKNESS) - OVERID_THICKNESS;
}
__device__ __forceinline__ float OverIdInnerSdf(float sdf) {
  return sdf - OVERID_THICKNESS;
}
__device__ __forceinline__ OverIdOut OverIdImpl(OverIdIn input_) {
  float distance1 = max(input_.distance1, -OverIdBorderSdf(input_.distance2));
  float distance2 = OverIdInnerSdf(input_.distance2);
  if (distance1 < distance2) {
    return OverIdOut{distance1, input_.levelid1};
  } else {
    return OverIdOut{distance2, input_.levelid2};
  }
}

struct UnionIdIn {
  float distance1;
  float levelid1;
  float distance2;
  float levelid2;
};
struct UnionIdOut {
  float distance;
  float levelid;
};
__device__ __forceinline__ UnionIdOut UnionIdImpl(UnionIdIn input_) {
  if (input_.distance1 < input_.distance2) {
    return UnionIdOut{input_.distance1, input_.levelid1};
  } else {
    return UnionIdOut{input_.distance2, input_.levelid2};
  }
}

struct DifferenceIdIn {
  float distance1;
  float levelid1;
  float distance2;
  float levelid2;
};
struct DifferenceIdOut {
  float distance;
  float levelid;
};
__device__ __forceinline__ DifferenceIdOut DifferenceIdImpl(DifferenceIdIn input_) {
  if (-input_.distance1 > input_.distance2) {
    return DifferenceIdOut{-input_.distance1, input_.levelid1};
  } else {
    return DifferenceIdOut{input_.distance2, input_.levelid2};
  }
}

struct IntersectionIdIn {
  float distance1;
  float levelid1;
  float distance2;
  float levelid2;
};
struct IntersectionIdOut {
  float distance;
  float levelid;
};
__device__ __forceinline__ IntersectionIdOut IntersectionIdImpl(IntersectionIdIn input_) {
  if (input_.distance1 > input_.distance2) {
    return IntersectionIdOut{input_.distance1, input_.levelid1};
  } else {
    return IntersectionIdOut{input_.distance2, input_.levelid2};
  }
}

struct ScaffoldIdIn {
  float distance1;
  float levelid1;
  float distance2;
  float levelid2;
  float view_scaffold;
};
struct ScaffoldIdOut {
  float distance;
  float levelid;
};
#define SCAFFOLDEDOVERID_THICKNESS 0.0025
__device__ __forceinline__ float ScaffoldIdBorderSdf(float sdf) {
  return abs(sdf + SCAFFOLDEDOVERID_THICKNESS) - SCAFFOLDEDOVERID_THICKNESS;
}
__device__ __forceinline__ float ScaffoldIdInnerSdf(float sdf) {
  return sdf + SCAFFOLDEDOVERID_THICKNESS;
}
__device__ __forceinline__ ScaffoldIdOut ScaffoldIdImpl(ScaffoldIdIn input_) {
  if(input_.view_scaffold == 0.0) {
    return ScaffoldIdOut{input_.distance2, input_.levelid2};
  }
  float distance1 = max(input_.distance1, -ScaffoldIdBorderSdf(input_.distance2));
  float distance2 = ScaffoldIdInnerSdf(input_.distance2);
  if (distance1 < distance2) {
    return ScaffoldIdOut{distance1, input_.levelid1};
  } else {
    return ScaffoldIdOut{distance2, input_.levelid2};
  }
}

struct Level1Output {
  float distance;
  float levelid;
};

struct Level2Output {
  float distance;
  float levelid;
};

struct Level3Output {
  float distance;
  float levelid;
};

struct Level4Output {
  float distance;
  float levelid;
};

struct Level5Output {
  float distance;
  float levelid;
};

struct Level6Output {
  float distance;
  float levelid;
};

struct Level7Output {
  float distance;
  float levelid;
};

struct Level8Output {
  float distance;
  float levelid;
};

struct Level9Output {
  float distance;
  float levelid;
};

struct Level10Output {
  float distance;
  float levelid;
};

struct PositionInput {
  float2 position;
  float view_scaffold;
};
