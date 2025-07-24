#include <nodes.cuh>


struct Params0Input {
  float theta;
  float2 trans;
  float spaceY;
  float thickness;
};

struct Params1Input {
  float amplitude;
  float frequence;
  float interpolation;
  float thickness;
};

struct Params2Input {
  float radius2;
  float inner2;
};

struct Consts0Input {
  float double_pi;
  float pi;
};

struct Consts1Input {
  float half_one;
  float ten;
};

struct Consts2Input {
  float2 two;
  float2 one;
};

struct Consts3Input {
  float zero;
};

struct Consts4Input {
  float point_one;
  float two;
};

struct Consts5Input {
  float levelid0;
  float levelid1;
  float levelid2;
};

struct LevelsOutput {
  Level1Output level1;
  Level2Output level2;
  Level3Output level3;
};
__device__ __forceinline__ LevelsOutput LevelsImpl(PositionInput position, Params0Input params0, Params1Input params1, Params2Input params2) {
  Consts0Input consts0 = Consts0Input{6.28, 3.14};
  Consts1Input consts1 = Consts1Input{0.5, 10.0};
  Consts2Input consts2 = Consts2Input{float2{2.0, 2.0}, float2{1.0, 1.0}};
  Consts3Input consts3 = Consts3Input{0.0};
  Consts4Input consts4 = Consts4Input{0.1, 2.0};
  Consts5Input consts5 = Consts5Input{0.0, 1.0, 2.0};
  MulOut mul_amplitude = MulImpl(MulIn{params1.amplitude, consts1.half_one});
  MulOut mul_frequence = MulImpl(MulIn{params1.frequence, consts1.ten});
  NormalizeOut normalize_theta = NormalizeImpl(NormalizeIn{params0.theta, consts0.double_pi, consts0.pi});
  Normalize2fOut normalize_trans = Normalize2fImpl(Normalize2fIn{params0.trans, consts2.two, consts2.one});
  RotationOut rotation = RotationImpl(RotationIn{position.position, normalize_theta.value});
  TranslationOut translation = TranslationImpl(TranslationIn{rotation.position, normalize_trans.value});
  ToVec2Out tovec2 = ToVec2Impl(ToVec2In{consts3.zero, params0.spaceY});
  RepetitionOut repetition = RepetitionImpl(RepetitionIn{translation.position, tovec2.value});
  StripeOut stripe = StripeImpl(StripeIn{repetition.position, params0.thickness});
  TriangleWaveOut triangle_wave = TriangleWaveImpl(TriangleWaveIn{repetition.position, mul_frequence.value, mul_amplitude.value, params1.thickness});
  SineWaveOut sine_wave = SineWaveImpl(SineWaveIn{repetition.position, mul_frequence.value, mul_amplitude.value, params1.thickness});
  InterpolationOut interpolation = InterpolationImpl(InterpolationIn{triangle_wave.distance, sine_wave.distance, params1.interpolation});
  DivOut div_spaceX = DivImpl(DivIn{consts4.point_one, params1.frequence});
  DivOut div_half_spaceX = DivImpl(DivIn{div_spaceX.value, consts4.two});
  FromVec2Out fromvec2 = FromVec2Impl(FromVec2In{normalize_trans.value});
  AddOut add_transX = AddImpl(AddIn{fromvec2.x, div_half_spaceX.value});
  ToVec2Out tovec2b = ToVec2Impl(ToVec2In{add_transX.value, fromvec2.y});
  TranslationOut translation2 = TranslationImpl(TranslationIn{rotation.position, tovec2b.value});
  ToVec2Out tovec2a = ToVec2Impl(ToVec2In{div_spaceX.value, params0.spaceY});
  RepetitionOut repetition2 = RepetitionImpl(RepetitionIn{translation2.position, tovec2a.value});
  CircleOut circle = CircleImpl(CircleIn{repetition2.position, params2.radius2, params2.inner2});
  Level1Output level1 = Level1Output{stripe.distance, consts5.levelid0};
  ScaffoldIdOut scaffoldid0 = ScaffoldIdImpl(ScaffoldIdIn{stripe.distance, consts5.levelid0, interpolation.distance, consts5.levelid1, position.view_scaffold});
  Level2Output level2 = Level2Output{scaffoldid0.distance, scaffoldid0.levelid};
  OverIdOut overid1 = OverIdImpl(OverIdIn{scaffoldid0.distance, scaffoldid0.levelid, circle.distance, consts5.levelid2});
  Level3Output level3 = Level3Output{overid1.distance, overid1.levelid};
  return LevelsOutput{level1, level2, level3};
}

