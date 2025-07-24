#include <nodes.cuh>

struct Params1Input {
  float theta;
  float2 trans;
  float2 space;
  float thickness;
};

struct Params2Input {
  float amplitude;
  float frequence;
  float interpolation;
  float thickness;
};

struct Params3Input {
  float size;
  float rotation;
};

struct Params4Input {
  float radius4;
  float inner4;
};

struct Params5Input {
  float distance;
  float angle;
  float offset;
  float radius;
};

struct Params6Input {
  float length;
  float curvature;
  float thickness;
};

struct Params7Input {
  float radius;
  float offset;
  float interpolation;
};

struct Consts0Input {
  float double_pi;
  float pi;
};

struct Consts1Input {
  float2 two;
  float2 one;
};

struct Consts2Input {
  float half_one;
  float ten;
};

struct Consts3Input {
  float zero;
};

struct Consts4Input {
  float point_one;
  float zero;
  float one;
  float two;
};

struct Consts5Input {
  float2 two;
  float2 min_one;
  float2 half_one;
  float2 zero;
};

struct Consts6Input {
  float one;
  float double_pi;
};

struct Consts7Input {
  float one;
};

struct Consts8Input {
  float one;
  float two;
};

struct Consts9Input {
  float one;
};

struct Consts10Input {
  float levelid0;
  float levelid1;
  float levelid2;
  float levelid3;
  float levelid4;
  float levelid5;
  float levelid6;
};

struct LevelsOutput {
  Level1Output level1;
  Level2Output level2;
  Level3Output level3;
  Level4Output level4;
  Level5Output level5;
  Level6Output level6;
  Level7Output level7;
};
__device__ __forceinline__ LevelsOutput LevelsImpl(PositionInput position, Params1Input params1, Params2Input params2, Params3Input params3, Params4Input params4, Params5Input params5, Params6Input params6, Params7Input params7) {
  Consts0Input consts0 = Consts0Input{6.28, 3.14};
  Consts1Input consts1 = Consts1Input{float2{2.0, 2.0}, float2{1.0, 1.0}};
  Consts2Input consts2 = Consts2Input{0.5, 10.0};
  Consts3Input consts3 = Consts3Input{0.0};
  Consts4Input consts4 = Consts4Input{0.1, 0.0, 1.0, 2.0};
  Consts5Input consts5 = Consts5Input{float2{2.0, 2.0}, float2{-1.0, -1.0}, float2{0.5, 0.5}, float2{0.0, 0.0}};
  Consts6Input consts6 = Consts6Input{1.0, 6.28};
  Consts7Input consts7 = Consts7Input{1.0};
  Consts8Input consts8 = Consts8Input{1.0, 2.0};
  Consts9Input consts9 = Consts9Input{1.0};
  Consts10Input consts10 = Consts10Input{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  NormalizeOut normalize_theta = NormalizeImpl(NormalizeIn{params1.theta, consts0.double_pi, consts0.pi});
  RotationOut rotation = RotationImpl(RotationIn{position.position, normalize_theta.value});
  Normalize2fOut normalize_trans = Normalize2fImpl(Normalize2fIn{params1.trans, consts1.two, consts1.one});
  TranslationOut translation = TranslationImpl(TranslationIn{rotation.position, normalize_trans.value});
  RepetitionOut repetition = RepetitionImpl(RepetitionIn{translation.position, params1.space});
  GridOut grid = GridImpl(GridIn{repetition.position, params1.thickness});
  Level1Output level1 = Level1Output{grid.distance, consts10.levelid0};
  MulOut amplitude = MulImpl(MulIn{params2.amplitude, consts2.half_one});
  MulOut frequence = MulImpl(MulIn{params2.frequence, consts2.ten});
  FromVec2Out fromvec2 = FromVec2Impl(FromVec2In{params1.space});
  ToVec2Out tovec2 = ToVec2Impl(ToVec2In{consts3.zero, fromvec2.y});
  RepetitionOut repetition2 = RepetitionImpl(RepetitionIn{translation.position, tovec2.value});
  TriangleWaveOut triangle_wave = TriangleWaveImpl(TriangleWaveIn{repetition2.position, frequence.value, amplitude.value, params2.thickness});
  SineWaveOut sine_wave = SineWaveImpl(SineWaveIn{repetition2.position, frequence.value, amplitude.value, params2.thickness});
  InterpolationOut wave = InterpolationImpl(InterpolationIn{triangle_wave.distance, sine_wave.distance, params2.interpolation});
  ScaffoldIdOut scaffold2 = ScaffoldIdImpl(ScaffoldIdIn{grid.distance, consts10.levelid0, wave.distance, consts10.levelid1, position.view_scaffold});
  Level2Output level2 = Level2Output{scaffold2.distance, scaffold2.levelid};
  DivOut row_space = DivImpl(DivIn{consts4.one, params2.frequence});
  MulOut mul_row_space = MulImpl(MulIn{row_space.value, consts4.point_one});
  DivOut half_row_space = DivImpl(DivIn{mul_row_space.value, consts4.two});
  NegOut neg = NegImpl(NegIn{half_row_space.value});
  ToVec2Out tovec2a = ToVec2Impl(ToVec2In{neg.value, consts4.zero});
  TranslationOut translation3 = TranslationImpl(TranslationIn{translation.position, tovec2a.value});
  ToVec2Out tovec2b = ToVec2Impl(ToVec2In{mul_row_space.value, fromvec2.y});
  RepetitionOut repetition3 = RepetitionImpl(RepetitionIn{translation3.position, tovec2b.value});
  NormalizeOut normalize_theta1 = NormalizeImpl(NormalizeIn{params3.rotation, consts0.double_pi, consts0.pi});
  RotationOut rotation3 = RotationImpl(RotationIn{repetition3.position, normalize_theta1.value});
  BoxOut box3 = BoxImpl(BoxIn{rotation3.position, params3.size});
  OverIdOut over3 = OverIdImpl(OverIdIn{scaffold2.distance, scaffold2.levelid, box3.distance, consts10.levelid2});
  Level3Output level3 = Level3Output{over3.distance, over3.levelid};
  Combine2Out transgrid = Combine2Impl(Combine2In{params1.trans, consts5.two, consts5.min_one, params1.space, consts5.half_one, consts5.zero});
  TranslationOut translation2 = TranslationImpl(TranslationIn{rotation.position, transgrid.value});
  RepetitionOut repetition4 = RepetitionImpl(RepetitionIn{translation2.position, params1.space});
  CircleOut circle4 = CircleImpl(CircleIn{repetition4.position, params4.radius4, params4.inner4});
  OverIdOut over4 = OverIdImpl(OverIdIn{over3.distance, over3.levelid, circle4.distance, consts10.levelid3});
  Level4Output level4 = Level4Output{over4.distance, over4.levelid};
  DivOut div_angle = DivImpl(DivIn{consts6.one, params5.angle});
  RoundOut round_angle = RoundImpl(RoundIn{div_angle.value});
  DivOut angle = DivImpl(DivIn{consts6.double_pi, round_angle.value});
  MulOut offset = MulImpl(MulIn{params5.offset, angle.value});
  RadialRepetitionOut radial_rep5 = RadialRepetitionImpl(RadialRepetitionIn{repetition4.position, offset.value, angle.value, params5.distance});
  CircleOut rad_circle51 = CircleImpl(CircleIn{radial_rep5.position1, params5.radius, consts7.one});
  CircleOut rad_circle52 = CircleImpl(CircleIn{radial_rep5.position2, params5.radius, consts7.one});
  UnionIdOut radial5 = UnionIdImpl(UnionIdIn{rad_circle51.distance, consts10.levelid4, rad_circle52.distance, consts10.levelid4});
  OverIdOut over5 = OverIdImpl(OverIdIn{over4.distance, over4.levelid, radial5.distance, radial5.levelid});
  Level5Output level5 = Level5Output{over5.distance, over5.levelid};
  NormalizeOut normalize1 = NormalizeImpl(NormalizeIn{params6.curvature, consts0.double_pi, consts0.pi});
  JointOut rad_joint1 = JointImpl(JointIn{radial_rep5.position1, params6.length, normalize1.value, params6.thickness});
  JointOut rad_joint2 = JointImpl(JointIn{radial_rep5.position2, params6.length, normalize1.value, params6.thickness});
  UnionIdOut radial_joints = UnionIdImpl(UnionIdIn{rad_joint1.distance, consts10.levelid5, rad_joint2.distance, consts10.levelid5});
  ScaffoldIdOut scaffold6 = ScaffoldIdImpl(ScaffoldIdIn{radial5.distance, radial5.levelid, radial_joints.distance, radial_joints.levelid, position.view_scaffold});
  OverIdOut over6 = OverIdImpl(OverIdIn{over4.distance, over4.levelid, scaffold6.distance, scaffold6.levelid});
  Level6Output level6 = Level6Output{over6.distance, over6.levelid};
  FromVec2Out fromvec2c = FromVec2Impl(FromVec2In{translation2.position});
  AddOut add_factor = AddImpl(AddIn{fromvec2c.x, consts8.one});
  DivOut div_factor = DivImpl(DivIn{add_factor.value, consts8.two});
  MulOut mul_factor = MulImpl(MulIn{params7.interpolation, div_factor.value});
  AddOut factor = AddImpl(AddIn{params7.offset, mul_factor.value});
  MulOut radius_factor = MulImpl(MulIn{factor.value, params7.radius});
  CircleOut circle7 = CircleImpl(CircleIn{repetition4.position, radius_factor.value, consts9.one});
  OverIdOut over7 = OverIdImpl(OverIdIn{over6.distance, over6.levelid, circle7.distance, consts10.levelid6});
  Level7Output level7 = Level7Output{over7.distance, over7.levelid};
  return LevelsOutput{level1, level2, level3, level4, level5, level6, level7};
}

