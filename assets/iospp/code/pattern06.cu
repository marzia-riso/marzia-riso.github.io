#include <nodes.cuh>

struct Params1Input {
  float theta;
  float2 trans;
  float scale;
  float thickness;
};

struct Params2Input {
  float offset;
  float radius1;
  float inner1;
};

struct Params3Input {
  float radius;
  float rotation;
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
  float half_pi;
};

struct Consts3Input {
  float2 repetition;
};

struct Consts4Input {
  float one;
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
__device__ __forceinline__ LevelsOutput LevelsImpl(PositionInput position, Params1Input params1, Params2Input params2, Params3Input params3) {
  Consts0Input consts0 = Consts0Input{6.28, 3.14};
  Consts1Input consts1 = Consts1Input{float2{2.0, 2.0}, float2{1.0, 1.0}};
  Consts2Input consts2 = Consts2Input{1.7320508076};
  Consts3Input consts3 = Consts3Input{float2{2.0, 3.4641016151}};
  Consts4Input consts4 = Consts4Input{1.0};
  Consts5Input consts5 = Consts5Input{0.0, 1.0, 2.0};
  ScaleOut scale = ScaleImpl(ScaleIn{position.position, params1.scale});
  NormalizeOut normalize_theta = NormalizeImpl(NormalizeIn{params1.theta, consts0.double_pi, consts0.pi});
  RotationOut rot = RotationImpl(RotationIn{scale.position, normalize_theta.value});
  Normalize2fOut normalize_trans = Normalize2fImpl(Normalize2fIn{params1.trans, consts1.two, consts1.one});
  TranslationOut trans = TranslationImpl(TranslationIn{rot.position, normalize_trans.value});
  HoneycombGridOut honeycomb = HoneycombGridImpl(HoneycombGridIn{trans.position, params1.thickness});
  MulOut honeycomb_dist = MulImpl(MulIn{honeycomb.distance, params1.scale});
  Level1Output level1 = Level1Output{honeycomb_dist.value, consts5.levelid0};
  RoundOut round_offset = RoundImpl(RoundIn{params2.offset});
  NegOut neg = NegImpl(NegIn{round_offset.value});
  FromVec2Out fromvec2 = FromVec2Impl(FromVec2In{normalize_trans.value});
  AddOut transX = AddImpl(AddIn{fromvec2.x, neg.value});
  MulOut mulOffset = MulImpl(MulIn{round_offset.value, consts2.half_pi});
  NegOut neg2 = NegImpl(NegIn{mulOffset.value});
  AddOut transY = AddImpl(AddIn{fromvec2.y, neg2.value});
  ToVec2Out tovec2 = ToVec2Impl(ToVec2In{transX.value, transY.value});
  TranslationOut trans2 = TranslationImpl(TranslationIn{rot.position, tovec2.value});
  RepetitionOut rep2 = RepetitionImpl(RepetitionIn{trans2.position, consts3.repetition});
  CircleOut circle = CircleImpl(CircleIn{rep2.position, params2.radius1, params2.inner1});
  OverIdOut over = OverIdImpl(OverIdIn{honeycomb_dist.value, consts5.levelid0, circle.distance, consts5.levelid1});
  Level2Output level2 = Level2Output{over.distance, over.levelid};
  SubOut sub_offset = SubImpl(SubIn{consts4.one, round_offset.value});
  AddOut transX2 = AddImpl(AddIn{fromvec2.x, sub_offset.value});
  MulOut mulOffset2 = MulImpl(MulIn{sub_offset.value, consts2.half_pi});
  NegOut neg3 = NegImpl(NegIn{mulOffset2.value});
  AddOut transY2 = AddImpl(AddIn{fromvec2.y, neg3.value});
  ToVec2Out tovec2a = ToVec2Impl(ToVec2In{transX2.value, transY2.value});
  TranslationOut trans3 = TranslationImpl(TranslationIn{rot.position, tovec2a.value});
  RepetitionOut rep3 = RepetitionImpl(RepetitionIn{trans3.position, consts3.repetition});
  RotationOut rot3 = RotationImpl(RotationIn{rep3.position, params3.rotation});
  BoxOut box = BoxImpl(BoxIn{rot3.position, params3.radius});
  OverIdOut over2 = OverIdImpl(OverIdIn{over.distance, over.levelid, box.distance, consts5.levelid2});
  Level3Output level3 = Level3Output{over2.distance, over2.levelid};
  return LevelsOutput{level1, level2, level3};
}

