#include <nodes.cuh>

struct Params1Input {
  float theta;
  float2 trans;
  float scale;
  float thickness;
};

struct Params2Input {
  float interp;
  float offset;
  float radius1;
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
  float one;
  float half_one;
};

struct Consts3Input {
  float2 repetition;
};

struct Consts4Input {
  float one;
};

struct Consts5Input {
  float min_one;
  float min_half_pi;
};

struct Consts6Input {
  float levelid0;
  float levelid1;
};

struct LevelsOutput {
  Level1Output level1;
  Level2Output level2;
};
__device__ __forceinline__ LevelsOutput LevelsImpl(PositionInput position, Params1Input params1, Params2Input params2) {
  Consts0Input consts0 = Consts0Input{6.28, 3.14};
  Consts1Input consts1 = Consts1Input{float2{2.0, 2.0}, float2{1.0, 1.0}};
  Consts2Input consts2 = Consts2Input{1.0, 0.5};
  Consts3Input consts3 = Consts3Input{float2{2.0, 3.4641016151}};
  Consts4Input consts4 = Consts4Input{1.0};
  Consts5Input consts5 = Consts5Input{-1.0, -1.7320508076};
  Consts6Input consts6 = Consts6Input{0.0, 1.0};
  ScaleOut scale = ScaleImpl(ScaleIn{position.position, params1.scale});
  NormalizeOut normalize_theta = NormalizeImpl(NormalizeIn{params1.theta, consts0.double_pi, consts0.pi});
  RotationOut rot = RotationImpl(RotationIn{scale.position, normalize_theta.value});
  Normalize2fOut normalize_trans = Normalize2fImpl(Normalize2fIn{params1.trans, consts1.two, consts1.one});
  TranslationOut trans = TranslationImpl(TranslationIn{rot.position, normalize_trans.value});
  HoneycombGridOut honeycomb = HoneycombGridImpl(HoneycombGridIn{trans.position, params1.thickness});
  MulOut honeycomb_dist = MulImpl(MulIn{honeycomb.distance, params1.scale});
  Level1Output level1 = Level1Output{honeycomb_dist.value, consts6.levelid0};
  FromVec2Out fromvec2a = FromVec2Impl(FromVec2In{rot.position});
  AddOut add = AddImpl(AddIn{fromvec2a.x, consts2.one});
  MulOut mul = MulImpl(MulIn{params2.interp, add.value});
  MulOut mul1 = MulImpl(MulIn{mul.value, consts2.half_one});
  AddOut factor = AddImpl(AddIn{params2.offset, mul1.value});
  MulOut radius = MulImpl(MulIn{factor.value, params2.radius1});
  TranslationOut trans2 = TranslationImpl(TranslationIn{rot.position, normalize_trans.value});
  RepetitionOut rep2 = RepetitionImpl(RepetitionIn{trans2.position, consts3.repetition});
  CircleOut circle = CircleImpl(CircleIn{rep2.position, radius.value, consts4.one});
  FromVec2Out fromvec2 = FromVec2Impl(FromVec2In{normalize_trans.value});
  AddOut transX = AddImpl(AddIn{fromvec2.x, consts5.min_one});
  AddOut transY = AddImpl(AddIn{fromvec2.y, consts5.min_half_pi});
  ToVec2Out tovec2 = ToVec2Impl(ToVec2In{transX.value, transY.value});
  TranslationOut trans3 = TranslationImpl(TranslationIn{rot.position, tovec2.value});
  RepetitionOut rep3 = RepetitionImpl(RepetitionIn{trans3.position, consts3.repetition});
  CircleOut circle2 = CircleImpl(CircleIn{rep3.position, radius.value, consts4.one});
  OverIdOut over = OverIdImpl(OverIdIn{honeycomb_dist.value, consts6.levelid0, circle.distance, consts6.levelid1});
  OverIdOut over2 = OverIdImpl(OverIdIn{over.distance, over.levelid, circle2.distance, consts6.levelid1});
  Level2Output level2 = Level2Output{over2.distance, over2.levelid};
  return LevelsOutput{level1, level2};
}

