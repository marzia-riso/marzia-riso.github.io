#include <nodes.cuh>

struct Params1Input {
  float theta;
  float2 trans;
  float2 space;
  float thickness;
};

struct Params2Input {
  float thickness;
};

struct Params3Input {
  float thickness;
};

struct Params4Input {
  float offset;
  float radius;
  float inner;
};

struct Params5Input {
  float theta;
  float size;
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
  float zero;
  float one;
  float two;
};

struct Consts4Input {
  float levelid0;
  float levelid1;
  float levelid2;
  float levelid3;
  float levelid4;
};

struct LevelsOutput {
  Level1Output level1;
  Level2Output level2;
  Level3Output level3;
  Level4Output level4;
  Level5Output level5;
};
__device__ __forceinline__ LevelsOutput LevelsImpl(PositionInput position, Params1Input params1, Params2Input params2, Params3Input params3, Params4Input params4, Params5Input params5) {
  Consts0Input consts0 = Consts0Input{6.28, 3.14};
  Consts1Input consts1 = Consts1Input{float2{2.0, 2.0}, float2{1.0, 1.0}};
  Consts2Input consts2 = Consts2Input{0.5, 0.0, -1.0, 2.0};
  Consts4Input consts4 = Consts4Input{0.0, 1.0, 2.0, 3.0, 4.0};
  NormalizeOut normalized_theta = NormalizeImpl(NormalizeIn{params1.theta, consts0.double_pi, consts0.pi});
  RotationOut rotation = RotationImpl(RotationIn{position.position, normalized_theta.value});
  Normalize2fOut normalized_trans = Normalize2fImpl(Normalize2fIn{params1.trans, consts1.two, consts1.one});
  TranslationOut translation = TranslationImpl(TranslationIn{rotation.position, normalized_trans.value});
  RepetitionOut repetition = RepetitionImpl(RepetitionIn{translation.position, params1.space});
  GridOut grid = GridImpl(GridIn{repetition.position, params1.thickness});
  Level1Output level1 = Level1Output{grid.distance, consts4.levelid0};
  StripeOut stripe = StripeImpl(StripeIn{repetition.position, params2.thickness});
  ScaffoldIdOut scaffold_id = ScaffoldIdImpl(ScaffoldIdIn{grid.distance, consts4.levelid0, stripe.distance, consts4.levelid1, position.view_scaffold});
  Level2Output level2 = Level2Output{scaffold_id.distance, scaffold_id.levelid};
  FromVec2Out fromvec21 = FromVec2Impl(FromVec2In{params1.space});
  MulOut mul1 = MulImpl(MulIn{fromvec21.y, consts2.two});
  MulOut mul2 = MulImpl(MulIn{fromvec21.x, consts2.half_one});
  MulOut mul3 = MulImpl(MulIn{mul1.value, consts2.half_one});
  MulOut mul4 = MulImpl(MulIn{fromvec21.y, consts2.half_one});
  ToVec2Out tovec2 = ToVec2Impl(ToVec2In{consts2.zero, mul4.value});
  TranslationOut translation2 = TranslationImpl(TranslationIn{translation.position, tovec2.value});
  RepetitionOut repetition2 = RepetitionImpl(RepetitionIn{translation2.position, params1.space});
  StripeOut stripe2 = StripeImpl(StripeIn{repetition2.position, params3.thickness});
  OverIdOut over_id2 = OverIdImpl(OverIdIn{scaffold_id.distance, scaffold_id.levelid, stripe2.distance, consts4.levelid2});
  Level3Output level3 = Level3Output{over_id2.distance, over_id2.levelid};
  ToVec2Out tovec21 = ToVec2Impl(ToVec2In{fromvec21.x, mul1.value});
  RoundOut round = RoundImpl(RoundIn{params4.offset});
  MulOut mul5 = MulImpl(MulIn{mul3.value, round.value});
  ToVec2Out tovec3 = ToVec2Impl(ToVec2In{consts2.zero, mul5.value});
  TranslationOut translation3 = TranslationImpl(TranslationIn{translation.position, tovec3.value});
  RepetitionOut repetition3 = RepetitionImpl(RepetitionIn{translation3.position, tovec21.value});
  CircleOut circle = CircleImpl(CircleIn{repetition3.position, params4.radius, params4.inner});
  OverIdOut over_id3 = OverIdImpl(OverIdIn{over_id2.distance, over_id2.levelid, circle.distance, consts4.levelid3});
  Level4Output level4 = Level4Output{over_id3.distance, over_id3.levelid};
  SubOut sub_round = SubImpl(SubIn{consts2.one, round.value});
  MulOut mul6 = MulImpl(MulIn{mul3.value, sub_round.value});
  ToVec2Out tovec4 = ToVec2Impl(ToVec2In{mul2.value, mul6.value});
  TranslationOut translation5 = TranslationImpl(TranslationIn{translation.position, tovec4.value});
  RepetitionOut repetition4 = RepetitionImpl(RepetitionIn{translation5.position, tovec21.value});
  RotationOut rotation2 = RotationImpl(RotationIn{repetition4.position, params5.theta});
  BoxOut box = BoxImpl(BoxIn{rotation2.position, params5.size});
  OverIdOut over_id4 = OverIdImpl(OverIdIn{over_id3.distance, over_id3.levelid, box.distance, consts4.levelid4});
  Level5Output level5 = Level5Output{over_id4.distance, over_id4.levelid};
  return LevelsOutput{level1, level2, level3, level4, level5};
}

