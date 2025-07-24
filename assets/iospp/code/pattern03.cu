#include <nodes.cuh>

struct Params0Input {
  float theta;
  float2 translation;
  float2 space;
  float thickness;
};

struct Params1Input {
  float radius;
  float inner;
  float step;
};

struct Params2Input {
  float theta;
  float distance;
  float angle;
  float radius;
};

struct Consts0Input {
  float double_pi;
  float pi;
  float one;
};

struct Consts1Input {
  float2 two;
  float2 one;
};

struct Consts2Input {
  float one;
};

struct Consts3Input {
  float levelid0;
  float levelid1;
  float levelid2;
};

struct LevelsOutput {
  Level1Output level1;
  Level2Output level2;
  Level3Output level3;
};
__device__ __forceinline__ LevelsOutput LevelsImpl(PositionInput position, Params0Input params1, Params1Input params2, Params2Input params3) {
  Consts0Input consts0 = Consts0Input{6.28, 3.14, 1.0};
  Consts1Input consts1 = Consts1Input{float2{2.0, 2.0}, float2{1.0, 1.0}};
  Consts2Input consts2 = Consts2Input{1.0};
  Consts3Input consts3 = Consts3Input{0.0, 1.0, 2.0};
  NormalizeOut normalize0 = NormalizeImpl(NormalizeIn{params1.theta, consts0.double_pi, consts0.pi});
  Normalize2fOut normalize2f0 = Normalize2fImpl(Normalize2fIn{params1.translation, consts1.two, consts1.one});
  RotationOut rotation0 = RotationImpl(RotationIn{position.position, normalize0.value});
  TranslationOut translation0 = TranslationImpl(TranslationIn{rotation0.position, normalize2f0.value});
  RepetitionOut repetition0 = RepetitionImpl(RepetitionIn{translation0.position, params1.space});
  GridOut grid0 = GridImpl(GridIn{repetition0.position, params1.thickness});
  Level1Output level1 = Level1Output{grid0.distance, consts3.levelid0};
  CircleOnionOut onion0 = CircleOnionImpl(CircleOnionIn{repetition0.position, params2.radius, params2.inner, params2.step});
  ScaffoldIdOut scaffoldid0 = ScaffoldIdImpl(ScaffoldIdIn{grid0.distance, consts3.levelid0, onion0.distance, consts3.levelid1, position.view_scaffold});
  Level2Output level2 = Level2Output{scaffoldid0.distance, scaffoldid0.levelid};
  NormalizeOut normalize1 = NormalizeImpl(NormalizeIn{params3.theta, consts0.double_pi, consts0.pi});
  DivOut div0 = DivImpl(DivIn{consts0.one, params3.angle});
  RoundOut round0 = RoundImpl(RoundIn{div0.value});
  DivOut div1 = DivImpl(DivIn{consts0.pi, round0.value});
  RadialRepetitionOut radialrepetition0 = RadialRepetitionImpl(RadialRepetitionIn{repetition0.position, normalize1.value, div1.value, params3.distance});
  CircleOut circle1 = CircleImpl(CircleIn{radialrepetition0.position1, params3.radius, consts2.one});
  CircleOut circle2 = CircleImpl(CircleIn{radialrepetition0.position2, params3.radius, consts2.one});
  UnionIdOut union0 = UnionIdImpl(UnionIdIn{circle1.distance, consts3.levelid2, circle2.distance, consts3.levelid2});
  OverIdOut overid2 = OverIdImpl(OverIdIn{scaffoldid0.distance, scaffoldid0.levelid, union0.distance, union0.levelid});
  Level3Output level3 = Level3Output{overid2.distance, overid2.levelid};
  return LevelsOutput{level1, level2, level3};
}

