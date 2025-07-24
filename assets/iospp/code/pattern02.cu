#include <nodes.cuh>

struct Params0Input {
  float angle;
  float2 translation;
  float2 space;
  float thickness;
};

struct Params1Input {
  float radius;
  float inner;
};

struct Params2Input {
  float radius;
  float inner;
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
  float2 two;
  float2 min_one;
  float2 half_one;
  float2 zero;
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
__device__ __forceinline__ LevelsOutput LevelsImpl(PositionInput position, Params0Input params0, Params1Input params1, Params2Input params2) {
  Consts0Input consts0 = Consts0Input{6.28, 3.14};
  Consts1Input consts1 = Consts1Input{float2{2.0, 2.0}, float2{1.0, 1.0}};
  Consts2Input consts2 = Consts2Input{float2{2.0, 2.0}, float2{-1.0, -1.0}, float2{0.5, 0.5}, float2{0.0, 0.0}};
  Consts3Input consts3 = Consts3Input{0.0, 1.0, 2.0};
  NormalizeOut normalize0 = NormalizeImpl(NormalizeIn{params0.angle, consts0.double_pi, consts0.pi});
  Normalize2fOut normalize2f0 = Normalize2fImpl(Normalize2fIn{params0.translation, consts1.two, consts1.one});
  RotationOut rotation0 = RotationImpl(RotationIn{position.position, normalize0.value});
  TranslationOut translation0 = TranslationImpl(TranslationIn{rotation0.position, normalize2f0.value});
  RepetitionOut repetition0 = RepetitionImpl(RepetitionIn{translation0.position, params0.space});
  GridOut grid0 = GridImpl(GridIn{repetition0.position, params0.thickness});
  CircleOut circle0 = CircleImpl(CircleIn{repetition0.position, params1.radius, params1.inner});
  ScaffoldIdOut scaffoldid0 = ScaffoldIdImpl(ScaffoldIdIn{grid0.distance, consts3.levelid0, circle0.distance, consts3.levelid1, position.view_scaffold});
  Level1Output level1 = Level1Output{grid0.distance, consts3.levelid0};
  Level2Output level2 = Level2Output{scaffoldid0.distance, scaffoldid0.levelid};
  Combine2Out combine20 = Combine2Impl(Combine2In{params0.translation, consts2.two, consts2.min_one, params0.space, consts2.half_one, consts2.zero});
  TranslationOut translation1 = TranslationImpl(TranslationIn{rotation0.position, combine20.value});
  RepetitionOut repetition1 = RepetitionImpl(RepetitionIn{translation1.position, params0.space});
  CircleOut circle2 = CircleImpl(CircleIn{repetition1.position, params2.radius, params2.inner});
  OverIdOut overid2 = OverIdImpl(OverIdIn{scaffoldid0.distance, scaffoldid0.levelid, circle2.distance, consts3.levelid2});
  Level3Output level3 = Level3Output{overid2.distance, overid2.levelid};
  return LevelsOutput{level1, level2, level3};
}

