#include <nodes.cuh>

struct Params0Input {
  float scale;
  float theta;
  float2 trans;
  float thickness;
};

struct Params1Input {
  float thickness;
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
  float2 space;
};

struct Consts3Input {
  float levelid0;
  float levelid1;
};

struct LevelsOutput {
  Level1Output level1;
  Level2Output level2;
};
__device__ __forceinline__ LevelsOutput LevelsImpl(PositionInput position, Params0Input params1, Params1Input params2) {
  Consts0Input consts0 = Consts0Input{6.28, 3.14};
  Consts1Input consts1 = Consts1Input{float2{2.0, 2.0}, float2{1.0, 1.0}};
  Consts2Input consts2 = Consts2Input{float2{1.0, 1.0}};
  Consts3Input consts3 = Consts3Input{0.0, 1.0};
  ScaleOut scale0 = ScaleImpl(ScaleIn{position.position, params1.scale});
  NormalizeOut normalize0 = NormalizeImpl(NormalizeIn{params1.theta, consts0.double_pi, consts0.pi});
  Normalize2fOut normalize2f0 = Normalize2fImpl(Normalize2fIn{params1.trans, consts1.two, consts1.one});
  RotationOut rotation0 = RotationImpl(RotationIn{scale0.position, normalize0.value});
  TranslationOut translation0 = TranslationImpl(TranslationIn{rotation0.position, normalize2f0.value});
  RepetitionOut repetition0 = RepetitionImpl(RepetitionIn{translation0.position, consts2.space});
  GridOut grid0 = GridImpl(GridIn{repetition0.position, params1.thickness});
  MulOut mul0 = MulImpl(MulIn{grid0.distance, params1.scale});
  Level1Output level1 = Level1Output{mul0.value, consts3.levelid0};
  TruchetOut truchet0 = TruchetImpl(TruchetIn{translation0.position, params2.thickness, params2.rotation});
  MulOut mul1 = MulImpl(MulIn{truchet0.distance, params1.scale});
  ScaffoldIdOut scaffoldid0 = ScaffoldIdImpl(ScaffoldIdIn{mul0.value, consts3.levelid0, mul1.value, consts3.levelid1, position.view_scaffold});
  Level2Output level2 = Level2Output{scaffoldid0.distance, scaffoldid0.levelid};
  return LevelsOutput{level1, level2};
}

