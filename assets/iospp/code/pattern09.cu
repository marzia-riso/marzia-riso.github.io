#include <nodes.cuh>

struct Params1Input {
  float theta;
  float2 trans;
  float2 space;
  float thickness;
};

struct Params2Input {
  float radius1;
  float inner1;
};

struct Params3Input {
  float theta;
  float distance;
  float angle;
  float radius;
};

struct Params4Input {
  float len;
  float curvature;
  float thickness;
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
  float double_pi;
  float pi;
};

struct Consts3Input {
  float one;
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
};
__device__ __forceinline__ LevelsOutput LevelsImpl(PositionInput position, Params1Input params1, Params2Input params2, Params3Input params3, Params4Input params4) {
  Consts0Input consts0 = Consts0Input{6.28, 3.14};
  Consts1Input consts1 = Consts1Input{float2{2.0, 2.0}, float2{1.0, 1.0}};
  Consts2Input consts2 = Consts2Input{6.28, 3.14};
  Consts3Input consts3 = Consts3Input{1.0};
  Consts4Input consts4 = Consts4Input{0.0, 1.0, 2.0, 3.0, 4.0};
  NormalizeOut normalize_theta = NormalizeImpl(NormalizeIn{params1.theta, consts0.double_pi, consts0.pi});
  RotationOut rotation = RotationImpl(RotationIn{position.position, normalize_theta.value});
  Normalize2fOut normalize_trans = Normalize2fImpl(Normalize2fIn{params1.trans, consts1.two, consts1.one});
  TranslationOut translation = TranslationImpl(TranslationIn{rotation.position, normalize_trans.value});
  RepetitionOut repetition = RepetitionImpl(RepetitionIn{translation.position, params1.space});
  GridOut grid = GridImpl(GridIn{repetition.position, params1.thickness});
  Level1Output level1 = Level1Output{grid.distance, consts4.levelid0};
  CircleOut circle = CircleImpl(CircleIn{repetition.position, params2.radius1, params2.inner1});
  ScaffoldIdOut scaffold_id0 = ScaffoldIdImpl(ScaffoldIdIn{grid.distance, consts4.levelid0, circle.distance, consts4.levelid1, position.view_scaffold});
  Level2Output level2 = Level2Output{scaffold_id0.distance, scaffold_id0.levelid};
  NormalizeOut normalize_angle = NormalizeImpl(NormalizeIn{params3.theta, consts2.double_pi, consts2.pi});
  DivOut div_angle = DivImpl(DivIn{consts3.one, params3.angle});
  RoundOut round_angle = RoundImpl(RoundIn{div_angle.value});
  DivOut div_angle2 = DivImpl(DivIn{consts2.pi, round_angle.value});
  RadialRepetitionOut radial_rep = RadialRepetitionImpl(RadialRepetitionIn{repetition.position, normalize_angle.value, div_angle2.value, params3.distance});
  CircleOut rad_circle1 = CircleImpl(CircleIn{radial_rep.position1, params3.radius, consts3.one});
  CircleOut rad_circle2 = CircleImpl(CircleIn{radial_rep.position2, params3.radius, consts3.one});
  UnionIdOut radial_union = UnionIdImpl(UnionIdIn{rad_circle1.distance, consts4.levelid2, rad_circle2.distance, consts4.levelid2});
  OverIdOut over = OverIdImpl(OverIdIn{scaffold_id0.distance, scaffold_id0.levelid, radial_union.distance, radial_union.levelid});
  Level3Output level3 = Level3Output{over.distance, over.levelid};
  NormalizeOut normalize1 = NormalizeImpl(NormalizeIn{params4.curvature, consts0.double_pi, consts0.pi});
  JointOut rad_joint1 = JointImpl(JointIn{radial_rep.position1, params4.len, normalize1.value, params4.thickness});
  JointOut rad_joint2 = JointImpl(JointIn{radial_rep.position2, params4.len, normalize1.value, params4.thickness});
  UnionIdOut radial_joints = UnionIdImpl(UnionIdIn{rad_joint1.distance, consts4.levelid3, rad_joint2.distance, consts4.levelid3});
  ScaffoldIdOut scaffold_id1 = ScaffoldIdImpl(ScaffoldIdIn{radial_union.distance, radial_union.levelid, radial_joints.distance, radial_joints.levelid, position.view_scaffold});
  OverIdOut over2 = OverIdImpl(OverIdIn{over.distance, over.levelid, scaffold_id1.distance, scaffold_id1.levelid});
  Level4Output level4 = Level4Output{over2.distance, over2.levelid};
  return LevelsOutput{level1, level2, level3, level4};
}

