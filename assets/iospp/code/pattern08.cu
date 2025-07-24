#include <nodes.cuh>

struct Params1Input {
  float radius1;
  float inner1;
  float step;
};

struct Params2Input {
  float theta;
  float angle;
  float distance;
  float radius;
};

struct Params3Input {
  float len;
  float curvature;
  float thickness;
};

struct Consts0Input {
  float double_pi;
  float pi;
};

struct Consts1Input {
  float2 trans;
};

struct Consts2Input {
  float one;
};

struct Consts3Input {
  float one;
};

struct Consts4Input {
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
  Consts1Input consts1 = Consts1Input{float2{0.0, 0.0}};
  Consts2Input consts2 = Consts2Input{1.0};
  Consts3Input consts3 = Consts3Input{1.0};
  Consts4Input consts4 = Consts4Input{0.0, 1.0, 2.0};
  TranslationOut translation = TranslationImpl(TranslationIn{position.position, consts1.trans});
  CircleOnionOut circleonion = CircleOnionImpl(CircleOnionIn{translation.position, params1.radius1, params1.inner1, params1.step});
  Level1Output level1 = Level1Output{circleonion.distance, consts4.levelid0};
  DivOut div_angle = DivImpl(DivIn{consts2.one, params2.angle});
  RoundOut round_angle = RoundImpl(RoundIn{div_angle.value});
  DivOut div_angle2 = DivImpl(DivIn{consts0.pi, round_angle.value});
  RadialRepetitionOut radial_rep = RadialRepetitionImpl(RadialRepetitionIn{translation.position, params2.theta, div_angle2.value, params2.distance});
  CircleOut rad_circle1 = CircleImpl(CircleIn{radial_rep.position1, params2.radius, consts3.one});
  CircleOut rad_circle2 = CircleImpl(CircleIn{radial_rep.position2, params2.radius, consts3.one});
  UnionIdOut radial_union = UnionIdImpl(UnionIdIn{rad_circle1.distance, consts4.levelid1, rad_circle2.distance, consts4.levelid1});
  OverIdOut over1 = OverIdImpl(OverIdIn{circleonion.distance, consts4.levelid0, radial_union.distance, radial_union.levelid});
  Level2Output level2 = Level2Output{over1.distance, over1.levelid};
  NormalizeOut normalize1 = NormalizeImpl(NormalizeIn{params3.curvature, consts0.double_pi, consts0.pi});
  JointOut rad_joint1 = JointImpl(JointIn{radial_rep.position1, params3.len, normalize1.value, params3.thickness});
  JointOut rad_joint2 = JointImpl(JointIn{radial_rep.position2, params3.len, normalize1.value, params3.thickness});
  UnionIdOut radial_joints = UnionIdImpl(UnionIdIn{rad_joint1.distance, consts4.levelid2, rad_joint2.distance, consts4.levelid2});
  ScaffoldIdOut scaffoldid1 = ScaffoldIdImpl(ScaffoldIdIn{over1.distance, over1.levelid, radial_joints.distance, radial_joints.levelid, position.view_scaffold});
  OverIdOut over3 = OverIdImpl(OverIdIn{circleonion.distance, consts4.levelid0, scaffoldid1.distance, scaffoldid1.levelid});
  Level3Output level3 = Level3Output{over3.distance, over3.levelid};
  return LevelsOutput{level1, level2, level3};
}

