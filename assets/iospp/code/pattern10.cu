#include <nodes.cuh>

struct Params1Input {
  float theta;
  float dist;
  float angle;
  float radius;
};

struct Params2Input {
  float thickness;
};

struct Params3Input {
  float dist3;
  float size3;
  float rotation3;
};

struct Params4Input {
  float dist4;
  float radius4;
  float inner4;
};

struct Params5Input {
  float radius5;
  float inner5;
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
  float pi;
};

struct Consts3Input {
  float one;
};

struct Consts4Input {
  float half_one;
};

struct Consts5Input {
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
  Consts1Input consts1 = Consts1Input{float2{0.0, 0.0}};
  Consts2Input consts2 = Consts2Input{1.0, 3.14};
  Consts3Input consts3 = Consts3Input{1.0};
  Consts4Input consts4 = Consts4Input{0.5};
  Consts5Input consts5 = Consts5Input{0.0, 1.0, 2.0, 3.0, 4.0};
  NormalizeOut normalize_theta = NormalizeImpl(NormalizeIn{params1.theta, consts0.double_pi, consts0.pi});
  DivOut div_angle = DivImpl(DivIn{consts2.one, params1.angle});
  RoundOut round_angle = RoundImpl(RoundIn{div_angle.value});
  DivOut angle = DivImpl(DivIn{consts2.pi, round_angle.value});
  TranslationOut translation = TranslationImpl(TranslationIn{position.position, consts1.trans});
  RadialRepetitionOut radial_rep = RadialRepetitionImpl(RadialRepetitionIn{translation.position, normalize_theta.value, angle.value, params1.dist});
  CircleOut rad_circle1 = CircleImpl(CircleIn{radial_rep.position1, params1.radius, consts3.one});
  CircleOut rad_circle2 = CircleImpl(CircleIn{radial_rep.position2, params1.radius, consts3.one});
  UnionIdOut radial = UnionIdImpl(UnionIdIn{rad_circle1.distance, consts5.levelid0, rad_circle2.distance, consts5.levelid0});
  Level1Output level1 = Level1Output{radial.distance, consts5.levelid0};
  RadialLineOut radial_line = RadialLineImpl(RadialLineIn{translation.position, normalize_theta.value, angle.value, params2.thickness});
  ScaffoldIdOut scaffoldid1 = ScaffoldIdImpl(ScaffoldIdIn{radial.distance, consts5.levelid0, radial_line.distance, consts5.levelid1, position.view_scaffold});
  Level2Output level2 = Level2Output{scaffoldid1.distance, scaffoldid1.levelid};
  RadialRepetitionOut radial_rep1 = RadialRepetitionImpl(RadialRepetitionIn{translation.position, normalize_theta.value, angle.value, params3.dist3});
  RotationOut radial_rep_rot1 = RotationImpl(RotationIn{radial_rep1.position1, params3.rotation3});
  RotationOut radial_rep_rot2 = RotationImpl(RotationIn{radial_rep1.position2, params3.rotation3});
  BoxOut rad_box1 = BoxImpl(BoxIn{radial_rep_rot1.position, params3.size3});
  BoxOut rad_box2 = BoxImpl(BoxIn{radial_rep_rot2.position, params3.size3});
  UnionIdOut radial_box = UnionIdImpl(UnionIdIn{rad_box1.distance, consts5.levelid2, rad_box2.distance, consts5.levelid2});
  ScaffoldIdOut scaffoldid2 = ScaffoldIdImpl(ScaffoldIdIn{scaffoldid1.distance, scaffoldid1.levelid, radial_box.distance, radial_box.levelid, position.view_scaffold});
  Level3Output level3 = Level3Output{scaffoldid2.distance, scaffoldid2.levelid};
  AddOut offset_theta = AddImpl(AddIn{normalize_theta.value, consts4.half_one});
  RadialRepetitionOut radial_rep2 = RadialRepetitionImpl(RadialRepetitionIn{translation.position, offset_theta.value, angle.value, params4.dist4});
  CircleOut rad_circle3 = CircleImpl(CircleIn{radial_rep2.position1, params4.radius4, params4.inner4});
  CircleOut rad_circle4 = CircleImpl(CircleIn{radial_rep2.position2, params4.radius4, params4.inner4});
  UnionIdOut radial_circle = UnionIdImpl(UnionIdIn{rad_circle3.distance, consts5.levelid3, rad_circle4.distance, consts5.levelid3});
  OverIdOut over3 = OverIdImpl(OverIdIn{scaffoldid2.distance, scaffoldid2.levelid, radial_circle.distance, radial_circle.levelid});
  Level4Output level4 = Level4Output{over3.distance, over3.levelid};
  CircleOut outer_circle = CircleImpl(CircleIn{translation.position, params5.radius5, params5.inner5});
  OverIdOut over4 = OverIdImpl(OverIdIn{over3.distance, over3.levelid, outer_circle.distance, consts5.levelid4});
  Level5Output level5 = Level5Output{over4.distance, over4.levelid};
  return LevelsOutput{level1, level2, level3, level4, level5};
}

