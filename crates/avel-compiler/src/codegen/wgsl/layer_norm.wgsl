let base = {{base}};
{{mean}}
var sq = 0.0;
for (var e = {{first}}; e < {{depth}}u; e += {{step}}) {
    let v = {{read}} - mean;
    sq += v * v;
}
let rstd = inverseSqrt({{sum}} / {{depth_f32}} + {{eps}});
for (var e = {{first}}; e < {{depth}}u; e += {{step}}) {
    {{store}}
}
