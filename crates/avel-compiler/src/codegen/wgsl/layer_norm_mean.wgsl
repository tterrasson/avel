var sum = 0.0;
for (var e = {{first}}; e < {{depth}}u; e += {{step}}) {
    sum += {{read}};
}
let mean = {{sum}} / {{depth_f32}};
