let base = {{base}};
let out_base = {{out_base}};
var maximum = avel_bits(0xff800000u);
for (var r = {{first}}; r < {{depth}}u; r += {{step}}u) {
    maximum = avel_max(maximum, {{read}});
}
let row_maximum = {{maximum}};
let finite_max = (bitcast<u32>(row_maximum) & 0x7fffffffu) < 0x7f800000u;
var sum = 0.0;
if (finite_max) {
    for (var r = {{first}}; r < {{depth}}u; r += {{step}}u) {
        sum += exp({{read}} - row_maximum);
    }
}
let row_sum = select(1.0, {{sum}}, finite_max);
{{finish}}
for (var r = {{first}}; r < {{depth}}u; r += {{step}}u) {
    var result = avel_bits(0x7fc00000u);
    if (finite_max) {
        result = {{result}};
    }
    {{store}}
}
