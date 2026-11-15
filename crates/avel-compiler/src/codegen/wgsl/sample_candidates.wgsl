{{coordinates}}
let base = {{logits_base}};
let index_base = {{indices_base}};
var maximum = {{first}};
for (var i = 1u; i < {{width}}u; i++) {
    maximum = avel_max(maximum, {{logit}});
}
var selected = 0u;
// Degenerate rows fall back to the first candidate.
if ((bitcast<u32>(maximum) & 0x7fffffffu) < 0x7f800000u) {
    var total = 0.0;
    for (var i = 0u; i < {{width}}u; i++) {
        total += exp({{logit}} - maximum);
    }
    // PCG RXS-M-XS hash of the seed and logical row. Use the top 24 bits
    // so conversion to f32 is exact and the uniform is strictly below 1.
    let state = (bitcast<u32>({{seed}}) + k * 0x9e3779b9u) * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let fraction = f32(((word >> 22u) ^ word) >> 8u) * (1.0 / 16777216.0);
    let threshold = fraction * total;
    var cumulative = 0.0;
    for (var i = 0u; i < {{width}}u; i++) {
        let weight = exp({{logit}} - maximum);
        cumulative += weight;
        if (weight > 0.0) { selected = i; }
        if (cumulative > threshold) { break; }
    }
}
let token = {{selected}};
{{store}}
