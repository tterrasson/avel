var m = {{lowest}};
var l = 0.0;
var acc: array<f32, {{value_depth}}>;
for (var d = 0u; d < {{value_depth}}u; d++) {
    acc[d] = 0.0;
}

{{prepare_query}}
let key_bound = {{key_bound}};
for (var j = 0u; j < key_bound; j++) {
    var score = 0.0;
    for (var d = 0u; d < {{depth}}u; d++) {
        score += {{query_read}} * {{key_read}};
    }
    score = {{scale}} * score;
    {{mask}}

    let m_new = max(m, score);
    let correction = exp(m - m_new);
    let p = exp(score - m_new);
    l = l * correction + p;
    for (var d = 0u; d < {{value_depth}}u; d++) {
        acc[d] = acc[d] * correction + p * {{value_read}};
    }
    m = m_new;
}

for (var d = 0u; d < {{value_depth}}u; d++) {
    {{store}}
}
