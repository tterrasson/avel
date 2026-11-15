let px = {{position_x}};
let py = {{position_y}};
let bx = floor(px);
let by = floor(py);
let wx = cubic_weights(px - bx);
let wy = cubic_weights(py - by);
var value = 0.0;
for (var j = 0; j < 4; j++) {
    let y = u32(clamp(i32(by) + j - 1, 0, {{last_y}}));
    var horizontal = 0.0;
    for (var i = 0; i < 4; i++) {
        let x = u32(clamp(i32(bx) + i - 1, 0, {{last_x}}));
        horizontal += {{read}} * wx[i];
    }
    value += horizontal * wy[j];
}
{{store}}
