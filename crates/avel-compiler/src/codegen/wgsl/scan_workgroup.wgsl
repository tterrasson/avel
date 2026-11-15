let base = {{base}};
let out_base = {{out_base}};
var carry = {{ty}}(0);
for (var tile = 0u; tile < {{depth}}u; tile += {{lanes}}u * 4u) {
    let start = tile + lane * 4u;
    var values: array<{{ty}}, 4>;
    var subtotal = {{ty}}(0);
    for (var j = 0u; j < 4u; j++) {
        let r = start + j;
        if (r < {{depth}}u) {
            subtotal += {{read}};
        }
        values[j] = subtotal;
    }
    let scanned = scan_lanes(subtotal, lane);
    for (var j = 0u; j < 4u; j++) {
        let r = start + j;
        if (r < {{depth}}u) {
            let prefix = carry + scanned.x + values[j];
            {{store}}
        }
    }
    carry += scanned.y;
}
