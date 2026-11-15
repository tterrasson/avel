let base = {{base}};
var acc = {{identity}};
for (var r = {{first}}; r < {{depth}}u; r += {{step}}u) {
    {{update}}
}
let total = {{total}};
if ({{owner}}) {
    let destination = {{out_index}};
    let k = destination;
    {{store}}
    {{second}}
}
