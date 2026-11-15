let base = {{base}};
let out_base = {{out_base}};
var prefix = {{zero}};
for (var r = 0u; r < {{depth}}u; r++) {
    prefix += {{read}};
    {{store}}
}
