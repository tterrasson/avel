let y_start = {{y_start}};
let y_end = {{y_end}};
let x_start = {{x_start}};
let x_end = {{x_end}};
var sum = 0.0;
for (var y = y_start; y < y_end; y++) {
    for (var x = x_start; x < x_end; x++) {
        sum += {{read}};
    }
}
{{store}}
