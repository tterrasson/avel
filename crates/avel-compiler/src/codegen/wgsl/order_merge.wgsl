let row = k / ({{groups}}u * {{keep}}u);
let block = (k / {{keep}}u) % {{groups}}u;
let diagonal = k % {{keep}}u;
let left = (row * {{old_groups}}u + 2u * block) * {{old_keep}}u;
let right = left + {{old_keep}}u;
let start = block * (2u * {{span}}u);
let remaining = {{width}}u - start;
let left_count = min(remaining, {{old_keep}}u);
let right_count = min(max(remaining, {{span}}u) - {{span}}u, {{old_keep}}u);
let base = {{base}};
var chosen = 0xffffffffu;
if (diagonal < left_count + right_count) {
    // Find the split of the prefix of length diagonal between the two runs.
    var low = max(diagonal, right_count) - right_count;
    var high = min(diagonal, left_count);
    while (low < high) {
        let mid = (low + high) >> 1u;
        let a = u32({{index_left}});
        let b = u32({{index_right}});
        if ({{compare}}) { low = mid + 1u; } else { high = mid; }
    }
    if (low < left_count) { chosen = u32({{selected_left}}); }
    if (diagonal - low < right_count) {
        let a = u32({{selected_right}});
        let b = chosen;
        if (b == 0xffffffffu) {
            chosen = a;
        } else if ({{compare}}) {
            chosen = a;
        }
    }
}
{{store}}
