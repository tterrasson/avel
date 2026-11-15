{{declarations}}

{{helpers}}

const N = {{n}}u;
var<workgroup> a: array<{{ty}}, {{area}}>;
var<workgroup> b: array<{{ty}}, {{area}}>;

@compute @workgroup_size({{tile}}, {{tile}})
fn {{entry_point}}(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let block = group.x + group.y * {{gx}}u;
    if (block >= {{groups}}u) {
        return;
    }
    {{batch}}
    let row = (block / {{columns}}u) * {{tile}}u + local.y;
    let col = (block % {{columns}}u) * {{tile}}u + local.x;
    let slot = local.y * {{tile}}u + local.x;
    var sum = 0.0;
    for (var base = 0u; base < {{inner}}u; base += {{tile}}u) {
        a[slot] = 0.0;
        b[slot] = 0.0;
        let ak = base + local.x;
        let bk = base + local.y;
        if (row < {{m}}u && ak < {{inner}}u) {
            a[slot] = {{lhs}};
        }
        if (bk < {{inner}}u && col < N) {
            b[slot] = {{rhs}};
        }
        workgroupBarrier();
        for (var k = 0u; k < {{tile}}u; k++) {
            sum += {{product}};
        }
        workgroupBarrier();
    }
    if (row < {{m}}u && col < N) {
        {{store}}
    }
}
