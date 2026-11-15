{{declarations}}
{{helpers}}
var<workgroup> values: array<{{ty}}, {{tile}}>;
var<workgroup> ids: array<u32, {{tile}}>;

@compute @workgroup_size({{lanes}})
fn {{entry_point}}(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
    let block = group.x + group.y * {{max_workgroups}}u;
    if (block >= {{groups}}u) { return; }
    let row = block / {{blocks}}u;
    let base = {{base}};
    let start = (block % {{blocks}}u) * {{tile}}u;
    for (var i = lane; i < {{tile}}u; i += {{lanes}}u) {
        let position = start + i;
        ids[i] = 0xffffffffu;
        if (position < {{width}}u) {
            values[i] = {{read}};
            ids[i] = position;
        }
    }
    workgroupBarrier();
    for (var run = 2u; run <= {{tile}}u; run <<= 1u) {
        for (var stride = run >> 1u; stride > 0u; stride >>= 1u) {
            for (var i = lane; i < {{tile}}u; i += {{lanes}}u) {
                let j = i ^ stride;
                if (i < j && ({{compare}} != ((i & run) == 0u))) {
                    let value = values[i];
                    let id = ids[i];
                    values[i] = values[j];
                    ids[i] = ids[j];
                    values[j] = value;
                    ids[j] = id;
                }
            }
            workgroupBarrier();
        }
    }
    for (var i = lane; i < {{keep}}u; i += {{lanes}}u) {
        let k = block * {{keep}}u + i;
        {{store}}
    }
}
