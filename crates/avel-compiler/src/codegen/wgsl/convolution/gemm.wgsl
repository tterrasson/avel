{{declarations}}

{{helpers}}

var<workgroup> tile_a: array<{{ty}}, {{area_a}}>;
var<workgroup> tile_b: array<{{ty}}, {{area_b}}>;

@compute @workgroup_size({{tile}}, {{tile}})
fn {{entry_point}}(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u) {
    let tile_row = (group.x + group.z * {{max_workgroups}}u) * {{bm}}u;
    let slice = group.y / {{tiles}}u;
    let g = (group.y % {{tiles}}u) / {{column_tiles}}u;
    let tile_column = (group.y % {{column_tiles}}u) * {{bn}}u;
    let start = slice * {{slice}}u;
    let end = min(start + {{slice}}u, {{reduction}}u);
    {{setup}}

    for (var r0 = start; r0 < end; r0 += {{tile}}u) {
        let ra = r0 + local.x;
        {{split_a}}
        {{load_a}}
        let rb = r0 + local.y;
        {{split_b}}
        {{load_b}}
        workgroupBarrier();

        for (var k = 0u; k < {{tile}}u; k++) {
            {{multiply}}
        }
        workgroupBarrier();
    }
    {{stores}}
}
