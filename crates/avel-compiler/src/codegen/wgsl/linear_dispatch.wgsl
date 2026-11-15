{{declarations}}

{{helpers}}

@compute @workgroup_size({{workgroup_size}})
fn {{entry_point}}(@builtin(global_invocation_id) id: vec3u) {
    let k = id.x + id.y * {{row_stride}}u;
    if (k >= {{invocations}}u) {
        return;
    }
    {{body}}
}
