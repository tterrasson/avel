{{declarations}}

{{helpers}}

@compute @workgroup_size({{lanes}})
fn {{entry_point}}(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
    // Uniform across the workgroup, so the barriers in the body stay uniform.
    let k = group.x + group.y * {{max_workgroups}}u;
    if (k >= {{rows}}u) {
        return;
    }
    {{body}}
}
