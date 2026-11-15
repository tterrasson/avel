{{declarations}}

{{helpers}}

@compute @workgroup_size({{lanes}})
fn {{entry_point}}(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
    // Uniform across the workgroup, so the barriers below stay uniform.
    let k = group.x + group.y * {{max_workgroups}}u;
    if (k >= {{rows}}u) {
        return;
    }
    let slice = group.z;
    let base = {{base}};
    let end = min((slice + 1u) * {{slice}}u, {{count}}u);
    var acc = 0.0;
    for (var r = slice * {{slice}}u + lane; r < end; r += {{lanes}}u) {
        acc += {{element}};
    }
    acc = workgroup_sum(acc, lane);
    if (lane == 0u) {
        {{store}}
    }
}
