var<workgroup> lanes: array<f32, {{lanes}}>;

fn workgroup_sum(value: f32, lane: u32) -> f32 {
    lanes[lane] = value;
    workgroupBarrier();
    for (var width = {{half}}u; width > 0u; width >>= 1u) {
        if (lane < width) {
            lanes[lane] += lanes[lane + width];
        }
        workgroupBarrier();
    }
    let total = lanes[0];
    // Every lane has read the total before the memory is reused.
    workgroupBarrier();
    return total;
}

