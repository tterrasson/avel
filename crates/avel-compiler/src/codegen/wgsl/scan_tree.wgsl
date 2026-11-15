var<workgroup> scan_partials: array<{{ty}}, {{lanes}}>;
var<workgroup> scan_total: {{ty}};

// Blelloch exclusive scan over lane totals, with O(lanes) additions.
fn scan_lanes(value: {{ty}}, lane: u32) -> vec2<{{ty}}> {
    scan_partials[lane] = value;
    workgroupBarrier();
    for (var stride = 1u; stride < {{lanes}}u; stride <<= 1u) {
        let index = (lane + 1u) * stride * 2u - 1u;
        if (index < {{lanes}}u) {
            scan_partials[index] += scan_partials[index - stride];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        scan_total = scan_partials[{{lanes}}u - 1u];
        scan_partials[{{lanes}}u - 1u] = {{ty}}(0);
    }
    workgroupBarrier();
    for (var stride = {{lanes}}u >> 1u; stride != 0u; stride >>= 1u) {
        let index = (lane + 1u) * stride * 2u - 1u;
        if (index < {{lanes}}u) {
            let left = scan_partials[index - stride];
            scan_partials[index - stride] = scan_partials[index];
            scan_partials[index] += left;
        }
        workgroupBarrier();
    }
    let result = vec2<{{ty}}>(scan_partials[lane], scan_total);
    workgroupBarrier();
    return result;
}
