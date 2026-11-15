let p{{axis}} = i32({{position}}) - {{padding}};
if (p{{axis}} < 0 || p{{axis}} >= {{size}}) {
    continue;
}
let i{{axis}} = u32(p{{axis}});
