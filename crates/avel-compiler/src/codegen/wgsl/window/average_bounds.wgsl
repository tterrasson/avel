let start{{axis}} = i32(o{{out_axis}} * {{stride}}u) - {{padding}};
let end{{axis}} = min(start{{axis}} + {{kernel}}, {{padded_size}});
