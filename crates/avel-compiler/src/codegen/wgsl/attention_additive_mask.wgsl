let mask_value = {{read}};
if (bitcast<u32>(mask_value) == 0xff800000u) {
    continue;
}
score += mask_value;
