// RDNA1 compatibility path. Tokens are 1-GiB descriptor-page offsets, NEVER
// physical addresses. Constant descriptor indices avoid requiring descriptor
// indexing/non-uniform-indexing features from the llama device. Bind all slots.
#ifdef KERYX_DESCRIPTOR_MEMORY
layout(std430, binding = 5) buffer Pages { uint w[]; } PAGES[16];
uint page_read32(uint64_t address) {
    uint slot = uint(address >> 30), word = uint(address & 0x3fffffffUL) >> 2;
    switch (slot) {
    case 0u: return PAGES[0].w[word];
    case 1u: return PAGES[1].w[word];
    case 2u: return PAGES[2].w[word];
    case 3u: return PAGES[3].w[word];
    case 4u: return PAGES[4].w[word];
    case 5u: return PAGES[5].w[word];
    case 6u: return PAGES[6].w[word];
    case 7u: return PAGES[7].w[word];
    case 8u: return PAGES[8].w[word];
    case 9u: return PAGES[9].w[word];
    case 10u: return PAGES[10].w[word];
    case 11u: return PAGES[11].w[word];
    case 12u: return PAGES[12].w[word];
    case 13u: return PAGES[13].w[word];
    case 14u: return PAGES[14].w[word];
    case 15u: return PAGES[15].w[word];
    default: return 0u; // host rejects tokens outside the descriptor table
    }
}
void page_write32(uint64_t address, uint value) {
    uint slot = uint(address >> 30), word = uint(address & 0x3fffffffUL) >> 2;
    switch (slot) {
    case 0u: PAGES[0].w[word] = value; break;
    case 1u: PAGES[1].w[word] = value; break;
    case 2u: PAGES[2].w[word] = value; break;
    case 3u: PAGES[3].w[word] = value; break;
    case 4u: PAGES[4].w[word] = value; break;
    case 5u: PAGES[5].w[word] = value; break;
    case 6u: PAGES[6].w[word] = value; break;
    case 7u: PAGES[7].w[word] = value; break;
    case 8u: PAGES[8].w[word] = value; break;
    case 9u: PAGES[9].w[word] = value; break;
    case 10u: PAGES[10].w[word] = value; break;
    case 11u: PAGES[11].w[word] = value; break;
    case 12u: PAGES[12].w[word] = value; break;
    case 13u: PAGES[13].w[word] = value; break;
    case 14u: PAGES[14].w[word] = value; break;
    case 15u: PAGES[15].w[word] = value; break;
    }
}
uint64_t page_read64(uint64_t address) {
    return uint64_t(page_read32(address)) | (uint64_t(page_read32(address + 4UL)) << 32);
}
uvec4 page_read128(uint64_t address) {
    return uvec4(page_read32(address), page_read32(address + 4UL),
                 page_read32(address + 8UL), page_read32(address + 12UL));
}
void page_write64(uint64_t address, uint64_t value) {
    page_write32(address, uint(value)); page_write32(address + 4UL, uint(value >> 32));
}
#endif
