export const datasets = [
    'All',
    'L26',
    'L22',
    'L23',
    'L30',
    'L27',
    'L28'
];

const L26_videos = ['All'];
for (let i = 200; i <= 219; i++) {
    L26_videos.push(`V${i}`);
}

const L22_videos = ['All'];
for (let i = 1; i <= 60; i++) {
    L22_videos.push(`V${String(i).padStart(3, '0')}`);
}

export const videos = {
    "All": ["All"],
    "L26": L26_videos,
    "L22": L22_videos,
};

export const rerankMethods = [
    'All',
    'rrf',
    'weighted_sum'
];

export const models = [
    'hf_clip_L',
    'hf_clip_H',
    'hf_siglip'
];