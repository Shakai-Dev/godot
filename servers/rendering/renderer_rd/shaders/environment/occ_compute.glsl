#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct ProcessVoxel {
	uint position;
	uint albedo_normal;
	uint emission;
	uint occlusion; // We will write to this: 4 bits per probe (8 probes total)
};

layout(set = 0, binding = 5, std430) restrict buffer ProcessVoxels {
	ProcessVoxel data[];
}
process_voxels;

#define REGION_SIZE 8

bool trace_ray_hdda(vec3 ray_pos, vec3 ray_dir, float p_distance, int p_cascade) {
	const int LEVEL_CASCADE = -1;
	const int LEVEL_REGION = 0;
	const int LEVEL_BLOCK = 1;
	const int LEVEL_VOXEL = 2;
	const int MAX_LEVEL = 3;

	const int fp_bits = 10;
	const int fp_block_bits = fp_bits + 2;
	const int fp_region_bits = fp_block_bits + 1;
	const int fp_cascade_bits = fp_region_bits + 4;

	bvec3 limit_dir = greaterThan(ray_dir, vec3(0.0));
	ivec3 step = mix(ivec3(0), ivec3(1), limit_dir);
	ivec3 ray_sign = ivec3(sign(ray_dir));

	ivec3 ray_dir_fp = ivec3(ray_dir * float(1 << fp_bits));

	bvec3 ray_zero = lessThan(abs(ray_dir), vec3(1.0 / 127.0));
	ivec3 inv_ray_dir_fp = ivec3(float(1 << fp_bits) / ray_dir);

	const ivec3 level_masks[MAX_LEVEL] = ivec3[](
			ivec3(1 << fp_region_bits) - ivec3(1),
			ivec3(1 << fp_block_bits) - ivec3(1),
			ivec3(1 << fp_bits) - ivec3(1));

	ivec3 region_offset_mask = (params.grid_size / REGION_SIZE) - ivec3(1);

	ivec3 limits[MAX_LEVEL];

	limits[LEVEL_REGION] = ((params.grid_size << fp_bits) - ivec3(1)) * step; // Region limit does not change, so initialize now.

	// Initialize to cascade
	int level = LEVEL_CASCADE;
	int cascade = p_cascade - 1;

	ivec3 cascade_base;
	ivec3 region_base;
	uvec2 block;
	bool hit = false;

	ivec3 pos;
	float distance = p_distance;
	ivec3 distance_limit;
	bool distance_limit_valid;

	while (true) {
		// This loop is written so there is only one single main iteration.
		// This ensures that different compute threads working on different
		// levels can still run together without blocking each other.

		if (level == LEVEL_VOXEL) {
			// The first level should be (in a worst case scenario) the most used
			// so it needs to appear first. The rest of the levels go from more to least used order.

			ivec3 block_local = (pos & level_masks[LEVEL_BLOCK]) >> fp_bits;
			uint block_index = uint(block_local.z * 16 + block_local.y * 4 + block_local.x);
			if (block_index < 32) {
				// Low 32 bits.
				if (bool(block.x & uint(1 << block_index))) {
					hit = true;
					break;
				}
			} else {
				// High 32 bits.
				block_index -= 32;
				if (bool(block.y & uint(1 << block_index))) {
					hit = true;
					break;
				}
			}
		} else if (level == LEVEL_BLOCK) {
			ivec3 block_local = (pos & level_masks[LEVEL_REGION]) >> fp_block_bits;
			block = imageLoad(voxel_cascades, region_base + block_local).rg;
			if (block != uvec2(0)) {
				// Have voxels inside
				level = LEVEL_VOXEL;
				limits[LEVEL_VOXEL] = pos - (pos & level_masks[LEVEL_BLOCK]) + step * (level_masks[LEVEL_BLOCK] + ivec3(1));
				continue;
			}
		} else if (level == LEVEL_REGION) {
			ivec3 region = pos >> fp_region_bits;
			region = (cascades.data[cascade].region_world_offset + region) & region_offset_mask; // Scroll to world
			region += cascade_base;
			bool region_used = imageLoad(voxel_region_cascades, region).r > 0;

			if (region_used) {
				// The region has contents.
				region_base = (region << 1);
				level = LEVEL_BLOCK;
				limits[LEVEL_BLOCK] = pos - (pos & level_masks[LEVEL_REGION]) + step * (level_masks[LEVEL_REGION] + ivec3(1));
				continue;
			}
		} else if (level == LEVEL_CASCADE) {
			// Return to global
			if (cascade >= p_cascade) {
				ray_pos = vec3(pos) / float(1 << fp_bits);
				ray_pos /= cascades.data[cascade].to_cell;
				ray_pos += cascades.data[cascade].offset;
				distance /= cascades.data[cascade].to_cell;
			}

			cascade++;
			if (cascade == params.max_cascades) {
				break;
			}

			ray_pos -= cascades.data[cascade].offset;
			ray_pos *= cascades.data[cascade].to_cell;

			pos = ivec3(ray_pos * float(1 << fp_bits));
			if (any(lessThan(pos, ivec3(0))) || any(greaterThanEqual(pos, params.grid_size << fp_bits))) {
				// Outside this cascade, go to next.
				continue;
			}

			distance *= cascades.data[cascade].to_cell;

			vec3 box = (vec3(params.grid_size * step) - ray_pos) / ray_dir;
			float advance_to_bounds = min(box.x, min(box.y, box.z));

			if (distance < advance_to_bounds) {
				// Can hit the distance in this cascade?
				distance_limit = pos + ray_sign * ivec3(distance * (1 << fp_bits));
				distance_limit_valid = true;
			} else {
				// We can't so subtract the advance to the end of the cascade.
				distance -= advance_to_bounds;
				distance_limit = ray_sign * 0xFFF << fp_bits; // Unreachable limit
				distance_limit_valid = false;
			}

			cascade_base = ivec3(0, int(params.grid_size.y / REGION_SIZE) * cascade, 0);
			level = LEVEL_REGION;
			continue;
		}

		// Fixed point, multi-level DDA.

		ivec3 mask = level_masks[level];
		ivec3 box = mask * step;
		ivec3 pos_diff = box - (pos & mask);
		ivec3 tv = mix((pos_diff * inv_ray_dir_fp), ivec3(0x7FFFFFFF), ray_zero) >> fp_bits;
		int t = min(tv.x, min(tv.y, tv.z));

		// The general idea here is that we _always_ need to increment to the closest next cell
		// (this is a DDA after all), so adv_box forces this increment for the minimum axis.

		ivec3 adv_box = pos_diff + ray_sign;
		ivec3 adv_t = (ray_dir_fp * t) >> fp_bits;

		pos += mix(adv_t, adv_box, equal(ivec3(t), tv));

		if (distance_limit_valid) { // Test against distance limit.
			bvec3 limit = lessThan(pos, distance_limit);
			bvec3 eq = equal(limit, limit_dir);
			if (!all(eq)) {
				break; // Reached limit, break.
			}
		}

		while (true) {
			bvec3 limit = lessThan(pos, limits[level]);
			bool inside = all(equal(limit, limit_dir));
			if (inside) {
				break;
			}
			level -= 1;
			if (level == LEVEL_CASCADE) {
				break;
			}
		}
	}

	return hit;
}

void main() {
	uint voxel_index = gl_GlobalInvocationID.x;
	if (voxel_index >= total_count) {
		return;
	}

	// Decode the position of the voxel
	ivec3 pos_i = ivec3((uvec3(process_voxels.data[voxel_index].position) >> uvec3(0, 7, 14)) & uvec3(0x7F));
	vec3 world_pos = (vec3(pos_i) + 0.5) / cascades.data[params.cascade].to_cell + cascades.data[params.cascade].offset;

	// Identify the 8 surrounding probes
	// In HD (High Density) mode, probe_cell_size is smaller, making these rays shorter and more precise
	ivec3 base_probe_pos = (pos_i / params.probe_cell_size) * params.probe_cell_size;

	uint packed_occlusion = 0;

	for (int i = 0; i < 8; i++) {
		ivec3 probe_offset = (ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1);
		ivec3 target_probe_i = base_probe_pos + (probe_offset * params.probe_cell_size);

		vec3 probe_world_pos = (vec3(target_probe_i) + 0.5) / cascades.data[params.cascade].to_cell + cascades.data[params.cascade].offset;

		vec3 dir = probe_world_pos - world_pos;
		float dist = length(dir);

		// Trace the ray using HDDAGI's high precision bitfield
		// We trace from the voxel towards the probe. If hit, the weight is set to 0.
		bool hit = trace_ray_hdda(world_pos, normalize(dir), dist, params.cascade);

		uint weight = hit ? 0 : 15; // 4-bit max value (15 = fully visible)
		packed_occlusion |= (weight << (i * 4));
	}

	process_voxels.data[voxel_index].occlusion = packed_occlusion;
}