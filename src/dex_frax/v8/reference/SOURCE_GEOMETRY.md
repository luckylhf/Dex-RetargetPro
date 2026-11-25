# Source surface geometry

lafan1_surfaces.json is derived locally from Ubisoft's official fbx/model_skeleton.fbx (SHA256 3d9349b5ec686248be6ab73ea3b918f486016a0d22e4b4b80d5418c8f3a608aa). Bone-local surfaces use dominant skin weights, with extra fingers folded into the original BVH Hand. Shoe landmarks and the sole normal are fixed reference-surface measurements, not fit per motion.

soma_surfaces.json is derived locally from NVIDIA soma_base_skel_minimal.usd (SHA256 4074b8d830f53af246b4d93d1871af38c8825e6b6b6bbaf034d4612673c89266), using its geometry bind and skeleton bind transforms. Official SOMA foot landmark configuration remains separate and unchanged.

The converter uses these JSON files as rigid per-bone convex envelopes. The original dataset and model terms and NVIDIA notices apply.

## Runtime setup

Keep the JSON filenames above and set `DEX_FRAX_SOURCE_GEOMETRY` to their parent directory. An existing authorized project copy can be referenced directly with the same environment variable. The LAFAN route requires the LAFAN table and the SOMA route requires the SOMA table. A wheel or source distribution omits both tables. Their effective content hashes are recorded with the other implementation inputs. A source checkout also looks in its ignored `.local-data/source_geometry/` directory.
