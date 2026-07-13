# P1 Input Diagnostics

- Sample windows: 32
- m0_shape_differs_from_m1: True
- m1_projected_text_nonzero: True
- z0_projected_text_all_zero: True
- s0_raw_text_hash_differs_from_m1: True
- m1_combined_differs_from_z0: True
- m1_combined_differs_from_s0: True
- embeddings_nonconstant: True

## Shapes
- M0_Numerical-only: combined [32, 24, 1], projected [32, 24, 0]
- M1_TaTS-Raw-Text: combined [32, 24, 13], projected [32, 24, 12]
- Z0_Zero-Text-Control: combined [32, 24, 13], projected [32, 24, 12]
- S0_Shuffled-Text-Diagnostic: combined [32, 24, 13], projected [32, 24, 12]

## Embedding Statistics
- variance: 67.16938018798828
- mean_norm: 227.0136260986328
- std_norm: 12.695108413696289
- pairwise_cosine_similarity_mean: 0.9985175728797913
- unique_embedding_ratio: 0.17838541666666666
